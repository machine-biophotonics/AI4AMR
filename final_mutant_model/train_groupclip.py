#!/usr/bin/env python3
"""
GroupCLIP: Cross-modal supervised contrastive learning for drug and mutant imaging.

Trains a shared embedding space where drugs (with MOA labels) and mutants
(with pathway labels) with similar cellular effects cluster together.

Architecture (DualMILEncoder):
  Shared EfficientNet-B0 backbone
  ├── Drug attention pool + classifier
  └── Mutant attention pool + classifier
  └── Shared GroupCLIP projection head

Loss: α · GroupCLIP + β · CE_drug + γ · CE_mutant

Reference: "Group Contrastive Learning for Weakly Paired Multimodal Data"
           Gorla et al., arXiv:2602.04021, 2026
"""

import os
import sys
import json
import time
import argparse
import glob
import re
import random
import warnings
warnings.filterwarnings("ignore", message=".*Not enough SMs to use max_autotune_gemm.*")

os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from collections import defaultdict, Counter
from datetime import datetime
from tqdm import tqdm
from functools import partial
from typing import Optional

from mil_model import DualMILEncoder, GROOVEModel, MultiCropDataset, extract_well_from_filename
from supcon_loss import GroupCLIPLoss

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='GroupCLIP: Cross-modal supervised contrastive learning')

# Data
parser.add_argument('--test_plate', type=str, default='P6', help='Test plate')
parser.add_argument('--run_all_folds', action='store_true', default=False)
parser.add_argument('--data_root', type=str, default=None)

# Architecture
parser.add_argument('--backbone', type=str, default='efficientnet_b0')
parser.add_argument('--pooling', type=str, default='attention', choices=['attention', 'simple_attention', 'mean', 'max'])
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--num_channels', type=int, default=1)
parser.add_argument('--pretrained', type=str, default='imagenet')

# Training
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=16, help='Per-modality batch size (total batch = 2x)')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--label_smoothing', type=float, default=0.1)

# Mode
parser.add_argument('--mode', type=str, default='groupclip', choices=['groupclip', 'groove'],
                    help='Training mode: groupclip or groove (with reconstruction + backtranslation)')

# Loss weights
parser.add_argument('--gc_weight', type=float, default=1.0, help='GroupCLIP loss weight')
parser.add_argument('--ce_weight', type=float, default=1.0, help='Combined CE loss weight')
parser.add_argument('--recon_weight', type=float, default=1.0, help='Reconstruction loss weight (groove mode)')
parser.add_argument('--bt_weight', type=float, default=1.0, help='Backtranslation loss weight (groove mode)')
parser.add_argument('--gc_temp', type=float, default=0.07, help='GroupCLIP temperature')
parser.add_argument('--recon_hidden', type=int, default=None, help='Decoder hidden dim (default: None = linear)')

# Crop extraction
parser.add_argument('--neighborhood', type=int, default=3, choices=[3, 5])
parser.add_argument('--grid_size', type=int, default=12)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--extraction_mode', type=str, default='neighborhood')

# Group balancing
parser.add_argument('--samples_per_group', type=int, default=8, help='Samples per group per batch')
parser.add_argument('--min_group_samples', type=int, default=2, help='Min samples per group to include')

args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Load mappings
# ---------------------------------------------------------------------------
with open(os.path.join(SCRIPT_DIR, 'group_mapping.json'), 'r') as f:
    GROUP_MAP = json.load(f)

with open(os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json'), 'r') as f:
    IC50_DATA = json.load(f)

with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json'), 'r') as f:
    MUTANT_DATA = json.load(f)

ANTIBIOTIC_TO_MOA = GROUP_MAP['ANTIBIOTIC_TO_MOA']
MOA_TO_GROUP = GROUP_MAP['MOA_TO_GROUP']
GENE_TO_PATHWAY = GROUP_MAP['GENE_TO_PATHWAY']
PATHWAY_TO_GROUP = GROUP_MAP['PATHWAY_TO_GROUP']

# ---------------------------------------------------------------------------
# Helper: extract labels and groups
# ---------------------------------------------------------------------------
def get_drug_info(plate_key, well):
    """Get (drug_class_label, moa, group_id) for a drug well."""
    if plate_key not in IC50_DATA or well not in IC50_DATA[plate_key]:
        return None
    info = IC50_DATA[plate_key][well]
    antibiotic = info.get('antibiotic', '')
    ic50 = info.get('ic50_multiple', '')
    if not antibiotic or not ic50:
        return None
    if ic50 == 'control':
        label = 'control'
        moa = 'Control'
    else:
        ic50_str = ic50 if 'x' in ic50 else f"{ic50}x"
        label = f"{antibiotic.replace(' ', '_')}_{ic50_str}"
        ab_clean = antibiotic.replace(' ', '_')
        moa = ANTIBIOTIC_TO_MOA.get(ab_clean, 'Unknown')
    group_id = MOA_TO_GROUP.get(moa, -1)
    return label, moa, group_id


def get_mutant_info(plate_key, well):
    """Get (mutant_label, pathway, group_id) for a mutant well."""
    row = well[0].upper()
    col = str(int(well[1:]))
    try:
        info = MUTANT_DATA[plate_key][row][col]
    except KeyError:
        return None
    label = info.get('id', '')
    gene = label.rsplit('_', 1)[0] if '_' in label else label
    pathway = GENE_TO_PATHWAY.get(gene, 'WT/NC' if 'WT' in label else 'Unknown')
    group_id = PATHWAY_TO_GROUP.get(pathway, -1)
    return label, pathway, group_id


def extract_drug_class_name(drug_label):
    """Get the antibiotic name from a drug class label like 'Ciprofloxacin_2x'."""
    if drug_label == 'control':
        return 'control'
    parts = drug_label.rsplit('_', 1)
    return parts[0]


# ---------------------------------------------------------------------------
# Build per-modality path lists with group labels
# ---------------------------------------------------------------------------
def collect_samples(plate, data_mode, drug_no_concentration=False):
    """
    Collect all samples for a plate, returning:
      drug_samples: [(path, class_label, group_id), ...]
      mutant_samples: [(path, class_label, group_id), ...]
    """
    plate_key = f"P{plate.split('_')[-1]}"
    drug_samples = []
    mutant_samples = []
    
    # Drug data
    drug_base = os.path.join(os.path.dirname(SCRIPT_DIR), 'Drugs_Data', plate_key)
    if data_mode in ['drug', 'both'] and os.path.exists(drug_base):
        for pattern in ['*.tif', '*.tiff']:
            for path in glob.glob(os.path.join(drug_base, '**', pattern), recursive=True):
                well = extract_well_from_filename(os.path.basename(path))
                if not well:
                    continue
                info = get_drug_info(plate_key, well)
                if info is not None:
                    label, moa, group_id = info
                    if group_id >= 0:
                        drug_samples.append((path, label, group_id))
    
    # Mutant data
    mutant_base = os.path.join(os.path.dirname(SCRIPT_DIR), 'Mutants_Data', plate_key)
    if data_mode in ['mutant', 'both'] and os.path.exists(mutant_base):
        for pattern in ['*.tif', '*.tiff']:
            for path in glob.glob(os.path.join(mutant_base, '**', pattern), recursive=True):
                well = extract_well_from_filename(os.path.basename(path))
                if not well:
                    continue
                info = get_mutant_info(plate_key, well)
                if info is not None:
                    label, pathway, group_id = info
                    if group_id >= 0:
                        mutant_samples.append((path, label, group_id))
    
    return drug_samples, mutant_samples


# ---------------------------------------------------------------------------
# Balanced Group Dataset for GroupCLIP
# ---------------------------------------------------------------------------
class GroupCLIPDataset(Dataset):
    """
    Multi-modal dataset that returns balanced group batches.
    
    Stores samples by group_id. At each epoch, rebalances so each group
    contributes equally, mixing drug and mutant samples when available.
    
    Each item returns:
        images, class_label, group_label, modality_id
    where modality_id=0 for drug, 1 for mutant.
    """
    
    def __init__(self, drug_samples, mutant_samples, transform_fn,
                 neighborhood=3, grid_size=12, crop_size=224,
                 extraction_mode='neighborhood', num_channels=1,
                 raster_crop_size=500, raster_resize_size=256,
                 samples_per_group=8):
        self.transform_fn = transform_fn
        self.neighborhood = neighborhood
        self.grid_size = grid_size
        self.crop_size = crop_size
        self.extraction_mode = extraction_mode
        self.num_channels = num_channels
        self.raster_crop_size = raster_crop_size
        self.raster_resize_size = raster_resize_size
        self.samples_per_group = samples_per_group
        
        # Build all classes and group mapping
        self.drug_classes = sorted(set(s[1] for s in drug_samples))
        self.mutant_classes = sorted(set(s[1] for s in mutant_samples))
        self.drug_class_to_idx = {c: i for i, c in enumerate(self.drug_classes)}
        self.mutant_class_to_idx = {c: i for i, c in enumerate(self.mutant_classes)}
        
        # Build per-group index
        self.grouped_samples = defaultdict(list)
        for path, label, gid in drug_samples:
            cidx = self.drug_class_to_idx[label]
            self.grouped_samples[gid].append((path, cidx, gid, 0))  # modality=0=drug
        for path, label, gid in mutant_samples:
            cidx = self.mutant_class_to_idx[label]
            self.grouped_samples[gid].append((path, cidx, gid, 1))  # modality=1=mutant
        
        self.group_ids = sorted(self.grouped_samples.keys())
        print(f"GroupCLIP dataset: {len(drug_samples)} drug, {len(mutant_samples)} mutant")
        print(f"Groups: {self.group_ids}")
        for gid in self.group_ids:
            drug_count = sum(1 for s in self.grouped_samples[gid] if s[3] == 0)
            mut_count = sum(1 for s in self.grouped_samples[gid] if s[3] == 1)
            print(f"  Group {gid}: {drug_count} drug + {mut_count} mutant = {len(self.grouped_samples[gid])}")
        
        # Precompute image size for crop extraction
        from PIL import Image
        sample_path = drug_samples[0][0] if drug_samples else mutant_samples[0][0]
        sample_img = Image.open(sample_path)
        w, h = sample_img.size
        self.image_size = w
        sample_img.close()
        
        # Precompute crop positions
        stride = (w - crop_size) // (grid_size - 1)
        self.stride = stride
        half_n = neighborhood // 2
        
        self.crop_positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * stride
                top = i * stride
                if (left + crop_size <= w and top + crop_size <= h
                    and left - half_n * stride >= 0
                    and left + half_n * stride + crop_size <= w
                    and top - half_n * stride >= 0
                    and top + half_n * stride + crop_size <= h):
                    self.crop_positions.append((left, top))
        
        self.epoch_indices = []
        self.set_epoch(0)
    
    def set_epoch(self, epoch):
        """Rebalance groups for this epoch."""
        rng = random.Random(SEED + epoch)
        self.epoch_indices = []
        
        for gid in self.group_ids:
            entries = self.grouped_samples[gid]
            if len(entries) < self.samples_per_group:
                idxs = list(range(len(entries)))
                rng.shuffle(idxs)
                idxs = idxs * (self.samples_per_group // len(entries) + 1)
                self.epoch_indices.extend([entries[i] for i in idxs[:self.samples_per_group]])
            else:
                idxs = rng.sample(range(len(entries)), self.samples_per_group)
                self.epoch_indices.extend([entries[i] for i in idxs])
        
        rng.shuffle(self.epoch_indices)
    
    def __len__(self):
        return len(self.epoch_indices)
    
    def _load_image(self, img_path):
        """Load and normalize image (same as MultiCropDataset)."""
        try:
            import tifffile
            img_array = tifffile.imread(img_path)
        except (ImportError, Exception):
            from PIL import Image as PILImage
            img_array = np.array(PILImage.open(img_path))
        
        if len(img_array.shape) == 3:
            img_array = img_array[:, :, 0]
        
        if img_array.dtype == np.uint16:
            img_array = img_array.astype(np.float32) / 65535.0
        elif img_array.dtype == np.uint8:
            img_array = img_array.astype(np.float32) / 255.0
        else:
            img_array = img_array.astype(np.float32)
        
        from PIL import Image as PILImage
        if self.num_channels == 1:
            return PILImage.fromarray((img_array * 255).astype(np.uint8), mode='L')
        return PILImage.fromarray((img_array * 255).astype(np.uint8), mode='L').convert('RGB')
    
    def _extract_crops(self, image):
        """Extract neighborhood crops (matches MultiCropDataset)."""
        from PIL import Image as PILImage
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        half_n = self.neighborhood // 2
        w, h = image.size
        
        rng = random.Random()
        crops_list = []
        
        for center_left, center_top in self.crop_positions:
            for di in range(-half_n, half_n + 1):
                for dj in range(-half_n, half_n + 1):
                    left = center_left + dj * self.stride + rng.randint(-self.stride // 4, self.stride // 4)
                    top = center_top + di * self.stride + rng.randint(-self.stride // 4, self.stride // 4)
                    left = max(0, min(left, w - self.crop_size))
                    top = max(0, min(top, h - self.crop_size))
                    
                    crop = image.crop((left, top, left + self.crop_size, top + self.crop_size))
                    crop = np.array(crop)
                    crops_list.append(self.transform_fn(image=crop)['image'])
        
        crops = torch.stack(crops_list)
        # Shuffle crops
        perm = torch.randperm(crops.shape[0])
        return crops[perm]
    
    def __getitem__(self, idx):
        path, class_idx, group_id, modality = self.epoch_indices[idx]
        image = self._load_image(path)
        crops = self._extract_crops(image)
        return crops, class_idx, group_id, modality


def collate_groupclip(batch):
    """Collate function for GroupCLIPDataset - separates by modality."""
    all_crops = []
    all_classes = []
    all_groups = []
    all_modalities = []
    
    for crops, cls, gid, mod in batch:
        all_crops.append(crops)
        all_classes.append(cls)
        all_groups.append(gid)
        all_modalities.append(mod)
    
    return {
        'images': torch.stack(all_crops),
        'class_labels': torch.tensor(all_classes, dtype=torch.long),
        'group_labels': torch.tensor(all_groups, dtype=torch.long),
        'modalities': torch.tensor(all_modalities, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_groupclip():
    test_plate = args.test_plate
    if 'Plate_' in test_plate:
        fold_key = test_plate
        plate_label = test_plate
    else:
        fold_key = f"Plate_{test_plate.replace('P', '')}"
        plate_label = test_plate
    
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'groupclip', f'fold_{fold_key}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"GroupCLIP Training: fold={fold_key}")
    print(f"{'='*60}\n")
    
    # Build train/val/test plate splits (same as train_mil.py)
    all_plates = ['Plate_1', 'Plate_2', 'Plate_3', 'Plate_4', 'Plate_5', 'Plate_6']
    test_plate_normalized = f"Plate_{plate_label.replace('P', '')}" if 'P' in plate_label else plate_label
    train_val_plates = [p for p in all_plates if p != test_plate_normalized]
    
    test_num = int(test_plate_normalized.split('_')[1])
    val_num = (test_num - 2) % 6 + 1
    val_plate = f"Plate_{val_num}"
    val_plates = [val_plate] if val_plate in train_val_plates else [train_val_plates[0]]
    train_plates = [p for p in train_val_plates if p not in val_plates][:4]
    
    print(f"Train plates: {train_plates}")
    print(f"Val plates: {val_plates}")
    print(f"Test plate: {test_plate_normalized}")
    
    # Collect samples
    train_drug, train_mutant = [], []
    val_drug, val_mutant = [], []
    test_drug, test_mutant = [], []
    
    for plate in train_plates:
        d, m = collect_samples(plate, 'both')
        train_drug.extend(d)
        train_mutant.extend(m)
    
    for plate in val_plates:
        d, m = collect_samples(plate, 'both')
        val_drug.extend(d)
        val_mutant.extend(m)
    
    for plate in [test_plate_normalized]:
        d, m = collect_samples(plate, 'both')
        test_drug.extend(d)
        test_mutant.extend(m)
    
    print(f"\nTrain: {len(train_drug)} drug, {len(train_mutant)} mutant")
    print(f"Val:   {len(val_drug)} drug, {len(val_mutant)} mutant")
    print(f"Test:  {len(test_drug)} drug, {len(test_mutant)} mutant")
    
    # Build transform
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    if args.num_channels == 1:
        norm_mean = [0.5]
        norm_std = [0.5]
    else:
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
    
    train_transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.3),
        A.Normalize(mean=norm_mean, std=norm_std),
        ToTensorV2(),
    ])
    
    eval_transform = A.Compose([
        A.Normalize(mean=norm_mean, std=norm_std),
        ToTensorV2(),
    ])
    
    # Create datasets
    train_dataset = GroupCLIPDataset(
        train_drug, train_mutant, train_transform,
        neighborhood=args.neighborhood, grid_size=args.grid_size,
        crop_size=args.crop_size, extraction_mode=args.extraction_mode,
        num_channels=args.num_channels,
        samples_per_group=args.samples_per_group,
    )
    
    # Determine num classes from dataset
    num_drug_classes = len(train_dataset.drug_classes)
    num_mutant_classes = len(train_dataset.mutant_classes)
    print(f"\nDrug classes: {num_drug_classes}, Mutant classes: {num_mutant_classes}")
    
    # Validation datasets (no augmentation, single center crop approach)
    # For efficiency, use MultiCropDataset with augment=False
    def build_val_dataset(samples, class_to_idx, transform):
        paths = [s[0] for s in samples]
        labels = [class_to_idx[s[1]] for s in samples]
        if not paths:
            return None
        return MultiCropDataset(
            paths, labels, None,
            neighborhood=args.neighborhood, grid_size=args.grid_size,
            augment=False, seed=SEED, num_channels=args.num_channels,
            extraction_mode=args.extraction_mode,
        )
    
    val_drug_dataset = build_val_dataset(val_drug, train_dataset.drug_class_to_idx, eval_transform)
    val_mutant_dataset = build_val_dataset(val_mutant, train_dataset.mutant_class_to_idx, eval_transform)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=len(train_dataset.group_ids),
        shuffle=True, num_workers=8, pin_memory=True,
        collate_fn=collate_groupclip, prefetch_factor=4,
    )
    
    val_drug_loader = DataLoader(
        val_drug_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=8, pin_memory=True,
    ) if val_drug_dataset else None
    
    val_mutant_loader = DataLoader(
        val_mutant_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=8, pin_memory=True,
    ) if val_mutant_dataset else None
    
    # Build model
    use_groove = (args.mode == 'groove')
    if use_groove:
        model = GROOVEModel(
            num_drug_classes=num_drug_classes,
            num_mutant_classes=num_mutant_classes,
            num_heads=args.num_heads,
            dropout=args.dropout,
            num_channels=args.num_channels,
            pretrained=args.pretrained,
            backbone=args.backbone,
            pooling=args.pooling,
            projection_dim=256,
            recon_hidden=args.recon_hidden,
        ).to(device)
        print(f"\nModel: GROOVEModel (GroupCLIP + Reconstruction + Backtranslation)")
    else:
        model = DualMILEncoder(
            num_drug_classes=num_drug_classes,
            num_mutant_classes=num_mutant_classes,
            num_heads=args.num_heads,
            dropout=args.dropout,
            num_channels=args.num_channels,
            pretrained=args.pretrained,
            backbone=args.backbone,
            pooling=args.pooling,
        ).to(device)
        print(f"\nModel: DualMILEncoder")
    
    print(f"  Backbone: {args.backbone}")
    print(f"  Pooling: {args.pooling}")
    print(f"  Drug classes: {num_drug_classes}, Mutant classes: {num_mutant_classes}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Losses
    criterion_gc = GroupCLIPLoss(temperature=args.gc_temp)
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion_mse = nn.MSELoss()
    
    # Optimizer
    backbone_params = []
    head_params = []
    for n, p in model.named_parameters():
        if 'backbone' in n:
            backbone_params.append(p)
        else:
            head_params.append(p)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': head_params, 'lr': args.lr},
    ], weight_decay=args.weight_decay, fused=torch.cuda.is_available())
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup, scheduler])
    
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    # TensorBoard + CSV logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_writer = SummaryWriter(log_dir=OUTPUT_DIR)
    csv_path = os.path.join(OUTPUT_DIR, f'training_groupclip_{timestamp}.csv')
    
    with open(csv_path, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        if use_groove:
            writer.writerow(['epoch', 'gc_loss', 'ce_loss', 'recon_loss', 'bt_loss', 'total_loss',
                            'train_drug_acc', 'train_mutant_acc',
                            'val_drug_acc', 'val_mutant_acc', 'val_drug_auc', 'val_mutant_auc'])
        else:
            writer.writerow(['epoch', 'gc_loss', 'ce_loss', 'total_loss',
                            'train_drug_acc', 'train_mutant_acc',
                            'val_drug_acc', 'val_mutant_acc', 'val_drug_auc', 'val_mutant_auc'])
    
    best_val_auc = 0.0
    best_val_drug_acc = 0.0
    best_val_mutant_acc = 0.0
    
    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_dataset.set_epoch(epoch)
        model.train()
        
        run_gc_loss = 0.0
        run_ce_loss = 0.0
        run_recon_loss = 0.0
        run_bt_loss = 0.0
        run_total_loss = 0.0
        drug_correct = 0
        drug_total = 0
        mutant_correct = 0
        mutant_total = 0
        
        for batch in tqdm(train_loader, desc=f'{args.mode.upper()} Epoch {epoch}', leave=False):
            images = batch['images'].to(device)
            class_labels = batch['class_labels'].to(device)
            group_labels = batch['group_labels'].to(device)
            modalities = batch['modalities'].to(device)
            
            optimizer.zero_grad()
            
            # Separate by modality
            drug_mask = modalities == 0
            mutant_mask = modalities == 1
            
            drug_imgs = images[drug_mask]
            mutant_imgs = images[mutant_mask]
            drug_classes_lab = class_labels[drug_mask]
            mutant_classes_lab = class_labels[mutant_mask]
            drug_groups = group_labels[drug_mask]
            mutant_groups = group_labels[mutant_mask]
            
            if drug_imgs.shape[0] < 2 or mutant_imgs.shape[0] < 2:
                continue
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                if use_groove:
                    # =====================================================
                    # GROOVE Step 1: GroupCLIP + CE + Reconstruction
                    # =====================================================
                    out = model.forward_groove_step1(drug_imgs, mutant_imgs)
                    
                    loss_gc = criterion_gc(out['drug_emb'], out['mutant_emb'],
                                           drug_groups, mutant_groups)
                    
                    loss_ce_drug = criterion_ce(out['drug_logits'], drug_classes_lab)
                    loss_ce_mutant = criterion_ce(out['mutant_logits'], mutant_classes_lab)
                    loss_ce = (loss_ce_drug + loss_ce_mutant) / 2.0
                    
                    # Get pooled features for reconstruction targets
                    with torch.no_grad():
                        _, drug_pooled = model.encode_to_latent(drug_imgs, 'drug')
                        _, mutant_pooled = model.encode_to_latent(mutant_imgs, 'mutant')
                    
                    loss_recon = (criterion_mse(out['drug_recon'], drug_pooled)
                                  + criterion_mse(out['mutant_recon'], mutant_pooled)) / 2.0
                    
                    loss_step1 = (args.gc_weight * loss_gc
                                  + args.ce_weight * loss_ce
                                  + args.recon_weight * loss_recon)
                    
                    # Scale, backward, unscale for Step 1
                    scaler.scale(loss_step1).backward(retain_graph=True)
                    scaler.unscale_(optimizer)
                    
                    # =====================================================
                    # GROOVE Step 2: Backtranslation cycle consistency
                    # =====================================================
                    bt_out = model.forward_groove_step2(out['drug_emb'], out['mutant_emb'])
                    loss_bt = (criterion_mse(bt_out['z_d_cycle'], out['drug_emb'])
                               + criterion_mse(bt_out['z_m_cycle'], out['mutant_emb'])) / 2.0
                    
                    loss_step2 = args.bt_weight * loss_bt
                    scaler.scale(loss_step2).backward()
                    
                    scaler.step(optimizer)
                    scaler.update()
                    
                    recon_val = loss_recon.item()
                    bt_val = loss_bt.item()
                    total_val = (loss_step1 + loss_step2).item()
                else:
                    # Standard GroupCLIP: GroupCLIP + CE
                    drug_logits, mutant_logits, drug_emb, mutant_emb = model(drug_imgs, mutant_imgs)
                    
                    loss_gc = criterion_gc(drug_emb, mutant_emb, drug_groups, mutant_groups)
                    loss_ce_drug = criterion_ce(drug_logits, drug_classes_lab)
                    loss_ce_mutant = criterion_ce(mutant_logits, mutant_classes_lab)
                    loss_ce = (loss_ce_drug + loss_ce_mutant) / 2.0
                    
                    loss = args.gc_weight * loss_gc + args.ce_weight * loss_ce
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    recon_val = 0.0
                    bt_val = 0.0
                    total_val = loss.item()
                    out = {'drug_logits': drug_logits, 'mutant_logits': mutant_logits}
            
            run_gc_loss += loss_gc.item()
            run_ce_loss += loss_ce.item()
            run_recon_loss += recon_val
            run_bt_loss += bt_val
            run_total_loss += total_val
            
            # Accuracy tracking
            _, drug_pred = out['drug_logits'].max(1)
            drug_correct += drug_pred.eq(drug_classes_lab).sum().item()
            drug_total += drug_classes_lab.size(0)
            
            _, mutant_pred = out['mutant_logits'].max(1)
            mutant_correct += mutant_pred.eq(mutant_classes_lab).sum().item()
            mutant_total += mutant_classes_lab.size(0)
        
        scheduler.step()
        
        train_drug_acc = 100. * drug_correct / max(drug_total, 1)
        train_mutant_acc = 100. * mutant_correct / max(mutant_total, 1)
        avg_gc = run_gc_loss / max(len(train_loader), 1)
        avg_ce = run_ce_loss / max(len(train_loader), 1)
        avg_recon = run_recon_loss / max(len(train_loader), 1)
        avg_bt = run_bt_loss / max(len(train_loader), 1)
        avg_total = run_total_loss / max(len(train_loader), 1)
        
        # -----------------------------------------------------------------------
        # Validation
        # -----------------------------------------------------------------------
        model.eval()
        
        def evaluate_modality(loader, modality='drug'):
            if loader is None:
                return 0.0, 0.0, 0.0, [], []
            correct = 0
            total = 0
            all_probs = []
            all_labels = []
            
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    batch_size = images.shape[0]
                    
                    # Create dummy for other modality
                    dummy = torch.zeros(1, images.shape[1], *images.shape[3:], device=device)
                    
                    if modality == 'drug':
                        logits, _, _, _ = model(images, dummy)
                    else:
                        _, logits, _, _ = model(dummy, images)
                    
                    probs = torch.softmax(logits, dim=1)
                    _, pred = logits.max(1)
                    correct += pred.eq(labels).sum().item()
                    total += labels.size(0)
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            acc = 100. * correct / max(total, 1)
            if len(set(all_labels)) >= 2:
                num_classes_actual = len(set(all_labels))
                labels_bin = label_binarize(all_labels, classes=sorted(set(all_labels)))
                probs_filtered = np.array(all_probs)[:, sorted(set(all_labels))]
                auc = roc_auc_score(labels_bin, probs_filtered, average='macro')
            else:
                auc = float('nan')
            
            return acc, auc, total, all_probs, all_labels
        
        val_drug_acc, val_drug_auc, _, _, _ = evaluate_modality(val_drug_loader, 'drug')
        val_mutant_acc, val_mutant_auc, _, _, _ = evaluate_modality(val_mutant_loader, 'mutant')
        
        val_auc_avg = np.nanmean([val_drug_auc, val_mutant_auc])
        
        if use_groove:
            print(
                f"Epoch {epoch:3d}: GC={avg_gc:.4f} CE={avg_ce:.4f} "
                f"Recon={avg_recon:.4f} BT={avg_bt:.4f} "
                f"DrugAcc={train_drug_acc:.1f}/{val_drug_acc:.1f} "
                f"MutAcc={train_mutant_acc:.1f}/{val_mutant_acc:.1f} "
                f"ValAUC={val_drug_auc:.4f}/{val_mutant_auc:.4f} "
                f"Time={time.time()-epoch_start:.0f}s"
            )
        else:
            print(
                f"Epoch {epoch:3d}: GC={avg_gc:.4f} CE={avg_ce:.4f} "
                f"DrugAcc={train_drug_acc:.1f}/{val_drug_acc:.1f} "
                f"MutAcc={train_mutant_acc:.1f}/{val_mutant_acc:.1f} "
                f"ValAUC={val_drug_auc:.4f}/{val_mutant_auc:.4f} "
                f"Time={time.time()-epoch_start:.0f}s"
            )
        
        # Logging
        log_prefix = 'GROOVE' if use_groove else 'GroupCLIP'
        tb_writer.add_scalars(f'{log_prefix}/Loss', {
            'gc': avg_gc, 'ce': avg_ce, 'total': avg_total
        }, epoch)
        if use_groove:
            tb_writer.add_scalars(f'{log_prefix}/Loss', {
                'recon': avg_recon, 'bt': avg_bt
            }, epoch)
        tb_writer.add_scalars(f'{log_prefix}/Drug_Acc', {
            'train': train_drug_acc, 'val': val_drug_acc
        }, epoch)
        tb_writer.add_scalars(f'{log_prefix}/Mutant_Acc', {
            'train': train_mutant_acc, 'val': val_mutant_acc
        }, epoch)
        
        with open(csv_path, 'a', newline='') as f:
            import csv
            writer = csv.writer(f)
            if use_groove:
                writer.writerow([epoch, avg_gc, avg_ce, avg_recon, avg_bt, avg_total,
                               train_drug_acc, train_mutant_acc,
                               val_drug_acc, val_mutant_acc, val_drug_auc, val_mutant_auc])
            else:
                writer.writerow([epoch, avg_gc, avg_ce, avg_total,
                               train_drug_acc, train_mutant_acc,
                               val_drug_acc, val_mutant_acc, val_drug_auc, val_mutant_auc])
        
        # Save best models
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'drug_class_to_idx': train_dataset.drug_class_to_idx,
            'mutant_class_to_idx': train_dataset.mutant_class_to_idx,
            'drug_classes': train_dataset.drug_classes,
            'mutant_classes': train_dataset.mutant_classes,
            'config': vars(args),
        }
        
        if not np.isnan(val_auc_avg) and val_auc_avg > best_val_auc:
            best_val_auc = val_auc_avg
            torch.save(ckpt, os.path.join(OUTPUT_DIR, 'best_model.pth'))
            torch.save(ckpt, os.path.join(OUTPUT_DIR, 'best_model_auc.pth'))
        
        if val_drug_acc > best_val_drug_acc:
            best_val_drug_acc = val_drug_acc
        
        if val_mutant_acc > best_val_mutant_acc:
            best_val_mutant_acc = val_mutant_acc
    
    print(f"\nTraining complete! Best val AUC: {best_val_auc:.4f}")
    
    # -----------------------------------------------------------------------
    # Final test evaluation
    # -----------------------------------------------------------------------
    print("\nTesting...")
    ckpt = torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    def build_test_loader(samples, class_to_idx):
        paths = [s[0] for s in samples]
        labels = [class_to_idx[s[1]] for s in samples if s[1] in class_to_idx]
        if not paths:
            return None, None, None
        group_ids = [s[2] for s in samples if s[1] in class_to_idx]
        dataset = MultiCropDataset(
            paths, labels, None,
            neighborhood=args.neighborhood, grid_size=args.grid_size,
            augment=False, seed=SEED, num_channels=args.num_channels,
            extraction_mode=args.extraction_mode,
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=8, pin_memory=True,
        )
        return loader, labels, group_ids
    
    test_drug_loader, test_drug_labels, test_drug_groups = build_test_loader(
        test_drug, train_dataset.drug_class_to_idx
    )
    test_mutant_loader, test_mutant_labels, test_mutant_groups = build_test_loader(
        test_mutant, train_dataset.mutant_class_to_idx
    )
    
    test_drug_acc, test_drug_auc, _, _, _ = evaluate_modality(test_drug_loader, 'drug')
    test_mutant_acc, test_mutant_auc, _, _, _ = evaluate_modality(test_mutant_loader, 'mutant')
    
    print(f"Test Drug: Acc={test_drug_acc:.2f}%, AUC={test_drug_auc:.4f}")
    print(f"Test Mutant: Acc={test_mutant_acc:.2f}%, AUC={test_mutant_auc:.4f}")
    
    # -----------------------------------------------------------------------
    # Extract and save test embeddings for cross-modal analysis
    # -----------------------------------------------------------------------
    drug_embeddings = []
    drug_emb_labels = []
    drug_emb_groups = []
    
    if test_drug_loader:
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            for images, labels in tqdm(test_drug_loader, desc='Drug embeddings'):
                images = images.to(device)
                emb = model.get_embeddings(images, modality='drug')
                drug_embeddings.append(emb.cpu().numpy())
                drug_emb_labels.extend(labels.numpy())
                drug_emb_groups.extend([train_dataset.drug_class_to_idx.get(
                    train_dataset.drug_classes[l], -1) for l in labels.numpy()])
    
    mutant_embeddings = []
    mutant_emb_labels = []
    mutant_emb_groups = []
    
    if test_mutant_loader:
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            for images, labels in tqdm(test_mutant_loader, desc='Mutant embeddings'):
                images = images.to(device)
                emb = model.get_embeddings(images, modality='mutant')
                mutant_embeddings.append(emb.cpu().numpy())
                mutant_emb_labels.extend(labels.numpy())
                mutant_emb_groups.extend([train_dataset.mutant_class_to_idx.get(
                    train_dataset.mutant_classes[l], -1) for l in labels.numpy()])
    
    # Save
    np.savez(
        os.path.join(OUTPUT_DIR, 'test_embeddings.npz'),
        drug_embeddings=np.concatenate(drug_embeddings) if drug_embeddings else np.array([]),
        mutant_embeddings=np.concatenate(mutant_embeddings) if mutant_embeddings else np.array([]),
        drug_labels=np.array(drug_emb_labels),
        mutant_labels=np.array(mutant_emb_labels),
        drug_groups=np.array(drug_emb_groups),
        mutant_groups=np.array(mutant_emb_groups),
        drug_class_names=np.array(train_dataset.drug_classes),
        mutant_class_names=np.array(train_dataset.mutant_classes),
    )
    
    # Final results
    results = {
        'config': vars(args),
        'test_drug_acc': float(test_drug_acc),
        'test_drug_auc': float(test_drug_auc),
        'test_mutant_acc': float(test_mutant_acc),
        'test_mutant_auc': float(test_mutant_auc),
        'best_val_auc': float(best_val_auc),
        'num_drug_classes': num_drug_classes,
        'num_mutant_classes': num_mutant_classes,
    }
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    tb_writer.close()
    print(f"Results saved to {OUTPUT_DIR}")
    print("Done!")


if __name__ == '__main__':
    train_groupclip()
