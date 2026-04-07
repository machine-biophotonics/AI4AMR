#!/usr/bin/env python3
"""
EfficientNet-B0 Training for Gene-Level Classification
Trains on 96 guide-level classes (including WT and NC controls)
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn
import torchvision
from torchvision.transforms import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, CenterCrop, Normalize, Compose, RandomCrop, Lambda, ColorJitter, RandomRotation, RandomAffine
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from typing import Optional, List, Dict
import random
from tqdm import tqdm
import csv
from datetime import datetime
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from collections import Counter
import io

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.set_num_threads(16)

print(f"PyTorch version: {torch.__version__}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# GradScaler for mixed precision training
scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--warmup_epochs', type=int, default=6, help='Warmup epochs')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
parser.add_argument('--min_delta', type=float, default=0.001, help='Min delta')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path')
parser.add_argument('--resume_csv', type=str, default=None, help='Continue writing to existing CSV file (use with --resume)')
parser.add_argument('--test_only', action='store_true', help='Only run test evaluation (use with --resume)')
parser.add_argument('--rho', type=float, default=0.1, help='SAM perturbation radius (0.1 for SAM, 2.0 for ASAM)')
parser.add_argument('--adaptive', action='store_true', help='Use Adaptive SAM (ASAM)')
parser.add_argument('--exclude_classes', nargs='*', default=[], help='List of class names to exclude from train/val/test')
parser.add_argument('--crop_size', type=int, default=224, help='Crop size for training (default: 224)')
parser.add_argument('--grid_size', type=int, default=12, help='Grid size for crops (default: 12x12)')
parser.add_argument('--center_loss', action='store_true', help='Use center loss for better feature discrimination')
parser.add_argument('--center_loss_weight', type=float, default=0.001, help='Weight for center loss')
parser.add_argument('--train_guides', nargs='+', type=int, default=[1,2,3], help='Guides to use for training (default: 1 2 3)')
parser.add_argument('--test_guide', type=int, default=None, help='Guide to use for testing (default: all)')
parser.add_argument('--guide_experiment', type=int, choices=[1,2,3], help='Run guide experiment: 1=train on g1,g2 test g3, 2=train on g1,g3 test g2, 3=train on g2,g3 test g1')
parser.add_argument('--debug', action='store_true', help='Enable debug logging for NaN detection')
args = parser.parse_args()

if args.debug:
    DEBUG_LOG_FILE = os.path.join(SCRIPT_DIR, f'debug_nan_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    print(f"Debug logging enabled: {DEBUG_LOG_FILE}")

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json'), 'r') as f:
    plate_data = json.load(f)

plate_maps = {}
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_maps[plate] = {}
    for row, wells in plate_data[plate].items():
        for col, info in wells.items():
            well = f"{row}{int(col):02d}"
            plate_maps[plate][well] = info['id']

def extract_gene(label):
    """Extract gene name from label - for 96 classes, use full label including guide number"""
    return label

def extract_well_from_filename(filename):
    match = re.search(r'Well(\w\d+)_', filename)
    return match.group(1) if match else None

all_genes = sorted(set(extract_gene(label) for pm in plate_maps.values() for label in pm.values()))
gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}
idx_to_gene = {idx: gene for gene, idx in gene_to_idx.items()}
num_classes = len(all_genes)
print(f"Number of gene classes: {num_classes}")
print(f"Genes: {all_genes}")

with open(os.path.join(SCRIPT_DIR, 'classes.txt'), 'w') as f:
    for i, gene in enumerate(all_genes):
        f.write(f"{i},{gene}\n")

def get_gene_from_path(img_path):
    dirname = os.path.dirname(img_path)
    plate = os.path.basename(dirname)
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if plate in plate_maps and well in plate_maps[plate]:
        label = plate_maps[plate][well]
        return extract_gene(label)
    return 'WT'

class GrayscaleMixedCropDataset(Dataset):
    """Dataset with random cropping (matching DINOv3 pipeline)"""
    
    def __init__(self, image_paths, labels, plates=None, crop_size=224, grid_size=12, augment=True, seed=42, epoch=0):
        self.image_paths = image_paths
        self.labels = labels
        self.plates = plates if plates is not None else [None] * len(labels)
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.augment = augment
        self.seed = seed
        self.epoch = epoch
        
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        self.image_size = w
        
        # Handle grid_size=1 (single crop centered)
        if grid_size == 1:
            stride = 0
        else:
            stride = (w - crop_size) // (grid_size - 1)
        self.stride = stride
        
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * stride
                top = i * stride
                if left + crop_size <= w and top + crop_size <= h:
                    positions.append((left, top))
        self.positions = positions
        
        self.shuffled_positions = self.positions.copy()
        if augment:
            rng = random.Random(seed + epoch)
            rng.shuffle(self.shuffled_positions)
        
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                         scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)}, rotate=(-15, 15), p=0.5),
                A.SomeOf([
                    A.ElasticTransform(alpha=50, sigma=5, p=1.0),
                    A.Perspective(scale=(0.02, 0.05), p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),
                    A.OpticalDistortion(distort_limit=0.05, p=1.0),
                ], n=1, replace=False, p=0.5),
                A.SomeOf([
                    A.GaussNoise(std_range=(0.05, 0.15), per_channel=False, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                ], n=1, replace=False, p=0.5),
                A.ImageCompression(quality_range=(85, 100), p=0.3),
                A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(16, 64), hole_width_range=(16, 64), p=0.4),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        
        print(f"Gene Dataset: {len(self.positions)} positions, augment={augment}")
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.augment:
            rng = random.Random(self.seed + epoch)
            self.shuffled_positions = self.positions.copy()
            rng.shuffle(self.shuffled_positions)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        left, top = self.shuffled_positions[idx % len(self.shuffled_positions)]
        crop = image.crop((left, top, left + self.crop_size, top + self.crop_size))
        crop = np.array(crop)
        crop = self.transform(image=crop)['image']
        
        if self.plates is not None and len(self.plates) > 0 and idx < len(self.plates):
            plate = self.plates[idx]
        else:
            plate = ""  # Empty string instead of None for DataLoader compatibility
        return crop, self.labels[idx], plate

def get_image_paths_for_plate(plate, base_dir):
    plate_dir = os.path.join(base_dir, plate)
    if not os.path.exists(plate_dir):
        return []
    
    patterns = ['*.tif', '*.tiff', '*.png']
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(plate_dir, '**', pattern), recursive=True))
    
    valid_paths = []
    for path in paths:
        well = extract_well_from_filename(os.path.basename(path))
        if well and well in plate_maps.get(plate, {}):
            valid_paths.append(path)
    
    return valid_paths

train_paths, train_labels, train_plates = [], [], []
val_paths, val_labels = [], []
test_paths, test_labels = [], []

for plate in ['P1', 'P2', 'P3', 'P4']:
    paths = get_image_paths_for_plate(plate, BASE_DIR)
    for path in paths:
        gene = get_gene_from_path(path)
        train_paths.append(path)
        train_labels.append(gene_to_idx[gene])
        train_plates.append(plate)

for plate in ['P5']:
    paths = get_image_paths_for_plate(plate, BASE_DIR)
    for path in paths:
        gene = get_gene_from_path(path)
        val_paths.append(path)
        val_labels.append(gene_to_idx[gene])

for plate in ['P6']:
    paths = get_image_paths_for_plate(plate, BASE_DIR)
    for path in paths:
        gene = get_gene_from_path(path)
        test_paths.append(path)
        test_labels.append(gene_to_idx[gene])

train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
test_labels = np.array(test_labels)

# Filter by guide if specified
def get_guide_from_label(label):
    """Extract guide number from label like 'ftsZ_1' -> 1"""
    if '_' in label:
        try:
            return int(label.rsplit('_', 1)[1])
        except ValueError:
            return None
    return None

# Apply guide filtering
train_guides = args.train_guides
test_guide = args.test_guide

if args.guide_experiment:
    exp_configs = {
        1: {'train': [1, 2], 'test': 3},
        2: {'train': [1, 3], 'test': 2},
        3: {'train': [2, 3], 'test': 1},
    }
    cfg = exp_configs[args.guide_experiment]
    train_guides = cfg['train']
    test_guide = cfg['test']

if train_guides != [1,2,3] or test_guide is not None:
    # Filter training data by guide
    if train_guides:
        print(f"Filtering training data to guides: {train_guides}")
        new_train_paths, new_train_labels, new_train_plates = [], [], []
        for path, label, plate in zip(train_paths, train_labels, train_plates):
            gene = idx_to_gene[label]
            guide = get_guide_from_label(gene)
            if guide in train_guides:
                new_train_paths.append(path)
                new_train_labels.append(label)
                new_train_plates.append(plate)
        train_paths = new_train_paths
        train_labels = new_train_labels
        train_plates = new_train_plates
        print(f"  After filter: {len(train_paths)} training samples")
    
    # Filter test data by guide
    if test_guide is not None:
        print(f"Filtering test data to guide: {test_guide}")
        new_test_paths, new_test_labels = [], []
        for path, label in zip(test_paths, test_labels):
            gene = idx_to_gene[label]
            guide = get_guide_from_label(gene)
            if guide == test_guide:
                new_test_paths.append(path)
                new_test_labels.append(label)
        test_paths = new_test_paths
        test_labels = new_test_labels
        print(f"  After filter: {len(test_paths)} test samples")

# Filter out excluded classes if specified
if args.exclude_classes:
    exclude_indices = [gene_to_idx[c] for c in args.exclude_classes if c in gene_to_idx]
    if exclude_indices:
        print(f"Excluding {len(exclude_indices)} classes: {args.exclude_classes}")
        
        # Filter training data
        train_mask = ~np.isin(train_labels, exclude_indices)
        train_paths = [p for p, m in zip(train_paths, train_mask) if m]
        train_labels = train_labels[train_mask]
        train_plates = [p for p, m in zip(train_plates, train_mask) if m]
        
        # Filter validation data
        val_mask = ~np.isin(val_labels, exclude_indices)
        val_paths = [p for p, m in zip(val_paths, val_mask) if m]
        val_labels = val_labels[val_mask]
        
        # Filter test data
        test_mask = ~np.isin(test_labels, exclude_indices)
        test_paths = [p for p, m in zip(test_paths, test_mask) if m]
        test_labels = test_labels[test_mask]
        
        # Create mapping from old indices to new contiguous indices
        all_labels = sorted(set(train_labels) | set(val_labels) | set(test_labels))
        label_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(all_labels)}
        
        # Remap all labels to contiguous indices
        train_labels = np.array([label_mapping[l] for l in train_labels])
        val_labels = np.array([label_mapping[l] for l in val_labels])
        test_labels = np.array([label_mapping[l] for l in test_labels])
        
        # Update num_classes
        num_classes = len(all_labels)
        print(f"After filtering: Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}, Classes: {num_classes}")

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
print(f"Class distribution: {Counter(train_labels)}")

# Class weights (inverse frequency, normalized) - rebuilt after filtering
class_counts = Counter(train_labels)
total = len(train_labels)
# Use actual class indices that exist in the filtered data
class_weights = torch.ones(num_classes, device=device)
for i in range(num_classes):
    if i in class_counts and class_counts[i] > 0:
        class_weights[i] = total / (num_classes * class_counts[i])
# Normalize
class_weights = class_weights / class_weights.sum() * num_classes

# Domain weights (per-plate, using n_d^-1/2 formula)
train_plates = np.array(train_plates)
plate_counts = Counter(train_plates)
n_plates = len(plate_counts)
domain_weights = {plate: 1.0 / np.sqrt(count) for plate, count in plate_counts.items()}
# Normalize domain weights
dom_sum = sum(domain_weights.values())
domain_weights = {k: v / dom_sum * n_plates for k, v in domain_weights.items()}
print(f"Domain weights: {domain_weights}")

# Compute combined weights for each sample
def get_combined_weights(labels, plates=None):
    """Compute class weights for samples (optionally combined with domain weights)"""
    weights = []
    for i, label in enumerate(labels):
        class_w = class_weights[label].item()
        if plates is not None:
            plate = plates[i]
            domain_w = domain_weights.get(plate, 1.0)
            class_w = class_w * domain_w
        weights.append(class_w)
    weights = np.array(weights)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.1, 5.0)  # Clamp to prevent gradient explosion
    return torch.tensor(weights, device=device)


DEBUG_LOG_FILE = None

def weighted_focal_loss(logits, targets, weights, alpha=0.25, gamma=2.0, label_smoothing=0.1):
    """Weighted Focal Loss with label smoothing (combined class and domain weights)."""
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none', label_smoothing=label_smoothing)
    
    ce_clamped = ce_loss.clamp(min=1e-8, max=50.0)
    pt = torch.exp(-ce_clamped)
    pt = pt.clamp(min=1e-10, max=1.0 - 1e-10)
    
    focal = alpha * (1 - pt) ** gamma * ce_clamped
    weighted = focal * weights
    
    loss = weighted.mean()
    
    if DEBUG_LOG_FILE and (torch.isnan(loss) or torch.isinf(loss)):
        with open(DEBUG_LOG_FILE, 'a') as f:
            f.write(f"[DEBUG] NaN/Inf detected!\n")
            f.write(f"  ce_loss: min={ce_loss.min().item():.4f}, max={ce_loss.max().item():.4f}, mean={ce_loss.mean().item():.4f}\n")
            f.write(f"  ce_clamped: min={ce_clamped.min().item():.4f}, max={ce_clamped.max().item():.4f}\n")
            f.write(f"  pt: min={pt.min().item():.6f}, max={pt.max().item():.6f}\n")
            f.write(f"  focal: min={focal.min().item():.4f}, max={focal.max().item():.4f}, mean={focal.mean().item():.4f}\n")
            f.write(f"  weights: min={weights.min().item():.6f}, max={weights.max().item():.6f}\n")
            f.write(f"  logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}\n")
            f.write(f"  Final loss: {loss.item()}\n")
            f.write(f"  Is nan: {torch.isnan(loss).item()}, Is inf: {torch.isinf(loss).item()}\n")
    
    return loss


class CenterLoss(nn.Module):
    """Center loss for better feature discrimination."""
    def __init__(self, num_classes, feat_dim, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        if use_gpu:
            self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)
        # Ensure centers have same dtype as features (handle AMP dtype mismatch)
        centers = self.centers.to(dtype=features.dtype)
        # Compute squared distances: ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x.c
        features_sq = torch.pow(features, 2).sum(dim=1, keepdim=True)  # (B, 1)
        centers_sq = torch.pow(centers, 2).sum(dim=1)  # (C,)
        # Use new addmm signature to avoid dtype issues
        distmat = torch.addmm(
            centers_sq.unsqueeze(0).expand(batch_size, -1) + features_sq.expand(-1, self.num_classes),
            features, centers.t(), beta=1, alpha=-2
        )
        
        classes = torch.arange(self.num_classes, dtype=torch.long, device=features.device)
        labels_expanded = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expanded.eq(classes.unsqueeze(0).expand(batch_size, -1))
        
        dist = distmat * mask.to(dtype=distmat.dtype)
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss

train_dataset = GrayscaleMixedCropDataset(train_paths, train_labels, train_plates, crop_size=args.crop_size, grid_size=args.grid_size, augment=True, seed=SEED)
val_dataset = GrayscaleMixedCropDataset(val_paths, val_labels, [], crop_size=args.crop_size, grid_size=args.grid_size, augment=False, seed=SEED)
test_dataset = GrayscaleMixedCropDataset(test_paths, test_labels, [], crop_size=args.crop_size, grid_size=args.grid_size, augment=False, seed=SEED)

print(f"Dataset config: crop_size={args.crop_size}, grid_size={args.grid_size}, crops_per_image={args.grid_size**2}")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')

# FULL FINETUNING - unfreeze entire model
for param in model.parameters():
    param.requires_grad = True

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(model.classifier[1].in_features, num_classes)
)

# Custom wrapper to return features when needed
class EfficientNetWithFeatures(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier
    
    def forward(self, x, return_features=False):
        features = self.features(x)
        pooled = self.avgpool(features).flatten(1)
        logits = self.classifier(pooled)
        if return_features:
            return logits, pooled
        return logits

model = EfficientNetWithFeatures(model)
model = model.to(device)

# Different learning rates for backbone vs classifier
backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n]

base_optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': args.lr * 0.1},  # Lower LR for backbone
    {'params': classifier_params, 'lr': args.lr}       # Higher LR for classifier
], weight_decay=0.01)

# Wrap with SAM optimizer
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        # Handle both optimizer class and optimizer instance
        if isinstance(base_optimizer, torch.optim.Optimizer):
            # base_optimizer is already an instance
            self.base_optimizer = base_optimizer
            # Need to add rho and adaptive to each param group from base optimizer
            for group in self.base_optimizer.param_groups:
                group['rho'] = rho
                group['adaptive'] = adaptive
        else:
            # base_optimizer is a class, instantiate it with param_groups
            self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            adaptive = group.get("adaptive", False)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                # Adaptive SAM: use element-wise scaling with p^2
                e_w = (torch.pow(p, 2) if adaptive else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group.get("adaptive", False) else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive)

# Center loss with separate optimizer (NOT using SAM to avoid perturbing centers)
if args.center_loss:
    feat_dim = model.classifier[1].in_features  # Feature dimension from classifier
    center_loss_fn = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)
    # Separate optimizer for center loss (SGD with higher LR, no SAM)
    center_optimizer = torch.optim.SGD(center_loss_fn.parameters(), lr=args.center_loss_weight * 10, momentum=0.9)
    print(f"Center loss enabled with weight={args.center_loss_weight}, feat_dim={feat_dim}")
else:
    center_loss_fn = None
    center_optimizer = None

num_training_steps = len(train_loader) * args.epochs
num_warmup_steps = len(train_loader) * args.warmup_epochs

# Track global step for scheduler (to handle resume correctly)
global_step = 0

def lr_lambda(step):
    if step < num_warmup_steps:
        return step / num_warmup_steps
    progress = (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
    return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# CSV logging
if args.resume_csv and os.path.exists(args.resume_csv):
    csv_path = args.resume_csv
    # Extract timestamp from existing CSV filename
    import re
    match = re.search(r'training_metrics_(\d{8}_\d{6})\.csv', csv_path)
    if match:
        timestamp = match.group(1)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Continuing CSV: {csv_path}")
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(SCRIPT_DIR, f'training_metrics_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_balanced_acc', 'val_roc_auc', 'lr'])

best_val_acc = 0.0
best_val_balanced_acc = 0.0
best_val_auc = 0.0
start_epoch = 0
train_losses, train_accs, val_losses, val_accs = [], [], [], []

# Early stopping patience counter
patience_counter = 0
if args.patience > 0:
    print(f"Early stopping enabled: patience={args.patience} epochs without improvement")
else:
    print("Early stopping disabled")

if args.resume:
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extract epoch from checkpoint (handle both with and without history)
    if 'train_losses' in checkpoint and len(checkpoint['train_losses']) > 0:
        train_losses = checkpoint['train_losses']
        train_accs = checkpoint['train_accs']
        val_losses = checkpoint['val_losses']
        val_accs = checkpoint['val_accs']
        best_val_acc = checkpoint['best_val_acc']
        if 'best_val_auc' in checkpoint:
            best_val_auc = checkpoint['best_val_auc']
        if 'best_val_balanced_acc' in checkpoint:
            best_val_balanced_acc = checkpoint['best_val_balanced_acc']
        start_epoch = len(train_losses)
    else:
        # Checkpoint only has model weights, extract epoch from checkpoint
        # checkpoint_e75 means epoch 75 is DONE, so next epoch is 76
        start_epoch = checkpoint.get('epoch', 0) + 1
        # Initialize best metrics since we don't have history
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
        print(f"Starting training from epoch {start_epoch}")
    
    # Calculate global step for scheduler
    # scheduler.step() is called once per batch (after both SAM steps), so no *2
    global_step = start_epoch * len(train_loader)
    if global_step > num_warmup_steps:
        # Already past warmup, set scheduler to current position
        for _ in range(global_step):
            scheduler.step()
    
    print(f"Resuming from epoch {start_epoch}, global_step={global_step}")

if args.test_only:
    print("Test-only mode: loading best model for evaluation...")
    
    checkpoint = torch.load(os.path.join(SCRIPT_DIR, 'best_model.pth'), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    crop_size = args.crop_size
    grid_size = args.grid_size
    image_size = test_dataset.image_size
    stride = (image_size - crop_size) // (grid_size - 1) if grid_size > 1 else 0
    positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            if grid_size == 1:
                left = (image_size - crop_size) // 2
                top = (image_size - crop_size) // 2
            else:
                left = j * stride
                top = i * stride
            if left + crop_size <= image_size and top + crop_size <= image_size:
                positions.append((left, top))
    
    print(f"Using crop_size={crop_size}, grid_size={grid_size}, {len(positions)} crops per image")
    
    results_data = []
    with torch.no_grad():
        for img_path, true_label in tqdm(zip(test_paths, test_labels), total=len(test_paths), desc="Processing"):
            image = Image.open(img_path).convert('RGB')
            img_name = os.path.basename(img_path)
            
            all_crops_preds = []
            for left, top in positions:
                crop = image.crop((left, top, left + crop_size, top + crop_size))
                crop = np.array(crop)
                crop = transform(image=crop)['image'].unsqueeze(0).to(device)
                
                output = model(crop)
                probs = torch.softmax(output, dim=1)
                all_crops_preds.append(probs.cpu().numpy()[0])
            
            all_crops_preds = np.array(all_crops_preds)
            avg_probs = all_crops_preds.mean(axis=0)
            pred = np.argmax(avg_probs)
            
            results_data.append({
                'image': img_name,
                'true_label': int(int(true_label)),
                'pred_label': int(int(pred)),
                'avg_probs': [float(x) for x in avg_probs.tolist()],
                'per_crop_preds': [int(x) for x in [np.argmax(p) for p in all_crops_preds]]
            })
    
    with open(os.path.join(SCRIPT_DIR, f'test_predictions_{timestamp}.json'), 'w') as f:
        json.dump(results_data, f, indent=2)
    
    correct = sum(1 for r in results_data if r['true_label'] == r['pred_label'])
    test_acc = 100. * correct / len(results_data)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Saved predictions to test_predictions_{timestamp}.json")
    print("Done!")
    exit(0)
else:
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for batch_idx, (images, labels, batch_plates) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)):
            images, labels = images.to(device), labels.to(device)
            batch_plates = list(batch_plates)
            
            # Compute combined weights (class × domain) - batch_plates has same length as labels
            weights = get_combined_weights(labels.cpu().tolist(), batch_plates)
            
            # First forward-backward pass (SAM)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                if center_loss_fn is not None:
                    outputs, features = model(images, return_features=True)
                    center_loss = center_loss_fn(features, labels)
                else:
                    outputs = model(images)
                
                loss = weighted_focal_loss(outputs, labels, weights)
                
                if center_loss_fn is not None:
                    total_loss = loss + args.center_loss_weight * center_loss
                    loss = total_loss
            
            # Debug: Check for NaN after first forward pass
            if DEBUG_LOG_FILE and (torch.isnan(loss) or torch.isinf(loss)):
                with open(DEBUG_LOG_FILE, 'a') as f:
                    f.write(f"[Epoch {epoch}] NaN/Inf after first forward pass at batch {batch_idx}\n")
                    f.write(f"  Loss value: {loss.item()}\n")
                    f.write(f"  Outputs: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}\n")
                    f.write(f"  Labels: {labels[:5].cpu().tolist()}\n")
                    f.write(f"  Weights: {weights[:5].cpu().tolist()}\n")
                    f.write(f"  Images stats: min={images.min().item():.4f}, max={images.max().item():.4f}\n")
                    f.write(f"  Images grad: {images.requires_grad}\n")
            
            if torch.isnan(loss) or torch.isinf(loss):
                with open(DEBUG_LOG_FILE or '/dev/null', 'a') as f:
                    if DEBUG_LOG_FILE:
                        f.write(f"[Epoch {epoch}] FATAL: NaN/Inf at batch {batch_idx}, breaking\n")
                print(f"[Epoch {epoch}] NaN/Inf detected at batch {batch_idx}, stopping training")
                break
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.first_step(zero_grad=True)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.first_step()
            
            # Second forward-backward pass (SAM)
            optimizer.zero_grad()
            stored_features = None
            with torch.amp.autocast('cuda'):
                if center_loss_fn is not None:
                    outputs, features = model(images, return_features=True)
                    stored_features = features.detach()  # Store for center loss (detached to avoid SAM interference)
                    center_loss = center_loss_fn(features, labels)
                else:
                    outputs = model(images)
                
                loss = weighted_focal_loss(outputs, labels, weights)
                
                if center_loss_fn is not None:
                    total_loss = loss + args.center_loss_weight * center_loss
                    loss = total_loss
            
            # Debug: Check for NaN after second forward pass
            if DEBUG_LOG_FILE and (torch.isnan(loss) or torch.isinf(loss)):
                with open(DEBUG_LOG_FILE, 'a') as f:
                    f.write(f"[Epoch {epoch}] NaN/Inf after second forward pass at batch {batch_idx}\n")
                    f.write(f"  Loss value: {loss.item()}\n")
                    f.write(f"  Outputs: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}\n")
            
            if torch.isnan(loss) or torch.isinf(loss):
                with open(DEBUG_LOG_FILE or '/dev/null', 'a') as f:
                    if DEBUG_LOG_FILE:
                        f.write(f"[Epoch {epoch}] FATAL: NaN/Inf at batch {batch_idx}, breaking\n")
                print(f"[Epoch {epoch}] NaN/Inf detected at batch {batch_idx}, stopping training")
                break
            running_loss += loss.item()
            if scaler is not None:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.second_step(zero_grad=True)
                scaler.update()  # Update at end of both SAM passes
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.second_step()
            
            # Update center loss separately (not with SAM) - using stored features from second pass
            if center_optimizer is not None and stored_features is not None:
                center_optimizer.zero_grad()
                if center_loss_fn is not None:
                    center_loss = center_loss_fn(stored_features, labels)
                    center_loss = args.center_loss_weight * center_loss
                    if scaler is not None:
                        scaler.scale(center_loss).backward()
                        scaler.step(center_optimizer)
                        scaler.update()
                    else:
                        center_loss.backward()
                        center_optimizer.step()
            
            running_loss += loss.item()
            
            # Update learning rate after both steps
            scheduler.step()
            global_step += 1
            
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    outputs_eval = model(images)
            
            _, predicted = outputs_eval.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Check if NaN detected during training - skip epoch logging
        if total == 0:
            print(f"[Epoch {epoch}] No valid training batches (NaN detected), skipping epoch")
            continue
        
        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = nn.functional.cross_entropy(outputs, labels)
                
                # Debug: Check for NaN in validation
                if DEBUG_LOG_FILE and (torch.isnan(loss) or torch.isinf(loss)):
                    with open(DEBUG_LOG_FILE, 'a') as f:
                        f.write(f"[Val Epoch {epoch}] NaN/Inf detected!\n")
                        f.write(f"  Loss: {loss.item()}\n")
                        f.write(f"  Outputs: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}\n")
                
                probs = torch.softmax(outputs, dim=1)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        all_probs = np.vstack(all_probs)
        avg_val_loss = running_loss / len(val_loader)
        
        # Check for NaN in validation
        if np.isnan(avg_val_loss) or np.isinf(avg_val_loss):
            print(f"[Epoch {epoch}] Validation loss is NaN/Inf, skipping epoch")
            continue
        
        val_acc = 100. * correct / total
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute ROC AUC (single pass)
        try:
            all_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
            per_class_auc = []
            for i in range(num_classes):
                # Skip classes with no positive samples
                if all_labels_bin[:, i].sum() > 0 and all_probs[:, i].std() > 0:
                    try:
                        fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
                        per_class_auc.append(auc(fpr, tpr))
                    except:
                        per_class_auc.append(0.5)  # Default for edge cases
                else:
                    per_class_auc.append(0.5)  # Default for missing/no-variance classes
            mean_val_auc = np.mean(per_class_auc)
        except Exception as e:
            print(f"Warning: Could not compute ROC AUC: {e}")
            mean_val_auc = 0.0
        
        per_class_correct = [np.sum((all_preds == i) & (all_labels == i)) for i in range(num_classes)]
        per_class_total = [np.sum(all_labels == i) for i in range(num_classes)]
        balanced_acc = np.mean([per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0 for i in range(num_classes)])
        
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%, Balanced Acc={balanced_acc*100:.2f}%, Val ROC AUC={mean_val_auc*100:.2f}%, LR={current_lr:.2e}")
        
        # Write to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, train_acc, avg_val_loss, val_acc, balanced_acc, mean_val_auc, current_lr])
        
        # Save LAST model (overwrite each epoch) - always save immediately
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc,
            'best_val_balanced_acc': best_val_balanced_acc,
            'best_val_auc': best_val_auc,
        }, os.path.join(SCRIPT_DIR, 'last_model.pth'))
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(SCRIPT_DIR, f'checkpoint_e{epoch}.pth'))
        
        # Save best model based on ROC AUC (primary metric) - save on ANY improvement
        if mean_val_auc > best_val_auc:
            best_val_auc = mean_val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs,
                'best_val_acc': best_val_acc,
                'best_val_balanced_acc': best_val_balanced_acc,
                'best_val_auc': best_val_auc,
            }, os.path.join(SCRIPT_DIR, 'best_model.pth'))
            print(f"  -> New best model! Val ROC AUC: {best_val_auc:.4f}")
            
            # Reset patience counter on improvement
            if args.patience > 0:
                patience_counter = 0
        else:
            # No improvement - increment patience counter
            if args.patience > 0:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs (no improvement for {args.patience} epochs)")
                    break
        
        # Also save based on balanced accuracy (for comparison) - save on ANY improvement
        if balanced_acc > best_val_balanced_acc:
            best_val_balanced_acc = balanced_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_balanced_acc': best_val_balanced_acc,
                'best_val_auc': best_val_auc,
            }, os.path.join(SCRIPT_DIR, 'best_model_balanced.pth'))
            print(f"  -> New best balanced model! Val Balanced Acc: {balanced_acc:.4f}")

    print("Training complete. Generating test results...")

checkpoint = torch.load(os.path.join(SCRIPT_DIR, 'best_model.pth'), map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for images, labels, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

test_acc = 100. * np.mean(all_preds == all_labels)
print(f"Test Accuracy: {test_acc:.2f}%")

test_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))

roc_auc = {}
ap = {}
for i in range(num_classes):
    if test_labels_bin[:, i].sum() > 0:
        roc_auc[i] = roc_auc_score(test_labels_bin[:, i], all_probs[:, i])
        ap[i] = average_precision_score(test_labels_bin[:, i], all_probs[:, i])

mean_roc_auc = np.mean(list(roc_auc.values()))
mean_ap = np.mean(list(ap.values()))
print(f"Mean ROC AUC: {mean_roc_auc:.4f}")
print(f"Mean AP: {mean_ap:.4f}")

results = {
    'timestamp': timestamp,
    'config': {
        'num_classes': num_classes,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
    },
    'results': {
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'mean_roc_auc': float(mean_roc_auc),
        'mean_ap': float(mean_ap),
    }
}

with open(os.path.join(SCRIPT_DIR, f'training_results_{timestamp}.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("Done!")
