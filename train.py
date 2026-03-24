import argparse
import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn
import torchvision
from torchvision.transforms.functional import rotate, adjust_brightness, adjust_contrast
from torchvision import datasets, models
from torchvision.transforms import v2
from torchvision.transforms import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, CenterCrop, Normalize, Compose, RandomCrop, Lambda, ColorJitter, RandomRotation, RandomAffine, RandomErasing, RandomChoice
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from typing import Optional, List, Dict, Callable
import random
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# Full Reproducibility Setup (seed=42)
# =============================================================================
SEED = 42

# Set seeds for all libraries
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Enable deterministic algorithms for full reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Set environment variable for CUDA determinism
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

torch.set_num_threads(16)

print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true', help='Resume training from best_model.pth')
parser.add_argument('--n_crops', type=int, default=144, help='Number of crops per image (default: 144, use 9 for quick test)')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=None, help='Batch size (auto-set based on crops if not specified)')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate for classifier')
parser.add_argument('--warmup_epochs', type=int, default=2, help='Number of warmup epochs')
args = parser.parse_args()

# Auto-set batch size based on n_crops for optimal GPU utilization
if args.batch_size is None:
    if args.n_crops <= 9:
        args.batch_size = 128
    elif args.n_crops <= 36:
        args.batch_size = 64
    elif args.n_crops <= 72:
        args.batch_size = 32
    else:
        args.batch_size = 16
print(f"Auto-configured: batch_size={args.batch_size} for n_crops={args.n_crops}")

# Load plate data from JSON (verified mapping)
with open(os.path.join(BASE_DIR, 'plate_well_id_path.json'), 'r') as f:
    plate_data = json.load(f)

plate_maps: dict[str, dict[str, str]] = {}
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_maps[plate] = {}
    for row, wells in plate_data[plate].items():
        for col, info in wells.items():
            well = f"{row}{int(col):02d}"
            plate_maps[plate][well] = info['id']


class RandomCenterCrop:
    """Random crop from center region only (avoids edges)"""
    def __init__(self, size: int, edge_margin: int = 200):
        self.size = size
        self.edge_margin = edge_margin  # pixels to avoid from edges
    
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        # Calculate center region (avoiding edges)
        center_w_start = self.edge_margin
        center_w_end = w - self.edge_margin
        center_h_start = self.edge_margin
        center_h_end = h - self.edge_margin
        
        # Random position within center region
        max_top = center_h_end - self.size
        max_left = center_w_end - self.size
        
        if max_top <= 0 or max_left <= 0:
            # If image is too small, just center crop
            left = (w - self.size) // 2
            top = (h - self.size) // 2
        else:
            top = random.randint(center_h_start, max_top)
            left = random.randint(center_w_start, max_left)
        
        return img.crop((left, top, left + self.size, top + self.size))


def extract_well_from_filename(filename: str) -> Optional[str]:
    match: Optional[re.Match] = re.search(r'Well(\w\d+)_', filename)
    if match:
        return match.group(1)
    return None

all_labels: list[str] = sorted(set(label for pm in plate_maps.values() for label in pm.values()))
label_to_idx: dict[str, int] = {label: idx for idx, label in enumerate(all_labels)}
idx_to_label: dict[int, str] = {idx: label for label, idx in label_to_idx.items()}
num_classes: int = len(all_labels)
print(f"Number of classes: {num_classes}")

with open(os.path.join(BASE_DIR, 'classes.txt'), 'w') as f:
    for i, label in enumerate(all_labels):
        f.write(f"{i},{label}\n")
print("Classes saved to classes.txt")


def get_label_from_path(img_path: str) -> Optional[str]:
    dirname: str = os.path.dirname(img_path)
    plate: str = os.path.basename(dirname)
    filename: str = os.path.basename(img_path)
    well: Optional[str] = extract_well_from_filename(filename)
    if plate in plate_maps and well in plate_maps[plate]:
        return plate_maps[plate][well]
    return None


class FixedGridDataset(Dataset):
    """Fixed Grid Dataset - uses ALL crops from entire image (no edge margin).
    
    Augmentation pipeline (matching reference):
    - RandomVerticalFlip
    - RandomHorizontalFlip  
    - RandomRotation (90 degrees)
    - ColorJitter
    - ToTensor
    - Normalize
    """
    def __init__(self, image_paths: list[str], transform: Optional[Compose], 
                 get_labels: bool, augment: bool = True):
        self.image_paths = image_paths
        self.transform = transform
        self.get_labels = get_labels
        self.augment = augment
        self.patch_size = 224
        
        # Calculate grid from first image
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        self.n_crops_w = w // self.patch_size
        self.n_crops_h = h // self.patch_size
        self.n_crops = self.n_crops_w * self.n_crops_h
        
        # Augmentation transforms (matching reference code)
        self.augment_transform = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(degrees=90),  # Like reference: RandomRotation(90)
            ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),  # Like reference
        ]) if augment else None
        
        print(f"Fixed Grid: {self.n_crops_w}x{self.n_crops_h} = {self.n_crops} crops/image (augment={augment})")
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        img_path: str = self.image_paths[idx]
        image: Image.Image = Image.open(img_path).convert('RGB')
        
        patches = []
        for i in range(self.n_crops_h):
            for j in range(self.n_crops_w):
                left = j * self.patch_size
                top = i * self.patch_size
                patch = image.crop((left, top, left + self.patch_size, top + self.patch_size))
                
                # Apply augmentation if enabled (PIL image level)
                if self.augment_transform:
                    patch = self.augment_transform(patch)
                
                # Apply tensor transform (ToTensor, Normalize)
                if self.transform:
                    patch = self.transform(patch)
                patches.append(patch)
        
        patches_tensor = torch.stack(patches)
        
        if self.get_labels:
            label_str: Optional[str] = get_label_from_path(img_path)
            if label_str is None:
                label_str = "Unknown"
            label: int = label_to_idx[label_str]
            return patches_tensor, label, img_path
        return patches_tensor, 0


class MultiPatchDataset(Dataset):
    """Original MultiPatchDataset with random crops from center region."""
    def __init__(self, image_paths: list[str], transform: Optional[Compose], 
                 get_labels: bool, n_patches: int = 1, random_patches: bool = True):
        self.image_paths = image_paths
        self.transform = transform
        self.get_labels = get_labels
        self.n_patches = n_patches
        self.random_patches = random_patches
        self.patch_size = 224
        self.edge_margin = 200
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        img_path: str = self.image_paths[idx]
        image: Image.Image = Image.open(img_path).convert('RGB')
        w, h = image.size
        
        patches = []
        center_w_start = self.edge_margin
        center_w_end = w - self.edge_margin
        center_h_start = self.edge_margin
        center_h_end = h - self.edge_margin
        
        center_w = center_w_end - center_w_start
        center_h = center_h_end - center_h_start
        
        for _ in range(self.n_patches):
            if self.random_patches:
                left = int(np.random.uniform(0, max(1, center_w - self.patch_size)))
                top = int(np.random.uniform(0, max(1, center_h - self.patch_size)))
            else:
                left = (center_w - self.patch_size) // 2
                top = (center_h - self.patch_size) // 2
            
            left = center_w_start + left
            top = center_h_start + top
            
            patch = image.crop((left, top, left + self.patch_size, top + self.patch_size))
            
            if self.transform:
                patch = self.transform(patch)
            patches.append(patch)
        
        patches_tensor = torch.stack(patches)
        
        if self.get_labels:
            label_str: Optional[str] = get_label_from_path(img_path)
            if label_str is None:
                label_str = "Unknown"
            label: int = label_to_idx[label_str]
            return patches_tensor, label, img_path
        return patches_tensor, 0


class ShuffledCropSampler:
    """Sampler that shuffles ALL crops across ALL images. Different order each epoch, no duplicates."""
    def __init__(self, total_crops: int, shuffle: bool = True):
        self.indices = list(range(total_crops))
        self.shuffle = shuffle
        if shuffle:
            random.shuffle(self.indices)
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)


class MixedCropDataset(Dataset):
    """Mixed Crop Dataset - uses crops from ENTIRE image grid with shuffled sampling.
    
    Same as train_bacnet.py:
    - Grid across WHOLE image (not just center)
    - Morphology-preserving augmentations only
    - On-demand image loading
    """
    def __init__(self, image_paths: list[str], labels: list[int], 
                 crop_size: int = 224, grid_size: int = 12,
                 n_crops_per_image: int = 144, augment: bool = True):
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.augment = augment
        
        # Get dimensions from first image
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        
        # Calculate grid positions across ENTIRE image (not just center)
        self.total_w = w - crop_size
        self.total_h = h - crop_size
        self.step_w = self.total_w / (grid_size - 1) if grid_size > 1 else 0
        self.step_h = self.total_h / (grid_size - 1) if grid_size > 1 else 0
        
        # Pre-calculate ALL crop positions (144 per image)
        self.crop_positions = []
        for img_idx in range(len(image_paths)):
            for i in range(grid_size):
                for j in range(grid_size):
                    left = int(j * self.step_w)
                    top = int(i * self.step_h)
                    self.crop_positions.append((img_idx, left, top))
        
        # Sample crops per image
        self.n_crops_per_image = min(n_crops_per_image, grid_size * grid_size)
        
        if self.n_crops_per_image < grid_size * grid_size:
            # For 9 crops: use centered 3x3 grid (positions 33, 34, 35, 45, 46, 47, 57, 58, 59)
            if self.n_crops_per_image == 9:
                center_start = grid_size // 2 - 1  # 12//2 - 1 = 5
                center_end = grid_size // 2 + 2    # 12//2 + 2 = 8
                indices = []
                for i in range(center_start, center_end):
                    for j in range(center_start, center_end):
                        indices.append(i * grid_size + j)
            else:
                indices = random.sample(range(grid_size * grid_size), self.n_crops_per_image)
            
            self.crop_positions = []
            for img_idx in range(len(image_paths)):
                for pos_idx in indices:
                    i = pos_idx // grid_size
                    j = pos_idx % grid_size
                    left = int(j * self.step_w)
                    top = int(i * self.step_h)
                    self.crop_positions.append((img_idx, left, top))
        
        # ON-DEMAND loading
        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Morphology-preserving augmentations
        self.flip_h = RandomHorizontalFlip(p=0.5)
        self.flip_v = RandomVerticalFlip(p=0.5)
        self.rotate_90 = RandomChoice([
            Lambda(lambda x: x.rotate(0)),
            Lambda(lambda x: x.rotate(90)),
            Lambda(lambda x: x.rotate(180)),
            Lambda(lambda x: x.rotate(270)),
        ])
        self.color = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
    
    def __len__(self):
        return len(self.crop_positions)
    
    def __getitem__(self, idx):
        img_idx, left, top = self.crop_positions[idx]
        
        # Load image ON-DEMAND from disk
        img = Image.open(self.image_paths[img_idx]).convert('RGB')
        
        # Extract crop directly at target size (NO RESIZE)
        crop = img.crop((left, top, left + self.crop_size, top + self.crop_size))
        
        # Morphology-preserving augmentations
        if self.augment:
            crop = self.flip_h(crop)
            crop = self.flip_v(crop)
            crop = self.rotate_90(crop)
            crop = self.color(crop)
        
        # Transform
        crop = self.to_tensor(crop)
        crop = self.normalize(crop)
        
        return crop, self.labels[img_idx]


# =============================================================================
# Gradual Unfreezing with State Tracking (Best Practice)
# =============================================================================
_unfrozen_state = {
    'classifier': False,
    'features_top': False,
    'features_mid': False,
    'features_bottom': False,
    'features_stem': False
}

def reset_unfreeze_state():
    """Reset unfreeze state (call before training starts)"""
    global _unfrozen_state
    _unfrozen_state = {
        'classifier': False,
        'features_top': False,
        'features_mid': False,
        'features_bottom': False,
        'features_stem': False
    }


def unfreeze_by_epoch(model, epoch):
    """Gradually unfreeze EfficientNet-B0 layers - ONLY unfreeze NEW layers, never re-freeze.
    
    Best practices from research (domain shift solution):
    1. Freeze backbone LONGER to avoid overfitting to domain shift
    2. Only unfreeze, never re-freeze already unfrozen layers
    3. Use differential learning rates (lower for backbone)
    """
    global _unfrozen_state
    
    newly_unfrozen = []
    
    # Phase 1: Epoch 1 - Classifier only (freeze backbone longer)
    if epoch >= 1 and not _unfrozen_state['classifier']:
        for param in model.backbone.classifier.parameters():
            param.requires_grad = True
        _unfrozen_state['classifier'] = True
        newly_unfrozen.append('classifier')
    
    # Phase 2: Epoch 5 - Unfreeze top MBConv blocks (stages 16-18) - SLOWER
    if epoch >= 5 and not _unfrozen_state['features_top']:
        for param in model.backbone.features[-3:].parameters():
            param.requires_grad = True
        _unfrozen_state['features_top'] = True
        newly_unfrozen.append('features_top')
    
    # Phase 3: Epoch 8 - Unfreeze middle MBConv blocks (stages 10-15) - SLOWER
    if epoch >= 8 and not _unfrozen_state['features_mid']:
        for param in model.backbone.features[-6:-3].parameters():
            param.requires_grad = True
        _unfrozen_state['features_mid'] = True
        newly_unfrozen.append('features_mid')
    
    # Phase 4: Epoch 11 - Unfreeze bottom MBConv blocks (stages 4-9) - SLOWER
    if epoch >= 11 and not _unfrozen_state['features_bottom']:
        for param in model.backbone.features[4:-6].parameters():
            param.requires_grad = True
        _unfrozen_state['features_bottom'] = True
        newly_unfrozen.append('features_bottom')
    
    # Phase 5: Epoch 14 - Unfreeze Stem and early layers (stages 0-3) - SLOWER
    if epoch >= 14 and not _unfrozen_state['features_stem']:
        for param in model.backbone.features[:4].parameters():
            param.requires_grad = True
        _unfrozen_state['features_stem'] = True
        newly_unfrozen.append('features_stem')
    
    return newly_unfrozen


def rebuild_optimizer_with_differential_lr(model, optimizer, epoch):
    """Rebuild optimizer with differential LR when layers are unfrozen.
    
    Research shows: classifier LR should be ~10x backbone LR for domain shift.
    """
    classifier_lr = args.lr
    backbone_lr = 1e-6
    
    if epoch == 5:
        classifier_lr = 1e-4
        backbone_lr = 1e-5  # ratio 10:1
    elif epoch == 8:
        classifier_lr = 5e-5
        backbone_lr = 5e-6  # ratio 10:1
    elif epoch == 11:
        classifier_lr = 3e-5
        backbone_lr = 3e-6  # ratio 10:1
    elif epoch >= 14:
        classifier_lr = 1e-5
        backbone_lr = 1e-6  # ratio 10:1
    else:
        return optimizer
    
    # Rebuild optimizer with new LRs
    param_groups = []
    
    # Classifier params
    classifier_params = [p for p in model.backbone.classifier.parameters() if p.requires_grad]
    if classifier_params:
        param_groups.append({'params': classifier_params, 'lr': classifier_lr})
    
    # Backbone params (features)
    backbone_params = [p for p in model.backbone.features.parameters() if p.requires_grad]
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': backbone_lr})
    
    # Create new optimizer only if there are trainable params
    if param_groups:
        new_optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)
        return new_optimizer
    
    return None


def get_differential_optimizer(model, classifier_lr=1e-3, backbone_lr=1e-5):
    """Create optimizer with differential learning rates for different layers."""
    
    # Count unfrozen parameters
    classifier_params = []
    backbone_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
                
    # Create optimizer with differential LR
    optimizer = torch.optim.AdamW([
        {'params': classifier_params, 'lr': classifier_lr},
        {'params': backbone_params, 'lr': backbone_lr},
    ], weight_decay=1e-5)
    
    return optimizer


train_paths: list[str] = []
for p in ['P1', 'P2', 'P3', 'P4']:
    train_paths.extend(glob.glob(os.path.join(BASE_DIR, p, '*.tif')))

val_paths: list[str] = glob.glob(os.path.join(BASE_DIR, 'P5', '*.tif'))
test_paths: list[str] = glob.glob(os.path.join(BASE_DIR, 'P6', '*.tif'))

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

# Compute class weights for imbalanced data
from collections import Counter
train_labels_list = []
for f in train_paths:
    label = get_label_from_path(f)
    if label:
        train_labels_list.append(label_to_idx[label])

class_counts = Counter(train_labels_list)
total_samples = len(train_labels_list)
class_weights = torch.tensor([total_samples / (num_classes * class_counts[i]) if class_counts[i] > 0 else 0 
                              for i in range(num_classes)], dtype=torch.float32).to(device)
print(f"Class weights computed (min: {class_weights.min():.2f}, max: {class_weights.max():.2f})")

# Transform for training (ToTensor + Normalize only, augmentation is in dataset)
train_transform: Compose = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# No augmentation for validation/testing (deterministic)
val_transform: Compose = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print(f"\n{'='*60}")
print("USING MIXED CROP DATASET - FULL GRID, SHUFFLED")
print("WITH GRADUAL UNFREEZING FOR EFFICIENTNET-B0")
print(f"{'='*60}")

# Create labels for each image path
def get_label_for_path(img_path):
    label_str = get_label_from_path(img_path)
    if label_str is None:
        label_str = "Unknown"
    return label_to_idx.get(label_str, 0)

train_labels = [get_label_for_path(p) for p in train_paths]
val_labels = [get_label_for_path(p) for p in val_paths]
test_labels = [get_label_for_path(p) for p in test_paths]

# Create datasets using MixedCropDataset
train_data = MixedCropDataset(
    train_paths, train_labels,
    crop_size=224, grid_size=12,
    n_crops_per_image=args.n_crops, augment=True
)
val_data = MixedCropDataset(
    val_paths, val_labels,
    crop_size=224, grid_size=12,
    n_crops_per_image=args.n_crops, augment=False
)
test_data = MixedCropDataset(
    test_paths, test_labels,
    crop_size=224, grid_size=12,
    n_crops_per_image=args.n_crops, augment=False
)

print(f"Using {args.n_crops} crops per image")
print(f"Train: {len(train_data)} crops, Val: {len(val_data)} crops, Test: {len(test_data)} crops")

def worker_init_fn(worker_id: int) -> None:
    """Set random seed for each DataLoader worker for reproducibility."""
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

train_loader: DataLoader = DataLoader(
    train_data, 
    batch_size=args.batch_size,  # Auto-configured based on n_crops
    sampler=ShuffledCropSampler(len(train_data), shuffle=True),
    shuffle=False, 
    num_workers=12,  # 36 cores / 3 = 12 workers
    pin_memory=True, 
    prefetch_factor=3,  
    persistent_workers=True,
    worker_init_fn=worker_init_fn,
    drop_last=True
)
val_loader: DataLoader = DataLoader(
    val_data, 
    batch_size=args.batch_size,  # Same as train for consistency
    sampler=ShuffledCropSampler(len(val_data), shuffle=False),
    shuffle=False, 
    num_workers=12, 
    pin_memory=True, 
    prefetch_factor=3, 
    persistent_workers=True,
    worker_init_fn=worker_init_fn
)
test_loader: DataLoader = DataLoader(
    test_data, 
    batch_size=args.batch_size,  # Same as train for consistency
    sampler=ShuffledCropSampler(len(test_data), shuffle=False),
    shuffle=False, 
    num_workers=12, 
    pin_memory=True, 
    prefetch_factor=3, 
    persistent_workers=True,
    worker_init_fn=worker_init_fn
)




class EfficientNetClassifier(nn.Module):
    """Single-task EfficientNet-B0 for gene classification.
    
    Returns 1 output:
    - gene_logits: 85 classes (main task)
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = 1280
        
        # Replace the classifier with single head
        # Keep the original avgpool and just replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)  # Returns logits (85 classes)
        return x

model: nn.Module = EfficientNetClassifier(num_classes=num_classes)
model = model.to(device)

print(f"Single-task: {num_classes} gene classes")

# Freeze backbone initially - ALL backbone params frozen, only classifier trains
# This is crucial for domain shift (ImageNet → microscopy)
for param in model.backbone.parameters():
    param.requires_grad = False
print("Backbone frozen initially - only classifier will train")

# Reset state first, then call with epoch=1 to unfreeze classifier
reset_unfreeze_state()
unfreeze_by_epoch(model, epoch=1)  # Unfreeze classifier at epoch 1
print("Classifier unfrozen - training starts")

# Single loss function for single-task
criterion: nn.Module = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
# OPTIMIZED SETTINGS:
# - Classifier LR: args.lr (default 1e-3)
# - Backbone LR: starts at 1e-6, gradually increases as backbone unfreezes
# - Weight decay: 1e-5 (NVIDIA standard for EfficientNet-B0)
# Create optimizer - only classifier is trainable initially
# Note: model.backbone.parameters() includes classifier, so use features only
opt_params = [
    {'params': model.backbone.classifier.parameters(), 'lr': args.lr},
]
backbone_features = [p for n, p in model.backbone.named_parameters() if 'features' in n and p.requires_grad]
if backbone_features:
    opt_params.append({'params': backbone_features, 'lr': 1e-6})
    
optimizer: torch.optim.AdamW = torch.optim.AdamW(opt_params, weight_decay=1e-5)

# AMP (Automatic Mixed Precision) for faster training
scaler = torch.cuda.amp.GradScaler()

num_epochs: int = args.epochs
best_val_acc: float = 0.0

# No separate scheduler - LR is controlled by rebuild_optimizer_with_differential_lr
# This ensures LR changes properly when layers are unfrozen
# Research shows 10:1 ratio is optimal (classifier 10x backbone)
print(f"LR Schedule: Manual adjustment via gradual unfreezing (no scheduler)")
print(f"  Epoch 1:   classifier_lr={args.lr}, backbone_lr=1e-6 (ratio 1000:1, classifier only)")
print(f"  Epoch 5:   classifier_lr=1e-4, backbone_lr=1e-5 (ratio 10:1, top features)")
print(f"  Epoch 8:   classifier_lr=5e-5, backbone_lr=5e-6 (ratio 10:1, mid features)")
print(f"  Epoch 11:  classifier_lr=3e-5, backbone_lr=3e-6 (ratio 10:1, bottom features)")
print(f"  Epoch 14:  classifier_lr=1e-5, backbone_lr=1e-6 (ratio 10:1, all unfrozen)")

train_losses: list[float] = []
train_accs: list[float] = []
val_losses: list[float] = []
val_accs: list[float] = []

start_epoch = 0
if args.resume:
    checkpoint_path = os.path.join(BASE_DIR, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
            train_accs = checkpoint['train_accs']
            val_losses = checkpoint['val_losses']
            val_accs = checkpoint['val_accs']
            best_val_acc = checkpoint['best_val_acc']
            start_epoch = len(train_losses)
            print(f"Resuming from epoch {start_epoch+1} with best_val_acc: {best_val_acc:.2f}%")
        else:
            print("Warning: Old checkpoint format, starting fresh")
    else:
        print("Warning: best_model.pth not found, starting fresh")


class EarlyStopping:
    def __init__(self, patience: int = 13, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Single-task evaluation function"""
    model.eval()
    running_loss: float = 0.0
    correct: float = 0
    total: float = 0
    pbar = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for crops, labels in pbar:
            crops = crops.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                gene_logits = model(crops)
                loss = criterion(gene_logits, labels)
            
            running_loss += loss.item()
            _, predicted = gene_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100. * correct / total


def get_all_predictions_and_labels(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_labels: list[int] = []
    all_probs: list[torch.Tensor] = []
    
    pbar = tqdm(loader, desc="Predicting", leave=False)
    with torch.no_grad():
        for crops, labels in pbar:
            crops = crops.to(device, non_blocking=True)
            gene_logits = model(crops)
            probs: torch.Tensor = torch.softmax(gene_logits, dim=1)
            _, predicted = gene_logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.append(probs.cpu())
    
    return np.array(all_preds), np.array(all_labels), torch.cat(all_probs).numpy()


early_stopping = EarlyStopping(patience=10, min_delta=0.01, mode='min')

for epoch in tqdm(range(start_epoch, num_epochs), desc="Epochs"):
    # Apply gradual unfreezing at the start of each epoch
    newly_unfrozen = unfreeze_by_epoch(model, epoch)
    
    # Rebuild optimizer with differential LR when new layers unfreeze
    if newly_unfrozen:
        print(f"  -> Unfrozen: {newly_unfrozen}")
        new_optimizer = rebuild_optimizer_with_differential_lr(model, optimizer, epoch)
        if new_optimizer:
            optimizer = new_optimizer
            print(f"  -> Rebuilt optimizer with new LR schedule")
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for crops, labels in pbar:
        # MixedCropDataset returns (crop, label) - crops already flattened batch
        crops = crops.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Use set_to_none=True for faster gradient zeroing
        optimizer.zero_grad(set_to_none=True)
        
        # Use AMP for faster training
        with torch.amp.autocast('cuda'):
            # Single-task model returns gene_logits only
            gene_logits = model(crops)
            
            # Single task loss
            loss = criterion(gene_logits, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = gene_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss': running_loss/(pbar.n+1), 'acc': 100.*correct/total})
    
    # No scheduler.step() - LR controlled by rebuild_optimizer_with_differential_lr
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    val_loss, val_acc = evaluate(model, val_loader, device)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # Print unfrozen layer info
    unfrozen_count = sum(1 for p in model.parameters() if p.requires_grad)
    total_count = sum(1 for _ in model.parameters())
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f} | Unfrozen: {unfrozen_count}/{total_count}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'label_to_idx': label_to_idx,
            'idx_to_label': idx_to_label,
            'num_classes': num_classes,
            'all_labels': all_labels,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc,
            'epoch': epoch + 1,
            'val_acc': val_acc,
        }, os.path.join(BASE_DIR, 'best_model.pth'))
        print(f"  -> Saved best model (val acc: {val_acc:.2f}%)")
    
    if early_stopping(val_loss):
        print(f"\nEarly stopping triggered at epoch {epoch+1}. Best val acc: {best_val_acc:.2f}%")
        break

print(f"\nTraining complete. Best val acc: {best_val_acc:.2f}%")

checkpoint = torch.load(os.path.join(BASE_DIR, 'best_model.pth'), map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
test_loss, test_acc = evaluate(model, test_loader, device)
print(f"Test Accuracy: {test_acc:.2f}%")

# Plot training curves
actual_epochs = len(train_losses)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(range(1, actual_epochs+1), train_losses, label='Train', marker='o')
axes[0].plot(range(1, actual_epochs+1), val_losses, label='Validation', marker='o')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curves')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(range(1, actual_epochs+1), train_accs, label='Train', marker='o')
axes[1].plot(range(1, actual_epochs+1), val_accs, label='Validation', marker='o')
axes[1].axhline(y=test_acc, color='r', linestyle='--', label=f'Test: {test_acc:.2f}%')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy Curves')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'training_plots.png'), dpi=150)
plt.close()

# ROC Curve and Precision-Recall Curve
test_preds, test_labels, test_probs = get_all_predictions_and_labels(model, test_loader, device)

# Binarize labels for multi-class ROC/PR curves
test_labels_bin = label_binarize(test_labels, classes=list(range(num_classes)))

# Find which classes have samples in test set
classes_with_samples = [i for i in range(test_labels_bin.shape[1]) if test_labels_bin[:, i].sum() > 0]

# Compute metrics for ALL classes
fpr: dict = {}
tpr: dict = {}
roc_auc: dict = {}
precision: dict = {}
recall: dict = {}
ap: dict = {}

for i in tqdm(classes_with_samples, desc="Computing metrics"):
    fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], test_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    precision[i], recall[i], _ = precision_recall_curve(test_labels_bin[:, i], test_probs[:, i])
    ap[i] = average_precision_score(test_labels_bin[:, i], test_probs[:, i])

# Get best 8 classes by AUC (ensure unique names)
sorted_by_auc = sorted(roc_auc.items(), key=lambda x: x[1], reverse=True)
best_classes = []
seen_names = set()
for i, auc_val in sorted_by_auc:
    name = idx_to_label[i]
    if name not in seen_names:
        best_classes.append(i)
        seen_names.add(name)
        if len(best_classes) >= 8:
            break

# Plot best 8 classes
fig_roc, axes_roc = plt.subplots(2, 4, figsize=(16, 8))
axes_roc = axes_roc.flatten()

for idx, i in enumerate(best_classes):
    axes_roc[idx].plot(fpr[i], tpr[i], label=f'AUC = {roc_auc[i]:.2f}')
    axes_roc[idx].plot([0, 1], [0, 1], 'k--')
    axes_roc[idx].set_xlabel('False Positive Rate')
    axes_roc[idx].set_ylabel('True Positive Rate')
    axes_roc[idx].set_title(f'{idx_to_label[i]}')
    axes_roc[idx].legend(loc='lower right')

for j in range(len(best_classes), len(axes_roc)):
    axes_roc[j].axis('off')

plt.suptitle('ROC Curves (Best 8 Classes by AUC)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'roc_curves.png'), dpi=150)
plt.close()

# Get best 8 classes by AP (ensure unique names)
sorted_by_ap = sorted(ap.items(), key=lambda x: x[1], reverse=True)
best_classes_ap = []
seen_names_ap = set()
for i, ap_val in sorted_by_ap:
    name = idx_to_label[i]
    if name not in seen_names_ap:
        best_classes_ap.append(i)
        seen_names_ap.add(name)
        if len(best_classes_ap) >= 8:
            break

# Precision-Recall Curve
fig_pr, axes_pr = plt.subplots(2, 4, figsize=(16, 8))
axes_pr = axes_pr.flatten()

for idx, i in enumerate(best_classes_ap):
    axes_pr[idx].plot(recall[i], precision[i], label=f'AP = {ap[i]:.2f}')
    axes_pr[idx].set_xlabel('Recall')
    axes_pr[idx].set_ylabel('Precision')
    axes_pr[idx].set_title(f'{idx_to_label[i]}')
    axes_pr[idx].legend(loc='lower left')

for j in range(len(best_classes_ap), len(axes_pr)):
    axes_pr[j].axis('off')

plt.suptitle('Precision-Recall Curves (Best 8 Classes by AP)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'precision_recall_curves.png'), dpi=150)
plt.close()

# Print average metrics for ALL classes
mean_roc_auc = np.mean([roc_auc[i] for i in classes_with_samples])
mean_ap = np.mean([ap[i] for i in classes_with_samples])

print(f"\n=== Performance Metrics (ALL {len(classes_with_samples)} classes) ===")
print(f"Average ROC AUC: {mean_roc_auc:.4f}")
print(f"Average Precision: {mean_ap:.4f}")

# Top and bottom performing classes by AUC
sorted_by_auc = sorted(roc_auc.items(), key=lambda x: x[1], reverse=True)
print(f"\nTop 5 Classes by AUC:")
for i, val in sorted_by_auc[:5]:
    print(f"  {idx_to_label[i]}: {val:.4f}")
print(f"Bottom 5 Classes by AUC:")
for i, val in sorted_by_auc[-5:]:
    print(f"  {idx_to_label[i]}: {val:.4f}")

# Top and bottom by AP
sorted_by_ap = sorted(ap.items(), key=lambda x: x[1], reverse=True)
print(f"\nTop 5 Classes by AP:")
for i, val in sorted_by_ap[:5]:
    print(f"  {idx_to_label[i]}: {val:.4f}")
print(f"Bottom 5 Classes by AP:")
for i, val in sorted_by_ap[-5:]:
    print(f"  {idx_to_label[i]}: {val:.4f}")

# Save all results
from datetime import datetime
import csv

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. Save comprehensive results to JSON
results_json = {
    "timestamp": timestamp,
    "config": {
        "num_classes": num_classes,
        "num_epochs": num_epochs,
        "batch_size": 16,
        "train_patches": 5,
        "val_patches": 5,
        "test_patches": 50,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "patch_size": 224,
        "edge_margin": 200
    },
    "dataset": {
        "train_samples": len(train_paths),
        "val_samples": len(val_paths),
        "test_samples": len(test_paths),
        "train_plates": ["P1", "P2", "P3", "P4"],
        "val_plate": "P5",
        "test_plate": "P6"
    },
    "results": {
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "mean_roc_auc": float(mean_roc_auc),
        "mean_ap": float(mean_ap),
        "train_losses": [float(x) for x in train_losses],
        "train_accs": [float(x) for x in train_accs],
        "val_losses": [float(x) for x in val_losses],
        "val_accs": [float(x) for x in val_accs]
    },
    "class_metrics": {
        idx_to_label[i]: {
            "idx": int(i),
            "auc": float(roc_auc.get(i, 0)),
            "ap": float(ap.get(i, 0)),
            "sample_count": int(test_labels_bin[:, i].sum())
        } for i in classes_with_samples
    },
    "top_5_auc": [{"class": idx_to_label[i], "auc": float(val)} for i, val in sorted_by_auc[:5]],
    "bottom_5_auc": [{"class": idx_to_label[i], "auc": float(val)} for i, val in sorted_by_auc[-5:]],
    "top_5_ap": [{"class": idx_to_label[i], "ap": float(val)} for i, val in sorted_by_ap[:5]],
    "bottom_5_ap": [{"class": idx_to_label[i], "ap": float(val)} for i, val in sorted_by_ap[-5:]]
}

json_path = os.path.join(BASE_DIR, f'training_results_{timestamp}.json')
with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"\nResults JSON saved: {json_path}")

# 2. Save class metrics to CSV
csv_path = os.path.join(BASE_DIR, f'class_metrics_{timestamp}.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['class_name', 'class_idx', 'auc', 'ap', 'test_samples'])
    for i in sorted(classes_with_samples):
        writer.writerow([idx_to_label[i], i, f"{roc_auc.get(i, 0):.4f}", f"{ap.get(i, 0):.4f}", int(test_labels_bin[:, i].sum())])
print(f"Class metrics CSV saved: {csv_path}")

# 3. Save training log
log_path = os.path.join(BASE_DIR, f'training_log_{timestamp}.txt')
with open(log_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("TRAINING LOG\n")
    f.write("="*60 + "\n\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Device: {device}\n\n")
    
    f.write("CONFIGURATION:\n")
    f.write("-"*40 + "\n")
    f.write(f"Number of classes: {num_classes}\n")
    f.write(f"Epochs: {num_epochs}\n")
    f.write(f"Batch size: 16\n")
    f.write(f"Train patches per image: 5\n")
    f.write(f"Val patches per image: 5\n")
    f.write(f"Test patches per image: 50\n")
    f.write(f"Learning rate: 0.001\n")
    f.write(f"Weight decay: 0.0001\n")
    f.write(f"Patch size: 224\n")
    f.write(f"Edge margin: 200\n\n")
    
    f.write("DATASET:\n")
    f.write("-"*40 + "\n")
    f.write(f"Train samples: {len(train_paths)} (Plates: P1, P2, P3, P4)\n")
    f.write(f"Val samples: {len(val_paths)} (Plate: P5)\n")
    f.write(f"Test samples: {len(test_paths)} (Plate: P6)\n\n")
    
    f.write("TRAINING HISTORY:\n")
    f.write("-"*40 + "\n")
    for epoch in range(len(train_losses)):
        f.write(f"Epoch {epoch+1}: Train Loss={train_losses[epoch]:.4f}, Train Acc={train_accs[epoch]:.2f}%, "
                f"Val Loss={val_losses[epoch]:.4f}, Val Acc={val_accs[epoch]:.2f}%\n")
    
    f.write("\n" + "="*60 + "\n")
    f.write("FINAL RESULTS:\n")
    f.write("="*60 + "\n")
    f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
    f.write(f"Test Accuracy: {test_acc:.2f}%\n")
    f.write(f"Mean ROC AUC (all classes): {mean_roc_auc:.4f}\n")
    f.write(f"Mean Average Precision (all classes): {mean_ap:.4f}\n\n")
    
    f.write("TOP 5 CLASSES BY AUC:\n")
    for i, val in sorted_by_auc[:5]:
        f.write(f"  {idx_to_label[i]}: {val:.4f}\n")
    
    f.write("\nBOTTOM 5 CLASSES BY AUC:\n")
    for i, val in sorted_by_auc[-5:]:
        f.write(f"  {idx_to_label[i]}: {val:.4f}\n")
    
    f.write("\nTOP 5 CLASSES BY AP:\n")
    for i, val in sorted_by_ap[:5]:
        f.write(f"  {idx_to_label[i]}: {val:.4f}\n")
    
    f.write("\nBOTTOM 5 CLASSES BY AP:\n")
    for i, val in sorted_by_ap[-5:]:
        f.write(f"  {idx_to_label[i]}: {val:.4f}\n")
    
    f.write("\n" + "="*60 + "\n")
    f.write("OUTPUT FILES:\n")
    f.write("-"*40 + "\n")
    f.write(f"Model: best_model.pth\n")
    f.write(f"Classes: classes.txt\n")
    f.write(f"Plots: training_plots.png, roc_curves.png, precision_recall_curves.png\n")
    f.write(f"Results: {json_path}\n")
    f.write(f"CSV: {csv_path}\n")

print(f"Training log saved: {log_path}")

print("\n" + "="*60)
print("ALL FILES SAVED:")
print("="*60)
print(f"  - Model: best_model.pth")
print(f"  - Classes: classes.txt")
print(f"  - Plots: training_plots.png, roc_curves.png, precision_recall_curves.png")
print(f"  - Results JSON: training_results_{timestamp}.json")
print(f"  - Class CSV: class_metrics_{timestamp}.csv")
print(f"  - Log: training_log_{timestamp}.txt")
print("="*60)
