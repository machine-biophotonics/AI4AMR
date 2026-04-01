import argparse
import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn
import torchvision
from torchvision.transforms.functional import rotate, adjust_brightness, adjust_contrast
from torchvision import datasets, models
from torchvision.transforms import v2
from torchvision.transforms import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, CenterCrop, Normalize, Compose, RandomCrop, Lambda, RandomRotation, RandomAffine, RandomErasing, RandomChoice
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import glob
import json
import re
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from typing import Optional, List, Dict, Callable
import random
from tqdm import tqdm
from collections import Counter
import csv
from datetime import datetime


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Focal Loss implementation."""
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


def weighted_focal_loss(logits, targets, weights, alpha=0.25, gamma=2.0):
    """Weighted Focal Loss (combined class and domain weights)."""
    # logits: (B, C), targets: (B,), weights: (B,)
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    # Apply weights
    weighted = focal * weights
    return weighted.mean()


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # parent directory where P1-P6 are located

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
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

torch.set_num_threads(16)

print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# Argument Parser
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, default=None, help='Resume training from checkpoint path')
parser.add_argument('--n_crops', type=int, default=144, help='Deprecated, ignored. Kept for compatibility.')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=None, help='Batch size (default: 64)')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate for classifier')
parser.add_argument('--warmup_epochs', type=int, default=2, help='Number of warmup epochs')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
parser.add_argument('--min_delta', type=float, default=0.01, help='Early stopping min delta')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()

# Set seed from args
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Auto-set batch size (default 64)
if args.batch_size is None:
    args.batch_size = 64
print(f"Auto-configured: batch_size={args.batch_size}")

# Load plate data from JSON (verified mapping)
with open(os.path.join(BASE_DIR, 'plate maps', 'plate_well_id_path.json'), 'r') as f:
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

with open(os.path.join(SCRIPT_DIR, 'classes.txt'), 'w') as f:
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


# =============================================================================
# Grayscale-friendly MixedCropDataset
# =============================================================================
class GrayscaleMixedCropDataset(Dataset):
    """Mixed Crop Dataset with DINOv3's exact augmentation pipeline.
    
    Same grid as DINOv3:
    - Grid across WHOLE image (not just center)
    - Training: each image yields one random crop per epoch (position selected from grid)
    - Validation/Testing: each image yields center crop only
    - Exact same albumentations pipeline as DINOv3 for fair comparison
    - On-demand image loading with caching
    - Grayscale images stored as RGB (3 identical channels)
    """
    
    @staticmethod
    def _get_plate(img_path: str) -> str:
        """Extract plate name (P1-P6) from image path."""
        # Example: /media/.../P1/WellA01_...
        parts = img_path.split(os.sep)
        for part in parts:
            if part in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
                return part
        # fallback: assume unknown plate
        return 'unknown'
    def __init__(self, image_paths: list[str], labels: list[int], 
                 crop_size: int = 224, grid_size: int = 12,
                 augment: bool = True, seed: int = 42, epoch: int = 0):
        self.image_paths = image_paths
        self.labels = labels
        self.plates = [self._get_plate(path) for path in image_paths]
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.augment = augment
        self.seed = seed
        self.epoch = epoch
        
        # Get dimensions from first image (assume all same size)
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        self.image_size = w  # assume square
        
        # Compute stride as integer (matching DINOv3)
        stride = (w - crop_size) // (grid_size - 1)
        self.stride = stride
        
        # Generate grid positions as multiples of stride
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * stride
                top = i * stride
                # Ensure crop fits within image (should be true for 2720x2720)
                if left + crop_size <= w and top + crop_size <= h:
                    positions.append((left, top))
        self.positions = positions  # list of (left, top) for a single image
        
        # Shuffle positions per epoch (like DINOv3) for random cropping
        self.shuffled_positions = self.positions.copy()
        if augment:
            rng = random.Random(seed + epoch)
            rng.shuffle(self.shuffled_positions)
        
        # Image cache for performance (max 100 images)
        self._image_cache = {}
        self._cache_max_size = 100
        
        # Define albumentations transforms (matching DINOv3 exactly)
        if augment:
            self.transform = A.Compose([
                # No resize, crop already 224x224
                # Geometric transforms
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                         scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
                         rotate=(-15, 15), p=0.5),
                # Pixel-level intensity augmentations (grayscale-friendly)
                A.SomeOf([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    A.RandomToneCurve(scale=0.3, p=1.0),
                    A.RandomShadow(p=0.3),
                ], n=2, replace=False, p=0.5),
                # Noise and blur (grayscale-friendly)
                A.SomeOf([
                    A.GaussNoise(std_range=(0.1, 0.5), per_channel=False, p=1.0),  # monochrome noise
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                ], n=1, replace=False, p=0.5),
                # Cutout-like dropout
                A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(16, 64), hole_width_range=(16, 64), p=0.4),
                # Compression artifacts
                A.ImageCompression(quality_range=(50, 100), p=0.3),
                # Sharpening
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
                # Elastic distortions (more aggressive)
                A.ElasticTransform(alpha=50, sigma=5, p=0.2),
                # Normalization (using ImageNet stats, but channels are identical)
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            # No augmentation for validation/testing (center crop only)
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        
        print(f"GrayscaleMixedCrop: {len(self.positions)} positions per image, augment={augment}, epoch={epoch}")
        
    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch
        if self.augment:
            rng = random.Random(self.seed + epoch)
            self.shuffled_positions = self.positions.copy()
            rng.shuffle(self.shuffled_positions)
    
    def __len__(self):
        return len(self.image_paths)
    
    def _load_image(self, img_path):
        """Load image with caching."""
        if img_path not in self._image_cache:
            img = Image.open(img_path).convert('RGB')
            self._image_cache[img_path] = img
            # Keep cache size manageable
            if len(self._image_cache) > self._cache_max_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._image_cache))
                del self._image_cache[oldest_key]
        return self._image_cache[img_path]
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        plate = self.plates[idx]
        
        # Load image ON-DEMAND from disk
        img = self._load_image(img_path)
        w, h = img.size
        
        # Determine crop position
        if self.augment:
            # Random cropping: select position based on epoch and sample index (cycling)
            position_index = (self.epoch + idx) % len(self.shuffled_positions)
            left, top = self.shuffled_positions[position_index]
        else:
            # Center crop for validation/testing
            left = (w - self.crop_size) // 2
            top = (h - self.crop_size) // 2
        
        # Extract crop directly at target size (NO RESIZE)
        crop = img.crop((left, top, left + self.crop_size, top + self.crop_size))
        
        # Convert PIL image to numpy array (HWC, RGB)
        crop_np = np.array(crop)
        
        # Apply albumentations transform
        transformed = self.transform(image=crop_np)
        crop_tensor = transformed['image']
        
        return crop_tensor, label, plate


# =============================================================================
# Shuffled Crop Sampler (for DataLoader)
# =============================================================================
class ShuffledCropSampler:
    """Sampler that shuffles ALL crops across ALL images. Different order each epoch, no duplicates."""
    def __init__(self, total_crops: int, shuffle: bool = True, seed: int = 42, epoch: int = 0):
        self.indices = list(range(total_crops))
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch
        if shuffle:
            rng = random.Random(seed + epoch)
            rng.shuffle(self.indices)
    
    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch
        if self.shuffle:
            rng = random.Random(self.seed + epoch)
            rng.shuffle(self.indices)
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)


# =============================================================================
# Gradual Unfreezing with State Tracking
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
    """Gradually unfreeze EfficientNet-B0 layers.
    
    Phase schedule:
    - Epoch 1: Classifier only
    - Epoch 5: Top MBConv blocks (stages 16-18)
    - Epoch 8: Middle MBConv blocks (stages 10-15)
    - Epoch 11: Bottom MBConv blocks (stages 4-9)
    - Epoch 14: Stem and early layers (stages 0-3)
    """
    global _unfrozen_state
    
    newly_unfrozen = []
    
    # Phase 1: Epoch 1 - Classifier only
    if epoch >= 1 and not _unfrozen_state['classifier']:
        for param in model.backbone.classifier.parameters():
            param.requires_grad = True
        _unfrozen_state['classifier'] = True
        newly_unfrozen.append('classifier')
    
    # Phase 2: Epoch 5 - Unfreeze top MBConv blocks (stages 16-18)
    if epoch >= 5 and not _unfrozen_state['features_top']:
        for param in model.backbone.features[-3:].parameters():
            param.requires_grad = True
        _unfrozen_state['features_top'] = True
        newly_unfrozen.append('features_top')
    
    # Phase 3: Epoch 8 - Unfreeze middle MBConv blocks (stages 10-15)
    if epoch >= 8 and not _unfrozen_state['features_mid']:
        for param in model.backbone.features[-6:-3].parameters():
            param.requires_grad = True
        _unfrozen_state['features_mid'] = True
        newly_unfrozen.append('features_mid')
    
    # Phase 4: Epoch 11 - Unfreeze bottom MBConv blocks (stages 4-9)
    if epoch >= 11 and not _unfrozen_state['features_bottom']:
        for param in model.backbone.features[4:-6].parameters():
            param.requires_grad = True
        _unfrozen_state['features_bottom'] = True
        newly_unfrozen.append('features_bottom')
    
    # Phase 5: Epoch 14 - Unfreeze Stem and early layers (stages 0-3)
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


# =============================================================================
# Dataset Setup
# =============================================================================
train_paths: list[str] = []
for p in ['P1', 'P2', 'P3', 'P4']:
    train_paths.extend(glob.glob(os.path.join(BASE_DIR, p, '*.tif')))

val_paths: list[str] = glob.glob(os.path.join(BASE_DIR, 'P5', '*.tif'))
test_paths: list[str] = glob.glob(os.path.join(BASE_DIR, 'P6', '*.tif'))

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

# Compute class weights for imbalanced data
train_labels_list = []
for f in train_paths:
    label = get_label_from_path(f)
    if label:
        train_labels_list.append(label_to_idx[label])

class_counts = Counter(train_labels_list)
total_samples = len(train_labels_list)

# Create labels for each image path
def get_label_for_path(img_path):
    label_str = get_label_from_path(img_path)
    if label_str is None:
        label_str = "Unknown"
    return label_to_idx.get(label_str, 0)

train_labels = [get_label_for_path(p) for p in train_paths]
val_labels = [get_label_for_path(p) for p in val_paths]
test_labels = [get_label_for_path(p) for p in test_paths]

print(f"\n{'='*60}")
print("GRAYSCALE-FRIENDLY EFFICIENTNET-B0 TRAINING")
print(f"{'='*60}")

# Create datasets using GrayscaleMixedCropDataset
train_data = GrayscaleMixedCropDataset(
    train_paths, train_labels,
    crop_size=224, grid_size=12,
    augment=True, seed=SEED, epoch=0
)
val_data = GrayscaleMixedCropDataset(
    val_paths, val_labels,
    crop_size=224, grid_size=12,
    augment=False, seed=SEED, epoch=0
)
test_data = GrayscaleMixedCropDataset(
    test_paths, test_labels,
    crop_size=224, grid_size=12,
    augment=False, seed=SEED, epoch=0
)

print(f"Using {len(train_data.positions)} possible crop positions per image")
print(f"Train: {len(train_data)} images, Val: {len(val_data)} images, Test: {len(test_data)} images")

def worker_init_fn(worker_id: int) -> None:
    """Set random seed for each DataLoader worker for reproducibility."""
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

# Compute domain weights per plate (training plates only)
plate_counts = {}
for plate in train_data.plates:
    plate_counts[plate] = plate_counts.get(plate, 0) + 1
total_samples = sum(plate_counts.values())
print(f"Training samples per plate: {plate_counts}")

# Compute weights as per DINOv3: u_d = n_d^{-1/2}, v_d = min(u_d, 3 * min_j u_j), w_d = v_d / sum(v_j)
u = {p: n ** -0.5 for p, n in plate_counts.items()}
min_u = min(u.values())
v = {p: min(val, 3 * min_u) for p, val in u.items()}
sum_v = sum(v.values())
domain_weights_dict = {p: w / sum_v for p, w in v.items()}
print(f"Domain weights: {domain_weights_dict}")

# Normalize class weights to sum to num_classes (as in DINOv3)
class_counts = Counter(train_labels_list)  # already computed earlier
total_samples = len(train_labels_list)
n_classes = num_classes
class_weights_dict = {}
for c in range(n_classes):
    count = class_counts.get(c, 1)  # avoid division by zero
    class_weights_dict[c] = total_samples / (n_classes * count)
# Normalize class weights to sum to n_classes
sum_class_weights = sum(class_weights_dict.values())
for c in class_weights_dict:
    class_weights_dict[c] = class_weights_dict[c] * n_classes / sum_class_weights
# Convert to tensor on device
class_weights = torch.tensor([class_weights_dict[c] for c in range(n_classes)], dtype=torch.float32).to(device)
print(f"Class weights (normalized): min {class_weights.min():.2f}, max {class_weights.max():.2f}")

# Helper functions to get weights per batch
def get_domain_weights(plates: List[str]) -> torch.Tensor:
    weights = [domain_weights_dict[p] for p in plates]
    return torch.tensor(weights, dtype=torch.float32, device=device)

def get_class_weights(class_idxs: List[int]) -> torch.Tensor:
    weights = [class_weights_dict[c] for c in class_idxs]
    return torch.tensor(weights, dtype=torch.float32, device=device)

def get_combined_weights(plates: List[str], class_idxs: List[int]) -> torch.Tensor:
    class_w = get_class_weights(class_idxs)
    domain_w = get_domain_weights(plates)
    combined = class_w * domain_w
    # Normalize to mean=1 for stable loss scale
    mean_weight = combined.mean()
    if mean_weight > 0:
        combined = combined / mean_weight
    return combined

# No samplers needed - dataset handles position selection per epoch

train_loader: DataLoader = DataLoader(
    train_data, 
    batch_size=args.batch_size,
    shuffle=False,  # image order fixed, crop positions shuffled via set_epoch
    num_workers=12,
    pin_memory=True, 
    prefetch_factor=3,  
    persistent_workers=True,
    worker_init_fn=worker_init_fn,
    drop_last=True
)
val_loader: DataLoader = DataLoader(
    val_data, 
    batch_size=args.batch_size,
    shuffle=False, 
    num_workers=12, 
    pin_memory=True, 
    prefetch_factor=3, 
    persistent_workers=True,
    worker_init_fn=worker_init_fn
)
test_loader: DataLoader = DataLoader(
    test_data, 
    batch_size=args.batch_size,
    shuffle=False, 
    num_workers=12, 
    pin_memory=True, 
    prefetch_factor=3, 
    persistent_workers=True,
    worker_init_fn=worker_init_fn
)


# =============================================================================
# Model Definition
# =============================================================================
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
for param in model.backbone.parameters():
    param.requires_grad = False
print("Backbone frozen initially - only classifier will train")

# Reset state first, then call with epoch=1 to unfreeze classifier
reset_unfreeze_state()
unfreeze_by_epoch(model, epoch=1)  # Unfreeze classifier at epoch 1
print("Classifier unfrozen - training starts")

# Focal loss with combined weights computed per batch
ALPHA = 0.25
GAMMA = 2.0

# Create optimizer - only classifier is trainable initially
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

# LR Schedule: Manual adjustment via gradual unfreezing
print(f"LR Schedule: Manual adjustment via gradual unfreezing (no scheduler)")
print(f"  Epoch 1:   classifier_lr={args.lr}, backbone_lr=1e-6 (classifier only)")
print(f"  Epoch 5:   classifier_lr=1e-4, backbone_lr=1e-5 (top features)")
print(f"  Epoch 8:   classifier_lr=5e-5, backbone_lr=5e-6 (mid features)")
print(f"  Epoch 11:  classifier_lr=3e-5, backbone_lr=3e-6 (bottom features)")
print(f"  Epoch 14:  classifier_lr=1e-5, backbone_lr=1e-6 (all unfrozen)")

train_losses: list[float] = []
train_accs: list[float] = []
val_losses: list[float] = []
val_accs: list[float] = []

start_epoch = 0
if args.resume:
    checkpoint_path = args.resume
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
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
        print(f"Warning: {checkpoint_path} not found, starting fresh")


# =============================================================================
# Early Stopping
# =============================================================================
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.01, mode: str = 'min'):
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


# =============================================================================
# Evaluation Function
# =============================================================================
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, return_preds: bool = False) -> tuple:
    """Single-task evaluation function with focal loss and combined weights."""
    model.eval()
    running_loss: float = 0.0
    correct: float = 0
    total: float = 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[torch.Tensor] = []
    
    pbar = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for crops, labels, plates in pbar:
            crops = crops.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                gene_logits = model(crops)
                # Compute combined weights (class * domain) normalized
                combined_weights = get_combined_weights(plates, labels.tolist())
                loss = weighted_focal_loss(gene_logits, labels, combined_weights, alpha=ALPHA, gamma=GAMMA)
            
            running_loss += loss.item()
            probs = torch.softmax(gene_logits, dim=1)
            _, predicted = gene_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if return_preds:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.append(probs.cpu())
    
    if return_preds:
        return running_loss / len(loader), 100. * correct / total, np.array(all_preds), np.array(all_labels), torch.cat(all_probs).numpy()
    return running_loss / len(loader), 100. * correct / total


# =============================================================================
# Training Loop
# =============================================================================
early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta, mode='min')

# CSV logging
csv_path = os.path.join(SCRIPT_DIR, f'training_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr', 'unfrozen_params', 'total_params'])

for epoch in tqdm(range(start_epoch, num_epochs), desc="Epochs"):
    # Set epoch for datasets (shuffles crop positions for training)
    train_data.set_epoch(epoch)
    val_data.set_epoch(epoch)
    test_data.set_epoch(epoch)
    
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
    
    for crops, labels, plates in pbar:
        crops = crops.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Use set_to_none=True for faster gradient zeroing
        optimizer.zero_grad(set_to_none=True)
        
        # Use AMP for faster training
        with torch.amp.autocast('cuda'):
            gene_logits = model(crops)
            # Compute combined weights (class * domain) normalized
            combined_weights = get_combined_weights(plates, labels.tolist())
            loss = weighted_focal_loss(gene_logits, labels, combined_weights, alpha=ALPHA, gamma=GAMMA)
        
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
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    # Validate every epoch
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
    
    # Log to CSV
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc, current_lr, unfrozen_count, total_count])
    
    # Save best model
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
        }, os.path.join(SCRIPT_DIR, 'best_model.pth'))
        print(f"  -> Saved best model (val acc: {val_acc:.2f}%)")
    
    if early_stopping(val_loss):
        print(f"\nEarly stopping triggered at epoch {epoch+1}. Best val acc: {best_val_acc:.2f}%")
        break

print(f"\nTraining complete. Best val acc: {best_val_acc:.2f}%")

# =============================================================================
# Test Evaluation
# =============================================================================
checkpoint = torch.load(os.path.join(SCRIPT_DIR, 'best_model.pth'), map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(model, test_loader, device, return_preds=True)
print(f"Test Accuracy: {test_acc:.2f}%")

# =============================================================================
# Plot Training Curves
# =============================================================================
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
plt.savefig(os.path.join(SCRIPT_DIR, 'training_plots.png'), dpi=150)
plt.close()

# =============================================================================
# ROC and Precision-Recall Curves
# =============================================================================

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
plt.savefig(os.path.join(SCRIPT_DIR, 'roc_curves.png'), dpi=150)
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
plt.savefig(os.path.join(SCRIPT_DIR, 'precision_recall_curves.png'), dpi=150)
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

# =============================================================================
# Save Results
# =============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. Save comprehensive results to JSON
results_json = {
    "timestamp": timestamp,
    "config": {
        "num_classes": num_classes,
        "num_epochs": num_epochs,
        "batch_size": args.batch_size,
        "n_crops": args.n_crops,
        "learning_rate": args.lr,
        "weight_decay": 1e-5,
        "patch_size": 224,
        "grid_size": 12,
        "seed": SEED,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "augmentation": "grayscale_friendly (no color jitter)"
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

json_path = os.path.join(SCRIPT_DIR, f'training_results_{timestamp}.json')
with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"\nResults JSON saved: {json_path}")

# 2. Save class metrics to CSV
class_csv_path = os.path.join(SCRIPT_DIR, f'class_metrics_{timestamp}.csv')
with open(class_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['class_name', 'class_idx', 'auc', 'ap', 'test_samples'])
    for i in sorted(classes_with_samples):
        writer.writerow([idx_to_label[i], i, f"{roc_auc.get(i, 0):.4f}", f"{ap.get(i, 0):.4f}", int(test_labels_bin[:, i].sum())])
print(f"Class metrics CSV saved: {class_csv_path}")

# 3. Save training log
log_path = os.path.join(SCRIPT_DIR, f'training_log_{timestamp}.txt')
with open(log_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("GRAYSCALE-FRIENDLY EFFICIENTNET-B0 TRAINING LOG\n")
    f.write("="*60 + "\n\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Device: {device}\n")
    f.write(f"Seed: {SEED}\n\n")
    
    f.write("CONFIGURATION:\n")
    f.write("-"*40 + "\n")
    f.write(f"Number of classes: {num_classes}\n")
    f.write(f"Epochs: {num_epochs}\n")
    f.write(f"Batch size: {args.batch_size}\n")
    f.write(f"Crops per image: {args.n_crops}\n")
    f.write(f"Learning rate: {args.lr}\n")
    f.write(f"Weight decay: 1e-5\n")
    f.write(f"Patch size: 224\n")
    f.write(f"Grid size: 12x12\n")
    f.write(f"Early stopping patience: {args.patience}\n")
    f.write(f"Early stopping min delta: {args.min_delta}\n")
    f.write(f"Augmentation: Grayscale-friendly (NO color jitter)\n\n")
    
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
    f.write(f"Class CSV: {class_csv_path}\n")
    f.write(f"Training CSV: {csv_path}\n")
    f.write(f"Log: {log_path}\n")

print(f"Training log saved: {log_path}")

print("\n" + "="*60)
print("ALL FILES SAVED:")
print("="*60)
print(f"  - Model: best_model.pth")
print(f"  - Classes: classes.txt")
print(f"  - Plots: training_plots.png, roc_curves.png, precision_recall_curves.png")
print(f"  - Results JSON: training_results_{timestamp}.json")
print(f"  - Class CSV: class_metrics_{timestamp}.csv")
print(f"  - Training CSV: {csv_path}")
print(f"  - Log: {log_path}")
print("="*60)
