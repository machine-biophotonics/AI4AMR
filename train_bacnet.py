"""
BacNet_train.py - Optimized Custom Lightweight CNN with Mixed Crop Sampler
=========================================================================
OPTIMIZATIONS:
1. Mixed Precision (AMP) - 2-3x speedup on modern GPUs
2. torch.compile() - Graph optimization
3. Pre-load images - Fast data access
4. High num_workers - Parallel data loading
5. Gradient clipping - Stable training
6. NO resize - Crops directly at target size

APPROACH:
1. ALL crop positions from each image (12×12 grid = 144 crops per image)
2. Batches contain MIXED crops from DIFFERENT images
3. NO duplicate crops within the same epoch
4. Effectively covers the WHOLE IMAGE
"""

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn
import torchvision
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import os
import glob
import json
import re
import random
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import time
import numpy as np
from typing import List, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch: {torch.__version__}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true', help='Resume training')
parser.add_argument('--no-compile', action='store_true', help='Skip torch.compile')
parser.add_argument('--grid_size', type=int, default=12, help='Grid size for crops (n×n)')
parser.add_argument('--crop_size', type=int, default=224, help='Size of each crop')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--n_crops', type=int, default=None, help='Number of crops per image (default: all 144)')
args = parser.parse_args()

CONFIG = {
    'grid_size': args.grid_size,
    'crop_size': args.crop_size,
    'batch_size': args.batch_size,
    'epochs': args.epochs,
    'lr': 3e-3,
    'weight_decay': 1e-3,
    'dropout': 0.4,
    'label_smoothing': 0.1,
    'num_workers': 32,
    'use_amp': True,
    'use_compile': not args.no_compile,
    'n_crops_per_image': args.n_crops if args.n_crops else args.grid_size ** 2,
}
print(f"\n{'='*60}")
print("BacNet V3 Training - GhostNetV2-inspired + Multi-Task")
print(f"{'='*60}")
print(f"Grid: {CONFIG['grid_size']}×{CONFIG['grid_size']} = {CONFIG['grid_size']**2} positions available")
print(f"Crops used: {CONFIG['n_crops_per_image']} per image")
print(f"Augmentation: Morphology-preserving (flip, rotate 90°, color)")
print(f"Multi-task: Gene (85) + Pathway (7) + Binary (WT vs KD)")
print(f"DFC Attention: Enabled (GhostNetV2)")
print(f"LR: {CONFIG['lr']} with 2-epoch warmup + Cosine")
print(f"{'='*60}\n")

print(f"\n{'='*60}")
print("BacNet Training - OPTIMIZED")
print(f"{'='*60}")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
print(f"  n_crops_per_image: {CONFIG['grid_size']}×{CONFIG['grid_size']} = {CONFIG['n_crops_per_image']}")
print(f"{'='*60}\n")

# Load plate data
with open(os.path.join(BASE_DIR, 'plate_well_id_path.json'), 'r') as f:
    plate_data = json.load(f)

plate_maps = {}
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_maps[plate] = {}
    for row, wells in plate_data[plate].items():
        for col, info in wells.items():
            well = f"{row}{int(col):02d}"
            plate_maps[plate][well] = info['id']

pathway_json_path = os.path.join(BASE_DIR, 'class_pathway_order.json')
with open(pathway_json_path) as f:
    pathway_data = json.load(f)
pathway_order = pathway_data['pathway_order']
all_labels_set = set(label for pm in plate_maps.values() for label in pm.values())
all_labels = [label for label in pathway_order if label in all_labels_set]
for label in sorted(all_labels_set - set(all_labels)):
    all_labels.append(label)
label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
num_classes = len(all_labels)
print(f"Classes: {num_classes}")

with open(os.path.join(BASE_DIR, 'classes_bacnet.txt'), 'w') as f:
    for i, label in enumerate(all_labels):
        f.write(f"{i},{label}\n")

# Create pathway labels mapping (gene_idx -> pathway_idx)
# Based on class_pathway_order.json pathway_mapping
pathway_mapping = pathway_data['pathway_mapping']
gene_to_pathway = {}
for gene_name, info in pathway_mapping.items():
    pathway_idx = info['pathway_order']
    for label in all_labels:
        if label.startswith(gene_name + '_') or (gene_name == 'WT' and label == 'WT'):
            gene_to_pathway[label_to_idx[label]] = pathway_idx

# Create binary labels mapping (gene_idx -> binary_label)
# WT = 0 (control), KD = 1 (knockdown)
gene_to_binary = {}
for gene_idx, label in idx_to_label.items():
    gene_to_binary[gene_idx] = 0 if label == 'WT' else 1

# Create lookup tensors for fast mapping
gene_to_pathway_tensor = torch.tensor(
    [gene_to_pathway.get(i, 0) for i in range(num_classes)], dtype=torch.long
)
gene_to_binary_tensor = torch.tensor(
    [gene_to_binary.get(i, 1) for i in range(num_classes)], dtype=torch.long
)

print(f"Pathway mapping: {len(gene_to_pathway)} genes mapped to 7 pathways")
print(f"Binary mapping: {(gene_to_binary_tensor == 0).sum()} WT, {(gene_to_binary_tensor == 1).sum()} KD")

def extract_well_from_filename(filename):
    match = re.search(r'Well(\w)(\d+)_', filename)
    if match:
        return f"{match.group(1)}{int(match.group(2)):02d}"
    return None

def get_label_from_path(img_path):
    dirname = os.path.basename(os.path.dirname(img_path))
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if dirname in plate_maps and well in plate_maps[dirname]:
        return plate_maps[dirname][well]
    return None


# =============================================================================
# BacNet V2: GhostNet-inspired Lightweight CNN (2026 Research)
# =============================================================================
class GhostModule(nn.Module):
    """
    Ghost Module from GhostNet - generates more features with fewer parameters.
    Uses cheap linear operations to generate "ghost" features from primary features.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, dw_kernel_size=3, ratio=2):
        super().__init__()
        self.primary_channels = out_channels // ratio
        self.ghost_channels = out_channels - self.primary_channels
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.primary_channels, kernel_size, bias=False),
            nn.BatchNorm2d(self.primary_channels),
            nn.ReLU(inplace=True)
        )
        
        self.ghost_conv = nn.Sequential(
            nn.Conv2d(self.primary_channels, self.ghost_channels, dw_kernel_size, 
                     stride=1, padding=dw_kernel_size//2, groups=self.primary_channels, bias=False),
            nn.BatchNorm2d(self.ghost_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        primary = self.primary_conv(x)
        ghost = self.ghost_conv(primary)
        return torch.cat([primary, ghost], dim=1)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excite(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial Attention Module (from CBAM) - focuses on where to attend"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class DFCAttention(nn.Module):
    """
    Decoupled Fully Connected (DFC) Attention from GhostNetV2.
    Captures long-range spatial dependencies efficiently.
    Hardware-friendly alternative to self-attention.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        fc_h = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        fc_w = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        self.fc_h = fc_h
        self.fc_w = fc_w
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Horizontal FC
        x_h = x.permute(0, 3, 2, 1).contiguous()  # [b, w, h, c]
        x_h = x_h.view(-1, c)
        x_h = self.fc_h(x_h)
        x_h = x_h.view(b, w, h, c).permute(0, 2, 1, 3)  # [b, h, w, c]
        
        # Vertical FC
        x_w = x.permute(0, 2, 3, 1).contiguous()  # [b, h, w, c]
        x_w = x_w.view(-1, c)
        x_w = self.fc_w(x_w)
        x_w = x_w.view(b, h, w, c)  # [b, h, w, c]
        
        # Combine and reshape
        attn = (x_h + x_w).permute(0, 3, 1, 2).contiguous()  # [b, c, h, w]
        attn = self.sigmoid(attn)
        
        return x * attn


class GhostBottleneck(nn.Module):
    """
    Ghost Bottleneck with MobileNetV2-style inverted residuals + SE + Spatial attention.
    Based on GhostNetV3 research (2024-2026).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=3, use_se=True):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        
        # Expand (pointwise conv)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
        
        # Depthwise conv
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=stride, 
                     padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        ])
        
        # SE attention
        if use_se:
            layers.append(SEBlock(hidden_dim, reduction=4))
        
        # Spatial attention
        layers.append(SpatialAttention())
        
        # Project (pointwise conv)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        
        # Shortcut connection
        if not (self.use_residual):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            return out + x
        else:
            return out + self.shortcut(x)


class BacNet(nn.Module):
    """
    BacNet V3: GhostNet-inspired Lightweight CNN with Multi-Task Learning
    
    Architecture based on 2024-2026 research:
    - Ghost modules: Generate more features with fewer parameters
    - MobileNetV2-style inverted residuals: Expand-collapse channels
    - SE attention: Channel recalibration
    - Spatial attention: Focus on relevant regions
    - DFC attention (GhostNetV2): Long-range spatial dependencies
    - Multi-task heads: Gene (85) + Pathway (7) + Binary (WT vs KD)
    
    ~671K parameters
    """
    def __init__(self, num_gene_classes=85, num_pathway_classes=7, dropout=0.4):
        super().__init__()
        self.num_gene_classes = num_gene_classes
        self.num_pathway_classes = num_pathway_classes
        
        # Stem: Standard conv
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Ghost Bottleneck stages
        # Stage 1: 112x112 -> 56x56
        self.stage1 = nn.Sequential(
            GhostBottleneck(16, 32, kernel_size=3, stride=2, expand_ratio=3, use_se=True),
            GhostBottleneck(32, 32, kernel_size=3, stride=1, expand_ratio=3, use_se=True)
        )
        
        # Stage 2: 56x56 -> 28x28
        self.stage2 = nn.Sequential(
            GhostBottleneck(32, 48, kernel_size=3, stride=2, expand_ratio=3, use_se=True),
            GhostBottleneck(48, 48, kernel_size=3, stride=1, expand_ratio=3, use_se=True)
        )
        
        # Stage 3: 28x28 -> 14x14
        self.stage3 = nn.Sequential(
            GhostBottleneck(48, 96, kernel_size=3, stride=2, expand_ratio=3, use_se=True),
            GhostBottleneck(96, 96, kernel_size=3, stride=1, expand_ratio=3, use_se=True)
        )
        
        # Stage 4: 14x14 -> 7x7
        self.stage4 = nn.Sequential(
            GhostBottleneck(96, 160, kernel_size=3, stride=2, expand_ratio=3, use_se=True),
            GhostBottleneck(160, 160, kernel_size=3, stride=1, expand_ratio=3, use_se=True)
        )
        
        # Final conv with DFC attention
        self.final_conv = nn.Sequential(
            nn.Conv2d(160, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # DFC attention after final conv (captures global context)
        self.dfc_attention = DFCAttention(256, reduction=8)
        
        # Shared pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
        
        # Multi-task heads
        self.gene_head = nn.Linear(256, num_gene_classes)
        self.pathway_head = nn.Linear(256, num_pathway_classes)
        self.binary_head = nn.Linear(256, 2)  # WT vs KD
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.final_conv(x)
        x = self.dfc_attention(x)
        
        # Shared features
        features = self.pool(x)
        features = self.flatten(features)
        features = self.dropout(features)
        
        # Multi-task outputs
        gene_logits = self.gene_head(features)
        pathway_logits = self.pathway_head(features)
        binary_logits = self.binary_head(features)
        
        return gene_logits, pathway_logits, binary_logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# MixedCropDataset: Optimized Dataset (NO RESIZE)
# Morphology-preserving augmentations only
# =============================================================================
class MixedCropDataset(Dataset):
    """
    Optimized dataset with NO resize.
    Crops directly at target size (crop_size).
    
    Augmentations (MORPHOLOGY-PRESERVING ONLY):
    - Horizontal flip (mirror, no distortion)
    - Vertical flip (mirror, no distortion)
    - 90° rotation steps (no interpolation, preserves fine features)
    - Color jitter (brightness, contrast, saturation, hue)
    
    NOT included (distorts morphology):
    - Random rotation (0-360° due to interpolation)
    - Shear transforms
    - Elastic deformation
    - CutMix/MixUp
    """
    def __init__(self, image_paths: List[str], labels: List[int], 
                 crop_size: int, grid_size: int,
                 norm_mean: List[float], norm_std: List[float],
                 n_crops_per_image: int = None,
                 use_augmentation: bool = True):
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.use_augmentation = use_augmentation
        
        # Get dimensions
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        
        # Calculate grid positions
        self.total_w = w - crop_size
        self.total_h = h - crop_size
        self.step_w = self.total_w / (grid_size - 1) if grid_size > 1 else 0
        self.step_h = self.total_h / (grid_size - 1) if grid_size > 1 else 0
        
        # Pre-calculate crop positions
        self.all_positions = []
        for img_idx in range(len(image_paths)):
            for i in range(grid_size):
                for j in range(grid_size):
                    left = int(j * self.step_w)
                    top = int(i * self.step_h)
                    self.all_positions.append((img_idx, left, top))
        
        # Sample crops per image
        self.n_crops_per_image = n_crops_per_image if n_crops_per_image else grid_size * grid_size
        
        if self.n_crops_per_image < grid_size * grid_size:
            # Sample subset of positions - NO duplicates
            positions_per_image = grid_size * grid_size
            crops_to_sample = min(self.n_crops_per_image, positions_per_image)
            indices = random.sample(range(positions_per_image), crops_to_sample)
            self.crop_positions = []
            for img_idx in range(len(image_paths)):
                for pos_idx in indices:
                    i = pos_idx // grid_size
                    j = pos_idx % grid_size
                    left = int(j * self.step_w)
                    top = int(i * self.step_h)
                    self.crop_positions.append((img_idx, left, top))
        else:
            self.crop_positions = self.all_positions
        
        print(f"  Using {self.n_crops_per_image} crops per image")
        print(f"  Total crops: {len(self.crop_positions)}")
        
        # ON-DEMAND loading
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=norm_mean, std=norm_std)
        
        # Morphology-preserving augmentations
        self.flip_h = T.RandomHorizontalFlip(p=0.5)
        self.flip_v = T.RandomVerticalFlip(p=0.5)
        # Discrete 90° rotation steps only - no interpolation
        self.rotate_90 = T.RandomChoice([
            T.Lambda(lambda x: x.rotate(0)),
            T.Lambda(lambda x: x.rotate(90)),
            T.Lambda(lambda x: x.rotate(180)),
            T.Lambda(lambda x: x.rotate(270)),
        ])
        # Color jitter - no morphological changes
        self.color = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
    
    def __len__(self):
        return len(self.crop_positions)
    
    def __getitem__(self, idx):
        img_idx, left, top = self.crop_positions[idx]
        
        # Load image ON-DEMAND from disk
        img = Image.open(self.image_paths[img_idx]).convert('RGB')
        
        # Extract crop directly at target size (NO RESIZE)
        crop = img.crop((left, top, left + self.crop_size, top + self.crop_size))
        
        # Morphology-preserving augmentations
        if self.use_augmentation:
            crop = self.flip_h(crop)
            crop = self.flip_v(crop)
            crop = self.rotate_90(crop)
            crop = self.color(crop)
        
        # Transform
        crop = self.to_tensor(crop)
        crop = self.normalize(crop)
        
        return crop, self.labels[img_idx]


class ShuffledCropSampler(Sampler):
    """Sampler that shuffles ALL crops. Different order each epoch, no duplicates."""
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


# =============================================================================
# Compute Dataset Statistics
# =============================================================================
def compute_dataset_stats(image_paths, sample_size=1000):
    """Compute dataset-specific mean and std."""
    print(f"\nComputing dataset statistics from {sample_size} images...")
    
    to_tensor = T.ToTensor()
    means = []
    
    indices = np.random.choice(len(image_paths), min(sample_size, len(image_paths)), replace=False)
    for i, idx in enumerate(tqdm(indices, desc="Computing stats")):
        img = np.array(Image.open(image_paths[idx]).convert('RGB'))
        tensor = to_tensor(img)
        means.append(tensor.mean(dim=(1, 2)).numpy())
    
    mean_r, mean_g, mean_b = np.array(means).mean(axis=0)
    
    std_sample = []
    for idx in indices[:200]:
        img = np.array(Image.open(image_paths[idx]).convert('RGB'))
        tensor = to_tensor(img)
        std_sample.append(tensor.std(dim=(1, 2)).numpy())
    
    std_r, std_g, std_b = np.array(std_sample).mean(axis=0)
    
    print(f"\nDataset statistics:")
    print(f"  Mean: R={mean_r:.4f}, G={mean_g:.4f}, B={mean_b:.4f}")
    print(f"  Std:  R={std_r:.4f}, G={std_g:.4f}, B={std_b:.4f}")
    
    return [mean_r, mean_g, mean_b], [std_r, std_g, std_b]


# =============================================================================
# Dataset Preparation
# =============================================================================
train_paths, val_paths, test_paths = [], [], []

for plate in ['P1', 'P2', 'P3', 'P4']:
    train_paths.extend(glob.glob(os.path.join(BASE_DIR, plate, '*.tif')))
val_paths = glob.glob(os.path.join(BASE_DIR, 'P5', '*.tif'))
test_paths = glob.glob(os.path.join(BASE_DIR, 'P6', '*.tif'))

print(f"\nDataset: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")

train_labels = [label_to_idx.get(get_label_from_path(p), 0) for p in train_paths]
val_labels = [label_to_idx.get(get_label_from_path(p), 0) for p in val_paths]
test_labels = [label_to_idx.get(get_label_from_path(p), 0) for p in test_paths]

# Compute normalization
print("\n" + "="*60)
print("COMPUTING DATASET NORMALIZATION")
print("="*60)
NORM_MEAN, NORM_STD = compute_dataset_stats(train_paths, sample_size=1000)
print("="*60 + "\n")

# Create datasets
train_dataset = MixedCropDataset(
    train_paths, train_labels, 
    crop_size=CONFIG['crop_size'],
    grid_size=CONFIG['grid_size'], 
    norm_mean=NORM_MEAN, 
    norm_std=NORM_STD,
    n_crops_per_image=CONFIG['n_crops_per_image'],
    use_augmentation=True
)
val_dataset = MixedCropDataset(
    val_paths, val_labels, 
    crop_size=CONFIG['crop_size'],
    grid_size=CONFIG['grid_size'], 
    norm_mean=NORM_MEAN, 
    norm_std=NORM_STD,
    n_crops_per_image=CONFIG['n_crops_per_image'],
    use_augmentation=False  # No augmentation for val/test
)
test_dataset = MixedCropDataset(
    test_paths, test_labels, 
    crop_size=CONFIG['crop_size'],
    grid_size=CONFIG['grid_size'], 
    norm_mean=NORM_MEAN, 
    norm_std=NORM_STD,
    n_crops_per_image=CONFIG['grid_size'] ** 2,  # Always use all crops for test
    use_augmentation=False
)

CONFIG['total_crops_per_epoch'] = len(train_dataset)
print(f"\nTotal crops/epoch: {CONFIG['total_crops_per_epoch']:,}")

# Create samplers
train_sampler = ShuffledCropSampler(len(train_dataset), shuffle=True)
val_sampler = ShuffledCropSampler(len(val_dataset), shuffle=False)
test_sampler = ShuffledCropSampler(len(test_dataset), shuffle=False)

# Create dataloaders (OPTIMIZED)
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], sampler=train_sampler,
                         num_workers=CONFIG['num_workers'], pin_memory=True, drop_last=True,
                         persistent_workers=True, prefetch_factor=4, worker_init_fn=seed_worker)

val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], sampler=val_sampler,
                       num_workers=CONFIG['num_workers'], pin_memory=True,
                       persistent_workers=True, prefetch_factor=4, worker_init_fn=seed_worker)

test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], sampler=test_sampler,
                        num_workers=CONFIG['num_workers'], pin_memory=True,
                        worker_init_fn=seed_worker)

# Class weights
class_counts = Counter(train_labels)
total = len(train_labels)
class_weights = torch.tensor([
    total / (num_classes * class_counts[i]) if class_counts[i] > 0 else 0
    for i in range(num_classes)], dtype=torch.float32).to(device)
print(f"Class weights: min={class_weights.min():.2f}, max={class_weights.max():.2f}")

# =============================================================================
# Model, Loss, Optimizer
# =============================================================================
model = BacNet(num_gene_classes=num_classes, num_pathway_classes=7, dropout=CONFIG['dropout'])
model = model.to(device)

num_params = count_parameters(model)
print(f"\nModel: BacNet - {num_params:,} params ({num_params/1e6:.2f}M)")

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=CONFIG['label_smoothing'])
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])

# LR Schedule: Linear Warmup (5 epochs) + Cosine Annealing
warmup_epochs = 2
scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CONFIG['epochs'] - warmup_epochs, eta_min=1e-6
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[scheduler_warmup, scheduler_cosine],
    milestones=[warmup_epochs]
)
print(f"LR Schedule: {warmup_epochs} epochs warmup (0.01x→1x) + Cosine Annealing")

# =============================================================================
# AMP Setup (OPTIMIZATION)
# =============================================================================
use_amp = CONFIG['use_amp'] and torch.cuda.is_available()
scaler = torch.amp.GradScaler('cuda') if use_amp else None
if use_amp:
    print("AMP (Mixed Precision) enabled")

# Track if model is compiled for save/load
model_compiled = False

# =============================================================================
# torch.compile (OPTIMIZATION)
# =============================================================================
if CONFIG['use_compile'] and not args.no_compile:
    print("Compiling model with torch.compile()...")
    compile_start = time.time()
    model = torch.compile(model, mode="reduce-overhead")
    model_compiled = True
    print(f"Compilation done in {time.time() - compile_start:.1f}s")

# =============================================================================
# Resume Training
# =============================================================================
start_epoch = 0
train_losses, train_accs, val_losses, val_accs = [], [], [], []
best_val_acc = 0.0

if args.resume:
    checkpoint_path = os.path.join(BASE_DIR, 'best_model_bacnet.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict = checkpoint['model_state_dict']
        
        if model_compiled:
            model._orig_mod.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
            train_accs = checkpoint['train_accs']
            val_losses = checkpoint['val_losses']
            val_accs = checkpoint['val_accs']
            best_val_acc = checkpoint['best_val_acc']
            start_epoch = len(train_losses)
        print(f"Resumed from epoch {start_epoch}, best_val_acc: {best_val_acc:.2f}%")
    else:
        print("Warning: best_model_bacnet.pth not found")

# =============================================================================
# Training Functions (OPTIMIZED with AMP)
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, device, scaler, epoch,
                gene_to_pathway_tensor, gene_to_binary_tensor,
                alpha=1.0, beta=0.3, gamma=0.2):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pathway_correct, binary_correct = 0, 0
    epoch_start = time.time()
    
    # Multi-task loss weights
    # alpha: gene (main task)
    # beta: pathway (auxiliary)
    # gamma: binary (auxiliary)
    
    # Re-shuffle for each epoch
    if hasattr(loader.sampler, 'indices'):
        loader.sampler.indices = list(range(len(loader.dataset)))
        random.shuffle(loader.sampler.indices)
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} Training")
    for crops, labels in pbar:
        crops = crops.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Get pathway and binary labels from gene labels
        pathway_labels = gene_to_pathway_tensor[labels].to(device)
        binary_labels = gene_to_binary_tensor[labels].to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler:
            with torch.amp.autocast('cuda'):
                gene_logits, pathway_logits, binary_logits = model(crops)
                
                # Multi-task loss
                L_gene = criterion(gene_logits, labels)
                L_pathway = criterion(pathway_logits, pathway_labels)
                L_binary = criterion(binary_logits, binary_labels)
                loss = alpha * L_gene + beta * L_pathway + gamma * L_binary
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            gene_logits, pathway_logits, binary_logits = model(crops)
            
            # Multi-task loss
            L_gene = criterion(gene_logits, labels)
            L_pathway = criterion(pathway_logits, pathway_labels)
            L_binary = criterion(binary_logits, binary_labels)
            loss = alpha * L_gene + beta * L_pathway + gamma * L_binary
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Track accuracy
        total_loss += loss.item()
        correct += (gene_logits.argmax(1) == labels).sum().item()
        pathway_correct += (pathway_logits.argmax(1) == pathway_labels).sum().item()
        binary_correct += (binary_logits.argmax(1) == binary_labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}', 
            'acc': f'{100.*correct/total:.1f}%',
            'path': f'{100.*pathway_correct/total:.1f}%'
        })
    
    epoch_time = time.time() - epoch_start
    samples_per_sec = total / epoch_time
    
    return (total_loss / len(loader), 
            100. * correct / total,
            100. * pathway_correct / total,
            100. * binary_correct / total,
            samples_per_sec)


def evaluate(model, loader, criterion, device, scaler=None,
             gene_to_pathway_tensor=None, gene_to_binary_tensor=None,
             alpha=1.0, beta=0.3, gamma=0.2):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    pathway_correct, binary_correct = 0, 0
    
    with torch.no_grad():
        for crops, labels in tqdm(loader, desc="Evaluating"):
            crops = crops.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            pathway_labels = None
            binary_labels = None
            if gene_to_pathway_tensor is not None:
                pathway_labels = gene_to_pathway_tensor[labels].to(device)
                binary_labels = gene_to_binary_tensor[labels].to(device)
            
            if use_amp and scaler:
                with torch.amp.autocast('cuda'):
                    gene_logits, pathway_logits, binary_logits = model(crops)
                    loss = alpha * criterion(gene_logits, labels) + \
                           beta * criterion(pathway_logits, pathway_labels) + \
                           gamma * criterion(binary_logits, binary_labels)
            else:
                gene_logits, pathway_logits, binary_logits = model(crops)
                loss = alpha * criterion(gene_logits, labels) + \
                       beta * criterion(pathway_logits, pathway_labels) + \
                       gamma * criterion(binary_logits, binary_labels)
            
            total_loss += loss.item()
            correct += (gene_logits.argmax(1) == labels).sum().item()
            pathway_correct += (pathway_logits.argmax(1) == pathway_labels).sum().item()
            binary_correct += (binary_logits.argmax(1) == binary_labels).sum().item()
            total += labels.size(0)
    
    return (total_loss / len(loader), 
            100. * correct / total,
            100. * pathway_correct / total,
            100. * binary_correct / total)


def get_all_predictions(model, loader, device, scaler=None):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for crops, labels in tqdm(loader, desc="Predicting"):
            crops = crops.to(device, non_blocking=True)
            
            if use_amp and scaler:
                with torch.amp.autocast('cuda'):
                    gene_logits, _, _ = model(crops)
            else:
                gene_logits, _, _ = model(crops)
            
            probs = torch.softmax(gene_logits, dim=1)
            
            all_preds.extend(gene_logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.append(probs.cpu())
    
    return np.array(all_preds), np.array(all_labels), torch.cat(all_probs).numpy()


# =============================================================================
# Training Loop
# =============================================================================
total_train_time = 0
batches_per_epoch = len(train_loader)

print(f"\n{'='*60}")
print("BacNet TRAINING STARTED")
print(f"{'='*60}")
print(f"Grid: {CONFIG['grid_size']}×{CONFIG['grid_size']} = {CONFIG['n_crops_per_image']} positions/image")
print(f"Total crops/epoch: {CONFIG['total_crops_per_epoch']:,}")
print(f"Batch: {CONFIG['batch_size']} | AMP: {use_amp} | Compile: {CONFIG['use_compile']}")
print(f"Normalization: mean={NORM_MEAN}, std={NORM_STD}")
print(f"{'='*60}\n")

def save_epoch_stats(epoch, train_loss, train_acc, val_loss, val_acc, lr, best_val_acc):
    """Save per-epoch statistics to JSON for later plotting"""
    stats_file = os.path.join(BASE_DIR, 'training_stats.json')
    
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            all_stats = json.load(f)
    else:
        all_stats = {
            'epochs': [], 'train_loss': [], 'train_acc': [], 
            'val_loss': [], 'val_acc': [], 'lr': [], 'best_val_acc': []
        }
    
    all_stats['epochs'].append(epoch)
    all_stats['train_loss'].append(train_loss)
    all_stats['train_acc'].append(train_acc)
    all_stats['val_loss'].append(val_loss)
    all_stats['val_acc'].append(val_acc)
    all_stats['lr'].append(lr)
    all_stats['best_val_acc'].append(best_val_acc)
    
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)

for epoch in range(start_epoch, CONFIG['epochs']):
    epoch_start = time.time()
    print(f"Epoch {epoch+1}/{CONFIG['epochs']} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    train_loss, train_acc, train_pathway_acc, train_binary_acc, samples_per_sec = train_epoch(
        model, train_loader, criterion, optimizer, device, scaler, epoch,
        gene_to_pathway_tensor, gene_to_binary_tensor,
        alpha=1.0, beta=0.3, gamma=0.2
    )
    val_loss, val_acc, val_pathway_acc, val_binary_acc = evaluate(
        model, val_loader, criterion, device, scaler,
        gene_to_pathway_tensor, gene_to_binary_tensor,
        alpha=1.0, beta=0.3, gamma=0.2
    )
    
    scheduler.step()
    
    epoch_time = time.time() - epoch_start
    total_train_time += epoch_time
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f"Train: {train_loss:.4f} / {train_acc:.2f}% (Path: {train_pathway_acc:.2f}%, Binary: {train_binary_acc:.2f}%) | Val: {val_loss:.4f} / {val_acc:.2f}% (Path: {val_pathway_acc:.2f}%, Binary: {val_binary_acc:.2f}%) | Time: {epoch_time:.1f}s | Speed: {samples_per_sec:.0f} crops/s")
    
    # Save per-epoch statistics to JSON
    save_epoch_stats(epoch + 1, train_loss, train_acc, val_loss, val_acc, 
                     optimizer.param_groups[0]['lr'], best_val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # Handle compiled model saving
        if model_compiled and hasattr(model, '_orig_mod'):
            model_state = model._orig_mod.state_dict()
        else:
            model_state = model.state_dict()
        
        save_dict = {
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'label_to_idx': label_to_idx,
            'idx_to_label': idx_to_label,
            'num_classes': num_classes,
            'config': CONFIG,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc,
            'epoch': epoch + 1,
            'model_compiled': model_compiled,
        }
        torch.save(save_dict, os.path.join(BASE_DIR, 'best_model_bacnet.pth'))
        print(f"  -> Saved best model ({val_acc:.2f}%)")

print(f"\nTotal training time: {total_train_time/60:.1f} minutes")
print(f"Best val accuracy: {best_val_acc:.2f}%")

# =============================================================================
# Test Evaluation
# =============================================================================
print(f"\n{'='*60}")
print("TEST EVALUATION")
print(f"{'='*60}")

checkpoint = torch.load(os.path.join(BASE_DIR, 'best_model_bacnet.pth'), map_location=device, weights_only=True)
# Create fresh model for evaluation (don't load into compiled model)
eval_model = BacNet(num_gene_classes=num_classes, num_pathway_classes=7, dropout=CONFIG['dropout'])
eval_model.load_state_dict(checkpoint['model_state_dict'])
eval_model = eval_model.to(device)
if model_compiled:
    eval_model = torch.compile(eval_model, mode="reduce-overhead")
test_loss, test_acc, test_pathway_acc, test_binary_acc = evaluate(
    eval_model, test_loader, criterion, device, scaler,
    gene_to_pathway_tensor, gene_to_binary_tensor,
    alpha=1.0, beta=0.3, gamma=0.2
)
print(f"Test Accuracy: {test_acc:.2f}% (Pathway: {test_pathway_acc:.2f}%, Binary: {test_binary_acc:.2f}%)")

# =============================================================================
# Plots
# =============================================================================
actual_epochs = len(train_losses)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(range(1, actual_epochs+1), train_losses, label='Train', marker='o')
axes[0].plot(range(1, actual_epochs+1), val_losses, label='Val', marker='o')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curves')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(range(1, actual_epochs+1), train_accs, label='Train', marker='o')
axes[1].plot(range(1, actual_epochs+1), val_accs, label='Val', marker='o')
axes[1].axhline(y=test_acc, color='r', linestyle='--', label=f'Test: {test_acc:.2f}%')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy Curves')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'training_plots_bacnet.png'), dpi=150)
plt.close()

# =============================================================================
# ROC & PR Curves
# =============================================================================
test_preds, test_labels_arr, test_probs = get_all_predictions(eval_model, test_loader, device, scaler)
test_labels_bin = label_binarize(test_labels_arr, classes=list(range(num_classes)))
classes_with_samples = [i for i in range(test_labels_bin.shape[1]) if test_labels_bin[:, i].sum() > 0]

fpr, tpr, roc_auc = {}, {}, {}
precision, recall, ap = {}, {}, {}

for i in tqdm(classes_with_samples, desc="Computing metrics"):
    fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], test_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    precision[i], recall[i], _ = precision_recall_curve(test_labels_bin[:, i], test_probs[:, i])
    ap[i] = average_precision_score(test_labels_bin[:, i], test_probs[:, i])

sorted_by_auc = sorted(roc_auc.items(), key=lambda x: x[1], reverse=True)

fig_roc, axes_roc = plt.subplots(2, 4, figsize=(16, 8))
axes_roc = axes_roc.flatten()

for idx, (i, _) in enumerate(sorted_by_auc[:8]):
    axes_roc[idx].plot(fpr[i], tpr[i], label=f'AUC = {roc_auc[i]:.2f}')
    axes_roc[idx].plot([0, 1], [0, 1], 'k--')
    axes_roc[idx].set_xlabel('FPR')
    axes_roc[idx].set_ylabel('TPR')
    axes_roc[idx].set_title(f'{idx_to_label[i]}')
    axes_roc[idx].legend(loc='lower right')

for j in range(8, len(axes_roc)):
    axes_roc[j].axis('off')

plt.suptitle('ROC Curves (Best 8 Classes by AUC)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'roc_curves_bacnet.png'), dpi=150)
plt.close()

sorted_by_ap = sorted(ap.items(), key=lambda x: x[1], reverse=True)

fig_pr, axes_pr = plt.subplots(2, 4, figsize=(16, 8))
axes_pr = axes_pr.flatten()

for idx, (i, _) in enumerate(sorted_by_ap[:8]):
    axes_pr[idx].plot(recall[i], precision[i], label=f'AP = {ap[i]:.2f}')
    axes_pr[idx].set_xlabel('Recall')
    axes_pr[idx].set_ylabel('Precision')
    axes_pr[idx].set_title(f'{idx_to_label[i]}')
    axes_pr[idx].legend(loc='lower left')

for j in range(8, len(axes_pr)):
    axes_pr[j].axis('off')

plt.suptitle('Precision-Recall Curves (Best 8 Classes by AP)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'precision_recall_curves_bacnet.png'), dpi=150)
plt.close()

# =============================================================================
# Metrics Summary
# =============================================================================
mean_roc_auc = np.mean([roc_auc[i] for i in classes_with_samples])
mean_ap = np.mean([ap[i] for i in classes_with_samples])

print(f"\n=== Performance Metrics ===")
print(f"Average ROC AUC: {mean_roc_auc:.4f}")
print(f"Average Precision: {mean_ap:.4f}")

print(f"\nTop 5 Classes by AUC:")
for i, val in sorted_by_auc[:5]:
    print(f"  {idx_to_label[i]}: {val:.4f}")

# =============================================================================
# Save Results
# =============================================================================
from datetime import datetime
import csv

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

results_json = {
    "timestamp": timestamp,
    "config": {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v for k, v in CONFIG.items()},
    "results": {
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "mean_roc_auc": float(mean_roc_auc),
        "mean_ap": float(mean_ap),
    }
}

json_path = os.path.join(BASE_DIR, f'training_results_bacnet_{timestamp}.json')
with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=2)

csv_path = os.path.join(BASE_DIR, f'class_metrics_bacnet_{timestamp}.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['class_name', 'class_idx', 'auc', 'ap', 'samples'])
    for i in sorted(classes_with_samples):
        writer.writerow([idx_to_label[i], i, f"{roc_auc.get(i, 0):.4f}",
                       f"{ap.get(i, 0):.4f}", int(test_labels_bin[:, i].sum())])

print(f"\n{'='*60}")
print("ALL FILES SAVED")
print("="*60)
print(f"  - Model: best_model_bacnet.pth")
print(f"  - Plots: training_plots_bacnet.png, roc_curves_bacnet.png, precision_recall_curves_bacnet.png")
print(f"  - Results: training_results_bacnet_{timestamp}.json")
print(f"  - CSV: class_metrics_bacnet_{timestamp}.csv")
print("="*60)
