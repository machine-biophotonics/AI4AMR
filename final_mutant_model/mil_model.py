"""
MIL with cycle-based crop extraction + 5x5 neighborhood (25 crops)
- Training: 25 crops (center + 5x5 grid with jitter)
- Validation/Test: 25 crops (center + 5x5 grid, no jitter)
- True multi-head attention: 4 heads × 64-dim = 256-dim
- Bottleneck: 1280 → 256
- Classifier per paper: BatchNorm → Linear → ReLU → Dropout → Linear
"""

import torch
import torch.nn as nn
import torchvision
import random
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import re
import os


class AttentionPooling(nn.Module):
    """Gated attention MIL pooling with true multi-head attention (Ilse et al. 2018)"""
    def __init__(self, in_features, num_heads=4, hidden_dim=256):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads  # 256 / 4 = 64-dim per head
        
        # Gated attention: V and U project full features to head_dim per head
        # After projection, split into heads
        self.V = nn.Linear(in_features, self.head_dim * num_heads)
        self.U = nn.Linear(in_features, self.head_dim * num_heads)
        # w produces single attention score per (instance)
        self.w = nn.Linear(self.head_dim, 1)
        
        # Output projection to combine heads back
        self.out_proj = nn.Linear(self.head_dim * num_heads, in_features)
    
    def forward(self, x, temperature=0.5):
        batch_size, num_instances, _ = x.shape  # (B, N, 256)
        
        # Project to all heads at once: 256 → 4*64 = 256
        V_out = self.V(x)  # (B, N, 256)
        U_out = self.U(x)  # (B, N, 256)
        
        # Reshape to split into heads: (B, N, H, 64)
        V_heads = V_out.view(batch_size, num_instances, self.num_heads, self.head_dim)
        U_heads = U_out.view(batch_size, num_instances, self.num_heads, self.head_dim)
        
        # Permute to (B, H, N, 64) for attention computation
        V_heads = V_heads.permute(0, 2, 1, 3)  # (B, H, N, 64)
        U_heads = U_heads.permute(0, 2, 1, 3)  # (B, H, N, 64)
        
        # Gated attention per head: tanh(V) ⊙ sigmoid(U)
        A_heads = torch.tanh(V_heads) * torch.sigmoid(U_heads)  # (B, H, N, 64)
        
        # Compute attention scores per head
        attn_scores = self.w(A_heads).squeeze(-1)  # (B, H, N)
        
        # Apply temperature and softmax over heads (dim=1) - each instance gets attention over heads
        attn_weights = torch.softmax(attn_scores / temperature, dim=1)  # (B, H, N)
        
        # Re-permute x for weighted sum: (B, N, H, 64) → (B, H, N, 64)
        x_heads = x.view(batch_size, num_instances, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Weighted sum: (B, H, N) × (B, H, N, 64) → (B, H, 64)
        pooled = torch.einsum('bhn,bhnf->bhf', attn_weights, x_heads)
        
        # Concatenate heads: (B, H*64) = (B, 256)
        pooled = pooled.reshape(batch_size, -1)
        
        # Output projection
        pooled = self.out_proj(pooled)
        
        # Average attention weights for visualization (B, N)
        attn_weights_avg = attn_weights.mean(dim=1)
        
        return pooled, attn_weights_avg


class AttentionMILModel(nn.Module):
    def __init__(self, num_classes, num_heads=4, attention_temp=0.5, bottleneck_dim=256):
        super().__init__()
        # Use EfficientNet features with proper flattening
        base_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(
            base_model.features,
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten()
        )
        feature_dim = 1280
        self.bottleneck_dim = bottleneck_dim
        
        # Bottleneck projection: 1280 -> 256
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        
        # Gated attention pooling on 256-dim features
        self.attention_pool = AttentionPooling(bottleneck_dim, num_heads)
        self.attention_temp = attention_temp
        
        # Classifier: bottleneck_dim -> num_classes (per paper: GMP -> BatchNorm -> Linear -> ReLU -> L2 -> Dropout -> Linear)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(bottleneck_dim),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(bottleneck_dim, num_classes)
        )
    
    def forward(self, x, return_attention=False):
        batch_size, num_crops = x.shape[:2]
        
        # Extract features
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        x = x.view(batch_size, num_crops, -1)  # (B, N, 1280)
        
        # Bottleneck projection: 1280 -> 256
        x = self.bottleneck(x)  # (B, N, 256)
        
        # Attention pooling with temperature
        pooled, attn_weights = self.attention_pool(x, temperature=self.attention_temp)
        
        # Attention pooling returns already projected 256-dim output
        pooled, attn_weights = self.attention_pool(x, temperature=self.attention_temp)
        
        output = self.classifier(pooled)
        
        if return_attention:
            return output, attn_weights
        return output


class MultiCropDataset(Dataset):
    """Cycle-based crop extraction with 9 neighbors for MIL"""
    
    def __init__(self, image_paths, labels, plate_well_map, crop_size=224, grid_size=12, augment=True, seed=42, epoch=0):
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.augment = augment
        self.seed = seed
        self.epoch = epoch
        self.single_crop = False  # Default to multi-crop
        
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        self.image_size = w
        
        stride = (w - crop_size) // (grid_size - 1)
        self.stride = stride
        
        # Only positions with full 5x5 neighborhood (skip edges)
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * stride
                top = i * stride
                if left + crop_size <= w and top + crop_size <= h:
                    # Need 2 strides of space on each side for 5x5
                    can_left = left - 2 * stride >= 0
                    can_right = left + 2 * stride + crop_size <= w
                    can_top = top - 2 * stride >= 0
                    can_bottom = top + 2 * stride + crop_size <= h
                    if can_left and can_right and can_top and can_bottom:
                        positions.append((left, top))
        
        self.positions = positions
        self.num_neighbors = 24  # 5x5 = 25 crops
        
        if augment:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.5, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], seed=seed)
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], seed=seed)
        
        print(f"MIL: {len(positions)} positions, {self.num_neighbors + 1} crops/image, augment={augment}")
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        num_pos = len(self.positions)
        num_images = len(self.image_paths)
        
        if not self.augment:
            # Val/test: use TRUE image center
            center_left = (self.image_size - self.crop_size) // 2
            center_top = (self.image_size - self.crop_size) // 2
            self.epoch_centers = {i: (center_left, center_top) for i in range(num_images)}
            self.single_crop = False
            return
        
        # Train: cycle-based
        cycle = epoch // num_pos
        pos_in_cycle = epoch % num_pos
        rng = random.Random(self.seed + cycle)
        shuffled = self.positions.copy()
        rng.shuffle(shuffled)
        
        self.epoch_centers = {}
        for idx in range(num_images):
            assigned_idx = (idx + pos_in_cycle) % num_pos
            self.epoch_centers[idx] = shuffled[assigned_idx]
        
        self.single_crop = False
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        center_left, center_top = self.epoch_centers[idx]
        
        if self.single_crop:
            # Single crop mode
            crop = image.crop((center_left, center_top, center_left + self.crop_size, center_top + self.crop_size))
            crop = np.array(crop)
            crop = self.transform(image=crop)['image']
            crops = crop.unsqueeze(0)
        else:
            # 5x5 grid around center with jitter
            jitter_range = self.stride // 4
            crops_list = []
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    if self.augment:
                        jitter_x = random.randint(-jitter_range, jitter_range)
                        jitter_y = random.randint(-jitter_range, jitter_range)
                    else:
                        jitter_x = jitter_y = 0
                    left = center_left + dj * self.stride + jitter_x
                    top = center_top + di * self.stride + jitter_y
                    left = max(0, min(left, self.image_size - self.crop_size))
                    top = max(0, min(top, self.image_size - self.crop_size))
                    crop = image.crop((left, top, left + self.crop_size, top + self.crop_size))
                    crop = np.array(crop)
                    crop = self.transform(image=crop)['image']
                    crops_list.append(crop)
            
            # Shuffle crop order
            if self.augment:
                perm = list(range(25))
                random.shuffle(perm)
                crops_list = [crops_list[i] for i in perm]
            
            crops = torch.stack(crops_list)
        
        return crops, self.labels[idx]


def extract_well_from_filename(filename):
    match = re.search(r'Well(\w\d+)_', filename)
    return match.group(1) if match else None


def get_gene_from_path(img_path, plate_maps):
    dirname = os.path.dirname(img_path)
    plate = os.path.basename(dirname)
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if plate in plate_maps and well in plate_maps[plate]:
        return plate_maps[plate][well]
    return 'WT'