"""
MIL with cycle-based crop extraction
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
    """Gated attention MIL pooling (Ilse et al. 2018) with temperature"""
    def __init__(self, in_features, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        
        # Gated attention: V and U learn what to attend to
        self.V = nn.Linear(in_features, in_features // 4)
        self.U = nn.Linear(in_features, in_features // 4)
        self.w = nn.Linear(in_features // 4, num_heads)
    
    def forward(self, x, temperature=0.5):
        # Gated attention: tanh(V) * sigmoid(U)
        A = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        attn_weights = self.w(A)  # (B, N, H)
        
        # Temperature scaling to prevent attention collapse
        attn_weights = torch.softmax(attn_weights / temperature, dim=1)
        
        # Weighted sum: (B, H, N) x (B, N, F) -> (B, H, F)
        pooled = torch.einsum('bnh,bnf->bhf', attn_weights, x)
        
        # Average heads (more stable than flattening)
        pooled = pooled.mean(dim=1)  # (B, F)
        
        return pooled, attn_weights


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
        
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        self.image_size = w
        
        stride = (w - crop_size) // (grid_size - 1)
        self.stride = stride
        
        # Only positions with full 3x3 neighborhood (skip edges)
        margin = stride
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * stride
                top = i * stride
                if left + crop_size <= w and top + crop_size <= h:
                    can_left = left - stride >= 0
                    can_right = left + stride + crop_size <= w
                    can_top = top - stride >= 0
                    can_bottom = top + stride + crop_size <= h
                    if can_left and can_right and can_top and can_bottom:
                        positions.append((left, top))
        
        self.positions = positions
        self.num_neighbors = 8
        
        if augment:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)}, rotate=(-10, 10), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.5, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        
        print(f"MIL: {len(positions)} positions, {self.num_neighbors + 1} crops/image, augment={augment}")
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        num_pos = len(self.positions)
        num_images = len(self.image_paths)
        
        if not self.augment:
            # Val/test: use center crop but still with neighbors (9 crops)
            center_idx = num_pos // 2
            center_pos = self.positions[center_idx]
            self.epoch_centers = {i: center_pos for i in range(num_images)}
            self.single_crop = False  # Use 9 crops for val/test too
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
            # Val/test: single center crop
            crop = image.crop((center_left, center_top, center_left + self.crop_size, center_top + self.crop_size))
            crop = np.array(crop)
            crop = self.transform(image=crop)['image']
            crops = crop.unsqueeze(0)  # Add crop dimension
        else:
            # Train: 3x3 grid around center with jitter
            jitter_range = self.stride // 4
            crops_list = []
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    # Add random jitter for more diversity
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
            
            # Shuffle crop order for regularization (fixes position overfitting)
            if self.augment:
                perm = torch.randperm(9)
                crops_list = [crops_list[i] for i in perm]
            
            crops = torch.stack(crops_list)
        
        return crops, self.labels[idx]


class AttentionMILModel(nn.Module):
    def __init__(self, num_classes, num_heads=4, attention_temp=0.5):
        super().__init__()
        # Use EfficientNet features with proper flattening
        base_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        feature_dim = 1280
        
        # Positional encoding for 9 positions (3x3 grid)
        self.pos_embedding = nn.Parameter(torch.randn(9, feature_dim) * 0.02)
        
        self.attention_pool = AttentionPooling(feature_dim, num_heads)
        self.attention_temp = attention_temp
        
        # Pool then project (instead of large flatten)
        self.proj = nn.Linear(feature_dim, feature_dim)
        
        # Multi-head projection (keep head diversity)
        self.head_proj = nn.Linear(feature_dim * num_heads, feature_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x, return_attention=False):
        batch_size, num_crops = x.shape[:2]
        
        # Extract features
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        x = x.view(batch_size, num_crops, -1)
        
        # Add positional encoding (3x3 grid: 9 positions)
        # Relative positions: (-1,-1), (-1,0), ..., (1,1)
        pos_emb = self.pos_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_emb
        
        # Instance-level dropout (drop entire crops, not features)
        mask = (torch.rand(batch_size, num_crops, 1, device=x.device) > 0.1).float()
        x = x * mask
        
        # Attention pooling with temperature
        pooled, attn_weights = self.attention_pool(x, temperature=self.attention_temp)
        
        # Flatten heads and project (keep diversity)
        pooled = pooled.view(batch_size, -1)
        pooled = self.head_proj(pooled)
        
        output = self.classifier(pooled)
        
        if return_attention:
            return output, attn_weights
        return output


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