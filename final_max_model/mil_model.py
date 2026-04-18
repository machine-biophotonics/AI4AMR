"""
MIL with cycle-based crop extraction + 5x5 neighborhood (25 crops)
- Training: 25 crops (center + 5x5 grid with jitter)
- Validation/Test: 25 crops (center + 5x5 grid, no jitter)
- True multi-head attention: 4 heads × 64-dim = 256-dim (default)
- Also supports: max, mean, gmp, certainty pooling
- Bottleneck: 1280 → 256
- Classifier per paper: BatchNorm → Linear → ReLU → Dropout → Linear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import re
import os


class MaxPoolingMIL(nn.Module):
    """Max-pooling based MIL (FocusMIL style) - AvgPoolCNN style
    Operates directly on 1280-dim features from backbone"""
    def __init__(self, in_features=1280, bottleneck_dim=1280):
        super().__init__()
        # No bottleneck - pass through directly
        
    def forward(self, x):
        # x shape: (B, N, 1280)
        pooled, indices = torch.max(x, dim=1)  # (B, 1280)
        
        attn_weights = torch.zeros(x.size(0), x.size(1), device=x.device)
        attn_weights.scatter_(1, indices, 1.0)
        
        return pooled, attn_weights


class MeanPoolingMIL(nn.Module):
    """Mean-pooling based MIL - AvgPoolCNN style
    Directly averages 1280-dim features from backbone (no bottleneck)"""
    def __init__(self, in_features=1280, bottleneck_dim=1280):
        super().__init__()
        # No bottleneck - pass through directly
        
    def forward(self, x):
        # x shape: (B, N, 1280)
        pooled = torch.mean(x, dim=1)  # (B, 1280)
        
        attn_weights = torch.ones(x.size(0), x.size(1), device=x.device) / x.size(1)
        
        return pooled, attn_weights


class GeneralizedMeanPoolingMIL(nn.Module):
    """Generalized Mean Pooling (GMP) - arXiv:2008.10548
    AvgPoolCNN style - operates on 1280-dim features"""
    def __init__(self, in_features=1280, bottleneck_dim=1280, p=3.0, learnable_p=True):
        super().__init__()
        # No bottleneck - operate directly on 1280-dim
        if learnable_p:
            self.p = nn.Parameter(torch.tensor(p, dtype=torch.float32))
        else:
            self.register_buffer('p', torch.tensor(p, dtype=torch.float32))
        
    def forward(self, x):
        # x shape: (B, N, 1280)
        p = torch.clamp(self.p, min=1.0, max=10.0)
        
        # Generalized mean: (mean(x^p))^(1/p)
        pooled = torch.pow(torch.mean(torch.pow(x + 1e-8, p), dim=1), 1.0 / p)  # (B, 1280)
        
        attn_weights = F.softmax(x.mean(dim=-1), dim=1)
        
        return pooled, attn_weights


class CertaintyPoolingMIL(nn.Module):
    """Certainty Pooling - arXiv:2008.10548
    AvgPoolCNN style - operates on 1280-dim features"""
    def __init__(self, in_features=1280, bottleneck_dim=1280, num_classes=96):
        super().__init__()
        self.certainty_head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # x shape: (B, N, 1280)
        instance_logits = self.certainty_head(x)  # (B, N, num_classes)
        instance_probs = F.softmax(instance_logits, dim=-1)
        
        max_probs, _ = torch.max(instance_probs, dim=-1)  # (B, N)
        certainty = max_probs.pow(2)
        
        attn_weights = certainty / (certainty.sum(dim=1, keepdim=True) + 1e-8)
        
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # (B, 1280)
        
        return pooled, attn_weights
        
        instance_logits = self.certainty_head(x)
        instance_probs = F.softmax(instance_logits, dim=-1)
        
        max_probs, _ = torch.max(instance_probs, dim=-1)
        certainty = max_probs.pow(2)
        
        attn_weights = certainty / (certainty.sum(dim=1, keepdim=True) + 1e-8)
        
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        
        return pooled, attn_weights


class AttentionPooling(nn.Module):
    """Gated attention MIL pooling with true multi-head attention and learnable temperature"""
    def __init__(self, in_features, num_heads=4, hidden_dim=256):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads  # 256 / 4 = 64-dim per head
        
        # Learnable temperature (initialized at 0.5)
        self.temperature = nn.Parameter(torch.tensor(0.5))
        
        # Gated attention: V and U project full features to head_dim per head
        # After projection, split into heads
        self.V = nn.Linear(in_features, self.head_dim * num_heads)
        self.U = nn.Linear(in_features, self.head_dim * num_heads)
        # w produces single attention score per (instance)
        self.w = nn.Linear(self.head_dim, 1)
        
        # Output projection to combine heads back
        self.out_proj = nn.Linear(self.head_dim * num_heads, in_features)
    
    def forward(self, x):
        batch_size, num_instances, _ = x.shape  # (B, N, 256)
        
        # Use learnable temperature
        temp = self.temperature.clamp(0.1, 5.0)  # Clamp to prevent collapse
        
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
        
        # Apply temperature and softmax over crops (dim=2) - each head normalizes across 25 crops
        attn_weights = torch.softmax(attn_scores / temp, dim=2)  # (B, H, N)
        
        # Re-permute x for weighted sum: (B, N, H, 64) → (B, H, N, 64)
        x_heads = x.view(batch_size, num_instances, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Weighted sum: (B, H, N) × (B, H, N, 64) → (B, H, 64)
        pooled = torch.einsum('bhn,bhnf->bhf', attn_weights, x_heads)
        
        # Concatenate heads: (B, H*64) = (B, 256)
        pooled = pooled.reshape(batch_size, -1)
        
        # Output projection
        pooled = self.out_proj(pooled)
        
        # Average attention weights across heads for visualization (B, N)
        attn_weights_avg = attn_weights.mean(dim=1)
        
        return pooled, attn_weights_avg


class AttentionMILModel(nn.Module):
    """MIL model with configurable pooling strategy.
    
    Pooling types:
        - attention: Gated multi-head attention (default, same as final_mutant_model)
        - max: Max-pooling (FocusMIL) - 1280-dim
        - mean: Mean-pooling (average) - 1280-dim (AvgPoolCNN style)
        - gmp: Generalized Mean Pooling (arXiv:2008.10548) - 1280-dim
        - certainty: Certainty Pooling (arXiv:2008.10548) - 1280-dim
    
    Architecture matches AvgPoolCNN:
        - Backbone: EfficientNet-B0 → 1280-dim features
        - Pooling: Configurable (attention uses 256-dim projection, others use 1280-dim)
        - Classifier: BN → Linear → ReLU → Dropout → Linear
    """
    def __init__(self, num_classes, num_heads=4, attention_temp=0.5, bottleneck_dim=1280, pooling_type='attention'):
        super().__init__()
        base_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(
            base_model.features,
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten()
        )
        feature_dim = 1280
        self.bottleneck_dim = bottleneck_dim
        self.pooling_type = pooling_type.lower()
        
        if self.pooling_type == 'attention':
            # Attention uses 256-dim projection (like final_mutant_model)
            self.bottleneck = nn.Linear(feature_dim, 256)
            self.pooling = AttentionPooling(256, num_heads)
            classifier_dim = 256
        else:
            # Other pooling use 1280-dim directly (AvgPoolCNN style)
            self.bottleneck = nn.Identity()  # No projection
            if self.pooling_type == 'max':
                self.pooling = MaxPoolingMIL(feature_dim, feature_dim)
            elif self.pooling_type == 'mean':
                self.pooling = MeanPoolingMIL(feature_dim, feature_dim)
            elif self.pooling_type in ['gmp', 'generalized_mean']:
                self.pooling = GeneralizedMeanPoolingMIL(feature_dim, feature_dim)
            elif self.pooling_type == 'certainty':
                self.pooling = CertaintyPoolingMIL(feature_dim, feature_dim, num_classes)
            else:
                raise ValueError(f"Unknown pooling type: {pooling_type}")
            classifier_dim = feature_dim
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(classifier_dim),
            nn.Linear(classifier_dim, classifier_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(classifier_dim, num_classes)
        )
    
    def forward(self, x, return_attention=False):
        batch_size, num_crops = x.shape[:2]
        
        # Extract features with backbone
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        x = x.view(batch_size, num_crops, -1)  # (B, N, 1280)
        
        # Apply bottleneck (identity for non-attention, linear for attention)
        x = self.bottleneck(x)
        
        if self.pooling_type == 'attention':
            pooled, attn_weights = self.pooling(x)
        else:
            pooled, attn_weights = self.pooling(x)
        
        output = self.classifier(pooled)
        
        if return_attention:
            return output, attn_weights
        return output


class MultiCropDataset(Dataset):
    """Cycle-based crop extraction with configurable neighborhood (3x3 or 5x5) for MIL
    
    Args:
        crop_neighborhood: 3 for 3x3 (9 crops), 5 for 5x5 (25 crops)
    """
    
    def __init__(self, image_paths, labels, plate_well_map, crop_size=224, grid_size=12, augment=True, seed=42, epoch=0, crop_neighborhood=5):
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.augment = augment
        self.seed = seed
        self.epoch = epoch
        self.single_crop = False
        self.crop_neighborhood = crop_neighborhood  # 3 for 3x3, 5 for 5x5
        
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        self.image_size = w
        
        stride = (w - crop_size) // (grid_size - 1)
        self.stride = stride
        
        # Only positions with full neighborhood (skip edges)
        half_neighbor = (crop_neighborhood - 1) // 2
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * stride
                top = i * stride
                if left + crop_size <= w and top + crop_size <= h:
                    can_left = left - half_neighbor * stride >= 0
                    can_right = left + half_neighbor * stride + crop_size <= w
                    can_top = top - half_neighbor * stride >= 0
                    can_bottom = top + half_neighbor * stride + crop_size <= h
                    if can_left and can_right and can_top and can_bottom:
                        positions.append((left, top))
        
        self.positions = positions
        self.num_neighbors = crop_neighborhood * crop_neighborhood - 1
        
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
            # Configurable neighborhood grid around center with jitter
            half = (self.crop_neighborhood - 1) // 2
            jitter_range = self.stride // 4
            crops_list = []
            for di in range(-half, half + 1):
                for dj in range(-half, half + 1):
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
                n_crops = self.crop_neighborhood * self.crop_neighborhood
                perm = list(range(n_crops))
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