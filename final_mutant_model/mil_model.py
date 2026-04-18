"""
Simplified MIL model with GMP + 6-layer classification head
Based on best practices from paper: EfficientNet-B0 + GMP + BN + FC(256) + ReLU + L1L2 + Dropout + Softmax
Optimizer: Adamax
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


class ClassificationHead(nn.Module):
    """
    6-layer classification head as described in the paper:
    1. GMP (Global Max Pool) - already in backbone
    2. BatchNorm1d
    3. Dense(256) + ReLU
    4. L1 + L2 Regularization (via optimizer weight decay)
    5. Dropout(0.2)
    6. Softmax output
    """
    def __init__(self, in_features, num_classes, hidden_dim=256, dropout=0.2):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.head(x)


class AttentionPooling(nn.Module):
    """Simple gated attention MIL pooling"""
    def __init__(self, in_features, num_heads=4, attention_temp=0.5):
        super().__init__()
        self.num_heads = num_heads
        
        self.V = nn.Linear(in_features, in_features // 4)
        self.U = nn.Linear(in_features, in_features // 4)
        self.w = nn.Linear(in_features // 4, num_heads)
        self.attention_temp = attention_temp
    
    def forward(self, x):
        A = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        attn_weights = self.w(A)
        attn_weights = torch.softmax(attn_weights / self.attention_temp, dim=1)
        pooled = torch.einsum('bnh,bnf->bhf', attn_weights, x)
        return pooled, attn_weights


class AttentionMILModel(nn.Module):
    """
    MIL model with EfficientNet-B0 backbone + GMP + Classification head
    Optional: Gated attention pooling (can be disabled for simpler model)
    """
    def __init__(self, num_classes, num_heads=4, attention_temp=0.5, use_attention=True, hidden_dim=256):
        super().__init__()
        
        # EfficientNet-B0 backbone (ImageNet pretrained)
        base_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        # Extract features but use Global Max Pooling instead of Global Average Pooling
        self.backbone = nn.Sequential(
            base_model.features,
            nn.AdaptiveMaxPool2d(1),  # GMP - Key modification from paper
            nn.Flatten()
        )
        feature_dim = 1280
        
        self.use_attention = use_attention
        
        if use_attention:
            # Optional: Attention pooling
            self.attention_pool = AttentionPooling(feature_dim, num_heads, attention_temp)
            self.head_proj = nn.Linear(feature_dim * num_heads, feature_dim)
            self.classifier = ClassificationHead(feature_dim, num_classes, hidden_dim)
        else:
            # Direct classification without attention
            self.classifier = ClassificationHead(feature_dim, num_classes, hidden_dim)
    
    def forward(self, x, return_attention=False):
        batch_size, num_crops = x.shape[:2]
        
        # Extract features for all crops
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        x = x.view(batch_size, num_crops, -1)
        
        if self.use_attention:
            # Attention pooling
            pooled, attn_weights = self.attention_pool(x)
            pooled = pooled.reshape(batch_size, -1)
            pooled = self.head_proj(pooled)
            output = self.classifier(pooled)
            
            if return_attention:
                return output, attn_weights
            return output
        else:
            # Global max pooling across crops + classification
            pooled, _ = torch.max(x, dim=1)  # Max over crop dimension
            output = self.classifier(pooled)
            
            if return_attention:
                # Return uniform attention for compatibility
                attn_weights = torch.ones(batch_size, num_crops, 1, device=x.device) / num_crops
                return output, attn_weights
            return output


class MultiCropDataset(Dataset):
    """Cycle-based crop extraction with 5x5 neighbors for MIL"""
    
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
        
        # Only positions with full 5x5 neighborhood (skip edges)
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * stride
                top = i * stride
                if left + crop_size <= w and top + crop_size <= h:
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
            center_left = (self.image_size - self.crop_size) // 2
            center_top = (self.image_size - self.crop_size) // 2
            self.epoch_centers = {i: (center_left, center_top) for i in range(num_images)}
            return
        
        # Cycle-based position assignment
        cycle = epoch // num_pos
        pos_in_cycle = epoch % num_pos
        rng = random.Random(self.seed + cycle)
        shuffled = self.positions.copy()
        rng.shuffle(shuffled)
        
        self.epoch_centers = {}
        for idx in range(num_images):
            assigned_idx = (idx + pos_in_cycle) % num_pos
            self.epoch_centers[idx] = shuffled[assigned_idx]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        center_left, center_top = self.epoch_centers[idx]
        
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
        
        # Shuffle crop order during training
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