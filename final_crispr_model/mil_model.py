"""
MIL with cycle-based crop extraction - improved version
Based on Ilse et al. 2018 gated attention
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
    """Gated attention MIL pooling (Ilse et al. 2018) - single head"""
    def __init__(self, in_features):
        super().__init__()
        
        # Single-head gated attention
        self.V = nn.Linear(in_features, in_features // 4)
        self.U = nn.Linear(in_features, in_features // 4)
        self.w = nn.Linear(in_features // 4, 1)
    
    def forward(self, x):
        # Gated attention: tanh(V(x)) * sigmoid(U(x))
        A = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        attn_weights = self.w(A).squeeze(-1)  # (B, N)
        
        # Softmax over instances
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum
        pooled = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)  # (B, F)
        
        return pooled, attn_weights


class AttentionMILModel(nn.Module):
    """MIL with gated attention pooling"""
    def __init__(self, num_classes):
        super().__init__()
        
        # EfficientNet backbone
        base_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        feature_dim = 1280
        
        # Single-head gated attention
        self.attention_pool = AttentionPooling(feature_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x, return_attention=False):
        batch_size, num_crops = x.shape[:2]
        
        # Extract features for all crops
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        x = x.view(batch_size, num_crops, -1)
        
        # Gated attention pooling
        pooled, attn_weights = self.attention_pool(x)
        
        output = self.classifier(pooled)
        
        if return_attention:
            return output, attn_weights
        return output


class MultiCropDataset(Dataset):
    """Cycle-based crop extraction with 9 neighbors for MIL"""
    
    def __init__(self, image_paths, labels, crop_size=224, grid_size=12, augment=True, seed=42, epoch=0):
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.augment = augment
        self.seed = seed
        self.epoch = epoch
        
        # Get image dimensions (handles rectangular images)
        sample_img = Image.open(image_paths[0]).convert('RGB')
        self.img_w, self.img_h = sample_img.size
        
        # Compute strides for x and y separately (handles rectangular)
        self.stride_x = (self.img_w - crop_size) // (grid_size - 1)
        self.stride_y = (self.img_h - crop_size) // (grid_size - 1)
        
        # Valid positions with full 3x3 neighborhood
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * self.stride_x
                top = i * self.stride_y
                if left + crop_size <= self.img_w and top + crop_size <= self.img_h:
                    can_left = left - self.stride_x >= 0
                    can_right = left + self.stride_x + crop_size <= self.img_w
                    can_top = top - self.stride_y >= 0
                    can_bottom = top + self.stride_y + crop_size <= self.img_h
                    if can_left and can_right and can_top and can_bottom:
                        positions.append((left, top))
        
        self.positions = positions
        
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
        
        print(f"MIL: {len(positions)} positions, 9 crops/image, augment={augment}")
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        num_pos = len(self.positions)
        num_images = len(self.image_paths)
        
        if not self.augment:
            # Val/test: use true image center (for rectangular)
            center_x = (self.img_w - self.crop_size) // 2
            center_y = (self.img_h - self.crop_size) // 2
            self.epoch_centers = {i: (center_x, center_y) for i in range(num_images)}
            return
        
        # Train: cycle through positions
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
        
        center_x, center_y = self.epoch_centers[idx]
        
        # Extract 3x3 grid around center
        crops_list = []
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                # Add small jitter during training
                if self.augment:
                    jx = random.randint(-self.stride_x // 4, self.stride_x // 4)
                    jy = random.randint(-self.stride_y // 4, self.stride_y // 4)
                else:
                    jx = jy = 0
                
                left = center_x + dx * self.stride_x + jx
                top = center_y + dy * self.stride_y + jy
                
                # Clamp to valid bounds
                left = max(0, min(left, self.img_w - self.crop_size))
                top = max(0, min(top, self.img_h - self.crop_size))
                
                crop = image.crop((left, top, left + self.crop_size, top + self.crop_size))
                crop = np.array(crop)
                crop = self.transform(image=crop)['image']
                crops_list.append(crop)
        
        # Shuffle crops for regularization
        if self.augment:
            perm = list(range(9))
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