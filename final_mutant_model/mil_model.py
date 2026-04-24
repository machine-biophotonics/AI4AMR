"""
MIL with 4 center positions -> 12 groups (36 crops total)
Each epoch: use 4 different center positions, cycling through them
Each center: 3 groups x 3 crops = 9 crops
Total per image: 4 centers x 3 groups = 12 groups
"""

import torch
import torch.nn as nn
import torchvision
import random
import numpy as np
from PIL import Image
import albumentations as A
from torch.utils.data import Dataset
import re
import os


class AttentionPooling(nn.Module):
    """Gated attention MIL pooling (Ilse et al. 2018)"""
    def __init__(self, in_features, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        
        self.V = nn.Linear(in_features, in_features // 4)
        self.U = nn.Linear(in_features, in_features // 4)
        self.w = nn.Linear(in_features // 4, num_heads)
    
    def forward(self, x, temperature=0.5):
        A = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        attn_weights = self.w(A)
        attn_weights = torch.softmax(attn_weights / temperature, dim=1)
        pooled = torch.einsum("bnh,bnf->bhf", attn_weights, x)
        return pooled, attn_weights


class AttentionMILModel(nn.Module):
    def __init__(self, num_classes, num_heads=4, attention_temp=0.5):
        super().__init__()
        base_model = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        feature_dim = 1280
        
        self.attention_pool = AttentionPooling(feature_dim, num_heads)
        self.attention_temp = attention_temp
        self.head_proj = nn.Linear(feature_dim * num_heads, feature_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x, return_attention=False):
        batch_size, num_crops = x.shape[:2]
        
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        x = x.view(batch_size, num_crops, -1)
        
        pooled, attn_weights = self.attention_pool(x, temperature=self.attention_temp)
        
        pooled = pooled.reshape(batch_size, -1)
        pooled = self.head_proj(pooled)
        
        output = self.classifier(pooled)
        
        if return_attention:
            return output, attn_weights
        return output


class MultiCropDataset(Dataset):
    """4 center positions -> 12 groups (36 crops total)"""
    
    def __init__(self, image_paths, labels, plate_well_map, crop_size=224, grid_size=12, num_centers=4, augment=False, seed=42, epoch=0):
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.num_centers = num_centers
        self.augment = augment
        self.seed = seed
        self.epoch = epoch
        
        sample_img = Image.open(image_paths[0]).convert("L")
        w, h = sample_img.size
        self.image_size = w
        
        stride = (w - crop_size) // (grid_size - 1)
        self.stride = stride
        
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
        
        center_left = (w - crop_size) // 2
        center_top = (h - crop_size) // 2
        quarter_w = w // 4
        quarter_h = h // 4
        
        self.center_positions = [
            (center_left, center_top),
            (center_left - quarter_w // 2, center_top - quarter_h // 2),
            (center_left + quarter_w // 2, center_top + quarter_h // 2),
            (center_left - quarter_w // 2, center_top + quarter_h // 2),
        ]
        self.center_positions = [
            (max(0, min(left, self.image_size - self.crop_size)),
             max(0, min(top, self.image_size - self.crop_size)))
            for left, top in self.center_positions
        ]
        
        print(f"MIL: {num_centers} centers x 3 groups = {num_centers * 3} groups per image")
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        
        if len(self.positions) == 0:
            raise ValueError("No positions available for crop extraction!")
        
        self.epoch_centers = self.center_positions
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")
        
        all_groups = []
        
        for center_idx, (center_left, center_top) in enumerate(self.epoch_centers):
            crop_positions_3x3 = [
                (-1, -1), (0, -1), (1, -1),
                (-1, 0),  (0, 0),  (1, 0),
                (-1, 1),  (0, 1),  (1, 1)
            ]
            
            crops_gray = []
            for di, dj in crop_positions_3x3:
                left = center_left + dj * self.stride
                top = center_top + di * self.stride
                left = max(0, min(left, self.image_size - self.crop_size))
                top = max(0, min(top, self.image_size - self.crop_size))
                crop = image.crop((left, top, left + self.crop_size, top + self.crop_size))
                crop_np = np.array(crop, dtype=np.float32) / 255.0
                crop_np = (crop_np - 0.456) / 0.224
                crop_tensor = torch.from_numpy(crop_np).unsqueeze(0)
                crops_gray.append(crop_tensor)
            
            ch_group_1 = torch.cat([crops_gray[0], crops_gray[1], crops_gray[2]], dim=0)
            ch_group_2 = torch.cat([crops_gray[3], crops_gray[4], crops_gray[5]], dim=0)
            ch_group_3 = torch.cat([crops_gray[6], crops_gray[7], crops_gray[8]], dim=0)
            
            all_groups.extend([ch_group_1, ch_group_2, ch_group_3])
        
        crops = torch.stack(all_groups, dim=0)
        
        return crops, self.labels[idx]


def extract_well_from_filename(filename):
    match = re.search(r"Well(\w\d+)_", filename)
    return match.group(1) if match else None


def get_gene_from_path(img_path, plate_maps):
    dirname = os.path.dirname(img_path)
    plate = os.path.basename(dirname)
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if plate in plate_maps and well in plate_maps[plate]:
        return plate_maps[plate][well]
    return "WT"
