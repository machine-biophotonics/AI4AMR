"""
MIL with cycle-based crop extraction + configurable neighborhood + Contrastive Learning
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


class AttentionPooling(nn.Module):
    """Gated attention MIL pooling (Ilse et al. 2018)"""
    def __init__(self, in_features, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = 256
        
        self.V = nn.Linear(in_features, self.hidden_dim)
        self.U = nn.Linear(in_features, self.hidden_dim)
        self.w = nn.Linear(self.hidden_dim, num_heads)
    
    def forward(self, x, temperature=0.5):
        A = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        attn_weights = self.w(A)
        attn_weights = torch.softmax(attn_weights / temperature, dim=1)
        pooled = torch.einsum('bnh,bnf->bhf', attn_weights, x)
        return pooled, attn_weights


class ContrastiveEncoder(nn.Module):
    """Encoder for contrastive learning"""
    def __init__(self, feature_dim=1280, projection_dim=256):
        super().__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(self, x):
        # x: (B, feature_dim)
        return self.projection_head(x)
    
    def get_embedding(self, x):
        # Return normalized embedding (without projection)
        with torch.no_grad():
            return F.normalize(x, dim=1)


class MILEncoder(nn.Module):
    """MIL encoder with optional contrastive head"""
    def __init__(self, num_classes, num_heads=4, attention_temp=0.5, dropout=0.2, use_contrastive=False, projection_dim=256):
        super().__init__()
        base_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.feature_dim = 1280
        self.use_contrastive = use_contrastive
        
        # Projection bottleneck (similar concept to MAMMOTH but simpler)
        mammoth_dim = 256
        self.use_mammoth = True
        self.proj_bottleneck = nn.Sequential(
            nn.Linear(self.feature_dim, mammoth_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(mammoth_dim, mammoth_dim),
        )
        
        self.attention_pool = AttentionPooling(mammoth_dim, num_heads)
        self.attention_temp = attention_temp
        
        self.head_proj = nn.Linear(mammoth_dim * num_heads, mammoth_dim)
        
        if use_contrastive:
            self.contrastive_head = ContrastiveEncoder(self.feature_dim, projection_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(mammoth_dim, num_classes)
        )
    
    def forward(self, x, return_attention=False, return_embedding=False, return_crop_embeddings=False):
        batch_size, num_crops = x.shape[:2]
        
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        crop_embeddings = x.view(batch_size, num_crops, -1)
        
        # Apply MAMMOTH-style projection
        crop_embeddings = crop_embeddings.view(batch_size * num_crops, -1)
        crop_embeddings = self.proj_bottleneck(crop_embeddings)
        crop_embeddings = crop_embeddings.view(batch_size, num_crops, -1)
        
        pooled, attn_weights = self.attention_pool(crop_embeddings, temperature=self.attention_temp)
        pooled = pooled.reshape(batch_size, -1)
        pooled = self.head_proj(pooled)
        
        if return_embedding and self.use_contrastive:
            embedding = self.contrastive_head.get_embedding(pooled)
            if return_attention:
                return embedding, attn_weights
            return embedding
        
        output = self.classifier(pooled)
        
        results = [output]
        if return_attention:
            results.append(attn_weights)
        if return_crop_embeddings:
            results.append(crop_embeddings)
        
        return results[0] if len(results) == 1 else tuple(results)
    
    def get_contrastive_embedding(self, x):
        """Get embedding for contrastive loss"""
        batch_size, num_crops = x.shape[:2]
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)  # (B*N, 1280)
        
        # For single crop per image, skip attention pooling
        if num_crops == 1:
            x = x.squeeze(1)  # (B, 1280)
            x = self.head_proj(x)
            return self.contrastive_head.get_embedding(x)
        
        # For multiple crops, use attention to aggregate
        x = x.view(batch_size, num_crops, -1)
        pooled, _ = self.attention_pool(x, temperature=self.attention_temp)
        pooled = pooled.reshape(batch_size, -1)
        pooled = self.head_proj(pooled)
        
        return self.contrastive_head.get_embedding(pooled)
    
    def get_backbone_features(self, x):
        """Get backbone features directly (for crop-level contrastive)"""
        batch_size, num_crops = x.shape[:2]
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        return x  # (B*N, 1280)
    
    def get_projected_features(self, x):
        """Get projected features (before contrastive head)"""
        # Handle input shape: (B, N, C, H, W) or (B, C, H, W)
        if len(x.shape) == 5:
            # Multiple crops per image
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            x = self.backbone(x)  # (B*N, 1280)
            x = x.view(B, N, -1)
            pooled, _ = self.attention_pool(x, temperature=self.attention_temp)
            pooled = pooled.reshape(B, -1)
        else:
            # Single crop per image
            x = self.backbone(x)  # (B, 1280)
            pooled = x
        
        pooled = self.head_proj(pooled)
        return pooled
    
    def get_mil_embeddings(self, x):
        """Get MIL bag embeddings (before classifier, for SC-MIL contrastive)"""
        batch_size, num_crops = x.shape[:2]
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        x = x.view(batch_size, num_crops, -1)
        
        # Attention pooling over all crops
        pooled, _ = self.attention_pool(x, temperature=self.attention_temp)
        pooled = pooled.reshape(batch_size, -1)
        
        # Project to feature dimension
        pooled = self.head_proj(pooled)
        
        # Apply activation and return (before classifier)
        return pooled
    
    def get_supcon_embeddings(self, x):
        """Get embeddings in [batch, n_crops, feature_dim] shape for SupCon"""
        batch_size, num_crops = x.shape[:2]
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        # Return: [batch, n_crops, feature_dim]
        return x.view(batch_size, num_crops, -1)
    
    def get_contrastive_embedding(self, x):
        """Get embedding for contrastive loss"""
        pooled = self.get_projected_features(x)
        return self.contrastive_head.get_embedding(pooled)
    
    def get_backbone_features(self, x):
        """Get backbone features directly"""
        x = self.backbone(x)
        return x


class AttentionMILModel(nn.Module):
    def __init__(self, num_classes, num_heads=4, attention_temp=0.5, dropout=0.2):
        super().__init__()
        base_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        feature_dim = 1280  # EfficientNet-B0 feature dimension
        
        self.attention_pool = AttentionPooling(feature_dim, num_heads)
        self.attention_temp = attention_temp
        
        self.head_proj = nn.Linear(feature_dim * num_heads, feature_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
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
    """Cycle-based crop extraction with configurable neighborhood for MIL"""
    
    def __init__(self, image_paths, labels, plate_well_map, crop_size=224, grid_size=12, neighborhood=3, augment=True, seed=42, epoch=0):
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.neighborhood = neighborhood
        self.augment = augment
        self.seed = seed
        self.epoch = epoch
        self.single_crop = False
        
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        self.image_size = w
        
        stride = (w - crop_size) // (grid_size - 1)
        self.stride = stride
        
        half_n = neighborhood // 2
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * stride
                top = i * stride
                if left + crop_size <= w and top + crop_size <= h:
                    can_left = left - half_n * stride >= 0
                    can_right = left + half_n * stride + crop_size <= w
                    can_top = top - half_n * stride >= 0
                    can_bottom = top + half_n * stride + crop_size <= h
                    if can_left and can_right and can_top and can_bottom:
                        positions.append((left, top))
        
        self.positions = positions
        self.num_neighbors = neighborhood * neighborhood - 1
        
        if augment:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        
        print(f"MIL: {len(positions)} positions, {self.neighborhood}x{self.neighborhood}={self.num_neighbors + 1} crops/image")
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        num_pos = len(self.positions)
        num_images = len(self.image_paths)
        
        if not self.augment:
            center_left = (self.image_size - self.crop_size) // 2
            center_top = (self.image_size - self.crop_size) // 2
            self.epoch_centers = {i: (center_left, center_top) for i in range(num_images)}
            self.single_crop = False
            return
        
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
            crop = image.crop((center_left, center_top, center_left + self.crop_size, center_top + self.crop_size))
            crop = np.array(crop)
            crop = self.transform(image=crop)['image']
            crops = crop.unsqueeze(0)
        else:
            jitter_range = self.stride // 4
            crops_list = []
            half_n = self.neighborhood // 2
            
            for di in range(-half_n, half_n + 1):
                for dj in range(-half_n, half_n + 1):
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
            
            num_crops = self.neighborhood * self.neighborhood
            if self.augment:
                perm = list(range(num_crops))
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