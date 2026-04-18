"""
MIL with 5x5 neighborhood (25 crops)
- Gated Attention MIL pooling with configurable heads
- Direct 1280-dim features to attention (no bottleneck)
- Simple classifier: Linear → ReLU → Dropout → Linear
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


class GatedAttentionMIL(nn.Module):
    """Gated attention MIL pooling (Ilse et al. 2018)"""
    def __init__(self, in_features, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads
        
        # Gated attention
        self.V = nn.Linear(in_features, in_features)
        self.U = nn.Linear(in_features, in_features)
        
        # Per-head attention score
        self.w = nn.Linear(self.head_dim, 1)
    
    def forward(self, x, temperature=0.5):
        batch_size, num_instances, _ = x.shape  # (B, N, 1280)
        
        # Gated attention: tanh(V) ⊙ sigmoid(U)
        A = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        
        # Reshape to heads: (B, N, H, 320)
        A_heads = A.view(batch_size, num_instances, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Per-head score: (B, H, N, 320) → (B, H, N)
        attn_scores = self.w(A_heads).squeeze(-1)
        
        # Softmax over instances: (B, H, N)
        attn_weights = torch.softmax(attn_scores / temperature, dim=2)
        
        # Reshape features for weighted sum
        x_heads = x.view(batch_size, num_instances, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Weighted sum: (B, H, N) × (B, H, N, 320) → (B, H, 320)
        pooled = torch.einsum('bhn,bhnf->bhf', attn_weights, x_heads)
        
        # Concatenate heads: (B, 1280)
        pooled = pooled.reshape(batch_size, -1)
        
        # Average attention for visualization
        attn_weights_avg = attn_weights.mean(dim=1)
        
        return pooled, attn_weights_avg


class AttentionMILModel(nn.Module):
    def __init__(self, num_classes, num_heads=4, attention_temp=0.5):
        super().__init__()
        # EfficientNet-B0 backbone with GMP (Global Max Pooling)
        base_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(
            base_model.features,
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten()
        )
        feature_dim = 1280
        self.num_heads = num_heads
        
        # Gated attention pooling
        self.attention = GatedAttentionMIL(feature_dim, num_heads)
        self.attention_temp = attention_temp
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x, return_attention=False):
        batch_size, num_crops = x.shape[:2]
        
        # Extract features: (B, N, 3, 224, 224) → (B, N, 1280)
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        x = x.view(batch_size, num_crops, -1)
        
        # Attention pooling: (B, N, 1280) → (B, 1280)
        pooled, attn_weights = self.attention(x, temperature=self.attention_temp)
        
        output = self.classifier(pooled)
        
        if return_attention:
            return output, attn_weights
        return output


class MultiCropDataset(Dataset):
    """Cycle-based crop extraction with 5x5 neighborhood (25 crops)"""
    
    def __init__(self, image_paths, labels, plate_well_map, crop_size=224, grid_size=12, augment=True, seed=42, epoch=0):
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.augment = augment
        self.seed = seed
        self.epoch = epoch
        
        # 5x5 neighborhood offsets
        self.offsets = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                self.offsets.append((dx, dy))
        
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        
        # Grid position from epoch-based cycle
        total_positions = self.grid_size * self.grid_size
        position_idx = (idx + self.epoch * len(self)) % total_positions
        grid_x, grid_y = position_idx % self.grid_size, position_idx // self.grid_size
        
        cell_w = w / self.grid_size
        cell_h = h / self.grid_size
        center_x = int((grid_x + 0.5) * cell_w)
        center_y = int((grid_y + 0.5) * cell_h)
        
        crops = []
        for dx, dy in self.offsets:
            crop_x = center_x + dx * int(cell_w)
            crop_y = center_y + dy * int(cell_h)
            
            crop_x = max(0, min(crop_x, w - self.crop_size))
            crop_y = max(0, min(crop_y, h - self.crop_size))
            
            crop = image.crop((crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size))
            crop = np.array(crop)
            
            if self.augment:
                crop = self._augment(crop)
            
            crop = crop.astype(np.float32) / 255.0
            crop = (crop - self.mean) / self.std
            crop = crop.transpose(2, 0, 1)
            crops.append(crop)
        
        crops = np.stack(crops, axis=0)
        return torch.tensor(crops, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
    def _augment(self, image):
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
        
        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k).copy()
        
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        return image


def get_gene_from_path(path):
    parts = path.split(os.sep)
    for part in parts:
        if '_' in part and 'Well' not in part:
            return part
    return parts[-1].split('_')[0] if parts else 'unknown'


def extract_well_from_filename(filename):
    match = re.search(r'Well([A-H]\d+)', filename)
    if match:
        return match.group(1)
    return 'unknown'


def get_data_splits(data_root, test_plate, val_plate='P6'):
    import json
    
    label_path = os.path.join(os.path.dirname(__file__), 'plate_well_id_path.json')
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            plate_well_map = json.load(f)
        gene_to_idx = {gene: idx for idx, gene in enumerate(sorted(set(plate_well_map.values())))}
    else:
        gene_to_idx = {}
        for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
            plate_path = os.path.join(data_root, plate)
            if os.path.exists(plate_path):
                for gene in os.listdir(plate_path):
                    if gene not in gene_to_idx:
                        gene_to_idx[gene] = len(gene_to_idx)
    
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    
    plates = {
        'train': [p for p in ['P1', 'P2', 'P3', 'P4', 'P5'] if p not in [test_plate, val_plate]],
        'val': [val_plate],
        'test': [test_plate]
    }
    
    for plate in plates['train']:
        plate_path = os.path.join(data_root, plate)
        if not os.path.exists(plate_path):
            continue
        for gene in os.listdir(plate_path):
            gene_path = os.path.join(plate_path, gene)
            if not os.path.isdir(gene_path):
                continue
            label = gene_to_idx.get(gene, 0)
            for root, _, files in os.walk(gene_path):
                for f in files:
                    if f.endswith('.tif'):
                        train_paths.append(os.path.join(root, f))
                        train_labels.append(label)
    
    for plate in plates['val']:
        plate_path = os.path.join(data_root, plate)
        if not os.path.exists(plate_path):
            continue
        for gene in os.listdir(plate_path):
            gene_path = os.path.join(plate_path, gene)
            if not os.path.isdir(gene_path):
                continue
            label = gene_to_idx.get(gene, 0)
            for root, _, files in os.walk(gene_path):
                for f in files:
                    if f.endswith('.tif'):
                        val_paths.append(os.path.join(root, f))
                        val_labels.append(label)
    
    for plate in plates['test']:
        plate_path = os.path.join(data_root, plate)
        if not os.path.exists(plate_path):
            continue
        for gene in os.listdir(plate_path):
            gene_path = os.path.join(plate_path, gene)
            if not os.path.isdir(gene_path):
                continue
            label = gene_to_idx.get(gene, 0)
            for root, _, files in os.walk(gene_path):
                for f in files:
                    if f.endswith('.tif'):
                        test_paths.append(os.path.join(root, f))
                        test_labels.append(label)
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, gene_to_idx


def focal_loss(preds, targets, alpha=0.25, gamma=2.0):
    ce_loss = nn.functional.cross_entropy(preds, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()