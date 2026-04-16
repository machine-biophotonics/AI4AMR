"""
MIL with class-bucket random sampling for diverse crop coverage.
- Training: 144 positions per image, 9 crops from 9 DIFFERENT images per class per epoch
- Val/Test: 9 crops from center + 3x3 neighborhood (same image)
- Configurable attention heads (default: 20)
- One epoch samples 9 pairs from each class
- Total epochs to exhaust: 12,096 / 9 = 1,344 epochs per class

Usage:
    python train_mil.py --epochs 200 --num_heads 20  # Default (20 heads)
    python train_mil.py --epochs 200 --num_heads 8   # Alternative (8 heads)
"""

from __future__ import annotations

import random
import re
import os
import time
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


NORMALIZE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORMALIZE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NUM_HEADS = 20  # Default: 20 heads. Configurable via --num_heads argument.


class AttentionPooling(nn.Module):
    """Gated attention MIL pooling (Ilse et al. 2018)."""
    
    def __init__(self, in_features: int, num_heads: int = NUM_HEADS) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = in_features // 4
        
        self.V = nn.Linear(in_features, head_dim)
        self.U = nn.Linear(in_features, head_dim)
        self.w = nn.Linear(head_dim, num_heads)
    
    def forward(self, x: torch.Tensor, temperature: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
        A_gate = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        attn_weights = self.w(A_gate)
        attn_weights = torch.softmax(attn_weights / temperature, dim=1)
        pooled = torch.einsum('bnh,bnf->bhf', attn_weights, x)
        return pooled, attn_weights


class AttentionMILModel(nn.Module):
    """Attention-based MIL model with EfficientNet-B0 backbone and configurable attention heads.
    
    Args:
        num_classes: Number of output classes
        num_heads: Number of attention heads (default: 8, recommended: 4-8)
        attention_temp: Temperature for attention softmax (default: 0.5)
    """
    
    def __init__(
        self,
        num_classes: int,
        num_heads: int = NUM_HEADS,
        attention_temp: float = 0.5
    ) -> None:
        super().__init__()
        
        import torchvision.models as torchvision_models
        
        base_model = torchvision_models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.feature_dim = 1280
        
        self.attention_pool = AttentionPooling(self.feature_dim, num_heads)
        self.attention_temp = attention_temp
        self.head_proj = nn.Linear(self.feature_dim * num_heads, self.feature_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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


class ClassBucketDataset(Dataset):
    """
    Class-bucket random sampling: 9 crops from 9 DIFFERENT images per class per epoch.
    
    Architecture:
    - Grid: 12×12 = 144 positions per image (all valid grid positions)
    - Per class: 84 images × 144 positions = 12,096 (image, position) pairs
    - Per epoch: 96 classes × 9 pairs = 864 pairs sampled
    - Epochs to exhaust: 12,096 / 9 = 1,344 epochs
    
    Time complexity: O(1) per sample retrieval
    """
    
    def __init__(
        self,
        image_paths: list[str],
        labels: np.ndarray,
        crop_size: int = 224,
        grid_size: int = 12,
        augment: bool = True,
        seed: int = 42,
        num_crops_per_class: int = 9
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.augment = augment
        self.seed = seed
        self.num_crops_per_class = num_crops_per_class
        self.current_epoch = 0
        
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        self.image_size = w
        
        self.stride = (w - crop_size) // (grid_size - 1)
        
        self.all_positions: list[tuple[int, int]] = []
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * self.stride
                top = i * self.stride
                if left + crop_size <= w and top + crop_size <= h:
                    self.all_positions.append((left, top))
        
        self.class_buckets: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self.class_to_images: dict[int, set[int]] = defaultdict(set)
        
        for img_idx, label in enumerate(labels):
            for pos_idx in range(len(self.all_positions)):
                self.class_buckets[label].append((img_idx, pos_idx))
                self.class_to_images[label].add(img_idx)
        
        self.num_images_per_class: dict[int, int] = {
            c: len(imgs) for c, imgs in self.class_to_images.items()
        }
        self.num_pairs_per_class = len(self.all_positions) * len(self.num_images_per_class)
        self.total_pairs = sum(len(v) for v in self.class_buckets.values())
        
        self.epoch_shuffle_seeds: dict[int, dict[int, random.Random]] = {}
        self.epoch_coverage: dict[int, list[tuple[int, int]]] = {}
        
        self.unique_classes = sorted(self.class_buckets.keys())
        self.class_to_idx = {c: i for i, c in enumerate(self.unique_classes)}
        
        self.num_classes = len(self.unique_classes)
        self.epochs_to_exhaust = self.num_pairs_per_class // num_crops_per_class
        
        print(f"ClassBucketDataset:")
        print(f"  - Classes: {self.num_classes}")
        print(f"  - Images per class: {np.mean(list(self.num_images_per_class.values())):.0f} (min: {min(self.num_images_per_class.values())}, max: {max(self.num_images_per_class.values())})")
        print(f"  - Positions per image: {len(self.all_positions)}")
        print(f"  - Pairs per class: {self.num_pairs_per_class}")
        print(f"  - Total pairs: {self.total_pairs}")
        print(f"  - Pairs per epoch: {self.num_classes * num_crops_per_class}")
        print(f"  - Epochs to exhaust: {self.epochs_to_exhaust}")
        
        if augment:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.5,
                    contrast_limit=0.5,
                    brightness_by_max=False,
                    p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
    
    def set_epoch(self, epoch: int) -> None:
        """Set epoch for cycle-based crop selection. O(n_classes) complexity."""
        self.current_epoch = epoch
        
        cycle_number = epoch // self.epochs_to_exhaust
        epoch_in_cycle = epoch % self.epochs_to_exhaust
        
        start_idx = epoch_in_cycle * self.num_crops_per_class
        
        for class_label in self.unique_classes:
            bucket = self.class_buckets[class_label]
            
            if class_label not in self.epoch_shuffle_seeds:
                self.epoch_shuffle_seeds[class_label] = {}
            
            if cycle_number not in self.epoch_shuffle_seeds[class_label]:
                seed_value = self.seed + int(class_label) * 10000 + int(cycle_number) * 1000
                self.epoch_shuffle_seeds[class_label][cycle_number] = random.Random(seed_value)
            
            rng = self.epoch_shuffle_seeds[class_label][cycle_number]
            shuffled = bucket.copy()
            rng.shuffle(shuffled)
            
            end_idx = start_idx + self.num_crops_per_class
            sampled = shuffled[start_idx:end_idx]
            self.epoch_coverage[class_label] = sampled
    
    def __len__(self) -> int:
        return self.num_classes
    
    def __getitem__(self, class_idx: int) -> tuple[torch.Tensor, int]:
        """Return 9 crops from 9 DIFFERENT images of the same class. O(1) per call."""
        class_label = self.unique_classes[class_idx]
        sampled = self.epoch_coverage[class_label]
        
        crops_list: list[torch.Tensor] = []
        for img_idx, pos_idx in sampled:
            img_path = self.image_paths[img_idx]
            left, top = self.all_positions[pos_idx]
            
            image = Image.open(img_path).convert('RGB')
            crop = image.crop((left, top, left + self.crop_size, top + self.crop_size))
            crop = np.array(crop)
            crop = self.transform(image=crop)['image']
            crops_list.append(crop)
        
        if self.augment:
            perm = list(range(self.num_crops_per_class))
            random.shuffle(perm)
            crops_list = [crops_list[i] for i in perm]
        
        crops = torch.stack(crops_list)
        label_idx = self.class_to_idx[class_label]
        
        return crops, label_idx
    
    def get_coverage_report(self) -> dict[str, int]:
        """Report on epoch coverage."""
        cycle_number = self.current_epoch // self.epochs_to_exhaust
        epoch_in_cycle = self.current_epoch % self.epochs_to_exhaust
        return {
            'epoch': self.current_epoch,
            'cycle': cycle_number,
            'epoch_in_cycle': epoch_in_cycle,
            'pairs_per_epoch': self.num_classes * self.num_crops_per_class,
            'epochs_per_cycle': self.epochs_to_exhaust
        }


class SingleImageDataset(Dataset):
    """
    9 crops from SAME image (center + 3x3 neighborhood) for val/test.
    
    Architecture:
    - Extracts center crop + 8 neighbors (3×3 grid)
    - Uses 100 valid positions (edge positions excluded for full neighborhood)
    """
    
    def __init__(
        self,
        image_paths: list[str],
        labels: np.ndarray,
        crop_size: int = 224,
        grid_size: int = 12,
        augment: bool = False,
        seed: int = 42
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.augment = augment
        self.seed = seed
        self.center = (0, 0)
        
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        self.image_size = w
        
        stride = (w - crop_size) // (grid_size - 1)
        self.stride = stride
        
        positions: list[tuple[int, int]] = []
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
        
        print(f"SingleImageDataset:")
        print(f"  - Images: {len(image_paths)}")
        print(f"  - Crops per image: 9 (center + 8 neighbors)")
        print(f"  - Valid positions: {len(self.positions)}")
        
        if augment:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)}, rotate=(-10, 10), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
    
    def set_epoch(self, epoch: int) -> None:
        """Set epoch (uses center for val/test)."""
        center_left = (self.image_size - self.crop_size) // 2
        center_top = (self.image_size - self.crop_size) // 2
        self.center = (center_left, center_top)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        center_left, center_top = self.center
        
        crops_list: list[torch.Tensor] = []
        for di in range(-1, 2):
            for dj in range(-1, 2):
                left = center_left + dj * self.stride
                top = center_top + di * self.stride
                crop = image.crop((left, top, left + self.crop_size, top + self.crop_size))
                crop = np.array(crop)
                crop = self.transform(image=crop)['image']
                crops_list.append(crop)
        
        crops = torch.stack(crops_list)
        
        return crops, int(self.labels[idx])


def extract_well_from_filename(filename: str) -> Optional[str]:
    """Extract well position from image filename."""
    match = re.search(r'Well(\w\d+)_', filename)
    return match.group(1) if match else None


def get_gene_from_path(img_path: str, plate_maps: dict) -> str:
    """Get gene label from image path."""
    dirname = os.path.dirname(img_path)
    plate = os.path.basename(dirname)
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if plate in plate_maps and well in plate_maps[plate]:
        return plate_maps[plate][well]
    return 'WT'


def worker_init_fn(worker_id: int, seed: int = 42) -> None:
    """Module-level worker init function for Windows Python 3.14 compatibility."""
    random.seed(seed + worker_id)
