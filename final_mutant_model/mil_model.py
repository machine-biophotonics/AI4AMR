"""
MIL with class-bucket random sampling for diverse crop coverage
- Training: 144 crops per image, 9 crops from 9 DIFFERENT images per class per epoch
- Val/Test: 9 crops from center + 3x3 neighborhood (same image)
- One epoch = exhaust all (image, position) pairs
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
from collections import defaultdict


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
        pooled = torch.einsum('bnh,bnf->bhf', attn_weights, x)
        return pooled, attn_weights


class AttentionMILModel(nn.Module):
    def __init__(self, num_classes, num_heads=4, attention_temp=0.5):
        super().__init__()
        base_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
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


class ClassBucketDataset(Dataset):
    """
    Class-bucket random sampling: 144 crops per image, 9 crops from 9 DIFFERENT images per class.
    
    - Grid: 12x12 = 144 positions per image (ALL positions, no neighborhood requirement)
    - Per class: 84 images × 144 positions = 12,096 (image, position) pairs
    - Per epoch: 96 classes × 9 pairs = 864 pairs sampled
    - Epochs to exhaust: 12,096 / 864 = 14 epochs
    """
    
    def __init__(self, image_paths, labels, crop_size=224, grid_size=12, augment=True, seed=42, num_crops_per_class=9):
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.augment = augment
        self.seed = seed
        self.num_crops_per_class = num_crops_per_class
        
        # Get image size from first image
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        self.image_size = w
        
        # Calculate stride for 144 positions (all grid positions)
        self.stride = (w - crop_size) // (grid_size - 1)
        
        # ALL 144 positions per image (no neighborhood requirement)
        self.all_positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * self.stride
                top = i * self.stride
                if left + crop_size <= w and top + crop_size <= h:
                    self.all_positions.append((left, top))
        
        print(f"Total positions per image: {len(self.all_positions)}")
        
        # Build class buckets: (image_idx, position_idx)
        self.class_buckets = defaultdict(list)
        self.class_to_images = defaultdict(set)
        
        for img_idx, label in enumerate(labels):
            for pos_idx, pos in enumerate(self.all_positions):
                self.class_buckets[label].append((img_idx, pos_idx))
                self.class_to_images[label].add(img_idx)
        
        # Pre-compute: positions available per image for each class
        self.image_positions = defaultdict(dict)
        for label in self.class_buckets:
            for img_idx, pos_idx in self.class_buckets[label]:
                if img_idx not in self.image_positions[label]:
                    self.image_positions[label][img_idx] = []
                self.image_positions[label][img_idx].append(pos_idx)
        
        # Statistics
        self.num_images_per_class = {c: len(imgs) for c, imgs in self.class_to_images.items()}
        self.num_pairs_per_class = len(self.all_positions) * len(self.num_images_per_class)
        self.total_pairs = sum(len(v) for v in self.class_buckets.values())
        
        # Epoch tracking for cycling
        self.epoch_shuffle_seeds = {}
        self.epoch_coverage = {}
        
        # Unique classes for batch iteration
        self.unique_classes = sorted(self.class_buckets.keys())
        
        # Class index mapping
        self.class_to_idx = {c: i for i, c in enumerate(self.unique_classes)}
        
        print(f"ClassBucketDataset:")
        print(f"  - Classes: {len(self.unique_classes)}")
        print(f"  - Images per class: avg {np.mean(list(self.num_images_per_class.values())):.0f}")
        print(f"  - Positions per image: {len(self.all_positions)}")
        print(f"  - Pairs per class: {self.num_pairs_per_class}")
        print(f"  - Total pairs: {self.total_pairs}")
        print(f"  - Pairs per epoch: {len(self.unique_classes) * num_crops_per_class}")
        print(f"  - Epochs to exhaust: {self.num_pairs_per_class / (len(self.unique_classes) * num_crops_per_class):.1f}")
        
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
    
    def set_epoch(self, epoch):
        """
        One epoch samples 9 pairs from each class.
        Cycles through all (image, position) pairs, then reshuffles with new seed.
        """
        self.current_epoch = epoch
        
        epochs_needed = self.num_pairs_per_class // self.num_crops_per_class
        
        cycle_number = epoch // epochs_needed
        epoch_in_cycle = epoch % epochs_needed
        
        for class_label in self.unique_classes:
            bucket = self.class_buckets[class_label]
            
            if class_label not in self.epoch_shuffle_seeds:
                self.epoch_shuffle_seeds[class_label] = {}
            
            if cycle_number not in self.epoch_shuffle_seeds[class_label]:
                self.epoch_shuffle_seeds[class_label][cycle_number] = random.Random(
                    self.seed + class_label * 10000 + cycle_number * 1000
                )
            
            rng = self.epoch_shuffle_seeds[class_label][cycle_number]
            
            shuffled = bucket.copy()
            rng.shuffle(shuffled)
            
            sampled = shuffled[:self.num_crops_per_class]
            self.epoch_coverage[class_label] = sampled
    
    def __len__(self):
        return len(self.unique_classes)
    
    def __getitem__(self, class_idx):
        """Return 9 crops from 9 DIFFERENT images of the same class."""
        class_label = self.unique_classes[class_idx]
        sampled = self.epoch_coverage[class_label]
        
        crops_list = []
        for img_idx, pos_idx in sampled:
            img_path = self.image_paths[img_idx]
            left, top = self.all_positions[pos_idx]
            
            image = Image.open(img_path).convert('RGB')
            crop = image.crop((left, top, left + self.crop_size, top + self.crop_size))
            crop = np.array(crop)
            crop = self.transform(image=crop)['image']
            crops_list.append(crop)
        
        # Shuffle crop order for augmentation
        if self.augment:
            perm = list(range(self.num_crops_per_class))
            random.shuffle(perm)
            crops_list = [crops_list[i] for i in perm]
        
        crops = torch.stack(crops_list)
        label_idx = self.class_to_idx[class_label]
        
        return crops, label_idx
    
    def get_coverage_report(self):
        """Report on epoch coverage."""
        epochs_needed = self.num_pairs_per_class // self.num_crops_per_class
        cycle_number = self.current_epoch // epochs_needed
        epoch_in_cycle = self.current_epoch % epochs_needed
        return {
            'epoch': self.current_epoch,
            'cycle': cycle_number,
            'epoch_in_cycle': epoch_in_cycle,
            'pairs_per_epoch': len(self.unique_classes) * self.num_crops_per_class,
            'epochs_per_cycle': epochs_needed
        }


class SingleImageDataset(Dataset):
    """9 crops from SAME image (center + 3x3 neighborhood) for val/test."""
    
    def __init__(self, image_paths, labels, crop_size=224, grid_size=12, augment=False, seed=42):
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.augment = augment
        self.seed = seed
        
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        self.image_size = w
        
        stride = (w - crop_size) // (grid_size - 1)
        self.stride = stride
        
        # Only positions with full neighborhood
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
    
    def set_epoch(self, epoch):
        center_left = (self.image_size - self.crop_size) // 2
        center_top = (self.image_size - self.crop_size) // 2
        self.center = (center_left, center_top)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        center_left, center_top = self.center
        
        crops_list = []
        for di in range(-1, 2):
            for dj in range(-1, 2):
                left = center_left + dj * self.stride
                top = center_top + di * self.stride
                crop = image.crop((left, top, left + self.crop_size, top + self.crop_size))
                crop = np.array(crop)
                crop = self.transform(image=crop)['image']
                crops_list.append(crop)
        
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


# Module-level worker init function for Windows Python 3.14 compatibility
def worker_init_fn(worker_id, seed=42):
    random.seed(seed + worker_id)
