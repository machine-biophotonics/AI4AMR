"""
MIL with class-bucket random sampling for diverse crop coverage
- Training: 9 crops from 9 DIFFERENT images of SAME class
- Val/Test: 9 crops from center + neighbors (same image)
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
    """Class-bucket random sampling: 9 crops from 9 DIFFERENT images per class"""
    
    def __init__(self, image_paths, labels, crop_size=224, grid_size=12, augment=True, seed=42, 
                 mode='train', num_crops_per_class=9):
        """
        mode: 'train' = sample from different images
              'val/test' = same image center + neighbors
        """
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.augment = augment
        self.seed = seed
        self.mode = mode
        self.num_crops_per_class = num_crops_per_class
        
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        self.image_size = w
        
        stride = (w - crop_size) // (grid_size - 1)
        self.stride = stride
        
        # All valid positions for cropping
        self.all_positions = []
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
                        self.all_positions.append((left, top))
        
        # Build class buckets: (image_idx, position) AND track unique images per class
        self.class_buckets = defaultdict(list)
        self.class_to_images = defaultdict(set)  # Track unique images per class
        for img_idx, label in enumerate(labels):
            for pos in self.all_positions:
                self.class_buckets[label].append((img_idx, pos))
                self.class_to_images[label].add(img_idx)
        
        # Get unique images count per class
        self.images_per_class = {c: len(imgs) for c, imgs in self.class_to_images.items()}
        
        # Pre-compute: positions available per image for each class
        self.image_positions = defaultdict(lambda: defaultdict(list))
        for label in self.class_buckets:
            for img_idx, pos in self.class_buckets[label]:
                self.image_positions[label][img_idx].append(pos)
        
        # Print bucket sizes
        bucket_sizes = [len(v) for v in self.class_buckets.values()]
        unique_img_counts = list(self.images_per_class.values())
        print(f"ClassBucketDataset: {len(self.class_buckets)} classes, "
              f"avg {np.mean(bucket_sizes):.0f} total positions, "
              f"avg {np.mean(unique_img_counts):.0f} unique images/class, "
              f"mode={mode}")
        
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
        
        # Tracking for exhaust-then-reset
        self.used_images = defaultdict(set)  # Track used image indices per class
        self.epoch_coverage = {}
        self.total_images = sum(len(v) for v in self.class_to_images.values())
        
        # Unique classes for batch iteration
        self.unique_classes = sorted(self.class_buckets.keys())
    
    def set_epoch(self, epoch):
        """Set up new random sample for epoch - ensure 9 DIFFERENT images"""
        self.current_epoch = epoch
        self.used_images.clear()
        self.epoch_coverage = {}
        
        rng = random.Random(self.seed + epoch)
        
        # For each class, sample 9 DIFFERENT images, then 1 random position from each
        for class_label in self.unique_classes:
            # Get available images for this class
            available_imgs = list(self.class_to_images[class_label])
            
            # Exhaust-then-reset: if we've used all images, reset
            if len(self.used_images[class_label]) >= len(available_imgs):
                self.used_images[class_label].clear()
            
            # Get unused images
            unused_imgs = [i for i in available_imgs if i not in self.used_images[class_label]]
            
            # If not enough unused, sample from all available
            if len(unused_imgs) < self.num_crops_per_class:
                self.used_images[class_label].clear()
                unused_imgs = available_imgs
            
            # Sample 9 different images
            if len(unused_imgs) >= self.num_crops_per_class:
                sampled_imgs = rng.sample(unused_imgs, self.num_crops_per_class)
            else:
                # If insufficient, allow repeats from remaining
                sampled_imgs = rng.choices(unused_imgs, k=self.num_crops_per_class)
            
            # Mark these images as used
            for img_idx in sampled_imgs:
                self.used_images[class_label].add(img_idx)
            
            # For each sampled image, pick 1 random position
            sampled = []
            for img_idx in sampled_imgs:
                positions = self.image_positions[class_label][img_idx]
                pos = rng.choice(positions)
                sampled.append((img_idx, pos))
            
            self.epoch_coverage[class_label] = sampled
    
    def __len__(self):
        return len(self.unique_classes)
    
    def __getitem__(self, class_idx):
        """Return 9 crops from 9 DIFFERENT images of the same class"""
        class_label = self.unique_classes[class_idx]
        
        if self.mode == 'val' or self.mode == 'test':
            # Val/test: use center crop + 3x3 neighborhood from SAME image
            img_idx = random.randint(0, len(self.image_paths) - 1)
            center_left = (self.image_size - self.crop_size) // 2
            center_top = (self.image_size - self.crop_size) // 2
            
            crops_list = []
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    left = center_left + dj * self.stride
                    top = center_top + di * self.stride
                    img_path = self.image_paths[img_idx]
                    image = Image.open(img_path).convert('RGB')
                    crop = image.crop((left, top, left + self.crop_size, top + self.crop_size))
                    crop = np.array(crop)
                    crop = self.transform(image=crop)['image']
                    crops_list.append(crop)
        else:
            # Train: sample 9 from DIFFERENT images + 1 random position each
            sampled = self.epoch_coverage[class_label]
            
            crops_list = []
            for img_idx, (left, top) in sampled:
                img_path = self.image_paths[img_idx]
                image = Image.open(img_path).convert('RGB')
                crop = image.crop((left, top, left + self.crop_size, top + self.crop_size))
                crop = np.array(crop)
                crop = self.transform(image=crop)['image']
                crops_list.append(crop)
                
                # Track coverage (mark image as used for exhaust tracking)
                self.used_images[class_label].add(img_idx)
        
        # Shuffle crop order
        if self.augment and self.mode == 'train':
            perm = list(range(self.num_crops_per_class))
            random.shuffle(perm)
            crops_list = [crops_list[i] for i in perm]
        
        crops = torch.stack(crops_list)
        
        # Get class index for label
        class_idx = self.unique_classes.index(class_label)
        
        return crops, class_idx
    
    def get_coverage_report(self):
        """Report on epoch coverage - how many unique images sampled"""
        total_used = sum(len(v) for v in self.used_images.values())
        return {
            'unique_images_sampled': total_used,
            'total_unique_images': self.total_images,
            'coverage_pct': total_used / self.total_images * 100
        }


class SingleImageDataset(Dataset):
    """Original: 9 crops from SAME image (for testing/validation)"""
    
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
                A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.5, p=0.3),
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