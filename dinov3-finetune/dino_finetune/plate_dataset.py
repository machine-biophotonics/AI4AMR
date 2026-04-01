import json
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import math

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from functools import lru_cache


class PlateDataset(Dataset):
    """
    Dataset for CRISPRi plate images with gene perturbation labels.
    Loads images from plate folders (P1-P6) and maps to class IDs.
    Uses albumentations for transforms.
    Supports multi-crop extraction from 2720x2720 images.
    """
    def __init__(
        self,
        plate_names: List[str],
        data_root: str,
        label_json_path: str,
        transform: Optional[A.Compose] = None,
        target_size: Tuple[int, int] = (224, 224),
        stain_augmentation: bool = False,
        all_crops: bool = True,  # if False, only center crop for val/test
        random_crops: bool = True,  # if True, randomly crop each iteration
        grid_size: int = 12,  # 12x12 = 144 crops per image
        seed: int = 42,  # random seed for reproducible crop shuffling
    ):
        self.plate_names = plate_names
        self.data_root = data_root
        self.target_size = target_size
        self.stain_augmentation = stain_augmentation
        self.all_crops = all_crops
        self.random_crops = random_crops
        self.grid_size = grid_size
        self.crop_size = target_size[0]  # assuming square
        self.seed = seed
        self.epoch = 0
        self.shuffled_positions = []  # will be filled after positions are computed
        
        # Load label mapping
        with open(label_json_path, 'r') as f:
            self.label_data = json.load(f)
        
        # Load pathway order to get all possible class IDs
        pathway_json = Path(label_json_path).parent / 'class_pathway_order.json'
        with open(pathway_json, 'r') as f:
            pathway_data = json.load(f)
        self.class_names = pathway_data['pathway_order']  # list of 85 strings
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        
        # Build sample list: each sample is (image_path, class_idx, plate_name)
        self.image_samples = []  # unique images
        for plate in plate_names:
            plate_dir = os.path.join(data_root, plate)
            if not os.path.isdir(plate_dir):
                raise FileNotFoundError(f"Plate directory not found: {plate_dir}")
            
            # Iterate over image files (tif, tiff, png, jpg)
            for img_file in sorted(os.listdir(plate_dir)):
                if not img_file.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                    continue
                # Extract well coordinate from filename
                # Example: WellA01_PointA01_0000_ChannelCam-DIA DIC Master Screening_Seq0000_sharpest_image_1.tif
                parts = img_file.split('_')
                if len(parts) < 4:
                    continue
                well_part = parts[0]  # 'WellA01'
                if not well_part.startswith('Well'):
                    continue
                well_id = well_part[4:]  # 'A01'
                row_letter = well_id[0].upper()
                try:
                    col = int(well_id[1:])
                except ValueError:
                    continue
                
                # Get label from JSON
                if (plate in self.label_data and
                    row_letter in self.label_data[plate] and
                    str(col) in self.label_data[plate][row_letter]):
                    class_name = self.label_data[plate][row_letter][str(col)]['id']
                    if class_name not in self.class_to_idx:
                        continue
                    class_idx = self.class_to_idx[class_name]
                else:
                    continue
                
                img_path = os.path.join(plate_dir, img_file)
                self.image_samples.append((img_path, class_idx, plate))
        
        # Compute crop positions
        self.samples = []  # list of (img_path, class_idx, plate, crop_x, crop_y)
        self.positions = []  # list of (x, y) for random cropping
        self.stride = None
        
        # Determine if we are using random crops
        use_random_crops = self.random_crops and self.all_crops  # random crops only when all_crops=True
        
        if use_random_crops:
            # Compute grid positions for random cropping
            stride = (2720 - self.crop_size) // (self.grid_size - 1)
            self.stride = stride
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    x = i * stride
                    y = j * stride
                    if x + self.crop_size <= 2720 and y + self.crop_size <= 2720:
                        self.positions.append((x, y))
            # Shuffle positions deterministically for epoch 0
            self.shuffled_positions = self.positions.copy()
            random.Random(self.seed).shuffle(self.shuffled_positions)
            # Store only image samples, crop randomly in __getitem__
            for img_path, class_idx, plate in self.image_samples:
                self.samples.append((img_path, class_idx, plate, None, None))
            print(f"Random cropping enabled: {len(self.positions)} possible positions per image")
        else:
            if self.all_crops:
                # Compute all crop positions
                stride = (2720 - self.crop_size) // (self.grid_size - 1)
                self.stride = stride
                positions = []
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        x = i * stride
                        y = j * stride
                        if x + self.crop_size <= 2720 and y + self.crop_size <= 2720:
                            positions.append((x, y))
                for img_path, class_idx, plate in self.image_samples:
                    for (x, y) in positions:
                        self.samples.append((img_path, class_idx, plate, x, y))
                print(f"All crops: {len(positions)} per image, total samples: {len(self.samples)}")
            else:
                # Center crop only
                center_x = (2720 - self.crop_size) // 2
                center_y = center_x
                for img_path, class_idx, plate in self.image_samples:
                    self.samples.append((img_path, class_idx, plate, center_x, center_y))
                print(f"Center crop only, total samples: {len(self.samples)}")
        
        print(f"Loaded {len(self.image_samples)} images from plates {plate_names}")
        print(f"Class distribution: {self._get_class_distribution()}")
        
        # Setup transforms
        self.transform = transform
        if self.transform is None:
            self.transform = self._default_transform()
        
        # Stain augmentation (optional)
        self.stain_transform = None
        if stain_augmentation:
            try:
                from stainaug import Augmentor
                self.stain_transform = Augmentor()
                print("Stain augmentation enabled")
            except ImportError:
                print("Warning: stainaug library not installed. Skipping stain augmentation.")
                self.stain_augmentation = False
        
        # Cache for loaded images
        self._image_cache = {}
        self.epoch = 0  # for deterministic crop selection
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        # Recompute shuffled positions for this epoch (only if random cropping enabled)
        if self.shuffled_positions:
            self.shuffled_positions = self.positions.copy()
            random.Random(self.seed + epoch).shuffle(self.shuffled_positions)
    
    def _default_transform(self):
        """Default transform: resize, normalize with ImageNet stats."""
        return A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    
    def _get_class_distribution(self):
        """Return dict of class index -> count."""
        dist = {}
        for _, class_idx, _, _, _ in self.samples:
            dist[class_idx] = dist.get(class_idx, 0) + 1
        return dist
    
    def __len__(self):
        return len(self.samples)
    
    def _load_image(self, img_path):
        """Load image with caching."""
        if img_path not in self._image_cache:
            # Load image using OpenCV
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Could not load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self._image_cache[img_path] = image
            # Keep cache size manageable (max 100 images)
            if len(self._image_cache) > 100:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._image_cache))
                del self._image_cache[oldest_key]
        return self._image_cache[img_path]
    
    def __getitem__(self, idx):
        img_path, class_idx, plate, x, y = self.samples[idx]
        # Load image
        image = self._load_image(img_path)
        
        # If x and y are None, we need to select a crop position deterministically
        if x is None or y is None:
            if self.shuffled_positions:
                # Determine position index based on epoch and sample index
                position_index = (self.epoch + idx) % len(self.shuffled_positions)
                x, y = self.shuffled_positions[position_index]
            elif self.positions:
                # fallback to unshuffled positions (should not happen for random cropping)
                position_index = (self.epoch + idx) % len(self.positions)
                x, y = self.positions[position_index]
            else:
                # fallback to center crop
                x = (2720 - self.crop_size) // 2
                y = x
        
        # Extract crop
        crop = image[y:y+self.crop_size, x:x+self.crop_size]
        
        # Apply stain augmentation if enabled
        if self.stain_augmentation and self.stain_transform is not None:
            crop = self.stain_transform.augment_HE(crop)
        
        # Apply transforms
        augmented = self.transform(image=crop)
        crop_tensor = augmented['image']
        
        # Metadata for later analysis
        metadata = {
            'image_path': img_path,
            'crop_x': x,
            'crop_y': y,
            'class_idx': class_idx,
            'plate': plate,
        }
        
        return crop_tensor, class_idx, plate, metadata


def get_plates_split():
    """Return predefined splits for train/val/test."""
    train_plates = ['P1', 'P2', 'P3', 'P4']
    val_plates = ['P5']
    test_plates = ['P6']
    return train_plates, val_plates, test_plates


def create_datasets(data_root: str, label_json_path: str, stain_augmentation: bool = False, target_size: Tuple[int, int] = (224, 224), seed: int = 42):
    """Create train, val, test datasets."""
    train_plates, val_plates, test_plates = get_plates_split()
    
    # Define augmentations for training - grayscale-friendly pipeline
    train_transform = A.Compose([
        # No resize, crop already 224x224
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                 scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
                 rotate=(-15, 15), p=0.5),
        # Pixel-level intensity augmentations (grayscale-friendly)
        A.SomeOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.RandomToneCurve(scale=0.3, p=1.0),
            A.RandomShadow(p=0.3),
        ], n=2, replace=False, p=0.5),
        # Noise and blur (grayscale-friendly)
        A.SomeOf([
            A.GaussNoise(std_range=(0.1, 0.5), per_channel=False, p=1.0),  # monochrome noise
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], n=1, replace=False, p=0.5),
        # Cutout-like dropout
        A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(16, 64), hole_width_range=(16, 64), p=0.4),
        # Compression artifacts
        A.ImageCompression(quality_range=(50, 100), p=0.3),
        # Sharpening
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
        # Elastic distortions (more aggressive)
        A.ElasticTransform(alpha=50, sigma=5, p=0.2),
        # Normalization (using ImageNet stats, but channels are identical)
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    # No augmentation for val/test (center crop)
    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    train_dataset = PlateDataset(
        plate_names=train_plates,
        data_root=data_root,
        label_json_path=label_json_path,
        transform=train_transform,
        target_size=target_size,
        stain_augmentation=stain_augmentation,
        all_crops=True,
        random_crops=True,
        grid_size=12,
        seed=seed,
    )
    val_dataset = PlateDataset(
        plate_names=val_plates,
        data_root=data_root,
        label_json_path=label_json_path,
        transform=val_transform,
        target_size=target_size,
        stain_augmentation=False,
        all_crops=False,  # center crop only
        random_crops=False,
        grid_size=12,
        seed=seed,
    )
    test_dataset = PlateDataset(
        plate_names=test_plates,
        data_root=data_root,
        label_json_path=label_json_path,
        transform=val_transform,
        target_size=target_size,
        stain_augmentation=False,
        all_crops=False,
        random_crops=False,
        grid_size=12,
        seed=seed,
    )
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Quick test
    data_root = "/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging"
    label_json_path = os.path.join(data_root, "plate maps/plate_well_id_path.json")
    train_ds, val_ds, test_ds = create_datasets(data_root, label_json_path, stain_augmentation=False)
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")
    # Load one sample
    img, label, plate, metadata = train_ds[0]
    print(f"Image shape: {img.shape}, Label: {label}, Plate: {plate}")
    print(f"Metadata: {metadata}")