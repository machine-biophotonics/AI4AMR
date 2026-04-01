#!/usr/bin/env python3
"""
Visualize augmentations applied to a real 224x224 crop from a plate image.

Loads an actual TIFF image, extracts a center crop, and applies each augmentation
individually, plus shows random examples of the full training pipeline.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import albumentations as A
import cv2

# Path to a sample image (choose one from P1)
SAMPLE_IMAGE_PATH = "/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/P1/WellA01_PointA01_0000_ChannelCam-DIA DIC Master Screening_Seq0000_sharpest_image_1.tif"

# Crop parameters
CROP_SIZE = 224
IMAGE_SIZE = 2720
CENTER = (IMAGE_SIZE - CROP_SIZE) // 2

def load_image(path):
    """Load TIFF image using OpenCV."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return img

def extract_center_crop(img):
    """Extract center 224x224 crop from 2720x2720 image."""
    h, w = img.shape[:2]
    assert h == IMAGE_SIZE and w == IMAGE_SIZE, f"Expected {IMAGE_SIZE}x{IMAGE_SIZE}, got {h}x{w}"
    crop = img[CENTER:CENTER+CROP_SIZE, CENTER:CENTER+CROP_SIZE, :]
    return crop

# Individual augmentations with same parameters as in training (p=1.0 for deterministic)
individual_transforms = [
    ("Original", None),
    ("HorizontalFlip", A.HorizontalFlip(p=1.0)),
    ("VerticalFlip", A.VerticalFlip(p=1.0)),
    ("RandomRotate90", A.RandomRotate90(p=1.0)),
    ("Affine (translate/scale/rotate)", A.Affine(
        translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
        scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
        rotate=(-15, 15), p=1.0)),
    ("RandomBrightnessContrast", A.RandomBrightnessContrast(
        brightness_limit=0.3, contrast_limit=0.3, p=1.0)),
    ("RandomGamma", A.RandomGamma(gamma_limit=(80, 120), p=1.0)),
    ("RandomToneCurve", A.RandomToneCurve(scale=0.3, p=1.0)),
    ("RandomShadow", A.RandomShadow(p=1.0)),
    ("GaussNoise (monochrome)", A.GaussNoise(std_range=(0.1, 0.5), per_channel=False, p=1.0)),
    ("GaussianBlur", A.GaussianBlur(blur_limit=(3, 7), p=1.0)),
    ("MotionBlur", A.MotionBlur(blur_limit=7, p=1.0)),
    ("CoarseDropout", A.CoarseDropout(num_holes_range=(1, 3), 
                                      hole_height_range=(16, 64), 
                                      hole_width_range=(16, 64), p=1.0)),
    ("ImageCompression", A.ImageCompression(quality_range=(50, 100), p=1.0)),
    ("Sharpen", A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0)),
    ("ElasticTransform", A.ElasticTransform(alpha=50, sigma=5, p=1.0)),
]

# Full training pipeline (grayscale-friendly, same as in plate_dataset.py)
full_train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
             scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
             rotate=(-15, 15), p=0.5),
    A.SomeOf([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        A.RandomToneCurve(scale=0.3, p=1.0),
        A.RandomShadow(p=0.3),
    ], n=2, replace=False, p=0.5),
    A.SomeOf([
        A.GaussNoise(std_range=(0.1, 0.5), per_channel=False, p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.MotionBlur(blur_limit=7, p=1.0),
    ], n=1, replace=False, p=0.5),
    A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(16, 64), hole_width_range=(16, 64), p=0.4),
    A.ImageCompression(quality_range=(50, 100), p=0.3),
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
    A.ElasticTransform(alpha=50, sigma=5, p=0.2),
])

def visualize_individual_augmentations(crop):
    """Show each augmentation applied to the same crop."""
    n_augs = len(individual_transforms)
    cols = 4
    rows = (n_augs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    axes = axes.flatten()
    
    for idx, (name, transform) in enumerate(individual_transforms):
        ax = axes[idx]
        if transform is None:
            augmented = crop
        else:
            augmented = transform(image=crop)['image']
        ax.imshow(augmented)
        ax.set_title(name, fontsize=10)
        ax.axis('off')
    
    for i in range(n_augs, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Individual Augmentations on Real Crop\n{os.path.basename(SAMPLE_IMAGE_PATH)}', fontsize=14)
    plt.tight_layout()
    plt.savefig('real_crop_individual.png', dpi=150)
    plt.close()
    print("Saved real_crop_individual.png")

def visualize_full_pipeline(crop, n_examples=6):
    """Show random examples of the full training pipeline applied to the same crop."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(n_examples):
        ax = axes[i]
        augmented = full_train_transform(image=crop)['image']
        ax.imshow(augmented)
        ax.set_title(f'Example {i+1}')
        ax.axis('off')
    
    plt.suptitle('Full Training Pipeline (random combinations) applied to real crop', fontsize=16)
    plt.tight_layout()
    plt.savefig('real_crop_pipeline.png', dpi=150)
    plt.close()
    print("Saved real_crop_pipeline.png")

def visualize_before_after(crop):
    """Show original crop and a few augmented versions side by side."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(crop)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    for i in range(1, 5):
        axes[i].imshow(full_train_transform(image=crop)['image'])
        axes[i].set_title(f'Augmented {i}')
        axes[i].axis('off')
    
    plt.suptitle('Original vs Augmented Crops', fontsize=14)
    plt.tight_layout()
    plt.savefig('real_crop_before_after.png', dpi=150)
    plt.close()
    print("Saved real_crop_before_after.png")

if __name__ == '__main__':
    print("=== Real Crop Augmentation Visualization ===")
    print(f"Loading image: {SAMPLE_IMAGE_PATH}")
    img = load_image(SAMPLE_IMAGE_PATH)
    print(f"Image shape: {img.shape}")
    
    crop = extract_center_crop(img)
    print(f"Extracted center crop shape: {crop.shape}")
    
    print("Generating individual augmentations...")
    visualize_individual_augmentations(crop)
    
    print("Generating full pipeline examples...")
    visualize_full_pipeline(crop)
    
    print("Generating before/after comparison...")
    visualize_before_after(crop)
    
    print("\nAll visualizations saved as PNG files.")