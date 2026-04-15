#!/usr/bin/env python3
"""
Visualize augmentations applied to a single 224x224 crop during training.

Shows the effect of each augmentation individually, plus the full pipeline.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Parameters
crop_size = 224

def create_sample_crop():
    """Create a synthetic 224x224 image with patterns to better see augmentations."""
    # Create a grid pattern with colors
    img = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    # Add grid lines
    for i in range(0, crop_size, 32):
        img[i:i+2, :, :] = 255  # horizontal white lines
        img[:, i:i+2, :] = 255  # vertical white lines
    # Add colored squares
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
    for i in range(6):
        for j in range(6):
            x = i * 32 + 8
            y = j * 32 + 8
            color = colors[(i + j) % len(colors)]
            img[x:x+16, y:y+16, :] = color
    # Add some texture (noise)
    noise = np.random.randint(0, 30, (crop_size, crop_size, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    return img

# Individual augmentations with same parameters as in training
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
    ("RGBShift", A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0)),
    ("RandomGamma", A.RandomGamma(gamma_limit=(80, 120), p=1.0)),
    ("RandomToneCurve", A.RandomToneCurve(scale=0.3, p=1.0)),
    ("RandomShadow", A.RandomShadow(p=1.0)),
    ("GaussNoise", A.GaussNoise(std_range=(0.1, 0.5), p=1.0)),
    ("ISONoise", A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0)),
    ("GaussianBlur", A.GaussianBlur(blur_limit=(3, 7), p=1.0)),
    ("MotionBlur", A.MotionBlur(blur_limit=7, p=1.0)),
    ("ColorJitter", A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0)),
    ("CoarseDropout", A.CoarseDropout(num_holes_range=(1, 3), 
                                      hole_height_range=(16, 64), 
                                      hole_width_range=(16, 64), p=1.0)),
    ("ImageCompression", A.ImageCompression(quality_range=(50, 100), p=1.0)),
    ("Sharpen", A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0)),
    ("ElasticTransform", A.ElasticTransform(alpha=50, sigma=5, p=1.0)),
]

# Full training pipeline (without normalization)
full_train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
             scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
             rotate=(-15, 15), p=0.5),
    A.SomeOf([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        A.RandomToneCurve(scale=0.3, p=1.0),
        A.RandomShadow(p=0.3),
    ], n=2, replace=False, p=0.5),
    A.SomeOf([
        A.GaussNoise(std_range=(0.1, 0.5), p=1.0),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.MotionBlur(blur_limit=7, p=1.0),
    ], n=1, replace=False, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(16, 64), hole_width_range=(16, 64), p=0.4),
    A.ImageCompression(quality_range=(50, 100), p=0.3),
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
    A.ElasticTransform(alpha=50, sigma=5, p=0.2),
])

def visualize_individual_augmentations(sample):
    """Show each augmentation individually."""
    n_augs = len(individual_transforms)
    cols = 4
    rows = (n_augs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    axes = axes.flatten()
    
    for idx, (name, transform) in enumerate(individual_transforms):
        ax = axes[idx]
        if transform is None:
            augmented = sample
        else:
            augmented = transform(image=sample)['image']
        ax.imshow(augmented)
        ax.set_title(name, fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_augs, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Individual Augmentations (applied with probability=1.0)', fontsize=16)
    plt.tight_layout()
    plt.savefig('augmentations_individual.png', dpi=150)
    plt.close()
    print("Saved augmentations_individual.png")

def visualize_full_pipeline(sample, n_examples=4):
    """Show random examples of the full training pipeline."""
    fig, axes = plt.subplots(1, n_examples, figsize=(4*n_examples, 4))
    for i in range(n_examples):
        ax = axes[i]
        augmented = full_train_transform(image=sample)['image']
        ax.imshow(augmented)
        ax.set_title(f'Example {i+1}')
        ax.axis('off')
    plt.suptitle('Full Training Augmentation Pipeline (random combinations)', fontsize=14)
    plt.tight_layout()
    plt.savefig('augmentations_pipeline.png', dpi=150)
    plt.close()
    print("Saved augmentations_pipeline.png")

if __name__ == '__main__':
    print("=== Augmentation Visualization ===")
    print("Creating sample 224x224 crop...")
    sample = create_sample_crop()
    
    print("Visualizing individual augmentations...")
    visualize_individual_augmentations(sample)
    
    print("Visualizing full pipeline examples...")
    visualize_full_pipeline(sample)
    
    print("\nAugmentation list (from plate_dataset.py):")
    for name, _ in individual_transforms[1:]:  # skip original
        print(f"  - {name}")
    
    print("\nNote: Some augmentations are applied probabilistically (p<1.0).")
    print("The full pipeline uses SomeOf to randomly select subsets.")