#!/usr/bin/env python3
"""
Visualize augmentations for 3 training configs:
- plate_fold (heavy/legacy)
- plate_fold_no_aug (paper-based minimal)
- final_crispr_model (paper-based)

Creates 6 plots: 3 configs × (augmentation samples + original crop)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CROP_SIZE = 224
GRID_SIZE = 12
FULL_IMG_SIZE = 2720
STRIDE = (FULL_IMG_SIZE - CROP_SIZE) // (GRID_SIZE - 1) if GRID_SIZE > 1 else 0

positions = [(j * STRIDE, i * STRIDE) for i in range(GRID_SIZE) for j in range(GRID_SIZE)
             if j * STRIDE + CROP_SIZE <= FULL_IMG_SIZE and i * STRIDE + CROP_SIZE <= FULL_IMG_SIZE]

# Find a sample image
all_images = []
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_dir = os.path.join(SCRIPT_DIR, plate)
    if os.path.exists(plate_dir):
        for f in sorted(os.listdir(plate_dir)):
            if f.endswith('.tif'):
                all_images.append(os.path.join(plate_dir, f))

np.random.seed(42)
sample_img_path = all_images[100]
print(f"Image: {os.path.basename(sample_img_path)}")

full_img = Image.open(sample_img_path).convert('RGB')
crop_left, crop_top = positions[70]
crop = full_img.crop((crop_left, crop_top, crop_left + CROP_SIZE, crop_top + CROP_SIZE))
crop_array = np.array(crop)
print(f"Crop: {CROP_SIZE}x{CROP_SIZE} at ({crop_left}, {crop_top})")

# === AUGMENTATION DEFINITIONS ===

# plate_fold - Heavy/Legacy
plate_fold_augs = [
    ("Original", None),
    ("HorizontalFlip", A.HorizontalFlip(p=1.0)),
    ("VerticalFlip", A.VerticalFlip(p=1.0)),
    ("RandomRotate90", A.RandomRotate90(p=1.0)),
    ("Affine (±10%, ±15°)", A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                                    scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)}, rotate=(-15, 15), p=1.0)),
    ("ElasticTransform", A.ElasticTransform(alpha=50, sigma=5, p=1.0)),
    ("Perspective", A.Perspective(scale=(0.02, 0.05), p=1.0)),
    ("GridDistortion", A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0)),
    ("GaussNoise", A.GaussNoise(std_range=(0.05, 0.15), per_channel=False, p=1.0)),
    ("GaussianBlur", A.GaussianBlur(blur_limit=(3, 5), p=1.0)),
    ("MotionBlur", A.MotionBlur(blur_limit=3, p=1.0)),
    ("CoarseDropout", A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(16, 64), 
                                   hole_width_range=(16, 64), p=1.0)),
]

# plate_fold_no_aug - Paper-based minimal
plate_fold_no_aug_augs = [
    ("Original", None),
    ("HorizontalFlip", A.HorizontalFlip(p=1.0)),
    ("VerticalFlip", A.VerticalFlip(p=1.0)),
    ("RandomRotate90", A.RandomRotate90(p=1.0)),
    ("Affine (±5%, ±10°)", A.Affine(translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)}, rotate=(-10, 10), p=1.0)),
    ("BrightnessContrast", A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.5, p=1.0)),
]

# final_crispr_model - Paper-based (same as plate_fold_no_aug but structured differently)
final_crispr_augs = plate_fold_no_aug_augs  # Same augmentations

# === PLOTTING FUNCTIONS ===

def plot_augmentations(augs, title, filename):
    n_augs = len(augs)
    n_cols = 4
    n_rows = (n_augs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, (name, transform) in enumerate(augs):
        if transform is None:
            aug_img = crop_array
        else:
            np.random.seed(42 + idx)  # Different seed for variety
            augmented = transform(image=crop_array)
            aug_img = augmented['image']
        
        axes[idx].imshow(aug_img)
        axes[idx].set_title(name, fontsize=9)
        axes[idx].axis('off')
    
    for idx in range(n_augs, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"{title}\nCrop: {CROP_SIZE}x{CROP_SIZE}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# === MAIN ===

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# Plot 1: plate_fold (heavy)
plot_augmentations(plate_fold_augs, "plate_fold: Heavy Augmentations", 
                 os.path.join(SCRIPT_DIR, "visualizations/plate_fold_augmentations.png"))

# Plot 2: plate_fold_no_aug (paper-based)
plot_augmentations(plate_fold_no_aug_augs, "plate_fold_no_aug: Paper-Based Augmentations", 
                 os.path.join(SCRIPT_DIR, "visualizations/plate_fold_no_aug_augmentations.png"))

# Plot 3: final_crispr_model (paper-based)
plot_augmentations(final_crispr_augs, "final_crispr_model: Paper-Based Augmentations", 
                 os.path.join(SCRIPT_DIR, "visualizations/final_crispr_model_augmentations.png"))

# === COMPARISON TABLE ===

print("\n" + "="*70)
print("AUGMENTATION COMPARISON TABLE")
print("="*70)

configs = {
    "plate_fold": {
        "Affine": "±10%, ±15°, scale 0.9-1.1",
        "Geometric": "Elastic, Perspective, GridDistortion, OpticalDistortion",
        "Noise/Blur": "GaussNoise, GaussianBlur, MotionBlur",
        "Other": "ImageCompression, CoarseDropout",
        "Probs": "Flip: 0.5, Distort: 0.5, Noise: 0.5",
    },
    "plate_fold_no_aug": {
        "Affine": "±5%, ±10°",
        "Geometric": "None (just flip/rotate)",
        "Noise/Blur": "GaussNoise (tight)",
        "Other": "BrightnessContrast, PixelDropout",
        "Probs": "Geometric: 0.5, Pixel: 0.3",
    },
    "final_crispr_model": {
        "Affine": "±5%, ±10°",
        "Geometric": "None (just flip/rotate)",
        "Noise/Blur": "GaussNoise (tight)",
        "Other": "BrightnessContrast, PixelDropout",
        "Probs": "Geometric: 0.5, Pixel: 0.3",
    },
}

# Create comparison figure
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

for idx, (config_name, config_data) in enumerate(configs.items()):
    ax = axes[idx]
    ax.axis('off')
    ax.set_title(config_name, fontsize=14, fontweight='bold')
    
    text = f"{config_name}\n"
    text += "="*30 + "\n\n"
    for key, val in config_data.items():
        text += f"{key}:\n  {val}\n\n"
    
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle("AUGMENTATION CONFIGURATIONS COMPARISON", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "visualizations/augmentation_comparison.png"), dpi=150, bbox_inches='tight')
print("Saved: visualizations/augmentation_comparison.png")
plt.close()

# === ORIGINAL CROP ===

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.imshow(crop_array)
ax.set_title(f"Original Crop: {CROP_SIZE}x{CROP_SIZE}\nPosition: ({crop_left}, {crop_top})", fontsize=12)
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "visualizations/original_crop.png"), dpi=150, bbox_inches='tight')
print("Saved: visualizations/original_crop.png")
plt.close()

print("\n" + "="*70)
print("ALL VISUALIZATIONS COMPLETE")
print("="*70)
print("Generated files:")
print("  - visualizations/plate_fold_augmentations.png")
print("  - visualizations/plate_fold_no_aug_augmentations.png")
print("  - visualizations/final_crispr_model_augmentations.png")
print("  - visualizations/augmentation_comparison.png")
print("  - visualizations/original_crop.png")