#!/usr/bin/env python3
"""
Visualize augmentations for sam_effnet, guide_effnet, and plate_fold (they use IDENTICAL augmentations).
Shows each augmentation applied to a sample crop from plate images.
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
BASE_DIR = os.path.dirname(SCRIPT_DIR)

CROP_SIZE = 224
GRID_SIZE = 12
FULL_IMG_SIZE = 2720
STRIDE = (FULL_IMG_SIZE - CROP_SIZE) // (GRID_SIZE - 1) if GRID_SIZE > 1 else 0

positions = [(j * STRIDE, i * STRIDE) for i in range(GRID_SIZE) for j in range(GRID_SIZE)
             if j * STRIDE + CROP_SIZE <= FULL_IMG_SIZE and i * STRIDE + CROP_SIZE <= FULL_IMG_SIZE]

# Find a sample image
all_images = []
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_dir = os.path.join(BASE_DIR, plate)
    if os.path.exists(plate_dir):
        for f in sorted(os.listdir(plate_dir)):
            if f.endswith('.tif'):
                all_images.append(os.path.join(plate_dir, f))

np.random.seed(42)
sample_img_path = all_images[100]
print(f"Image: {os.path.basename(sample_img_path)}")

full_img = Image.open(sample_img_path).convert('RGB')
crop_left, crop_top = positions[70]  # Pick a middle position
crop = full_img.crop((crop_left, crop_top, crop_left + CROP_SIZE, crop_top + CROP_SIZE))
crop_array = np.array(crop)
print(f"Crop: {CROP_SIZE}x{CROP_SIZE} at ({crop_left}, {crop_top})")

# SIMPLE AUGMENTATIONS (matching sam_effnet, guide_effnet, plate_fold)
augmentations = [
    ("1. Original (No Aug)", None),
    ("2. HorizontalFlip (p=0.5)", A.HorizontalFlip(p=1.0)),
    ("3. VerticalFlip (p=0.5)", A.VerticalFlip(p=1.0)),
    ("4. RandomRotate90 (p=0.5)", A.RandomRotate90(p=1.0)),
    ("5. Affine (p=0.5)", A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                                   scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)}, rotate=(-15, 15), p=1.0)),
    
    ("6. ElasticTransform (p=0.5)", A.ElasticTransform(alpha=50, sigma=5, p=1.0)),
    ("7. Perspective (p=0.5)", A.Perspective(scale=(0.02, 0.05), p=1.0)),
    ("8. GridDistortion (p=0.5)", A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0)),
    ("9. OpticalDistortion (p=0.5)", A.OpticalDistortion(distort_limit=0.05, p=1.0)),
    
    ("10. GaussNoise (p=0.5)", A.GaussNoise(std_range=(0.05, 0.15), per_channel=False, p=1.0)),
    ("11. GaussianBlur (p=0.5)", A.GaussianBlur(blur_limit=(3, 5), p=1.0)),
    ("12. MotionBlur (p=0.5)", A.MotionBlur(blur_limit=3, p=1.0)),
    
    ("13. ImageCompression (p=0.3)", A.ImageCompression(quality_range=(85, 100), p=1.0)),
    ("14. CoarseDropout (p=0.4)", A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(16, 64), 
                                                  hole_width_range=(16, 64), p=1.0)),
]

# Create figure
n_augs = len(augmentations)
n_cols = 4
n_rows = (n_augs + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
axes = axes.flatten()

for idx, (name, transform) in enumerate(augmentations):
    if transform is None:
        aug_img = crop_array
    else:
        augmented = transform(image=crop_array)
        aug_img = augmented['image']
    
    axes[idx].imshow(aug_img)
    axes[idx].set_title(name, fontsize=8)
    axes[idx].axis('off')

# Hide unused axes
for idx in range(n_augs, len(axes)):
    axes[idx].axis('off')

plt.suptitle(f"AUGMENTATIONS: sam_effnet, guide_effnet, plate_fold (IDENTICAL)\nCrop: {CROP_SIZE}x{CROP_SIZE}", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('augmentations.png', dpi=150, bbox_inches='tight')
print("\nSaved: augmentations.png")

# Also create a comparison table
print("\n" + "="*70)
print("AUGMENTATION COMPARISON: sam_effnet vs guide_effnet vs plate_fold")
print("="*70)
print(f"{'Augmentation':<35} {'sam_effnet':<12} {'guide_effnet':<12} {'plate_fold':<12}")
print("-"*70)

aug_list = [
    ("HorizontalFlip", "p=0.5"),
    ("VerticalFlip", "p=0.5"),
    ("RandomRotate90", "p=0.5"),
    ("Affine", "p=0.5"),
    ("ElasticTransform", "p=0.5"),
    ("Perspective", "p=0.5"),
    ("GridDistortion", "p=0.5"),
    ("OpticalDistortion", "p=0.5"),
    ("GaussNoise", "p=0.5"),
    ("GaussianBlur", "p=0.5"),
    ("MotionBlur", "p=0.5"),
    ("ImageCompression", "p=0.3"),
    ("CoarseDropout", "p=0.4"),
]

for aug, detail in aug_list:
    print(f"{aug:<35} Yes         Yes         Yes")

print("\nOPTIMIZER COMPARISON:")
print(f"{'sam_effnet':<35} SAM optimizer")
print(f"{'guide_effnet':<35} SAM optimizer")
print(f"{'plate_fold':<35} Basic AdamW")

print("\n⚠️  sam_effnet, guide_effnet, and plate_fold use IDENTICAL augmentations!")
print("    Only difference is optimizer: SAM vs basic AdamW")