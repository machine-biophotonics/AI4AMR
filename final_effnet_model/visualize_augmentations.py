#!/usr/bin/env python3
"""
Visualize augmentations - EXACTLY matching train.py
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
STRIDE = (FULL_IMG_SIZE - CROP_SIZE) // (GRID_SIZE - 1)

positions = [(j * STRIDE, i * STRIDE) for i in range(GRID_SIZE) for j in range(GRID_SIZE)
             if j * STRIDE + CROP_SIZE <= FULL_IMG_SIZE and i * STRIDE + CROP_SIZE <= FULL_IMG_SIZE]

all_images = []
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_dir = os.path.join(BASE_DIR, plate)
    if os.path.exists(plate_dir):
        for f in sorted(os.listdir(plate_dir)):
            if f.endswith('.tif'):
                all_images.append(os.path.join(plate_dir, f))

np.random.seed(42)
sample_img_path = all_images[np.random.randint(100, len(all_images)-1)]
print(f"Image: {os.path.basename(sample_img_path)}")

full_img = Image.open(sample_img_path).convert('RGB')
crop_left, crop_top = positions[117]
crop = full_img.crop((crop_left, crop_top, crop_left + CROP_SIZE, crop_top + CROP_SIZE))
crop_array = np.array(crop)
print(f"Crop: {CROP_SIZE}x{CROP_SIZE} at ({crop_left}, {crop_top})")

# AUGMENTATIONS EXACTLY MATCHING train.py (lines 180-203)
augmentations = [
    ("1. Original", None),
    # Basic transforms (always applied with p=0.5)
    ("2. HorizontalFlip (p=0.5)", A.HorizontalFlip(p=1.0)),
    ("3. VerticalFlip (p=0.5)", A.VerticalFlip(p=1.0)),
    ("4. RandomRotate90 (p=0.5)", A.RandomRotate90(p=1.0)),
    # Affine (p=0.5)
    ("5. Affine (p=0.5)", A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                                    scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)}, rotate=(-15, 15), p=1.0)),
    # SomeOf - applies 1 of these with p=0.5
    ("6. ElasticTransform (SomeOf, p=0.5)", A.ElasticTransform(alpha=50, sigma=5, p=1.0)),
    ("7. Perspective (SomeOf, p=0.5)", A.Perspective(scale=(0.02, 0.05), p=1.0)),
    ("8. GridDistortion (SomeOf, p=0.5)", A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0)),
    ("9. OpticalDistortion (SomeOf, p=0.5)", A.OpticalDistortion(distort_limit=0.05, p=1.0)),
    # Noise and blur (SomeOf with p=0.5)
    ("10. GaussNoise (SomeOf, p=0.5)", A.GaussNoise(std_range=(0.05, 0.15), per_channel=False, p=1.0)),
    ("11. GaussianBlur (SomeOf, p=0.5)", A.GaussianBlur(blur_limit=(3, 5), p=1.0)),
    ("12. MotionBlur (SomeOf, p=0.5)", A.MotionBlur(blur_limit=3, p=1.0)),
    # Image quality artifacts (p=0.3)
    ("13. ImageCompression (p=0.3)", A.ImageCompression(quality_range=(85, 100), p=1.0)),
    # Cutout (p=0.4)
    ("14. CoarseDropout (p=0.4)", A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(16, 64), hole_width_range=(16, 64), p=1.0)),
]

# Plot all augmentations (4x4 grid = 16)
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
axes = axes.flatten()

for idx, (name, aug) in enumerate(augmentations):
    if idx >= len(axes):
        break
    if aug is None:
        result = crop_array
    else:
        try:
            transformed = aug(image=crop_array)
            result = transformed['image']
        except Exception as e:
            print(f"Error in {name}: {e}")
            result = crop_array
    
    if result.max() <= 1.0:
        result = (result * 255).astype(np.uint8)
    
    axes[idx].imshow(result)
    axes[idx].set_title(name, fontsize=10)
    axes[idx].axis('off')

for idx in range(len(augmentations), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'augmentations_visualization.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved to: {SCRIPT_DIR}/augmentations_visualization.png")
