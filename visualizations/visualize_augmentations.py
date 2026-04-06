#!/usr/bin/env python3
"""
Visualize augmentations for sam_effnet and guide_effnet (they use IDENTICAL augmentations).
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

# AUGMENTATIONS EXACTLY MATCHING sam_effnet/train.py and guide_effnet/train.py
augmentations = [
    # === SYMMETRY TRANSFORMS ===
    ("1. Original (No Aug)", None),
    ("2. HorizontalFlip (p=0.5)", A.HorizontalFlip(p=1.0)),
    ("3. VerticalFlip (p=0.5)", A.VerticalFlip(p=1.0)),
    ("4. RandomRotate90 (p=0.5)", A.RandomRotate90(p=1.0)),
    ("5. Rotate 360° (p=0.5)", A.Rotate(limit=360, p=1.0)),
    ("6. Affine (p=0.5)", A.Affine(translate_percent={'x': (-0.15, 0.15), 'y': (-0.15, 0.15)},
                                   scale={'x': (0.85, 1.15), 'y': (0.85, 1.15)}, rotate=(-20, 20), p=1.0)),
    
    # === LIGHTING & CONTRAST ===
    ("7. CLAHE (p=0.3)", A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0)),
    ("8. RandomBrightnessContrast (p=0.4)", A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0)),
    ("9. RandomGamma (p=0.25)", A.RandomGamma(gamma_limit=(70, 130), p=1.0)),
    ("10. Equalize (p=0.1)", A.Equalize(p=1.0)),
    
    # === SHADOW SIMULATION ===
    ("11. RandomShadow (p=0.2)", A.RandomShadow(shadow_roi=(0.3, 0.3, 0.7, 0.7), num_shadows_limit=(1, 3), 
                                                 shadow_dimension=5, shadow_intensity_range=(0.3, 0.5), p=1.0)),
    
    # === GEOMETRIC DEFORMATIONS (SomeOf, p=0.4) ===
    ("12. ElasticTransform (p=0.4)", A.ElasticTransform(alpha=50, sigma=5, p=1.0)),
    ("13. Perspective (p=0.4)", A.Perspective(scale=(0.02, 0.05), p=1.0)),
    ("14. GridDistortion (p=0.4)", A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0)),
    ("15. OpticalDistortion (p=0.4)", A.OpticalDistortion(distort_limit=0.05, p=1.0)),
    
    # === NOISE & BLUR (SomeOf, p=0.4) ===
    ("16. GaussNoise (p=0.4)", A.GaussNoise(std_range=(0.1, 0.2), per_channel=False, p=1.0)),
    ("17. GaussianBlur (p=0.4)", A.GaussianBlur(blur_limit=(3, 7), p=1.0)),
    ("18. MotionBlur (p=0.4)", A.MotionBlur(blur_limit=5, p=1.0)),
    
    # === PIXEL DROPOUT ===
    ("19. PixelDropout (p=0.15)", A.PixelDropout(dropout_prob=0.05, drop_value=0, p=1.0)),
    
    # === NOISE ARTIFACTS ===
    ("20. SaltAndPepper (p=0.2)", A.SaltAndPepper(p=1.0)),
    ("21. ISONoise (p=0.15)", A.ISONoise(p=1.0)),
    
    # === ERASING ===
    ("22. Erasing (p=0.2)", A.Erasing(p=1.0)),
    
    # === QUALITY ARTIFACTS ===
    ("23. ImageCompression (p=0.3)", A.ImageCompression(quality_range=(80, 100), p=1.0)),
    ("24. CoarseDropout (p=0.3)", A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(16, 64), 
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

plt.suptitle(f"AUGMENTATIONS: sam_effnet AND guide_effnet (IDENTICAL)\nCrop: {CROP_SIZE}x{CROP_SIZE}", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('augmentations.png', dpi=150, bbox_inches='tight')
print("\nSaved: augmentations.png")

# Also create a comparison table
print("\n" + "="*70)
print("AUGMENTATION COMPARISON: sam_effnet vs guide_effnet vs dinov3-finetune")
print("="*70)
print(f"{'Augmentation':<35} {'sam_effnet':<12} {'guide_effnet':<12} {'dinov3-finetune':<15}")
print("-"*70)

aug_list = [
    ("HorizontalFlip", "p=0.5"),
    ("VerticalFlip", "p=0.5"),
    ("RandomRotate90", "p=0.5"),
    ("Rotate 360°", "p=0.5"),
    ("Affine", "p=0.5"),
    ("CLAHE", "p=0.3"),
    ("RandomShadow", "p=0.2"),
    ("ElasticTransform", "p=0.4"),
    ("Perspective", "p=0.4"),
    ("GridDistortion", "p=0.4"),
    ("OpticalDistortion", "p=0.4"),
    ("GaussNoise", "p=0.4"),
    ("GaussianBlur", "p=0.4"),
    ("MotionBlur", "p=0.4"),
    ("PixelDropout", "p=0.15"),
    ("SaltAndPepper", "p=0.2"),
    ("ISONoise", "p=0.15"),
    ("Erasing", "p=0.2"),
    ("ImageCompression", "p=0.3"),
    ("CoarseDropout", "p=0.3"),
    ("Focal Loss", "Yes"),
    ("Domain Weights", "Yes"),
    ("Center Loss", "Optional"),
]

for aug, detail in aug_list:
    print(f"{aug:<35} ✓ (p={detail.split('=')[1] if '=' in detail else 'default':<8} ✓ (same) ✓ (same)")

print("\n⚠️  sam_effnet and guide_effnet use IDENTICAL augmentations!")
print("    (They share the same train.py file)")
print("⚠️  dinov3-finetune also uses SAME augmentations")
print("    (plate_dataset.py has identical augmentation code)")