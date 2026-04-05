#!/usr/bin/env python3
"""
Visualize augmentations - EXACTLY matching sam_effnet/train.py
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

CROP_SIZE = 544
GRID_SIZE = 5
FULL_IMG_SIZE = 2720
STRIDE = (FULL_IMG_SIZE - CROP_SIZE) // (GRID_SIZE - 1) if GRID_SIZE > 1 else 0

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
crop_left, crop_top = positions[12]
crop = full_img.crop((crop_left, crop_top, crop_left + CROP_SIZE, crop_top + CROP_SIZE))
crop_array = np.array(crop)
print(f"Crop: {CROP_SIZE}x{CROP_SIZE} at ({crop_left}, {crop_top})")

# AUGMENTATIONS EXACTLY MATCHING sam_effnet/train.py
augmentations = [
    # === SYMMETRY TRANSFORMS ===
    ("1. Original", None),
    ("2. HorizontalFlip (p=0.5)", A.HorizontalFlip(p=1.0)),
    ("3. VerticalFlip (p=0.5)", A.VerticalFlip(p=1.0)),
    ("4. RandomRotate90 (p=0.5)", A.RandomRotate90(p=1.0)),
    ("5. Affine (p=0.5)", A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                                    scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)}, rotate=(-15, 15), p=1.0)),
    
    # === RANDOM RESIZE CROP ===
    ("6. RandomResizedCrop (p=0.3)", A.RandomResizedCrop(size=(CROP_SIZE, CROP_SIZE), scale=(0.7, 1.0), ratio=(0.9, 1.1), p=1.0)),
    
    # === LIGHTING & CONTRAST ===
    ("7. CLAHE (p=0.3)", A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0)),
    ("8. RandomBrightnessContrast (p=0.3)", A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)),
    ("9. RandomGamma (p=0.2)", A.RandomGamma(gamma_limit=(80, 120), p=1.0)),
    ("10. Equalize (p=0.1)", A.Equalize(p=1.0)),
    
    # === SHADOW SIMULATION ===
    ("11. RandomShadow (p=0.2)", A.RandomShadow(shadow_roi=(0.3, 0.3, 0.7, 0.7), num_shadows_limit=(1, 3), shadow_dimension=5, shadow_intensity_range=(0.3, 0.5), p=1.0)),
    
    # === GEOMETRIC DEFORMATIONS ===
    ("12. ElasticTransform", A.ElasticTransform(alpha=50, sigma=5, p=1.0)),
    ("13. Perspective", A.Perspective(scale=(0.02, 0.05), p=1.0)),
    ("14. GridDistortion", A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0)),
    ("15. OpticalDistortion", A.OpticalDistortion(distort_limit=0.05, p=1.0)),
    
    # === NOISE & BLUR ===
    ("16. GaussNoise", A.GaussNoise(std_range=(0.05, 0.15), per_channel=False, p=1.0)),
    ("17. GaussianBlur", A.GaussianBlur(blur_limit=(3, 5), p=1.0)),
    ("18. MotionBlur", A.MotionBlur(blur_limit=3, p=1.0)),
    
    # === PIXEL DROPOUT ===
    ("19. PixelDropout (p=0.2)", A.PixelDropout(dropout_prob=0.05, drop_value=0, p=1.0)),
    
    # === QUALITY ARTIFACTS ===
    ("20. ImageCompression (p=0.3)", A.ImageCompression(quality_range=(85, 100), p=1.0)),
    ("21. CoarseDropout (p=0.4)", A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(16, 64), hole_width_range=(16, 64), p=1.0)),
    
    # === NEW ADDITIONAL AUGMENTATIONS (grayscale-compatible) ===
    ("22. SaltAndPepper", A.SaltAndPepper(p=1.0)),
    ("23. Erasing", A.Erasing(p=1.0)),
    ("24. ISONoise", A.ISONoise(p=1.0)),
]

# Plot all augmentations individually (5x6 grid = 30)
fig, axes = plt.subplots(5, 6, figsize=(30, 25))
axes = axes.flatten()

for idx, (name, aug) in enumerate(augmentations):
    if idx >= len(axes):
        break
    if aug is None:
        result = crop_array.copy()
    else:
        try:
            transformed = aug(image=crop_array)
            result = transformed['image']
        except Exception as e:
            print(f"Error in {name}: {e}")
            result = crop_array.copy()
    
    if result.max() <= 1.0:
        result = (result * 255).astype(np.uint8)
    
    axes[idx].imshow(result, cmap='gray' if len(result.shape) == 2 else None)
    axes[idx].set_title(name, fontsize=9)
    axes[idx].axis('off')

for idx in range(len(augmentations), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Individual Augmentations (Applied at p=1.0 for visualization)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'augmentations_individual.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved individual: {SCRIPT_DIR}/augmentations_individual.png")

# === COMBINED AUGMENTATIONS ===
print("\nGenerating combined augmentation examples...")

combined_transform = A.Compose([
    # === SYMMETRY TRANSFORMS ===
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
             scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)}, rotate=(-15, 15), p=0.5),
    
    # === RANDOM RESIZE CROP ===
    A.RandomResizedCrop(size=(CROP_SIZE, CROP_SIZE), scale=(0.7, 1.0), ratio=(0.9, 1.1), p=0.3),
    
    # === LIGHTING & CONTRAST ===
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.2),
    A.Equalize(p=0.1),
    
    # === SHADOW SIMULATION ===
    A.RandomShadow(shadow_roi=(0.3, 0.3, 0.7, 0.7), num_shadows_limit=(1, 3), shadow_dimension=5, shadow_intensity_range=(0.3, 0.5), p=0.2),
    
    # === GEOMETRIC DEFORMATIONS ===
    A.SomeOf([
        A.ElasticTransform(alpha=50, sigma=5, p=1.0),
        A.Perspective(scale=(0.02, 0.05), p=1.0),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),
        A.OpticalDistortion(distort_limit=0.05, p=1.0),
    ], n=1, replace=False, p=0.5),
    
    # === NOISE & BLUR ===
    A.SomeOf([
        A.GaussNoise(std_range=(0.05, 0.15), per_channel=False, p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MotionBlur(blur_limit=3, p=1.0),
    ], n=1, replace=False, p=0.5),
    
    # === PIXEL DROPOUT ===
    A.PixelDropout(dropout_prob=0.05, drop_value=0, p=0.2),
    
    # === QUALITY ARTIFACTS ===
    A.ImageCompression(quality_range=(85, 100), p=0.3),
    A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(16, 64), hole_width_range=(16, 64), p=0.4),
    
    # === NEW ADDITIONAL AUGMENTATIONS (grayscale-compatible) ===
    A.SaltAndPepper(p=0.2),
    A.Erasing(p=0.2),
    A.ISONoise(p=0.1),
])

# Generate 10 random augmented versions
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
axes = axes.flatten()

for i in range(10):
    np.random.seed(i)
    try:
        transformed = combined_transform(image=crop_array)
        result = transformed['image']
    except Exception as e:
        print(f"Error in combined {i}: {e}")
        result = crop_array.copy()
    
    if result.max() <= 1.0:
        result = (result * 255).astype(np.uint8)
    
    axes[i].imshow(result, cmap='gray' if len(result.shape) == 2 else None)
    axes[i].set_title(f'Random Sample {i+1}', fontsize=10)
    axes[i].axis('off')

plt.suptitle('Combined Augmentations (10 Random Samples)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'augmentations_combined.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved combined: {SCRIPT_DIR}/augmentations_combined.png")

print("\nDone! Check:")
print(f"  - {SCRIPT_DIR}/augmentations_individual.png")
print(f"  - {SCRIPT_DIR}/augmentations_combined.png")