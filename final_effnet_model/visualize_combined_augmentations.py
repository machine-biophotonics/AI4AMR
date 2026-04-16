#!/usr/bin/env python3
"""
Visualize combined augmentations + MixUp - as used in training.
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

# Grid parameters
CROP_SIZE = 224
GRID_SIZE = 12
FULL_IMG_SIZE = 2720
STRIDE = (FULL_IMG_SIZE - CROP_SIZE) // (GRID_SIZE - 1)

positions = [(j * STRIDE, i * STRIDE) for i in range(GRID_SIZE) for j in range(GRID_SIZE)
             if j * STRIDE + CROP_SIZE <= FULL_IMG_SIZE and i * STRIDE + CROP_SIZE <= FULL_IMG_SIZE]

# Find images
all_images = []
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_dir = os.path.join(BASE_DIR, plate)
    if os.path.exists(plate_dir):
        for f in sorted(os.listdir(plate_dir)):
            if f.endswith('.tif'):
                all_images.append(os.path.join(plate_dir, f))

np.random.seed(42)
# Get multiple images for MixUp visualization
sample_paths = [all_images[np.random.randint(100, len(all_images)-1)] for _ in range(5)]

# Load and crop images
crops = []
for sample_img_path in sample_paths[:3]:
    full_img = Image.open(sample_img_path).convert('RGB')
    crop_left, crop_top = positions[117]
    crop = full_img.crop((crop_left, crop_top, crop_left + CROP_SIZE, crop_top + CROP_SIZE))
    crops.append(np.array(crop))

crop_array = crops[0]
print(f"Base crop: {CROP_SIZE}x{CROP_SIZE}")

# Combined augmentation pipeline (MATCHING train.py)
combined_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)},
             scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)}, rotate=(-30, 30), p=0.7),
    A.SomeOf([
        A.ElasticTransform(alpha=80, sigma=8, p=1.0),
        A.Perspective(scale=(0.05, 0.1), p=1.0),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
        A.OpticalDistortion(distort_limit=0.1, p=1.0),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=1.0),
    ], n=3, replace=False, p=0.7),
    A.SomeOf([
        A.GaussNoise(std_range=(0.1, 0.25), per_channel=False, p=1.0),
        A.GaussianBlur(blur_limit=(3, 9), p=1.0),
        A.MotionBlur(blur_limit=7, p=1.0),
        A.MedianBlur(blur_limit=5, p=1.0),
    ], n=2, replace=False, p=0.7),
    A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 4), shadow_intensity_range=(0.3, 0.9), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.RandomGamma(gamma_limit=(50, 150), p=0.4),
    A.SomeOf([
        A.Defocus(radius=(3, 6), p=1.0),
        A.PixelDropout(dropout_prob=0.03, p=1.0),
        A.Downscale(scale_range=(0.4, 0.8), p=1.0),
    ], n=2, replace=False, p=0.5),
    A.ImageCompression(quality_range=(70, 100), p=0.5),
    A.CoarseDropout(num_holes_range=(3, 6), hole_height_range=(16, 64), hole_width_range=(16, 64), p=0.6),
    A.Sharpen(alpha=(0.3, 0.7), lightness=(0.5, 1.0), p=0.5),
    A.OneOf([
        A.CLAHE(clip_limit=4, p=1.0),
        A.UnsharpMask(blur_limit=(3, 7), sigma_limit=(0.1, 1.0), p=1.0),
        A.Equalize(mode='cv', by_channels=False, p=1.0),
    ], p=0.4),
])

# Generate multiple augmented versions
augmented_crops = []
for i in range(12):
    np.random.seed(i * 42)  # Different seed for different augmentations
    transformed = combined_transform(image=crop_array)
    augmented_crops.append(transformed['image'])

# MixUp function (matching train.py)
def mixup_images(img1, img2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    mixed = lam * img1 + (1 - lam) * img2
    return mixed.astype(np.uint8), lam

# Create visualization
fig, axes = plt.subplots(4, 4, figsize=(16, 16))

# Row 1: Original + 3 different combined augmentations
axes[0, 0].imshow(crop_array, cmap='gray')
axes[0, 0].set_title("Original", fontsize=12)
axes[0, 0].axis('off')

for i in range(3):
    axes[0, i+1].imshow(augmented_crops[i], cmap='gray')
    axes[0, i+1].set_title(f"Combined Aug {i+1}", fontsize=12)
    axes[0, i+1].axis('off')

# Row 2: More augmented versions
for i in range(4):
    axes[1, i].imshow(augmented_crops[i+4], cmap='gray')
    axes[1, i].set_title(f"Combined Aug {i+5}", fontsize=12)
    axes[1, i].axis('off')

# Row 3: MixUp demonstrations
mixup_examples = []
for i in range(4):
    img1 = augmented_crops[i * 2]
    img2 = augmented_crops[i * 2 + 1]
    mixed, lam = mixup_images(img1, img2, alpha=0.4)
    mixup_examples.append((mixed, lam))

for i, (mixed, lam) in enumerate(mixup_examples):
    axes[2, i].imshow(mixed, cmap='gray')
    axes[2, i].set_title(f"MixUp λ={lam:.2f}", fontsize=12)
    axes[2, i].axis('off')

# Row 4: Show MixUp component images side by side with result
for i in range(4):
    img1 = augmented_crops[i * 2]
    img2 = augmented_crops[i * 2 + 1]
    mixed, lam = mixup_examples[i]
    
    # Create 3-panel: img1, img2, mixed
    if i < 2:
        axes[3, i*2].imshow(img1, cmap='gray')
        axes[3, i*2].set_title(f"Image A", fontsize=10)
        axes[3, i*2].axis('off')
        axes[3, i*2+1].imshow(img2, cmap='gray')
        axes[3, i*2+1].set_title(f"Image B", fontsize=10)
        axes[3, i*2+1].axis('off')

# Remove empty subplots
for i in range(2, 4):
    axes[3, i].axis('off')

plt.suptitle("Combined Augmentations + MixUp Visualization\n(Same as training pipeline)", 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'combined_augmentations_mixup.png'), dpi=150, bbox_inches='tight')
print(f"Saved to: {SCRIPT_DIR}/combined_augmentations_mixup.png")

# Also create a simpler version showing just original vs augmented vs mixup
fig2, axes2 = plt.subplots(1, 5, figsize=(20, 4))

axes2[0].imshow(crop_array, cmap='gray')
axes2[0].set_title("Original", fontsize=14)
axes2[0].axis('off')

np.random.seed(100)
transformed = combined_transform(image=crop_array)
axes2[1].imshow(transformed['image'], cmap='gray')
axes2[1].set_title("Combined Aug", fontsize=14)
axes2[1].axis('off')

np.random.seed(200)
transformed = combined_transform(image=crop_array)
axes2[2].imshow(transformed['image'], cmap='gray')
axes2[2].set_title("Combined Aug", fontsize=14)
axes2[2].axis('off')

img1 = augmented_crops[0]
img2 = augmented_crops[1]
mixed, lam = mixup_images(img1, img2, alpha=0.4)
axes2[3].imshow(mixed, cmap='gray')
axes2[3].set_title(f"MixUp (λ={lam:.2f})", fontsize=14)
axes2[3].axis('off')

img1 = augmented_crops[2]
img2 = augmented_crops[3]
mixed, lam = mixup_images(img1, img2, alpha=0.6)
axes2[4].imshow(mixed, cmap='gray')
axes2[4].set_title(f"MixUp (λ={lam:.2f})", fontsize=14)
axes2[4].axis('off')

plt.suptitle("Original → Augmented → MixUp Flow", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'augmentations_flow.png'), dpi=150, bbox_inches='tight')
print(f"Saved to: {SCRIPT_DIR}/augmentations_flow.png")

print("\n✅ Visualization complete!")
print("Files saved:")
print("  1. combined_augmentations_mixup.png")
print("  2. augmentations_flow.png")