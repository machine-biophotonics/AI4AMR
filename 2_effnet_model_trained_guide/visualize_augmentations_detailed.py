#!/usr/bin/env python3
"""
Visualize individual augmentations for 2_effnet_model_trained_guide
Shows each augmentation separately with description
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import albumentations as A
from PIL import Image
import numpy as np
import os
import glob

# Find a sample image
def find_sample_image():
    # Try different path formats
    paths_to_try = [
        "/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging",
        "/media/student/Data SSD 1-TB/2025_12_19 CRISPRi Reference Plate Imaging",
    ]
    for base_dir in paths_to_try:
        for plate in ['P1', 'P2', 'P3', 'P4']:
            pattern = os.path.join(base_dir, plate, "*.tif")
            files = glob.glob(pattern)
            if files:
                print(f"Found: {files[0]}")
                return files[0]
    return None

# Load image
img_path = find_sample_image()
print(f"Loading: {img_path}")
img = Image.open(img_path).convert('RGB')
w, h = img.size
crop_size = 224
left = (w - crop_size) // 2
top = (h - crop_size) // 2
img = img.crop((left, top, left + crop_size, top + crop_size))
img_np = np.array(img)

print(f"Image shape: {img_np.shape}")

# Define individual augmentations to test
augmentations = {
    "1. Original": None,
    "2. HorizontalFlip": A.HorizontalFlip(p=1.0),
    "3. VerticalFlip": A.VerticalFlip(p=1.0),
    "4. RandomRotate90": A.RandomRotate90(p=1.0),
    "5. Affine (translate+scale+rotate)": A.Affine(
        translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
        scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
        rotate=(-15, 15),
        p=1.0
    ),
    "6. Perspective": A.Perspective(scale=(0.1, 0.2), p=1.0),
    "7. ElasticTransform": A.ElasticTransform(alpha=1, sigma=50, border_mode=0, p=1.0),
    "8. BrightnessContrast": A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
    "9. RandomGamma": A.RandomGamma(gamma_limit=(70, 130), p=1.0),
    "10. RandomToneCurve": A.RandomToneCurve(scale=0.4, p=1.0),
    "11. CLAHE": A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
    "12. GaussNoise": A.GaussNoise(std_range=(0.1, 0.3), per_channel=False, p=1.0),
    "13. GaussianBlur": A.GaussianBlur(blur_limit=(5, 9), p=1.0),
    "14. MotionBlur": A.MotionBlur(blur_limit=7, p=1.0),
    "15. MedianBlur": A.MedianBlur(blur_limit=5, p=1.0),
    "16. Sharpen": A.Sharpen(alpha=(0.3, 0.5), lightness=(0.5, 1.0), p=1.0),
    "17. CoarseDropout": A.CoarseDropout(num_holes_range=(2, 4), hole_height_range=(24, 48), hole_width_range=(24, 48), p=1.0),
    "18. ColorJitter": A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0, p=1.0),
}

# Create figure
n_augs = len(augmentations)
n_cols = 4
n_rows = (n_augs + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
axes = axes.flatten()

for idx, (name, aug) in enumerate(augmentations.items()):
    if aug is None:
        aug_img = img_np.copy()
    else:
        aug_img = aug(image=img_np)['image']
    
    axes[idx].imshow(aug_img)
    axes[idx].set_title(name, fontsize=10, fontweight='bold')
    axes[idx].axis('off')

# Hide empty subplots
for idx in range(len(augmentations), len(axes)):
    axes[idx].axis('off')

plt.suptitle("Individual Augmentation Effects", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/2_effnet_model_trained_guide/augmentations_detailed.png", dpi=150, bbox_inches='tight')
print("Saved: augmentations_detailed.png")

# =============================================================================
# Create comparison: Current vs Extensive
# =============================================================================
CURRENT_AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}, scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)}, rotate=(-15, 15), p=0.5),
    A.SomeOf([A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0), A.RandomGamma(gamma_limit=(80, 120), p=1.0), A.RandomToneCurve(scale=0.3, p=1.0)], n=2, p=0.5),
    A.SomeOf([A.GaussNoise(std_range=(0.1, 0.5), per_channel=False, p=1.0), A.GaussianBlur(blur_limit=(3, 7), p=1.0)], n=1, p=0.5),
    A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(16, 64), hole_width_range=(16, 64), p=0.4),
])

EXTENSIVE_AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(translate_percent={'x': (-0.15, 0.15), 'y': (-0.15, 0.15)}, scale={'x': (0.85, 1.15), 'y': (0.85, 1.15)}, rotate=(-20, 20), p=0.5),
    A.Perspective(scale=(0.05, 0.15), p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, border_mode=0, p=0.2),
    A.SomeOf([A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0), A.RandomGamma(gamma_limit=(70, 130), p=1.0), A.RandomToneCurve(scale=0.4, p=1.0), A.CLAHE(clip_limit=4.0, p=1.0)], n=2, p=0.6),
    A.SomeOf([A.GaussNoise(std_range=(0.05, 0.3), per_channel=False, p=1.0), A.GaussianBlur(blur_limit=(3, 9), p=1.0), A.MotionBlur(blur_limit=7, p=1.0), A.MedianBlur(blur_limit=5, p=1.0)], n=2, p=0.5),
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
    A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 64), hole_width_range=(16, 64), p=0.4),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0, p=0.3),
])

# Generate samples for both
fig, axes = plt.subplots(2, 8, figsize=(20, 6))

# Current augmentations
for i in range(8):
    if i == 0:
        aug_img = img_np.copy()
    else:
        aug_img = CURRENT_AUG(image=img_np)['image']
    axes[0, i].imshow(aug_img)
    axes[0, i].set_title("Original" if i == 0 else f"Sample {i}", fontsize=10)
    axes[0, i].axis('off')

# Extensive augmentations
for i in range(8):
    if i == 0:
        aug_img = img_np.copy()
    else:
        aug_img = EXTENSIVE_AUG(image=img_np)['image']
    axes[1, i].imshow(aug_img)
    axes[1, i].set_title("Original" if i == 0 else f"Sample {i}", fontsize=10)
    axes[1, i].axis('off')

axes[0, 0].set_ylabel("CURRENT", fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel("EXTENSIVE", fontsize=12, fontweight='bold')

plt.suptitle("Current vs Extensive Augmentations Comparison", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/2_effnet_model_trained_guide/augmentations_comparison.png", dpi=150, bbox_inches='tight')
print("Saved: augmentations_comparison.png")

print("\nDone! Check the PNG files in 2_effnet_model_trained_guide/")
