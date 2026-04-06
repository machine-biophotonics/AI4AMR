#!/usr/bin/env python3
"""
Visualize augmentations for 2_effnet_model_trained_guide
Shows current vs proposed extensive augmentations
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Load a sample image
SAMPLE_IMAGES = [
    "/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/P1/WellA01_PointA01/WellA01_PointA01_0000_ChannelCam-DIA_DIC_Master_Screening_Seq0000_sharpest_image_1.tif",
    "/media/student/Data SSD 1-TB/2025_12_19 CRISPRi Reference Plate Imaging/P1/WellA01_PointA01/WellA01_PointA01_0000_ChannelCam-DIA DIC Master Screening_Seq0000_sharpest_image_1.tif"
]

def load_sample_image():
    """Load a sample image for visualization"""
    for path in SAMPLE_IMAGES:
        if os.path.exists(path):
            img = Image.open(path).convert('RGB')
            # Take center crop for visualization
            w, h = img.size
            crop_size = 224
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            img = img.crop((left, top, left + crop_size, top + crop_size))
            return np.array(img)
    # If no sample found, create dummy
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

# =============================================================================
# CURRENT AUGMENTATIONS (in train.py)
# =============================================================================
CURRENT_AUGMENTS = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(
        translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
        scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
        rotate=(-15, 15),
        p=0.5
    ),
    A.SomeOf([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        A.RandomToneCurve(scale=0.3, p=1.0),
    ], n=2, replace=False, p=0.5),
    A.SomeOf([
        A.GaussNoise(std_range=(0.1, 0.5), per_channel=False, p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    ], n=1, replace=False, p=0.5),
    A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(16, 64), hole_width_range=(16, 64), p=0.4),
], p=1.0)

# =============================================================================
# PROPOSED EXTENSIVE AUGMENTATIONS (grayscale-friendly, fixed)
# =============================================================================
EXTENSIVE_AUGMENTS = A.Compose([
    # Geometric transformations
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(
        translate_percent={'x': (-0.15, 0.15), 'y': (-0.15, 0.15)},
        scale={'x': (0.85, 1.15), 'y': (0.85, 1.15)},
        rotate=(-20, 20),
        p=0.5
    ),
    A.Perspective(scale=(0.05, 0.15), p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, border_mode=0, p=0.2),
    
    # Intensity adjustments (grayscale-friendly)
    A.SomeOf([
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
        A.RandomGamma(gamma_limit=(70, 130), p=1.0),
        A.RandomToneCurve(scale=0.4, p=1.0),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
    ], n=2, replace=False, p=0.6),
    
    # Noise and blur (grayscale-friendly)
    A.SomeOf([
        A.GaussNoise(std_range=(0.05, 0.3), per_channel=False, p=1.0),
        A.GaussianBlur(blur_limit=(3, 9), p=1.0),
        A.MotionBlur(blur_limit=7, p=1.0),
        A.MedianBlur(blur_limit=5, p=1.0),
    ], n=2, replace=False, p=0.5),
    
    # Sharpness
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
    
    # Dropout and cuts
    A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 64), hole_width_range=(16, 64), p=0.4),
    
    # Color jitter (subtle, works on grayscale too)
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0, p=0.3),
], p=1.0)


def visualize_augmentations(augments, name, num_samples=8):
    """Apply augmentation multiple times and show results"""
    img = load_sample_image()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(name, fontsize=14, fontweight='bold')
    
    # Original
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')
    
    # Augmented samples
    for i in range(num_samples - 1):
        augmented = augments(image=img)['image']
        row = (i + 1) // 4
        col = (i + 1) % 4
        axes[row, col].imshow(augmented)
        axes[row, col].set_title(f"Aug {i+1}")
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    print("Loading sample image...")
    img = load_sample_image()
    print(f"Image shape: {img.shape}")
    
    # Create figure with both
    fig = plt.figure(figsize=(20, 12))
    
    # Current augmentations
    print("Generating current augmentations...")
    for i in range(8):
        if i == 0:
            augmented = img.copy()
        else:
            augmented = CURRENT_AUGMENTS(image=img)['image']
        ax = fig.add_subplot(4, 8, i + 1)
        ax.imshow(augmented)
        ax.set_title("Original" if i == 0 else f"Aug {i}")
        ax.axis('off')
    
    # Extensive augmentations
    print("Generating extensive augmentations...")
    for i in range(8):
        if i == 0:
            augmented = img.copy()
        else:
            augmented = EXTENSIVE_AUGMENTS(image=img)['image']
        ax = fig.add_subplot(4, 8, i + 9)
        ax.imshow(augmented)
        ax.set_title("Original" if i == 0 else f"Aug {i}")
        ax.axis('off')
    
    # Add text labels
    fig.text(0.25, 0.92, "CURRENT AUGMENTATIONS", ha='center', fontsize=14, fontweight='bold')
    fig.text(0.75, 0.92, "EXTENSIVE AUGMENTATIONS", ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig("/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/2_effnet_model_trained_guide/augmentation_comparison.png", dpi=150)
    print("Saved: augmentation_comparison.png")
    
    # Print summary
    print("\n" + "="*60)
    print("CURRENT AUGMENTATIONS:")
    print("="*60)
    print("""
- HorizontalFlip (p=0.5)
- VerticalFlip (p=0.5)
- RandomRotate90 (p=0.5)
- Affine (translate/scale/rotate) (p=0.5)
- SomeOf (2 of 3):
  - RandomBrightnessContrast
  - RandomGamma  
  - RandomToneCurve
- SomeOf (1 of 2):
  - GaussNoise
  - GaussianBlur
- CoarseDropout (p=0.4)
""")
    
    print("="*60)
    print("EXTENSIVE AUGMENTATIONS (PROPOSED):")
    print("="*60)
    print("""
GEOMETRIC:
- HorizontalFlip (p=0.5)
- VerticalFlip (p=0.5)
- RandomRotate90 (p=0.5)
- Affine (p=0.5)
- Perspective (p=0.3)
- ElasticTransform (p=0.2)

INTENSITY:
- SomeOf (2 of 4):
  - RandomBrightnessContrast
  - RandomGamma
  - RandomToneCurve
  - CLAHE

NOISE/BLUR:
- SomeOf (2 of 4):
  - GaussNoise
  - GaussianBlur
  - MotionBlur
  - MedianBlur

COMPRESSION:
- ImageCompression (p=0.3)

SHARPENING:
- Sharpen (p=0.3)

DROPOUT:
- CoarseDropout (p=0.4)

COLOR:
- ColorJitter (p=0.3)
""")


if __name__ == "__main__":
    main()
