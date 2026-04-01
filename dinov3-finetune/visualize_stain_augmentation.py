#!/usr/bin/env python3
"""
Visualize stain augmentation effects on a real crop from a plate image.

Stain augmentation (H&E) simulates variations in histopathology staining,
which can dramatically alter the color appearance of the image.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from stainaug import Augmentor

# Path to a sample image
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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def extract_center_crop(img):
    """Extract center 224x224 crop from 2720x2720 image."""
    h, w = img.shape[:2]
    assert h == IMAGE_SIZE and w == IMAGE_SIZE, f"Expected {IMAGE_SIZE}x{IMAGE_SIZE}, got {h}x{w}"
    crop = img[CENTER:CENTER+CROP_SIZE, CENTER:CENTER+CROP_SIZE, :]
    return crop

def apply_stain_augmentation(crop, n_variations=5):
    """Apply stain augmentation multiple times to show variations."""
    augmentor = Augmentor()
    variations = [crop]  # include original
    for i in range(n_variations):
        augmented = augmentor.augment_HE(crop)
        variations.append(augmented)
    return variations

def visualize_stain_augmentations(crop, variations):
    """Plot original and stain-augmented crops."""
    n = len(variations)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    for i, ax in enumerate(axes):
        ax.imshow(variations[i])
        if i == 0:
            ax.set_title('Original')
        else:
            ax.set_title(f'Stain aug {i}')
        ax.axis('off')
    plt.suptitle('Stain Augmentation (H&E) Variations', fontsize=14)
    plt.tight_layout()
    plt.savefig('stain_augmentation_variations.png', dpi=150)
    plt.close()
    print("Saved stain_augmentation_variations.png")

def visualize_stain_before_after(crop):
    """Show original and one augmented side by side."""
    augmentor = Augmentor()
    augmented = augmentor.augment_HE(crop)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(crop)
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(augmented)
    axes[1].set_title('After Stain Augmentation')
    axes[1].axis('off')
    plt.suptitle('Effect of Stain Augmentation on Color', fontsize=14)
    plt.tight_layout()
    plt.savefig('stain_augmentation_before_after.png', dpi=150)
    plt.close()
    print("Saved stain_augmentation_before_after.png")

def visualize_color_channels(crop, augmented):
    """Compare color histograms before and after stain augmentation."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0,0].imshow(crop)
    axes[0,0].set_title('Original Image')
    axes[0,0].axis('off')
    
    # Augmented image
    axes[0,1].imshow(augmented)
    axes[0,1].set_title('After Stain Augmentation')
    axes[0,1].axis('off')
    
    # Color histograms
    color_names = ['Red', 'Green', 'Blue']
    for i, (color, channel) in enumerate(zip(color_names, [0,1,2])):
        hist_orig, _ = np.histogram(crop[:,:,channel], bins=256, range=(0,256))
        hist_aug, _ = np.histogram(augmented[:,:,channel], bins=256, range=(0,256))
        axes[1,0].plot(hist_orig, alpha=0.5, label=f'Original {color}')
        axes[1,1].plot(hist_aug, alpha=0.5, label=f'Augmented {color}')
    
    axes[1,0].set_title('Original Color Histograms')
    axes[1,0].legend()
    axes[1,0].set_xlim(0, 256)
    axes[1,1].set_title('Augmented Color Histograms')
    axes[1,1].legend()
    axes[1,1].set_xlim(0, 256)
    
    plt.suptitle('Color Distribution Changes due to Stain Augmentation', fontsize=14)
    plt.tight_layout()
    plt.savefig('stain_augmentation_histograms.png', dpi=150)
    plt.close()
    print("Saved stain_augmentation_histograms.png")

if __name__ == '__main__':
    print("=== Stain Augmentation Visualization ===")
    print("Loading image...")
    img = load_image(SAMPLE_IMAGE_PATH)
    crop = extract_center_crop(img)
    print(f"Loaded crop shape: {crop.shape}")
    
    print("Generating stain augmentation variations...")
    variations = apply_stain_augmentation(crop, n_variations=4)
    
    print("Plotting variations...")
    visualize_stain_augmentations(crop, variations)
    
    print("Creating before/after comparison...")
    visualize_stain_before_after(crop)
    
    print("Creating color histogram comparison...")
    visualize_color_channels(crop, variations[1])  # compare with first augmentation
    
    print("\nStain augmentation changes the color appearance significantly,")
    print("simulating variations in H&E staining concentration and balance.")
    print("This helps the model generalize across stain variability in histopathology.")