#!/usr/bin/env python3
"""
Visualization: How crops are extracted from plate images for EfficientNet training

This script shows:
1. Original plate image
2. 12×12 grid overlay (144 crops)
3. How each crop is 224×224 pixels
4. The augmentation pipeline
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import json
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging'
EFFNET_DIR = os.path.join(BASE_DIR, 'effnet_model')
OUTPUT_DIR = os.path.join(EFFNET_DIR, 'crop_extraction_viz')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("CROP EXTRACTION VISUALIZATION FOR EFFICIENTNET TRAINING")
print("="*70)

# =============================================================================
# STEP 1: Show crop extraction grid
# =============================================================================
print("\n1. Creating crop extraction grid visualization...")

# Load a sample image from training
train_dir = '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/P1'
if os.path.exists(train_dir):
    tif_files = [f for f in os.listdir(train_dir) if f.endswith('.tif')]
    if tif_files:
        sample_path = os.path.join(train_dir, tif_files[0])
        sample_img = Image.open(sample_path).convert('RGB')
    else:
        # Create dummy image if no files found
        sample_img = Image.new('RGB', (2688, 2688), color='gray')
else:
    sample_img = Image.new('RGB', (2688, 2688), color='gray')

# Crop extraction parameters (from train.py)
PATCH_SIZE = 224
GRID_SIZE = 12  # 12×12 = 144 crops
EDGE_MARGIN = 0  # MixedCropDataset uses entire image

w, h = sample_img.size
print(f"Image size: {w} × {h} pixels")
print(f"Patch size: {PATCH_SIZE} × {PATCH_SIZE} pixels")
print(f"Grid: {GRID_SIZE} × {GRID_SIZE} = {GRID_SIZE * GRID_SIZE} crops per image")

# Calculate grid positions
total_w = w - PATCH_SIZE
total_h = h - PATCH_SIZE
step_w = total_w / (GRID_SIZE - 1) if GRID_SIZE > 1 else 0
step_h = total_h / (GRID_SIZE - 1) if GRID_SIZE > 1 else 0

print(f"Step size: {step_w:.1f} × {step_h:.1f} pixels")

# Create visualization showing the grid
fig, axes = plt.subplots(2, 2, figsize=(16, 16))

# Original image
axes[0, 0].imshow(sample_img)
axes[0, 0].set_title('Original Plate Image', fontsize=14)
axes[0, 0].axis('off')

# Grid overlay (all 144 crops)
ax = axes[0, 1]
ax.imshow(sample_img)
ax.set_title(f'12×12 Grid = 144 Crops (each {PATCH_SIZE}×{PATCH_SIZE}px)', fontsize=14)
ax.axis('off')

# Draw all grid positions
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        left = int(j * step_w)
        top = int(i * step_h)
        
        # Only draw every 3rd line to avoid clutter
        if i % 3 == 0 and j % 3 == 0:
            rect = patches.Rectangle((left, top), PATCH_SIZE, PATCH_SIZE,
                                     linewidth=0.5, edgecolor='cyan', facecolor='none')
            ax.add_patch(rect)

# Highlight a few sample crops
sample_positions = [(3, 3), (6, 6), (9, 9)]
colors = ['red', 'green', 'blue']
for idx, (i, j) in enumerate(sample_positions):
    left = int(j * step_w)
    top = int(i * step_h)
    rect = patches.Rectangle((left, top), PATCH_SIZE, PATCH_SIZE,
                             linewidth=2, edgecolor=colors[idx], facecolor='none')
    ax.add_patch(rect)
    ax.text(left + 5, top + 15, f'Crop {i*GRID_SIZE+j}', color=colors[idx], fontsize=8, fontweight='bold')

# Zoom into one crop
ax = axes[1, 0]
i, j = 6, 6
left = int(j * step_w)
top = int(i * step_h)
crop = sample_img.crop((left, top, left + PATCH_SIZE, top + PATCH_SIZE))
ax.imshow(crop)
ax.set_title(f'Example Crop: position ({i},{j}) → 224×224 pixels', fontsize=14)
ax.axis('off')

# Show grid spacing
ax = axes[1, 1]
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect('equal')
ax.set_title('Grid Layout Diagram', fontsize=14)

# Draw grid diagram
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        x = j * (100 / GRID_SIZE)
        y = i * (100 / GRID_SIZE)
        size = 100 / GRID_SIZE
        
        # Color based on position
        if i < 4:
            color = 'lightblue'  # Top
        elif i < 8:
            color = 'lightgreen'  # Middle
        else:
            color = 'lightcoral'  # Bottom
            
        rect = patches.Rectangle((x, y), size, size,
                                 linewidth=0.5, edgecolor='black', facecolor=color, alpha=0.5)
        ax.add_patch(rect)
        
        # Label corner crops
        if (i, j) in [(0, 0), (0, 11), (11, 0), (11, 11), (6, 6)]:
            ax.text(x + size/2, y + size/2, f'{i*GRID_SIZE+j}', 
                   ha='center', va='center', fontsize=6, fontweight='bold')

ax.text(50, -5, f'← {GRID_SIZE} columns × {GRID_SIZE} rows = {GRID_SIZE*GRID_SIZE} crops →', 
        ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_crop_extraction_grid.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_crop_extraction_grid.png")

# =============================================================================
# STEP 2: Show crop extraction process step by step
# =============================================================================
print("\n2. Creating step-by-step extraction process...")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Step 1: Load image
axes[0].imshow(sample_img)
axes[0].set_title('Step 1: Load Image\n(2688×2688 pixels)', fontsize=12)
axes[0].axis('off')

# Step 2: Define grid
axes[1].imshow(sample_img)
for i in range(0, GRID_SIZE, 2):
    for j in range(0, GRID_SIZE, 2):
        left = int(j * step_w)
        top = int(i * step_h)
        rect = patches.Rectangle((left, top), PATCH_SIZE, PATCH_SIZE,
                                 linewidth=0.5, edgecolor='cyan', facecolor='cyan', alpha=0.2)
        axes[1].add_patch(rect)
axes[1].set_title('Step 2: Define Grid\n(12×12 positions)', fontsize=12)
axes[1].axis('off')

# Step 3: Extract one crop
axes[2].imshow(sample_img)
i, j = 6, 6
left = int(j * step_w)
top = int(i * step_h)
rect = patches.Rectangle((left, top), PATCH_SIZE, PATCH_SIZE,
                         linewidth=3, edgecolor='red', facecolor='red', alpha=0.3)
axes[2].add_patch(rect)
axes[2].set_title('Step 3: Extract Crop\n(position 6,6)', fontsize=12)
axes[2].axis('off')

# Step 4: 224×224 patch
crop = sample_img.crop((left, top, left + PATCH_SIZE, top + PATCH_SIZE))
axes[3].imshow(crop)
axes[3].set_title(f'Step 4: 224×224 Patch\n→ Feed to EfficientNet', fontsize=12)
axes[3].axis('off')

plt.suptitle('Crop Extraction Pipeline', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_extraction_process.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_extraction_process.png")

# =============================================================================
# STEP 3: Show augmentation pipeline
# =============================================================================
print("\n3. Creating augmentation pipeline visualization...")

# Get a sample crop
i, j = 6, 6
left = int(j * step_w)
top = int(i * step_h)
base_crop = sample_img.crop((left, top, left + PATCH_SIZE, top + PATCH_SIZE))

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Original
axes[0, 0].imshow(base_crop)
axes[0, 0].set_title('Original Crop', fontsize=11)
axes[0, 0].axis('off')

# RandomHorizontalFlip
flipped_h = base_crop.transpose(Image.FLIP_LEFT_RIGHT)
axes[0, 1].imshow(flipped_h)
axes[0, 1].set_title('RandomHorizontalFlip', fontsize=11)
axes[0, 1].axis('off')

# RandomVerticalFlip
flipped_v = base_crop.transpose(Image.FLIP_TOP_BOTTOM)
axes[0, 2].imshow(flipped_v)
axes[0, 2].set_title('RandomVerticalFlip', fontsize=11)
axes[0, 2].axis('off')

# RandomRotation(90)
rotated = base_crop.rotate(90)
axes[0, 3].imshow(rotated)
axes[0, 3].set_title('Rotation (90°)', fontsize=11)
axes[0, 3].axis('off')

# ColorJitter examples
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation

bright = adjust_brightness(base_crop, brightness_factor=1.3)
axes[1, 0].imshow(bright)
axes[1, 0].set_title('Brightness +30%', fontsize=11)
axes[1, 0].axis('off')

contrast = adjust_contrast(base_crop, contrast_factor=1.3)
axes[1, 1].imshow(contrast)
axes[1, 1].set_title('Contrast +30%', fontsize=11)
axes[1, 1].axis('off')

saturation = adjust_saturation(base_crop, saturation_factor=1.3)
axes[1, 2].imshow(saturation)
axes[1, 2].set_title('Saturation +30%', fontsize=11)
axes[1, 2].axis('off')

# Combined augmentation
augmented = base_crop
augmented = augmented.transpose(Image.FLIP_LEFT_RIGHT)
augmented = augmented.rotate(90)
augmented = adjust_brightness(augmented, brightness_factor=1.2)
axes[1, 3].imshow(augmented)
axes[1, 3].set_title('Combined Augmentation', fontsize=11)
axes[1, 3].axis('off')

plt.suptitle('Data Augmentation Pipeline (applied during training)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_augmentation_pipeline.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03_augmentation_pipeline.png")

# =============================================================================
# STEP 4: Summary diagram
# =============================================================================
print("\n4. Creating summary diagram...")

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Title
ax.text(50, 95, 'CRISPRi EfficientNet Training: Crop Extraction Summary', 
        ha='center', fontsize=18, fontweight='bold')

# Input
ax.text(50, 85, 'INPUT: Full Plate Images (2688×2688 pixels)', 
        ha='center', fontsize=14, bbox=dict(boxstyle='round', facecolor='lightblue'))

# Arrow
ax.annotate('', xy=(65, 72), xytext=(50, 78),
            arrowprops=dict(arrowstyle='->', lw=2))

# Grid
ax.text(80, 70, 'Extract 144 crops per image\n(12×12 grid, 224×224 each)\nStep size: ~226 pixels', 
        ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen'))

# Arrows to dataset splits
ax.annotate('', xy=(20, 58), xytext=(50, 65),
            arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('', xy=(50, 58), xytext=(50, 65),
            arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('', xy=(80, 58), xytext=(50, 65),
            arrowprops=dict(arrowstyle='->', lw=2))

# Dataset splits
ax.text(20, 52, 'Training\nP1, P2, P3, P4\n~1344 images\n~193,536 crops', 
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.text(50, 52, 'Validation\nP5\n~336 images\n~48,384 crops', 
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.text(80, 52, 'Testing\nP6\n~336 images\n~48,384 crops', 
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'))

# Arrow to augmentation
ax.annotate('', xy=(50, 38), xytext=(50, 44),
            arrowprops=dict(arrowstyle='->', lw=2))

# Augmentation
ax.text(50, 32, 'During Training: Apply Augmentations\n• RandomHorizontalFlip (50%)\n• RandomVerticalFlip (50%)\n• Rotation (0°, 90°, 180°, 270°)\n• ColorJitter (brightness, contrast, saturation)', 
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral'))

# Arrow to model
ax.annotate('', xy=(50, 18), xytext=(50, 24),
            arrowprops=dict(arrowstyle='->', lw=2))

# Model
ax.text(50, 12, 'EfficientNet-B0 → 85-class Classification\nInput: 3×224×224 → Output: 85 logits', 
        ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightpink'))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_summary_diagram.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_summary_diagram.png")

# =============================================================================
# Print summary
# =============================================================================
print("\n" + "="*70)
print("CROP EXTRACTION SUMMARY")
print("="*70)

print(f"""
Image Processing Pipeline:
-------------------------
1. Input Image: {w}×{h} pixels (plate image)
2. Grid: {GRID_SIZE}×{GRID_SIZE} = {GRID_SIZE*GRID_SIZE} positions
3. Patch Size: {PATCH_SIZE}×{PATCH_SIZE} pixels
4. Step Size: {step_w:.1f}×{step_h:.1f} pixels (overlapping patches)

Dataset Statistics:
------------------
• Training (P1-P4): ~1344 images × 144 crops = ~193,536 crops
• Validation (P5):  ~336 images × 144 crops = ~48,384 crops  
• Testing (P6):     ~336 images × 144 crops = ~48,384 crops

Augmentation (training only):
----------------------------
• RandomHorizontalFlip (p=0.5)
• RandomVerticalFlip (p=0.5)
• RandomRotation (0°, 90°, 180°, 270°)
• ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
""")

print("\nGenerated files in crop_extraction_viz/:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  - {f}")
