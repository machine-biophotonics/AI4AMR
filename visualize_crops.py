"""
visualize_crops.py - Visualize crop grid and training data pipeline

Creates visualizations showing:
1. Grid overlay on sample images
2. Crop positions
3. Augmentation examples
4. Batch mixing
5. Dataset statistics
6. Training summary

Usage:
    python visualize_crops.py
"""

import os
import sys
import json
import glob
import re
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from PIL import Image
import torch
from torchvision import transforms as T

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

GRID_SIZE = 12
CROP_SIZE = 224
IMAGE_SIZE = 2720

def extract_well(filename):
    match = re.search(r'Well(\w)(\d+)_', filename)
    if match:
        return f"{match.group(1)}{int(match.group(2)):02d}"
    return None

def get_label(path):
    with open(os.path.join(BASE_DIR, 'plate_well_id_path.json'), 'r') as f:
        plate_data = json.load(f)
    
    dirname = os.path.basename(os.path.dirname(path))
    filename = os.path.basename(path)
    well = extract_well(filename)
    
    if dirname in plate_data and well in plate_data[dirname]:
        return plate_data[dirname][well]['id']
    return None

def compute_grid_positions(grid_size=GRID_SIZE, crop_size=CROP_SIZE, image_size=IMAGE_SIZE):
    total = image_size - crop_size
    step = total / (grid_size - 1)
    
    positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            top = int(i * step)
            left = int(j * step)
            positions.append((i, j, top, left))
    
    return positions, step

def get_sample_images(n=4):
    train_paths = []
    for plate in ['P1', 'P2', 'P3', 'P4']:
        train_paths.extend(glob.glob(os.path.join(BASE_DIR, plate, '*.tif')))
    
    samples = []
    for _ in range(n):
        path = random.choice(train_paths)
        label = get_label(path)
        samples.append((path, label))
    
    return samples

def visualize_grid_overlay(image_path, positions, step):
    """Show image with crop grid overlay"""
    fig, ax = plt.subplots(figsize=(12, 12))
    
    img = Image.open(image_path).convert('RGB')
    ax.imshow(img)
    ax.set_title(f'Crop Grid Overlay\n{GRID_SIZE}×{GRID_SIZE} = {GRID_SIZE**2} positions | Step: {step:.1f}px', 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    colors = ['yellow', 'cyan', 'lime', 'magenta']
    
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            top = int(i * step)
            left = int(j * step)
            
            color = colors[(i + j) % len(colors)]
            alpha = 0.35 if (i + j) % 2 == 0 else 0.2
            
            rect = patches.Rectangle(
                (left, top), CROP_SIZE, CROP_SIZE,
                linewidth=0.5, edgecolor=color, facecolor=color, alpha=alpha
            )
            ax.add_patch(rect)
    
    ax.text(10, 30, f'Image: {IMAGE_SIZE}×{IMAGE_SIZE}', color='white', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax.text(10, 60, f'Crops: {GRID_SIZE*GRID_SIZE} per image', color='white', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax.text(10, 90, f'Crop: {CROP_SIZE}×{CROP_SIZE}px', color='white', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    return fig

def visualize_crop_samples(image_path, positions):
    """Show sample crops from different grid positions"""
    img = Image.open(image_path).convert('RGB')
    
    positions_to_show = [
        (0, 0), (0, GRID_SIZE//2), (0, GRID_SIZE-1),
        (GRID_SIZE//2, 0), (GRID_SIZE//2, GRID_SIZE//2), (GRID_SIZE//2, GRID_SIZE-1),
        (GRID_SIZE-1, 0), (GRID_SIZE-1, GRID_SIZE//2), (GRID_SIZE-1, GRID_SIZE-1),
    ]
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes = axes.flatten()
    
    step = (IMAGE_SIZE - CROP_SIZE) / (GRID_SIZE - 1)
    
    for idx, (row, col) in enumerate(positions_to_show):
        top = int(row * step)
        left = int(col * step)
        crop = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
        
        axes[idx].imshow(crop)
        axes[idx].set_title(f'({row}, {col})', fontsize=10)
        axes[idx].axis('off')
    
    for idx in range(len(positions_to_show), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Sample Crops from Different Grid Positions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def visualize_augmentations(image_path, positions):
    """Show different augmentation types"""
    step = (IMAGE_SIZE - CROP_SIZE) / (GRID_SIZE - 1)
    top = int(GRID_SIZE//2 * step)
    left = int(GRID_SIZE//2 * step)
    
    img = Image.open(image_path).convert('RGB')
    crop = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original
    axes[0, 0].imshow(crop)
    axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Horizontal flip
    flip_h = T.RandomHorizontalFlip(p=1.0)
    axes[0, 1].imshow(flip_h(crop))
    axes[0, 1].set_title('Horizontal Flip\n(p=0.5)', fontsize=10)
    axes[0, 1].axis('off')
    
    # Vertical flip
    flip_v = T.RandomVerticalFlip(p=1.0)
    axes[0, 2].imshow(flip_v(crop))
    axes[0, 2].set_title('Vertical Flip\n(p=0.5)', fontsize=10)
    axes[0, 2].axis('off')
    
    # 90 degree rotation
    axes[0, 3].imshow(crop.rotate(90))
    axes[0, 3].set_title('Rotation 90°\n(Discrete)', fontsize=10)
    axes[0, 3].axis('off')
    
    # 180 degree rotation
    axes[1, 0].imshow(crop.rotate(180))
    axes[1, 0].set_title('Rotation 180°\n(Discrete)', fontsize=10)
    axes[1, 0].axis('off')
    
    # 270 degree rotation
    axes[1, 1].imshow(crop.rotate(270))
    axes[1, 1].set_title('Rotation 270°\n(Discrete)', fontsize=10)
    axes[1, 1].axis('off')
    
    # Color jitter
    color = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
    axes[1, 2].imshow(color(crop))
    axes[1, 2].set_title('Color Jitter\n(Brightness, Contrast)', fontsize=10)
    axes[1, 2].axis('off')
    
    # Combined
    combined = flip_h(crop)
    combined = combined.rotate(90)
    combined = color(combined)
    axes[1, 3].imshow(combined)
    axes[1, 3].set_title('Combined\n(Flip+Rotate+Color)', fontsize=10)
    axes[1, 3].axis('off')
    
    plt.suptitle('Morphology-Preserving Augmentations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def visualize_batch_mixing():
    """Visualize how batches mix crops from different images"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 12))
    
    n_positions = GRID_SIZE * GRID_SIZE
    
    for batch_num in range(4):
        for i in range(8):
            img_idx = (batch_num * 8 + i) % 12
            pos_in_img = random.randint(0, n_positions - 1)
            row = pos_in_img // GRID_SIZE
            col = pos_in_img % GRID_SIZE
            
            x = (batch_num % 2) * 6 + (i % 4)
            y = (batch_num // 2) * 12 + (i // 4) * 6
            
            color = colors[img_idx]
            rect = FancyBboxPatch((x * 0.9, 23 - y), 0.7, 0.9,
                                   boxstyle="round,pad=0.02",
                                   facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
            ax.text(x * 0.9 + 0.35, 24 - y, f'{row},{col}', ha='center', va='bottom', fontsize=6)
    
    for batch_num in range(4):
        ax.axvline(x=batch_num * 5.4 - 0.2, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(batch_num * 5.4 + 1.5, 25, f'Batch {batch_num+1}', fontsize=12, 
                ha='center', fontweight='bold', color='red')
    
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.5, 26)
    ax.set_title('Batch Mixing Visualization\nEach color = different image | Numbers = (row, col) position',
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=colors[i], edgecolor='black', 
                       label=f'Image {i}') for i in range(min(8, 12))]
    ax.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=8)
    
    plt.tight_layout()
    return fig

def visualize_dataset_stats():
    """Compute and visualize dataset statistics"""
    train_paths = []
    for plate in ['P1', 'P2', 'P3', 'P4']:
        train_paths.extend(glob.glob(os.path.join(BASE_DIR, plate, '*.tif')))
    
    sample_paths = random.sample(train_paths, min(200, len(train_paths)))
    
    labels = [get_label(p) for p in sample_paths]
    labels = [l for l in labels if l is not None]
    
    label_counts = {}
    for l in labels:
        label_counts[l] = label_counts.get(l, 0) + 1
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sorted_labels = sorted(label_counts.keys())
    counts = [label_counts[l] for l in sorted_labels]
    
    axes[0].bar(range(len(counts)), counts, color='steelblue')
    axes[0].set_xlabel('Class Index')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Class Distribution (Sample of {len(sample_paths)} images)\n{len(label_counts)} unique classes')
    expected = len(sample_paths) / len(label_counts) if label_counts else 0
    axes[0].axhline(y=expected, color='red', linestyle='--', label=f'Expected: {expected:.1f}')
    axes[0].legend()
    
    sorted_counts = sorted(counts)
    axes[1].plot(range(len(sorted_counts)), sorted_counts, 'o-', color='steelblue')
    axes[1].set_xlabel('Class Rank')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Class Distribution (Sorted)')
    axes[1].axhline(y=np.mean(counts), color='red', linestyle='--', label=f'Mean: {np.mean(counts):.1f}')
    axes[1].legend()
    
    plt.suptitle('Dataset Statistics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def visualize_coverage():
    """Visualize full image coverage by crops"""
    step = (IMAGE_SIZE - CROP_SIZE) / (GRID_SIZE - 1)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    ax.set_xlim(-50, IMAGE_SIZE + 50)
    ax.set_ylim(-50, IMAGE_SIZE + 50)
    ax.set_aspect('equal')
    
    ax.fill([0, IMAGE_SIZE, IMAGE_SIZE, 0], [0, 0, IMAGE_SIZE, IMAGE_SIZE], 
            color='lightgray', alpha=0.3, label='Full Image')
    
    colors = ['yellow', 'cyan', 'lime', 'magenta']
    
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            top = int(i * step)
            left = int(j * step)
            color = colors[(i + j) % len(colors)]
            
            rect = Rectangle((left, top), CROP_SIZE, CROP_SIZE,
                             linewidth=1, edgecolor=color, facecolor=color, alpha=0.3)
            ax.add_patch(rect)
    
    coverage = (GRID_SIZE * CROP_SIZE) / IMAGE_SIZE * 100
    
    ax.set_title(f'Full Image Coverage\n{GRID_SIZE}×{GRID_SIZE} grid | Crop: {CROP_SIZE}px | Coverage: {coverage:.1f}%',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Pixels')
    ax.set_ylabel('Pixels')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_summary():
    """Create comprehensive summary figure"""
    fig = plt.figure(figsize=(20, 14))
    
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    ax1.text(0.5, 0.9, 'MODEL', fontsize=14, fontweight='bold', ha='center', transform=ax1.transAxes)
    ax1.text(0.5, 0.6, 'BacNet V2', fontsize=20, fontweight='bold', ha='center', transform=ax1.transAxes)
    ax1.text(0.5, 0.35, '~671K parameters', fontsize=11, ha='center', transform=ax1.transAxes)
    ax1.text(0.5, 0.15, 'GhostNet-inspired\nLightweight CNN', fontsize=10, ha='center', transform=ax1.transAxes)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.text(0.5, 0.9, 'DATA', fontsize=14, fontweight='bold', ha='center', transform=ax2.transAxes)
    ax2.text(0.5, 0.65, 'Train: 8,064 images', fontsize=11, ha='center', transform=ax2.transAxes)
    ax2.text(0.5, 0.45, 'Val: 336 images', fontsize=11, ha='center', transform=ax2.transAxes)
    ax2.text(0.5, 0.25, 'Test: 336 images', fontsize=11, ha='center', transform=ax2.transAxes)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    ax3.text(0.5, 0.9, 'CROPPING', fontsize=14, fontweight='bold', ha='center', transform=ax3.transAxes)
    ax3.text(0.5, 0.65, f'Grid: {GRID_SIZE}×{GRID_SIZE} = {GRID_SIZE**2}', fontsize=11, ha='center', transform=ax3.transAxes)
    ax3.text(0.5, 0.45, f'Per image: {GRID_SIZE**2} crops', fontsize=11, ha='center', transform=ax3.transAxes)
    ax3.text(0.5, 0.25, f'Total: {8064 * GRID_SIZE**2:,}/epoch', fontsize=11, ha='center', transform=ax3.transAxes)
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    ax4.text(0.5, 0.9, 'AUGMENTATIONS', fontsize=14, fontweight='bold', ha='center', transform=ax4.transAxes)
    ax4.text(0.5, 0.65, '✓ Horizontal Flip', fontsize=10, ha='center', transform=ax4.transAxes)
    ax4.text(0.5, 0.45, '✓ Vertical Flip', fontsize=10, ha='center', transform=ax4.transAxes)
    ax4.text(0.5, 0.25, '✓ 90°/180°/270° Rotation', fontsize=10, ha='center', transform=ax4.transAxes)
    ax4.text(0.5, 0.05, '✓ Color Jitter', fontsize=10, ha='center', transform=ax4.transAxes)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    ax5.text(0.5, 0.9, 'MORPHOLOGY PRESERVING', fontsize=14, fontweight='bold', ha='center', transform=ax5.transAxes)
    ax5.text(0.5, 0.65, '✓ No distortion', fontsize=10, ha='center', transform=ax5.transAxes)
    ax5.text(0.5, 0.45, '✓ No interpolation blur', fontsize=10, ha='center', transform=ax5.transAxes)
    ax5.text(0.5, 0.25, '✓ No resize', fontsize=10, ha='center', transform=ax5.transAxes)
    ax5.text(0.5, 0.05, '✓ Discrete rotations only', fontsize=10, ha='center', transform=ax5.transAxes)
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    ax6.text(0.5, 0.9, 'NO DUPLICATES', fontsize=14, fontweight='bold', ha='center', transform=ax6.transAxes)
    ax6.text(0.5, 0.65, f'✓ All {GRID_SIZE**2} positions used', fontsize=10, ha='center', transform=ax6.transAxes)
    ax6.text(0.5, 0.45, '✓ Random sample per epoch', fontsize=10, ha='center', transform=ax6.transAxes)
    ax6.text(0.5, 0.25, '✓ Unique in each epoch', fontsize=10, ha='center', transform=ax6.transAxes)
    ax6.text(0.5, 0.05, '✓ Shuffled batches', fontsize=10, ha='center', transform=ax6.transAxes)
    
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    train_crops = 8064 * GRID_SIZE * GRID_SIZE
    val_crops = 336 * GRID_SIZE * GRID_SIZE
    
    summary_text = f"""
    TRAINING CONFIGURATION
    ═══════════════════════════════════════════════════════════════════════
    • Model:       BacNet V2 (~671K params, GhostNet-inspired, SE + Spatial attention)
    • Crops:       {GRID_SIZE}×{GRID_SIZE} = {GRID_SIZE**2} positions per image | NO RESIZE
    • Train:       {train_crops:,} crops/epoch ({8064} images × {GRID_SIZE**2} positions)
    • Val/Test:    {val_crops:,} crops/epoch each
    • Batch:       64 crops per batch (mixed from different images)
    • Augment:     Morphology-preserving (flip, rotate 90°, color jitter)
    • LR:          5 epochs warmup + Cosine Annealing
    • Regularize:  Dropout 0.4, Weight decay 1e-3, Label smoothing 0.1, Gradient clip 1.0
    • Optimized:   AMP, torch.compile, 32 workers, persistent workers
    ═══════════════════════════════════════════════════════════════════════
    """
    
    ax7.text(0.5, 0.5, summary_text, fontsize=10, ha='center', va='center', 
             transform=ax7.transAxes, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('CRISPRi Image Classification - Training Pipeline Summary', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    return fig

def main():
    print("="*60)
    print("CROP VISUALIZATION SCRIPT")
    print("="*60)
    
    samples = get_sample_images(n=4)
    positions, step = compute_grid_positions()
    
    print(f"\nGrid: {GRID_SIZE}×{GRID_SIZE} = {GRID_SIZE**2} positions")
    print(f"Step: {step:.2f} pixels")
    print(f"Sample images: {len(samples)}")
    
    print("\nGenerating visualizations...")
    
    # 1. Grid overlay
    print("  1. Grid overlay...")
    sample_path, sample_label = samples[0]
    fig = visualize_grid_overlay(sample_path, positions, step)
    fig.savefig(os.path.join(BASE_DIR, 'viz_crop_grid_overlay.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Crop samples
    print("  2. Crop samples...")
    fig = visualize_crop_samples(sample_path, positions)
    fig.savefig(os.path.join(BASE_DIR, 'viz_crop_samples.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 3. Augmentations
    print("  3. Augmentations...")
    fig = visualize_augmentations(sample_path, positions)
    fig.savefig(os.path.join(BASE_DIR, 'viz_augmentations.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 4. Batch mixing
    print("  4. Batch mixing...")
    fig = visualize_batch_mixing()
    fig.savefig(os.path.join(BASE_DIR, 'viz_batch_mixing.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 5. Dataset stats
    print("  5. Dataset statistics...")
    fig = visualize_dataset_stats()
    fig.savefig(os.path.join(BASE_DIR, 'viz_dataset_stats.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 6. Coverage
    print("  6. Coverage...")
    fig = visualize_coverage()
    fig.savefig(os.path.join(BASE_DIR, 'viz_coverage.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 7. Summary
    print("  7. Summary...")
    fig = create_summary()
    fig.savefig(os.path.join(BASE_DIR, 'viz_summary.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS SAVED")
    print("="*60)
    print("Files created:")
    print("  1. viz_crop_grid_overlay.png  - Image with crop grid")
    print("  2. viz_crop_samples.png      - Crops from different positions")
    print("  3. viz_augmentations.png     - Augmentation examples")
    print("  4. viz_batch_mixing.png      - Batch mixing visualization")
    print("  5. viz_dataset_stats.png      - Class distribution")
    print("  6. viz_coverage.png           - Full image coverage")
    print("  7. viz_summary.png           - Comprehensive summary")
    print("="*60)

if __name__ == "__main__":
    main()
