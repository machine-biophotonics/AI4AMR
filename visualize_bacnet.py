"""
visualize_bacnet.py - Visualization script for BacNet training
Shows:
1. Crop grid on sample images
2. Batch mixing visualization
3. Training data flow
4. Sample crops from different positions
"""

import os
import sys
import json
import glob
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GRID_SIZE = 12
CROP_SIZE = 224

def extract_well_from_filename(filename):
    match = re.search(r'Well(\w)(\d+)_', filename)
    if match:
        return f"{match.group(1)}{int(match.group(2)):02d}"
    return None

def get_label_from_path(img_path):
    with open(os.path.join(BASE_DIR, 'plate_well_id_path.json'), 'r') as f:
        plate_data = json.load(f)
    
    dirname = os.path.basename(os.path.dirname(img_path))
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    
    if dirname in plate_data and well in plate_data[dirname]:
        return plate_data[dirname][well]['id']
    return "Unknown"

def get_sample_images(n=4):
    """Get sample images from each plate"""
    samples = []
    for plate in ['P1', 'P2', 'P3', 'P4']:
        paths = glob.glob(os.path.join(BASE_DIR, plate, '*.tif'))
        if paths:
            path = np.random.choice(paths)
            label = get_label_from_path(path)
            samples.append((path, plate, label))
            if len(samples) >= n:
                break
    return samples

def compute_grid_positions(img_size, crop_size, grid_size):
    """Compute exact grid positions"""
    total = img_size - crop_size
    step = total / (grid_size - 1)
    positions = [(int(i * step), int(j * step)) for i in range(grid_size) for j in range(grid_size)]
    return positions

def visualize_single_image_grid(ax, img_path, grid_size=12, crop_size=224, title=""):
    """Visualize crop grid on a single image"""
    img = np.array(Image.open(img_path).convert('RGB'))
    h, w = img.shape[:2]
    
    ax.imshow(img)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    
    step_h = (h - crop_size) / (grid_size - 1)
    step_w = (w - crop_size) / (grid_size - 1)
    
    for i in range(grid_size):
        for j in range(grid_size):
            top = int(i * step_h)
            left = int(j * step_w)
            
            color = 'yellow' if (i + j) % 2 == 0 else 'cyan'
            alpha = 0.3
            
            rect = patches.Rectangle(
                (left, top), crop_size, crop_size,
                linewidth=0.5, edgecolor=color, facecolor=color, alpha=alpha
            )
            ax.add_patch(rect)
    
    ax.text(5, 20, f"Image: {h}×{w}", color='white', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax.text(5, 40, f"Crops: {grid_size*grid_size}", color='white', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

def visualize_crop_samples(samples, grid_size=12, crop_size=224):
    """Show sample crops from different positions"""
    img = np.array(Image.open(samples[0][0]).convert('RGB'))
    h, w = img.shape[:2]
    step_h = (h - crop_size) / (grid_size - 1)
    step_w = (w - crop_size) / (grid_size - 1)
    
    positions_to_show = [
        (0, 0), (0, 11), (11, 0), (11, 11),    # Corners
        (6, 6),                                   # Center
        (3, 3), (3, 8), (8, 3), (8, 8),          # Around center
    ]
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for idx, (row, col) in enumerate(positions_to_show):
        top = int(row * step_h)
        left = int(col * step_w)
        crop = img[top:top + crop_size, left:left + crop_size]
        
        axes[idx].imshow(crop)
        axes[idx].set_title(f"Pos ({row},{col})", fontsize=10)
        axes[idx].axis('off')
    
    for idx in range(len(positions_to_show), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle("Sample Crops from Different Grid Positions", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'visualization_crops_samples.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: visualization_crops_samples.png")

def visualize_batch_mixing(n_images=16, batch_size=32, grid_size=12):
    """Visualize how batches mix crops from different images"""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    n_positions = grid_size * grid_size
    total_crops = n_images * n_positions
    
    colors = plt.cm.tab20(np.linspace(0, 1, n_images))
    
    crop_idx = 0
    batch_starts = []
    
    for batch_num in range(min(4, total_crops // batch_size)):
        batch_start = batch_num * batch_size
        batch_starts.append(batch_start)
        
        for i in range(batch_size):
            if crop_idx >= total_crops:
                break
            
            img_idx = crop_idx // n_positions
            pos_in_img = crop_idx % n_positions
            row = pos_in_img // grid_size
            col = pos_in_img % grid_size
            
            x = (batch_num * 4) + (i % 4)
            y = (i // 4)
            
            color = colors[img_idx % len(colors)]
            rect = FancyBboxPatch((x * 0.8, 15 - y), 0.6, 0.8,
                                   boxstyle="round,pad=0.02",
                                   facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
            ax.text(x * 0.8 + 0.3, 15.4 - y, f"{row},{col}", ha='center', va='bottom', fontsize=6)
            
            crop_idx += 1
    
    for batch_num, start in enumerate(batch_starts):
        ax.axvline(x=batch_num * 4 - 0.2, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(batch_num * 4 + 1.5, 16.5, f"Batch {batch_num+1}", fontsize=12, 
                ha='center', fontweight='bold', color='red')
    
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(-0.5, 17)
    ax.set_title("Batch Mixing Visualization\n(Each color = different image, numbers = grid position in that image)",
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=colors[i], edgecolor='black', label=f'Image {i}')
                       for i in range(min(8, n_images))]
    ax.legend(handles=legend_elements[:8], loc='upper right', ncol=2, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'visualization_batch_mixing.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: visualization_batch_mixing.png")

def visualize_data_flow():
    """Create a flowchart-style visualization of the training data flow"""
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    box_style = dict(boxstyle="round,pad=0.3", facecolor="lightblue", edgecolor="black", linewidth=2)
    arrow_style = dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="black", lw=2)
    
    boxes = [
        (1, 10, "8064 Training Images\n(P1-P4)", "lightgreen"),
        (1, 7, "336 Images\n(P5 - Validation)", "lightyellow"),
        (1, 4, "336 Images\n(P6 - Test)", "lightpink"),
        (8, 10, "12×12 Grid\n(144 positions)", "lightblue"),
        (8, 7, "1,161,216\nCrops/Epoch", "orange"),
        (8, 4, "336×144\nCrops", "orange"),
        (14, 10, "Shuffled\nMixed Batches", "purple"),
        (14, 7, "Val Batches\n(No Shuffle)", "purple"),
        (14, 4, "Test Batches\n(No Shuffle)", "purple"),
    ]
    
    for x, y, text, color in boxes:
        box = FancyBboxPatch((x, y), 5, 2, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor="black", linewidth=2)
        ax.add_patch(box)
        ax.text(x + 2.5, y + 1, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    arrows = [
        (3.5, 11, 6.5, 11),      # Train images -> Grid
        (3.5, 8, 6.5, 8),        # Val images -> Grid
        (3.5, 5, 6.5, 5),        # Test images -> Grid
        (10.5, 11, 12.5, 11),    # Grid -> Train crops
        (10.5, 8, 12.5, 8),      # Grid -> Val crops
        (10.5, 5, 12.5, 5),      # Grid -> Test crops
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_style)
    
    ax.text(9, 11.5, "All positions\nfrom all images", ha='center', fontsize=9, style='italic')
    ax.text(9, 8.5, "All positions\nfrom all images", ha='center', fontsize=9, style='italic')
    ax.text(9, 5.5, "All positions\nfrom all images", ha='center', fontsize=9, style='italic')
    
    ax.text(1, 2, "Data Split", fontsize=14, fontweight='bold', ha='center')
    ax.text(8, 2, "Cropping", fontsize=14, fontweight='bold', ha='center')
    ax.text(14, 2, "Batching", fontsize=14, fontweight='bold', ha='center')
    
    ax.axhline(y=3, xmin=0.05, xmax=0.95, color='gray', linestyle=':', linewidth=2)
    ax.axvline(x=5.5, ymin=0, ymax=1, color='gray', linestyle=':', linewidth=1)
    ax.axvline(x=12, ymin=0, ymax=1, color='gray', linestyle=':', linewidth=1)
    
    plt.suptitle("BacNet Training Data Flow", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'visualization_data_flow.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: visualization_data_flow.png")

def visualize_epoch_structure(n_images_shown=4, grid_size=12):
    """Show how one epoch covers all images"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    n_positions = grid_size * grid_size
    
    x_epochs = list(range(1, 6))
    y_crops = [n_images_shown * n_positions] * 5
    axes[0].bar(x_epochs, y_crops, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Crops')
    axes[0].set_title(f'Each Epoch: Same # Crops\n({n_images_shown}×{n_positions}={y_crops[0]})')
    axes[0].set_ylim(0, max(y_crops) * 1.1)
    for i, v in enumerate(y_crops):
        axes[0].text(i+1, v + 20, str(v), ha='center', fontsize=10)
    
    x_images = list(range(1, n_images_shown + 1))
    y_positions = [n_positions] * n_images_shown
    bars = axes[1].bar(x_images, y_positions, color='coral', edgecolor='black')
    axes[1].set_xlabel('Image Index')
    axes[1].set_ylabel('Positions Covered')
    axes[1].set_title(f'Each Image: {n_positions} Crops\n(All 12×12 grid positions)')
    axes[1].set_ylim(0, n_positions * 1.1)
    for bar, v in zip(bars, y_positions):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 2, str(v), ha='center', fontsize=10)
    
    positions = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    rows, cols = zip(*positions)
    axes[2].scatter(cols, [grid_size - 1 - r for r in rows], s=100, c='green', alpha=0.7, edgecolors='black')
    axes[2].set_xlabel('Column')
    axes[2].set_ylabel('Row')
    axes[2].set_title('Grid Coverage\n(Every position = 1 crop per image)')
    axes[2].set_xlim(-0.5, grid_size - 0.5)
    axes[2].set_ylim(-0.5, grid_size - 0.5)
    axes[2].set_xticks(range(grid_size))
    axes[2].set_yticks(range(grid_size))
    axes[2].invert_yaxis()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle("Epoch Structure Visualization", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'visualization_epoch_structure.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: visualization_epoch_structure.png")

def visualize_augmentations():
    """Show the augmentation pipeline"""
    img_path = glob.glob(os.path.join(BASE_DIR, 'P1', '*.tif'))[0]
    img = Image.open(img_path).convert('RGB')
    img = img.resize((300, 300))
    img_array = np.array(img)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    from torchvision import transforms as T
    
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title("Original", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    flip_h = T.RandomHorizontalFlip(p=1.0)
    axes[0, 1].imshow(np.array(flip_h(img_array if isinstance(img_array, Image.Image) else Image.fromarray(img_array))))
    axes[0, 1].set_title("Horizontal Flip\n(p=0.5)", fontsize=12)
    axes[0, 1].axis('off')
    
    flip_v = T.RandomVerticalFlip(p=1.0)
    axes[0, 2].imshow(np.array(flip_v(img_array if isinstance(img_array, Image.Image) else Image.fromarray(img_array))))
    axes[0, 2].set_title("Vertical Flip\n(p=0.5)", fontsize=12)
    axes[0, 2].axis('off')
    
    rotate = T.RandomRotation(degrees=(0, 360))
    axes[0, 3].imshow(np.array(rotate(img_array if isinstance(img_array, Image.Image) else Image.fromarray(img_array))))
    axes[0, 3].set_title("Random Rotation\n(0-360°)", fontsize=12)
    axes[0, 3].axis('off')
    
    color = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
    axes[1, 0].imshow(np.array(color(img_array if isinstance(img_array, Image.Image) else Image.fromarray(img_array))))
    axes[1, 0].set_title("Color Jitter\n(brightness, contrast)", fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].text(0.5, 0.5, "Morphology\nPreserving\n\n✓ No Resize\n✓ No Crop\n✓ No Cutout", 
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   transform=axes[1, 1].transAxes)
    axes[1, 1].set_title("Key Point", fontsize=12, fontweight='bold', color='green')
    axes[1, 1].axis('off')
    
    axes[1, 2].text(0.5, 0.5, "Image Size\nPreserved\n\n2720×2720\n→ 224×224 crop", 
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   transform=axes[1, 2].transAxes)
    axes[1, 2].set_title("No Resize", fontsize=12, fontweight='bold', color='blue')
    axes[1, 2].axis('off')
    
    axes[1, 3].axis('off')
    
    plt.suptitle("Augmentation Pipeline (Morphology Preserving)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'visualization_augmentations.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: visualization_augmentations.png")

def main():
    print("=" * 60)
    print("BacNet Training Visualization Generator")
    print("=" * 60)
    
    samples = get_sample_images(n=4)
    print(f"\nFound {len(samples)} sample images")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, (path, plate, label) in enumerate(samples):
        title = f"{plate} - {label}"
        visualize_single_image_grid(axes[idx], path, GRID_SIZE, CROP_SIZE, title)
    
    plt.suptitle("Sample Images with Crop Grid Overlay\n(Yellow/Cyan = crop boundaries)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'visualization_crop_grid.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: visualization_crop_grid.png")
    
    print("\nGenerating visualizations...")
    visualize_crop_samples(samples, GRID_SIZE, CROP_SIZE)
    visualize_batch_mixing(n_images=16, batch_size=32, grid_size=GRID_SIZE)
    visualize_data_flow()
    visualize_epoch_structure(n_images_shown=4, grid_size=GRID_SIZE)
    visualize_augmentations()
    
    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS SAVED")
    print("=" * 60)
    print("Files created:")
    print("  1. visualization_crop_grid.png      - Sample images with crop grid")
    print("  2. visualization_crops_samples.png  - Sample crops from different positions")
    print("  3. visualization_batch_mixing.png   - How batches mix crops from different images")
    print("  4. visualization_data_flow.png      - Overall training data flow")
    print("  5. visualization_epoch_structure.png - Epoch structure explanation")
    print("  6. visualization_augmentations.png  - Augmentation pipeline")
    print("=" * 60)

if __name__ == "__main__":
    main()
