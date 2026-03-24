#!/usr/bin/env python3
"""
DINOv3 Crop Visualization
Visualize how 2720x2720 images are cropped into 25x 512x512 patches (5x5 grid)
"""

import os
import glob
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CROP_SIZE = 512
GRID_SIZE = 5

OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'dinov3')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_sample_image():
    """Get sample image path."""
    paths = glob.glob(os.path.join(BASE_DIR, 'P1', '*.tif'))
    return paths[0] if paths else None

def extract_crops(img_path, crop_size=512, grid_size=5):
    """Extract crops from image."""
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    
    step_w = (w - crop_size) / (grid_size - 1) if grid_size > 1 else 0
    step_h = (h - crop_size) / (grid_size - 1) if grid_size > 1 else 0
    
    crop_indices = list(range(grid_size * grid_size))
    
    crops = []
    positions = []
    for pos_idx in crop_indices:
        i = pos_idx // grid_size
        j = pos_idx % grid_size
        left = int(j * step_w)
        top = int(i * step_h)
        crop = img.crop((left, top, left + crop_size, top + crop_size))
        crops.append(crop)
        positions.append((left, top, i, j))
    
    return img, crops, positions, step_w, step_h

def visualize_crops(img_path=None, save_path=None):
    """Main visualization function."""
    
    if img_path is None:
        img_path = get_sample_image()
    
    if img_path is None:
        print("No sample image found")
        return
    
    print(f"Visualizing: {img_path}")
    
    img, crops, positions, step_w, step_h = extract_crops(img_path, CROP_SIZE, GRID_SIZE)
    
    w, h = img.size
    print(f"Extracted {len(crops)} crops from {w}x{h} image")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE} = {GRID_SIZE*GRID_SIZE} crops")
    print(f"Crop positions: {positions[:5]}... (showing first 5)")
    
    fig = plt.figure(figsize=(18, 14))
    
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(img)
    ax1.set_title(f"Original Image ({w}x{h})\n5x5 grid = 25 crops of {CROP_SIZE}x{CROP_SIZE}", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    for i in range(GRID_SIZE + 1):
        ax1.axhline(i * step_h, color='red', linewidth=1.5, alpha=0.8)
        ax1.axvline(i * step_w, color='red', linewidth=1.5, alpha=0.8)
    
    for idx, (x, y, row, col) in enumerate(positions):
        ax1.text(x + CROP_SIZE//2, y + CROP_SIZE//2, str(idx), 
                color='white', ha='center', va='center', 
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor='red', alpha=0.9))
    
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title(f"All 25 Crops ({CROP_SIZE}x{CROP_SIZE} each)", fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    for idx, crop in enumerate(crops):
        row = idx // GRID_SIZE
        col = idx % GRID_SIZE
        ax = fig.add_subplot(5, 5, idx + 1)
        ax.imshow(crop)
        ax.axis('off')
        ax.set_title(f"{idx}", fontsize=8)
    
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('off')
    ax3.set_title("Crop Extraction Details", fontsize=12, fontweight='bold')
    info_text = f"""
Original Image: {w} × {h} pixels
Crop Size: {CROP_SIZE} × {CROP_SIZE} pixels
Grid: {GRID_SIZE} × {GRID_SIZE} = {GRID_SIZE*GRID_SIZE} crops
Stride: {step_w:.1f} × {step_h:.1f} pixels

After extraction:
  → Resize each crop 512×512 → 256×256
  → DINOv3 ViT-L input (matches SAT-493M training)
  → Extract CLS token (1024-dim)
  → Average 25 CLS tokens per image
    """
    ax3.text(0.05, 0.5, info_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    ax4.set_title("Sample Crops at Original Size", fontsize=12, fontweight='bold')
    
    sample_indices = [0, 12, 24]
    for i, idx in enumerate(sample_indices):
        ax = fig.add_subplot(3, 3, 7 + i)
        ax.imshow(crops[idx])
        x, y, row, col = positions[idx]
        ax.set_title(f"Crop {idx}: row={row}, col={col}\n({x}, {y})", fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, 'crop_visualization.png')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {save_path}")
    plt.close()

if __name__ == '__main__':
    visualize_crops()