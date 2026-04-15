#!/usr/bin/env python3
"""
Visualize crop positions for DINOv3 training.

Shows how crops are extracted from 2720x2720 images using a 12x12 grid (144 positions).
Each epoch, the 144 positions are randomly shuffled (seed + epoch), and each image gets a 
deterministic crop position based on (epoch + idx) % 144 within that shuffled order.
Over 144 epochs, each image will see each of the 144 positions exactly once.
"""

import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Parameters
image_size = 2720
crop_size = 224
grid_size = 12
stride = (image_size - crop_size) // (grid_size - 1)  # 226

# Compute all possible crop positions (same as in plate_dataset.py)
positions = []
for i in range(grid_size):
    for j in range(grid_size):
        x = i * stride
        y = j * stride
        if x + crop_size <= image_size and y + crop_size <= image_size:
            positions.append((x, y))

seed = 42  # global seed for reproducible shuffling

print(f"Total crop positions: {len(positions)}")
print(f"Grid stride: {stride}")
print(f"First few positions: {positions[:5]}")

# Simulate crop positions for a given image index across epochs
def get_crop_position(image_idx, epoch):
    """Return (x, y) for given image index and epoch."""
    # Shuffle positions deterministically for this epoch
    shuffled = positions.copy()
    random.Random(seed + epoch).shuffle(shuffled)
    position_index = (epoch + image_idx) % len(shuffled)
    return shuffled[position_index]

# Create a dummy image (or load a real one if available)
def create_dummy_image():
    """Create a dummy 2720x2720 image with grid lines."""
    img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 200  # light gray
    # Draw grid lines
    for i in range(0, image_size, stride):
        img[i:i+2, :, :] = 0  # horizontal black lines
        img[:, i:i+2, :] = 0  # vertical black lines
    # Add some pattern to distinguish regions
    for i in range(grid_size):
        for j in range(grid_size):
            x = i * stride
            y = j * stride
            if x + crop_size <= image_size and y + crop_size <= image_size:
                # Shade each grid cell differently
                shade = 150 + (i * grid_size + j) % 100
                img[y:y+crop_size, x:x+crop_size, :] = shade
    return img

def visualize_crops_for_image(image_idx=0, epochs=6):
    """Plot crop positions for given image across multiple epochs."""
    img = create_dummy_image()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for epoch in range(epochs):
        ax = axes[epoch]
        ax.imshow(img)
        ax.set_title(f'Epoch {epoch}')
        ax.set_xlim(0, image_size)
        ax.set_ylim(image_size, 0)  # invert y axis
        
        x, y = get_crop_position(image_idx, epoch)
        rect = patches.Rectangle((x, y), crop_size, crop_size, 
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x + crop_size/2, y + crop_size/2, f'{epoch}', 
                color='white', fontsize=12, ha='center', va='center')
    
    plt.suptitle(f'Crop positions for image index {image_idx} across {epochs} epochs\n'
                 f'(Red box = 224x224 crop, grid 12x12, stride {stride})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'crop_positions_image_{image_idx}.png', dpi=150)
    plt.close()

def visualize_all_positions():
    """Plot all 144 crop positions on a single image."""
    img = create_dummy_image()
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img)
    ax.set_title('All 144 crop positions (12x12 grid)')
    ax.set_xlim(0, image_size)
    ax.set_ylim(image_size, 0)
    
    for idx, (x, y) in enumerate(positions):
        rect = patches.Rectangle((x, y), crop_size, crop_size,
                                 linewidth=1, edgecolor='blue', facecolor='none', alpha=0.7)
        ax.add_patch(rect)
        # Optional: label each position with index
        # ax.text(x+10, y+10, str(idx), fontsize=6, color='white')
    
    plt.tight_layout()
    plt.savefig('crop_positions_all.png', dpi=150)
    plt.close()

def visualize_epoch_pattern():
    """Show which positions are used by different images in a single epoch."""
    epoch = 0
    # For simplicity, show first 144 images (each will have distinct position)
    fig, axes = plt.subplots(12, 12, figsize=(20, 20))
    axes = axes.flatten()
    
    for img_idx in range(144):
        ax = axes[img_idx]
        # Create a small dummy image for each position
        dummy = np.ones((50, 50, 3), dtype=np.uint8) * 255
        ax.imshow(dummy)
        x, y = get_crop_position(img_idx, epoch)
        rect = patches.Rectangle((5, 5), 40, 40, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.set_title(f'Img {img_idx}\nPos {x},{y}', fontsize=8)
        ax.axis('off')
    
    plt.suptitle(f'Crop positions for first 144 images at epoch {epoch}\n'
                 f'Each image gets a unique position (cycling through all 144 positions)', fontsize=16)
    plt.tight_layout()
    plt.savefig('crop_positions_epoch_pattern.png', dpi=150)
    plt.close()

if __name__ == '__main__':
    print("=== Crop Extraction Explanation ===")
    print("1. Image size: 2720x2720 pixels")
    print("2. Crop size: 224x224 pixels")
    print("3. Grid: 12x12 = 144 possible positions")
    print("4. Stride: (2720-224)//(12-1) = 226 pixels")
    print("5. Position selection: shuffled each epoch (seed + epoch), then deterministic based on (epoch + image_index) % 144")
    print("   - Each epoch, the 144 positions are randomly shuffled (same order for all images)")
    print("   - Over 144 epochs, each image will see each of the 144 positions exactly once")
    print("   - Within an epoch, different images may share the same crop position")
    print()
    
    print("Generating visualizations...")
    print("\n1. All 144 positions on a single image:")
    visualize_all_positions()
    
    print("\n2. Crop positions for a single image across 6 epochs:")
    visualize_crops_for_image(image_idx=42, epochs=6)
    
    print("\n3. Crop positions for 144 different images at epoch 0:")
    visualize_epoch_pattern()