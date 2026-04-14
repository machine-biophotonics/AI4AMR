#!/usr/bin/env python3
'''
Visualize augmented versions of the crop group.

Usage:
    python visualize_augmented_crops.py --image_path path/to/image.tif --output_dir ./visualizations
'''

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import random


def get_valid_positions(image_size, crop_size, grid_size):
    '''Get 100 valid positions that have full 3x3 neighborhood.'''
    stride_x = (image_size[0] - crop_size) // (grid_size - 1)
    stride_y = (image_size[1] - crop_size) // (grid_size - 1)
    
    valid_positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            left = j * stride_x
            top = i * stride_y
            
            can_left = left - stride_x >= 0
            can_right = left + stride_x + crop_size <= image_size[0]
            can_top = top - stride_y >= 0
            can_bottom = top + stride_y + crop_size <= image_size[1]
            
            if can_left and can_right and can_top and can_bottom:
                valid_positions.append((left, top, i, j))
    
    return valid_positions, stride_x, stride_y


def extract_9_crops(image, center_left, center_top, stride_x, stride_y, crop_size):
    '''Extract 9 crops in 3x3 grid around center.'''
    crops = []
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            left = center_left + dx * stride_x
            top = center_top + dy * stride_y
            crop = image.crop((left, top, left + crop_size, top + crop_size))
            crops.append(crop)
    return crops


def apply_augmentations(crop, seed):
    '''Apply random augmentations to a crop.'''
    random.seed(seed)
    
    augmented = crop.copy()
    
    if random.random() < 0.5:
        augmented = augmented.transpose(Image.ROTATE_90)
    
    if random.random() < 0.5:
        augmented = ImageOps.mirror(augmented)
    
    if random.random() < 0.5:
        augmented = ImageOps.flip(augmented)
    
    if random.random() < 0.5:
        angle = random.choice([0, 90, 180, 270])
        augmented = augmented.rotate(angle)
    
    return augmented


def main():
    parser = argparse.ArgumentParser(description='Visualize augmented crop groups')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to sample image')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='Crop size (default: 224)')
    parser.add_argument('--grid_size', type=int, default=12,
                        help='Grid size (default: 12)')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                        help='Output directory')
    parser.add_argument('--example_pos', type=int, default=None,
                        help='Example position index (default: random)')
    parser.add_argument('--num_augmented', type=int, default=4,
                        help='Number of augmented versions to generate (default: 4)')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f'Loading image: {args.image_path}')
    image = Image.open(args.image_path).convert('RGB')
    w, h = image.size
    print(f'Image size: {w}x{h}')
    
    valid_positions, stride_x, stride_y = get_valid_positions((w, h), args.crop_size, args.grid_size)
    
    if args.example_pos is None:
        example_idx = 50
    else:
        example_idx = args.example_pos
    
    example_left, example_top, example_row, example_col = valid_positions[example_idx]
    print(f'Example position: index={example_idx}, row={example_row}, col={example_col}')
    
    # Extract original 9 crops
    original_crops = extract_9_crops(image, example_left, example_top, stride_x, stride_y, args.crop_size)
    
    # Create figure with original + augmented versions
    n_augmented = args.num_augmented
    rows = n_augmented + 1
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot original
    for idx, (ax, crop) in enumerate(zip(axes[0], original_crops)):
        ax.imshow(crop)
        row = idx // 3 - 1
        col = idx % 3 - 1
        ax.set_title(f'Original\ndx={col}, dy={row}', fontsize=8)
        ax.axis('off')
    
    # Plot augmented versions
    for aug_idx in range(n_augmented):
        aug_crops = [apply_augmentations(crop, aug_idx * 100 + i) for i, crop in enumerate(original_crops)]
        
        for idx, (ax, crop) in enumerate(zip(axes[aug_idx + 1], aug_crops)):
            ax.imshow(crop)
            if idx == 0:
                ax.set_title(f'Augmented v{aug_idx + 1}', fontsize=8)
            ax.axis('off')
    
    plt.suptitle(f'Original and {n_augmented} Augmented Versions (Same 9 Crops)', fontsize=12, y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(args.output_dir, 'crop_group_augmented.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {output_path}')
    
    print(f'\nAll outputs saved to: {args.output_dir}')


if __name__ == '__main__':
    main()