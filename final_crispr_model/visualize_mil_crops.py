#!/usr/bin/env python3
'''
Visualize MIL crop extraction: shows full image with position grid and crop neighbors.

Usage:
    python visualize_mil_crops.py --image_path path/to/image.tif --output_dir ./visualizations
'''

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
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


def get_edge_positions(image_size, crop_size, grid_size):
    '''Get positions at edges that don't have full 3x3 neighborhood.'''
    stride_x = (image_size[0] - crop_size) // (grid_size - 1)
    stride_y = (image_size[1] - crop_size) // (grid_size - 1)
    
    edge_positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            left = j * stride_x
            top = i * stride_y
            
            can_left = left - stride_x >= 0
            can_right = left + stride_x + crop_size <= image_size[0]
            can_top = top - stride_y >= 0
            can_bottom = top + stride_y + crop_size <= image_size[1]
            
            if not (can_left and can_right and can_top and can_bottom):
                edge_positions.append((left, top))
    
    return edge_positions, stride_x, stride_y


def extract_crop(image, left, top, crop_size):
    '''Extract a single crop from the image.'''
    crop = image.crop((left, top, left + crop_size, top + crop_size))
    return crop


def extract_9_crops(image, center_left, center_top, stride_x, stride_y, crop_size):
    '''Extract 9 crops in 3x3 grid around center.'''
    crops = []
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            left = center_left + dx * stride_x
            top = center_top + dy * stride_y
            crop = extract_crop(image, left, top, crop_size)
            crops.append(crop)
    return crops


def main():
    parser = argparse.ArgumentParser(description='Visualize MIL crop extraction')
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
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f'Loading image: {args.image_path}')
    image = Image.open(args.image_path).convert('RGB')
    w, h = image.size
    print(f'Image size: {w}x{h}')
    
    valid_positions, stride_x, stride_y = get_valid_positions((w, h), args.crop_size, args.grid_size)
    edge_positions, _, _ = get_edge_positions((w, h), args.crop_size, args.grid_size)
    
    print(f'Valid positions (with 3x3 neighborhood): {len(valid_positions)}')
    print(f'Edge positions (no full neighborhood): {len(edge_positions)}')
    
    if args.example_pos is None:
        example_idx = random.randint(0, len(valid_positions) - 1)
    else:
        example_idx = args.example_pos
    
    example_left, example_top, example_row, example_col = valid_positions[example_idx]
    print(f'Example position: index={example_idx}, row={example_row}, col={example_col}')
    print(f'  Center: ({example_left}, {example_top})')
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(image)
    
    # Draw edge positions (light red)
    for left, top in edge_positions:
        rect = plt.Rectangle((left, top), args.crop_size, args.crop_size,
                              fill=True, facecolor='#FFB3B3', alpha=0.4,
                              edgecolor='none', zorder=1)
        ax.add_patch(rect)
    
    # Draw all valid positions (light yellow)
    for left, top, _, _ in valid_positions:
        rect = plt.Rectangle((left, top), args.crop_size, args.crop_size,
                              fill=True, facecolor='#FFFFB3', alpha=0.4,
                              edgecolor='none', zorder=2)
        ax.add_patch(rect)
    
    # Highlight example 3x3 group (light green)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            left = example_left + dx * stride_x
            top = example_top + dy * stride_y
            rect = plt.Rectangle((left, top), args.crop_size, args.crop_size,
                                  fill=True, facecolor='#B3FFB3', alpha=0.6,
                                  edgecolor='#00FF00', linewidth=2, zorder=3)
            ax.add_patch(rect)
    
    ax.set_title(f'MIL Crop Positions\nYellow: Valid ({len(valid_positions)} positions) | Red: Edges (not used) | Green: Example 3x3 group',
                 fontsize=12)
    ax.axis('off')
    
    output_path = os.path.join(args.output_dir, 'crop_positions_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {output_path}')
    
    # Extract and save the 9 crops from example position
    crops = extract_9_crops(image, example_left, example_top, stride_x, stride_y, args.crop_size)
    
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for idx, (ax, crop) in enumerate(zip(axes.flat, crops)):
        ax.imshow(crop)
        row = idx // 3 - 1
        col = idx % 3 - 1
        ax.set_title(f'Crop {idx+1} (dx={col}, dy={row})', fontsize=8)
        ax.axis('off')
    
    output_path = os.path.join(args.output_dir, 'crop_group_example.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {output_path}')
    
    # Also save individual crops
    for idx, crop in enumerate(crops):
        crop.save(os.path.join(args.output_dir, f'crop_{idx+1:02d}.png'))
    print(f'Saved 9 individual crops to {args.output_dir}')
    
    print(f'\nAll outputs saved to: {args.output_dir}')


if __name__ == '__main__':
    main()