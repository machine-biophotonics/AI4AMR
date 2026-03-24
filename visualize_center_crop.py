import matplotlib
matplotlib.use('Agg')
import argparse
import os
import random
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class RandomCenterCrop:
    def __init__(self, size: int, edge_margin: int = 200):
        self.size = size
        self.edge_margin = edge_margin
    
    def get_center_region(self, w: int, h: int):
        center_w_start = self.edge_margin
        center_w_end = w - self.edge_margin
        center_h_start = self.edge_margin
        center_h_end = h - self.edge_margin
        
        max_top = center_h_end - self.size
        max_left = center_w_end - self.size
        
        if max_top <= 0 or max_left <= 0:
            left = (w - self.size) // 2
            top = (h - self.size) // 2
        else:
            top = random.randint(center_h_start, max_top)
            left = random.randint(center_w_start, max_left)
        
        return left, top
    
    def get_all_center_positions(self, w: int, h: int):
        center_w_start = self.edge_margin
        center_w_end = w - self.edge_margin
        center_h_start = self.edge_margin
        center_h_end = h - self.edge_margin
        
        positions = []
        for top in range(center_h_start, max(center_h_start, center_h_end - self.size + 1)):
            for left in range(center_w_start, max(center_w_start, center_w_end - self.size + 1)):
                if top + self.size <= h and left + self.size <= w:
                    positions.append((left, top))
        return positions


def visualize_center_crop(image_path: str, output_path: str, patch_size: int = 224, edge_margin: int = 200):
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    w, h = img.size
    
    crop = RandomCenterCrop(patch_size, edge_margin)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    ax = axes[0]
    ax.imshow(img_np)
    rect = patches.Rectangle(
        (edge_margin, edge_margin), 
        w - 2*edge_margin, 
        h - 2*edge_margin,
        linewidth=3, edgecolor='yellow', facecolor='yellow', alpha=0.3
    )
    ax.add_patch(rect)
    ax.set_title(f'Full Image ({w}x{h})\nYellow = Center Region', fontsize=12)
    ax.axis('off')
    
    ax = axes[1]
    ax.imshow(img_np)
    
    positions = crop.get_all_center_positions(w, h)
    for left, top in positions:
        rect = patches.Rectangle(
            (left, top), patch_size, patch_size,
            linewidth=0.5, edgecolor='lime', facecolor='lime', alpha=0.1
        )
        ax.add_patch(rect)
    
    left, top = crop.get_center_region(w, h)
    rect = patches.Rectangle(
        (left, top), patch_size, patch_size,
        linewidth=3, edgecolor='red', facecolor='red', alpha=0.3
    )
    ax.add_patch(rect)
    ax.set_title(f'Training: 1 Random Patch\n({len(positions)} possible positions)', fontsize=12)
    ax.axis('off')
    
    ax = axes[2]
    ax.imshow(img_np)
    
    sample_positions = random.sample(positions, min(5, len(positions)))
    colors = ['red', 'blue', 'orange', 'purple', 'cyan']
    for i, (left, top) in enumerate(sample_positions):
        rect = patches.Rectangle(
            (left, top), patch_size, patch_size,
            linewidth=2, edgecolor=colors[i], facecolor=colors[i], alpha=0.2
        )
        ax.add_patch(rect)
    
    ax.set_title(f'Val/Test: Multiple Patches\n(Average predictions)', fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize center crop extraction")
    parser.add_argument('--patch-size', type=int, default=224, help='Patch size')
    parser.add_argument('--edge-margin', type=int, default=200, help='Edge margin to avoid')
    parser.add_argument('--n-samples', type=int, default=5, help='Number of samples')
    parser.add_argument('--output', type=str, default='center_crop_visualization.png', help='Output path')
    args = parser.parse_args()
    
    image_paths = []
    for p in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
        image_paths.extend(glob.glob(os.path.join(BASE_DIR, p, '*.tif')))
    
    random.shuffle(image_paths)
    image_paths = image_paths[:args.n_samples]
    
    n = len(image_paths)
    fig, axes = plt.subplots(n, 3, figsize=(18, 6*n))
    
    if n == 1:
        axes = axes.reshape(1, -1)
    
    crop = RandomCenterCrop(args.patch_size, args.edge_margin)
    
    for row, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        w, h = img.size
        
        ax = axes[row, 0]
        ax.imshow(img_np)
        rect = patches.Rectangle(
            (args.edge_margin, args.edge_margin), 
            w - 2*args.edge_margin, 
            h - 2*args.edge_margin,
            linewidth=3, edgecolor='yellow', facecolor='yellow', alpha=0.3
        )
        ax.add_patch(rect)
        ax.set_title(f'{os.path.basename(img_path)}\nFull Image ({w}x{h})', fontsize=10)
        ax.axis('off')
        
        ax = axes[row, 1]
        ax.imshow(img_np)
        
        positions = crop.get_all_center_positions(w, h)
        for left, top in positions:
            rect = patches.Rectangle(
                (left, top), args.patch_size, args.patch_size,
                linewidth=0.5, edgecolor='lime', facecolor='lime', alpha=0.1
            )
            ax.add_patch(rect)
        
        left, top = crop.get_center_region(w, h)
        rect = patches.Rectangle(
            (left, top), args.patch_size, args.patch_size,
            linewidth=3, edgecolor='red', facecolor='red', alpha=0.3
        )
        ax.add_patch(rect)
        ax.set_title(f'Training: 1 Random Patch\n({len(positions)} positions)', fontsize=10)
        ax.axis('off')
        
        ax = axes[row, 2]
        ax.imshow(img_np)
        
        sample_positions = random.sample(positions, min(5, len(positions)))
        colors = ['red', 'blue', 'orange', 'purple', 'cyan']
        for i, (left, top) in enumerate(sample_positions):
            rect = patches.Rectangle(
                (left, top), args.patch_size, args.patch_size,
                linewidth=2, edgecolor=colors[i], facecolor=colors[i], alpha=0.2
            )
            ax.add_patch(rect)
        
        ax.set_title(f'Val/Test: 5 Patches\n(Average predictions)', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(BASE_DIR, args.output)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
