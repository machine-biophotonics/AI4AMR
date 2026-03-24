"""
Visualize crop extraction for train.py
Shows the grid positions and sample crops from MixedCropDataset
Supports both 144 crops (12x12) and 9 crops (3x3 centered)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import glob
import json
import random
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def extract_well_from_filename(filename):
    import re
    match = re.search(r'Well(\w)(\d+)_', filename)
    if match:
        return f"{match.group(1)}{int(match.group(2)):02d}"
    return None

with open(os.path.join(BASE_DIR, 'plate_well_id_path.json'), 'r') as f:
    plate_data = json.load(f)

plate_maps = {}
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_maps[plate] = {}
    for row, wells in plate_data[plate].items():
        for col, info in wells.items():
            well = f"{row}{int(col):02d}"
            plate_maps[plate][well] = info['id']

all_labels = sorted(set(label for pm in plate_maps.values() for label in pm.values()))
label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
num_classes = len(all_labels)

def get_label_from_path(img_path):
    dirname = os.path.basename(os.path.dirname(img_path))
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if dirname in plate_maps and well in plate_maps[dirname]:
        return plate_maps[dirname][well]
    return None


class MixedCropDataset:
    """MixedCropDataset with support for different crop counts"""
    def __init__(self, image_paths, labels, crop_size=224, grid_size=12, n_crops_per_image=144, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.n_crops_per_image = n_crops_per_image
        self.augment = augment
        
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        
        self.total_w = w - crop_size
        self.total_h = h - crop_size
        self.step_w = self.total_w / (grid_size - 1) if grid_size > 1 else 0
        self.step_h = self.total_h / (grid_size - 1) if grid_size > 1 else 0
        
        # Calculate ALL positions across entire image
        all_positions = []
        for img_idx in range(len(image_paths)):
            for i in range(grid_size):
                for j in range(grid_size):
                    left = int(j * self.step_w)
                    top = int(i * self.step_h)
                    all_positions.append((img_idx, left, top))
        
        # Sample n_crops_per_image positions (NO duplicates)
        if n_crops_per_image < grid_size * grid_size:
            # For 9 crops: use centered 3x3 grid
            if n_crops_per_image == 9:
                center_start = grid_size // 2 - 1
                center_end = grid_size // 2 + 2
                indices = []
                for i in range(center_start, center_end):
                    for j in range(center_start, center_end):
                        indices.append(i * grid_size + j)
            else:
                indices = random.sample(range(grid_size * grid_size), n_crops_per_image)
            
            self.crop_positions = []
            for img_idx in range(len(image_paths)):
                for local_idx in indices:
                    i = local_idx // grid_size
                    j = local_idx % grid_size
                    left = int(j * self.step_w)
                    top = int(i * self.step_h)
                    self.crop_positions.append((img_idx, left, top))
        else:
            self.crop_positions = all_positions
    
    def __len__(self):
        return len(self.crop_positions)
    
    def get_image_positions(self, img_idx, for_image_idx=None):
        """Get all crop positions for a specific image"""
        positions = []
        n_crops = self.n_crops_per_image
        grid = int(np.sqrt(self.grid_size * self.grid_size)) if self.n_crops_per_image == 9 else self.grid_size
        
        if self.n_crops_per_image == 9:
            center_start = self.grid_size // 2 - 1
            center_end = self.grid_size // 2 + 2
            for i in range(center_start, center_end):
                for j in range(center_start, center_end):
                    left = int(j * self.step_w)
                    top = int(i * self.step_h)
                    positions.append((left, top))
        else:
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    left = int(j * self.step_w)
                    top = int(i * self.step_h)
                    positions.append((left, top))
        return positions


def visualize_crops(n_crops_per_image=144, output_dir='visualizations'):
    """Main visualization function"""
    os.makedirs(os.path.join(BASE_DIR, output_dir), exist_ok=True)
    
    train_paths = []
    for p in ['P1', 'P2', 'P3', 'P4']:
        train_paths.extend(glob.glob(os.path.join(BASE_DIR, p, '*.tif')))
    
    random.shuffle(train_paths)
    train_labels = [label_to_idx[get_label_from_path(p)] for p in train_paths]
    
    grid_size = 12 if n_crops_per_image == 144 else 12  # Always 12, but sample 9 positions
    
    dataset = MixedCropDataset(
        train_paths, train_labels,
        crop_size=224, grid_size=grid_size,
        n_crops_per_image=n_crops_per_image, augment=False
    )
    
    print(f"Grid: {n_crops_per_image} crops per image ({n_crops_per_image} positions)")
    print(f"Total crops: {len(dataset)}")
    print(f"Step: W={dataset.step_w:.1f}, H={dataset.step_h:.1f}")
    
    sample_img_idx = 0
    sample_img_path = train_paths[sample_img_idx]
    sample_img = Image.open(sample_img_path).convert('RGB')
    w, h = sample_img.size
    
    fig = plt.figure(figsize=(20, 16))
    
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(sample_img)
    ax1.set_title(f"Original Image\nSize: {w}x{h}", fontsize=12)
    ax1.set_xlabel(f"Using {n_crops_per_image} crops per image")
    ax1.axis('off')
    
    positions = dataset.get_image_positions(sample_img_idx)
    
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(sample_img)
    
    crop_label = "9 (3x3 centered)" if n_crops_per_image == 9 else "144 (12x12 full)"
    ax2.set_title(f"Grid Overlay - {crop_label}", fontsize=12)
    
    for left, top in positions:
        rect = plt.Rectangle((left, top), 224, 224, 
                            fill=False, edgecolor='red', linewidth=0.5)
        ax2.add_patch(rect)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(sample_img)
    ax3.set_title(f"All {n_crops_per_image} Positions Covered", fontsize=12)
    
    for idx, (left, top) in enumerate(positions):
        center_x = left + 112
        center_y = top + 112
        ax3.plot(center_x, center_y, 'r.', markersize=2)
    ax3.axis('off')
    
    ax4 = fig.add_subplot(2, 3, 4)
    n_cols = 5 if n_crops_per_image == 144 else 3
    n_rows = (n_crops_per_image + n_cols - 1) // n_cols
    
    fig4, axes4 = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    if n_rows == 1:
        axes4 = axes4.reshape(1, -1)
    
    for idx, (left, top) in enumerate(positions):
        row = idx // n_cols
        col = idx % n_cols
        crop = sample_img.crop((left, top, left + 224, top + 224))
        axes4[row, col].imshow(crop)
        axes4[row, col].set_title(f"#{idx}", fontsize=8)
        axes4[row, col].axis('off')
    
    for idx in range(n_crops_per_image, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes4[row, col].axis('off')
    
    plt.suptitle(f"All {n_crops_per_image} Crop Positions", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, output_dir, f'all_crops_{n_crops_per_image}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: all_crops_{n_crops_per_image}.png")
    
    fig5, axes5 = plt.subplots(2, 4, figsize=(16, 8))
    
    augmentation_types = [
        ("Original", None),
        ("H-Flip", "flip_h"),
        ("V-Flip", "flip_v"),
        ("Rotate 90", 90),
        ("Rotate 180", 180),
        ("Rotate 270", 270),
        ("Color Bright", "bright"),
        ("Combined", "combined")
    ]
    
    test_positions = [positions[0], positions[len(positions)//2], positions[-1]]
    
    for idx, (left, top) in enumerate(test_positions):
        crop = sample_img.crop((left, top, left + 224, top + 224))
        axes5[0, idx].imshow(crop)
        axes5[0, idx].set_title(f"Position {idx}", fontsize=10)
        axes5[0, idx].axis('off')
    
    for idx, (title, aug) in enumerate(augmentation_types):
        row = 1
        col = idx % 4
        
        crop = sample_img.crop((positions[0][0], positions[0][1], 
                               positions[0][0] + 224, positions[0][1] + 224))
        
        if aug == "flip_h":
            crop = ImageOps.mirror(crop)
        elif aug == "flip_v":
            crop = ImageOps.flip(crop)
        elif aug == 90:
            crop = crop.rotate(90)
        elif aug == 180:
            crop = crop.rotate(180)
        elif aug == 270:
            crop = crop.rotate(270)
        elif aug == "bright":
            enhancer = ImageEnhance.Brightness(crop)
            crop = enhancer.enhance(1.3)
        elif aug == "combined":
            if random.random() > 0.5:
                crop = ImageOps.mirror(crop)
            if random.random() > 0.5:
                crop = ImageOps.flip(crop)
            crop = crop.rotate(random.choice([0, 90, 180, 270]))
        
        axes5[row, col].imshow(crop)
        axes5[row, col].set_title(title, fontsize=10)
        axes5[row, col].axis('off')
    
    plt.suptitle("Augmentation Examples", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, output_dir, f'augmentation_examples_{n_crops_per_image}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: augmentation_examples_{n_crops_per_image}.png")
    
    fig6, axes6 = plt.subplots(1, 3, figsize=(15, 5))
    
    axes6[0].text(0.5, 0.5, f"Batch = 32 crops\nfrom DIFFERENT images\n\nEach batch mixes crops\nfrom many different wells", 
                  ha='center', va='center', fontsize=14, transform=axes6[0].transAxes)
    axes6[0].axis('off')
    
    coverage = n_crops_per_image / (12*12) * 100
    axes6[1].text(0.5, 0.5, f"Image Coverage:\n{coverage:.1f}%\n\n{n_crops_per_image}/144 positions\nsampled per image", 
                  ha='center', va='center', fontsize=14, transform=axes6[1].transAxes)
    axes6[1].axis('off')
    
    axes6[2].text(0.5, 0.5, f"Shuffle Strategy:\n\n1. All crops from ALL images\n2. Shuffle together\n3. Batch 32 at a time\n4. Different order each epoch", 
                  ha='center', va='center', fontsize=14, transform=axes6[2].transAxes)
    axes6[2].axis('off')
    
    plt.suptitle("Batch & Sampling Strategy", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, output_dir, f'sampling_strategy_{n_crops_per_image}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: sampling_strategy_{n_crops_per_image}.png")


if __name__ == "__main__":
    print("=" * 60)
    print("Visualizing 144 crops (12x12 full grid)")
    print("=" * 60)
    visualize_crops(n_crops_per_image=144)
    
    print("\n" + "=" * 60)
    print("Visualizing 9 crops (3x3 centered grid)")
    print("=" * 60)
    visualize_crops(n_crops_per_image=9)
    
    print("\nVisualization complete!")
    print(f"Files saved to: {os.path.join(BASE_DIR, 'visualizations')}")
