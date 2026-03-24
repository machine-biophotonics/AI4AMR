"""
UMAP Visualization for CRISPRi Classification.
Alternative to t-SNE that preserves both local and global structure.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from PIL import Image
import os
import glob
import json
import re
import random
from tqdm import tqdm

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not installed. Installing...")
    os.system("pip install umap-learn")
    import umap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class RandomCenterCrop:
    def __init__(self, size, edge_margin=200):
        self.size = size
        self.edge_margin = edge_margin
    
    def __call__(self, img):
        w, h = img.size
        center_w_start = self.edge_margin
        center_w_end = w - self.edge_margin
        center_h_start = self.edge_margin
        center_h_end = h - self.edge_margin
        
        center_w = center_w_end - center_w_start
        center_h = center_h_end - center_h_start
        
        max_top = center_h_end - self.size
        max_left = center_w_end - self.size
        
        if max_top <= 0 or max_left <= 0:
            left = (w - self.size) // 2
            top = (h - self.size) // 2
        else:
            top = random.randint(center_h_start, max_top)
            left = random.randint(center_w_start, max_left)
        
        return img.crop((left, top, left + self.size, top + self.size))


class FeatureDataset(Dataset):
    def __init__(self, image_paths, transform=None, n_patches=10):
        self.image_paths = image_paths
        self.transform = transform
        self.n_patches = n_patches
        self.patch_size = 224
        self.edge_margin = 200
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        
        center_w_start = self.edge_margin
        center_w_end = w - self.edge_margin
        center_h_start = self.edge_margin
        center_h_end = h - self.edge_margin
        
        center_w = center_w_end - center_w_start
        center_h = center_h_end - center_h_start
        
        patches = []
        for _ in range(self.n_patches):
            left = int(np.random.uniform(0, max(1, center_w - self.patch_size)))
            top = int(np.random.uniform(0, max(1, center_h - self.patch_size)))
            left = center_w_start + left
            top = center_h_start + top
            patch = image.crop((left, top, left + self.patch_size, top + self.patch_size))
            patches.append(patch)
        
        images = torch.stack([self.transform(p) for p in patches])
        return images, img_path


def get_base_name(label):
    if label.startswith('WT') or label.startswith('NC'):
        return label.split('_')[0]
    parts = label.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return label


def get_variant_number(label):
    if label.startswith('WT') or label.startswith('NC'):
        return 1
    parts = label.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return 1


def extract_well_from_filename(filename):
    match = re.search(r'Well(\w)(\d+)_', filename)
    if match:
        row = match.group(1)
        col = int(match.group(2))
        return f"{row}{col:02d}"
    return None


def load_plate_data():
    with open(os.path.join(BASE_DIR, 'plate_well_id_path.json'), 'r') as f:
        plate_data = json.load(f)
    
    plate_maps = {}
    for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
        plate_maps[plate] = {}
        for row, wells in plate_data[plate].items():
            for col, info in wells.items():
                well = f"{row}{int(col):02d}"
                plate_maps[plate][well] = info['id']
    return plate_maps


def get_label_from_path(img_path, plate_maps):
    dirname = os.path.basename(os.path.dirname(img_path))
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if dirname in plate_maps and well in plate_maps[dirname]:
        return plate_maps[dirname][well]
    return None


def main():
    print("Loading model...")
    checkpoint = torch.load(os.path.join(BASE_DIR, 'best_model.pth'), map_location=device)
    
    label_to_idx = checkpoint['label_to_idx']
    idx_to_label = checkpoint['idx_to_label']
    num_classes = checkpoint['num_classes']
    all_labels = checkpoint['all_labels']
    
    print(f"Loaded model with {num_classes} classes")
    
    model = models.efficientnet_b0(weights=None)
    in_features = 1280
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    
    plate_maps = load_plate_data()
    
    test_paths = glob.glob(os.path.join(BASE_DIR, 'P6', '*.tif'))
    print(f"Found {len(test_paths)} test images")
    
    label_paths = {}
    for path in test_paths:
        label = get_label_from_path(path, plate_maps)
        if label and label in label_to_idx:
            if label not in label_paths:
                label_paths[label] = []
            label_paths[label].append(path)
    
    sampled_paths = []
    for label, paths in label_paths.items():
        sampled_paths.extend(paths[:50])
    
    print(f"Sampled {len(sampled_paths)} images from {len(label_paths)} classes")
    
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    N_PATCHES = 10
    dataset = FeatureDataset(sampled_paths, transform=transform, n_patches=N_PATCHES)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    
    features = []
    labels = []
    
    print(f"Extracting features with {N_PATCHES} random crops per image...")
    with torch.no_grad():
        for images, paths in tqdm(loader, desc="Extracting features"):
            batch_size, n_patches, C, H, W = images.shape
            images = images.view(-1, C, H, W).to(device)
            feat = feature_extractor(images)
            feat = feat.view(batch_size, n_patches, -1)
            feat_avg = feat.mean(dim=1)
            features.append(feat_avg.cpu().numpy())
            
            for path in paths:
                label = get_label_from_path(path, plate_maps)
                labels.append(label)
    
    features = np.vstack(features)
    labels = np.array(labels)
    print(f"Features shape: {features.shape}")
    
    unique_labels = sorted(set(labels))
    unique_bases = sorted(set(get_base_name(l) for l in unique_labels))
    
    print(f"Found {len(unique_bases)} unique genes")
    
    print("\nRunning UMAP (this may take a minute)...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42
    )
    
    features_2d = reducer.fit_transform(features)
    print("UMAP completed")
    
    n_bases = len(unique_bases)
    base_color_map = {}
    colors = plt.colormaps['tab20'](np.linspace(0, 1, min(20, n_bases)))
    for i, base in enumerate(unique_bases[:20]):
        base_color_map[base] = colors[i]
    if n_bases > 20:
        colors2 = plt.colormaps['Set3'](np.linspace(0, 1, n_bases - 20))
        for i, base in enumerate(unique_bases[20:]):
            base_color_map[base] = colors2[i]
    
    base_groups = {}
    for label in unique_labels:
        base = get_base_name(label)
        if base not in base_groups:
            base_groups[base] = []
        base_groups[base].append(label)
    
    fig, ax = plt.subplots(figsize=(24, 18))
    
    for base in unique_bases:
        group_labels = base_groups[base]
        base_color = base_color_map[base]
        
        for label in sorted(group_labels, key=get_variant_number):
            mask = labels == label
            ax.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[base_color],
                label=f"{label}",
                alpha=0.7,
                s=35,
                edgecolors='white',
                linewidths=0.3
            )
    
    ax.set_xlabel('UMAP Dimension 1', fontsize=14)
    ax.set_ylabel('UMAP Dimension 2', fontsize=14)
    ax.set_title(
        f'UMAP Visualization of CRISPRi Screen (Test Set)\n'
        f'{len(unique_bases)} gene families | {len(features)} samples',
        fontsize=13
    )
    
    from matplotlib.lines import Line2D
    legend_elements = []
    for base in sorted(unique_bases):
        group_labels = sorted(base_groups[base], key=get_variant_number)
        base_color = base_color_map[base]
        for label in group_labels:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=base_color, 
                       markersize=9, label=label, markeredgecolor='white', markeredgewidth=0.5)
            )
    
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left', 
              fontsize=7, ncol=3, framealpha=0.9)
    
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    
    output_path = os.path.join(BASE_DIR, 'umap_plot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nSaved: {output_path}")
    
    print("\n" + "="*50)
    print("DONE")
    print("="*50)
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
