import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn
import torchvision.models as models
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from collections import Counter
import os
import glob
import json
import re
import random
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

GENE_COLORS = {
    'mrcA': '#E57373', 'mrcB': '#EF5350', 'mrdA': '#F06292', 'ftsI': '#EC407A',
    'murA': '#FFB74D', 'murC': '#FFA726', 'lpxA': '#4DB6AC', 'lpxC': '#26A69A',
    'lptA': '#4DD0E1', 'lptC': '#26C6DA', 'msbA': '#80DEEA', 'gyrA': '#5C6BC0',
    'gyrB': '#3F51B5', 'parC': '#7986CB', 'parE': '#9FA8DA', 'dnaE': '#9575CD',
    'dnaB': '#B39DDB', 'rpoA': '#81C784', 'rpoB': '#66BB6A', 'rpsA': '#FFF176',
    'rpsL': '#FFEE58', 'rplA': '#FFD54F', 'rplC': '#FFCA28', 'folA': '#AED581',
    'folP': '#9CCC65', 'secY': '#80CBC4', 'secA': '#4DB6AC', 'ftsZ': '#F06292',
    'WT': '#424242'
}

SUBGROUP_MARKERS = {
    '1': 'circle', '2': 'square', '3': 'triangle-up'
}

def parse_gene_subgroup(label):
    if '_' in label and label != 'WT':
        parts = label.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in ['1', '2', '3']:
            return parts[0], parts[1]
    return label, None

def label_sort_key(label):
    gene, subgroup = parse_gene_subgroup(label)
    subgroup_order = {'1': 0, '2': 1, '3': 2}
    sub_idx = subgroup_order.get(subgroup, 3) if subgroup else 0
    return (0 if gene == 'WT' else 1, gene, sub_idx)


# NOTE: RandomCenterCrop is defined but NOT used - FeatureDataset does manual cropping
# class RandomCenterCrop:
#     """Random crop from center region only (avoids edges) - matches training"""
#     def __init__(self, size, edge_margin=200):
#         self.size = size
#         self.edge_margin = edge_margin
#     
#     def __call__(self, img):
#         w, h = img.size
#         center_w_start = self.edge_margin
#         center_w_end = w - self.edge_margin
#         center_h_start = self.edge_margin
#         center_h_end = h - self.edge_margin
#         
#         center_w = center_w_end - center_w_start
#         center_h = center_h_end - center_h_start
#         
#         max_top = center_h_end - self.size
#         max_left = center_w_end - self.size
#         
#         if max_top <= 0 or max_left <= 0:
#             left = (w - self.size) // 2
#             top = (h - self.size) // 2
#         else:
#             top = random.randint(center_h_start, max_top)
#             left = random.randint(center_w_start, max_left)
#         
#         return img.crop((left, top, left + self.size, top + self.size))


# NOTE: Unused - FeatureDataset does manual cropping instead
# random_center_crop = RandomCenterCrop(224, edge_margin=200)

train_transform = Compose([
    # random_center_crop,  # Not used - patches already cropped in FeatureDataset
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform = train_transform


def get_base_name(label):
    """Extract base name from label. e.g., 'dnaB_1' -> 'dnaB', 'WT' -> 'WT'"""
    if label.startswith('WT') or label.startswith('NC'):
        return label.split('_')[0]
    parts = label.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return label


def get_variant_number(label):
    """Extract variant number from label. e.g., 'dnaB_1' -> 1, 'WT' -> 1"""
    if label.startswith('WT') or label.startswith('NC'):
        return 1
    parts = label.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return 1

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(os.path.join(BASE_DIR, 'best_model.pth'), map_location=device)

label_to_idx = checkpoint['label_to_idx']
idx_to_label = checkpoint['idx_to_label']
num_classes = checkpoint['num_classes']
all_labels = checkpoint['all_labels']

print(f"Loaded model with {num_classes} classes")
print(f"Device: {device}")

# Recreate model with Dropout (matching training - p=0.5)
model = models.efficientnet_b0(weights=None)
in_features = 1280
model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.5),  # FIXED: was 0.3, now matches training
    nn.Linear(in_features, num_classes)
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Remove the classifier to get features
feature_extractor = nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval()

# Load plate data
with open(os.path.join(BASE_DIR, 'plate_well_id_path.json'), 'r') as f:
    plate_data = json.load(f)

plate_maps = {}
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_maps[plate] = {}
    for row, wells in plate_data[plate].items():
        for col, info in wells.items():
            well = f"{row}{int(col):02d}"
            plate_maps[plate][well] = info['id']

def extract_well_from_filename(filename):
    match = re.search(r'Well(\w)(\d+)_', filename)
    if match:
        row = match.group(1)
        col = int(match.group(2))
        return f"{row}{col:02d}"
    return None

def get_label_from_path(img_path):
    dirname = os.path.dirname(img_path)
    plate = os.path.basename(dirname)
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if plate in plate_maps and well in plate_maps[plate]:
        return plate_maps[plate][well]
    return None

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

# NOTE: RandomCenterCrop is defined but NOT used - FeatureDataset does manual cropping
# class RandomCenterCrop:
#     """Random crop from center region only (avoids edges) - matches training"""
#     def __init__(self, size, edge_margin=200):
#         self.size = size
#         self.edge_margin = edge_margin
#     
#     def __call__(self, img):
#         w, h = img.size
#         center_w_start = self.edge_margin
#         center_w_end = w - self.edge_margin
#         center_h_start = self.edge_margin
#         center_h_end = h - self.edge_margin
#         
#         center_w = center_w_end - center_w_start
#         center_h = center_h_end - center_h_start
#         
#         max_top = center_h_end - self.size
#         max_left = center_w_end - self.size
#         
#         if max_top <= 0 or max_left <= 0:
#             left = (w - self.size) // 2
#             top = (h - self.size) // 2
#         else:
#             top = random.randint(center_h_start, max_top)
#             left = random.randint(center_w_start, max_left)
#         
#         return img.crop((left, top, left + self.size, top + self.size))

# NOTE: Unused - FeatureDataset does manual cropping instead
# random_center_crop = RandomCenterCrop(224, edge_margin=200)

train_transform = Compose([
    # random_center_crop,  # Not used - patches already cropped in FeatureDataset
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform = train_transform

# Get all test images
test_paths = glob.glob(os.path.join(BASE_DIR, 'P6', '*.tif'))
print(f"Found {len(test_paths)} test images")

# Sample images per class (max 50 per class for speed)
label_paths = {}
for path in test_paths:
    label = get_label_from_path(path)
    if label and label in label_to_idx:
        if label not in label_paths:
            label_paths[label] = []
        label_paths[label].append(path)

sampled_paths = []
sampled_labels = []
for label, paths in label_paths.items():
    sampled = paths[:50]  # Max 50 per class
    sampled_paths.extend(sampled)
    sampled_labels.extend([label] * len(sampled))

print(f"Sampled {len(sampled_paths)} images from {len(label_paths)} classes")

N_PATCHES = 10  # Match training

# Create dataset and loader
dataset = FeatureDataset(sampled_paths, transform=transform, n_patches=N_PATCHES)
loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

# Extract features (average across multiple random crops, like training)
features = []
labels = []

print(f"Extracting features with {N_PATCHES} random crops per image...")
with torch.no_grad():
    for images, paths in tqdm(loader, desc="Extracting features"):
        batch_size, n_patches, C, H, W = images.shape
        images = images.view(-1, C, H, W).to(device)
        feat = feature_extractor(images)
        feat = feat.view(batch_size, n_patches, -1)
        feat_avg = feat.mean(dim=1)  # Average across patches
        features.append(feat_avg.cpu().numpy())
        
        for path in paths:
            label = get_label_from_path(path)
            labels.append(label)

features = np.vstack(features)
labels = np.array(labels)
print(f"Features shape: {features.shape}")

# Group genes by base name
unique_labels = sorted(set(labels))
unique_bases = sorted(set(get_base_name(l) for l in unique_labels))

# Assign colors to base names
base_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_bases)))
base_color_map = {base: base_colors[i] for i, base in enumerate(unique_bases)}

# Group labels by base name
base_groups = {}
for label in unique_labels:
    base = get_base_name(label)
    if base not in base_groups:
        base_groups[base] = []
    base_groups[base].append(label)

# Get max variant number per base for intensity scaling
max_variants = {}
for base, group_labels in base_groups.items():
    variant_nums = [get_variant_number(l) for l in group_labels]
    max_variants[base] = max(variant_nums) if variant_nums else 1

print(f"Found {len(unique_bases)} unique genes: {unique_bases}")

# Apply t-SNE with best practices
print("Running t-SNE (this may take a minute)...")

# Best practices from literature:
# - perplexity: 5-50, higher for larger datasets (we use 30-50)
# - n_iter: 1000-3000, more iterations = better convergence (we use 2000)
# - learning_rate: auto (sklearn default) or 200
# - init: 'pca' initialization is often better than 'random'

perplexity = min(50, max(5, int(np.sqrt(len(features)))))  # Rule of thumb: sqrt(n)
n_iter = 2000  # More iterations for better convergence

tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    n_iter=n_iter,
    learning_rate='auto',
    init='pca',  # PCA initialization is more stable
    random_state=42,
    method='barnes_hut'  # Faster for large datasets
)

features_2d = tsne.fit_transform(features)
print(f"t-SNE completed. KL divergence: {tsne.kl_divergence_:.4f}")

# Plot
print("Creating plot...")
fig, ax = plt.subplots(figsize=(24, 18))

# Use tab20 colormap - each gene gets ONE distinct color, no shading
n_bases = len(unique_bases)
base_color_map = {}
colors = plt.colormaps['tab20'](np.linspace(0, 1, min(20, n_bases)))
for i, base in enumerate(unique_bases[:20]):
    base_color_map[base] = colors[i]
if n_bases > 20:
    colors2 = plt.colormaps['Set3'](np.linspace(0, 1, n_bases - 20))
    for i, base in enumerate(unique_bases[20:]):
        base_color_map[base] = colors2[i]
base_color_map = {base: base_colors[i] for i, base in enumerate(unique_bases)}

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

ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
ax.set_title(
    f't-SNE Visualization of CRISPRi Screen (Test Set)\n'
    f'{len(unique_bases)} gene families | {len(features)} samples',
    fontsize=13
)

# Create legend grouped by gene family
legend_elements = []
for base in sorted(unique_bases):
    group_labels = sorted(base_groups[base], key=get_variant_number)
    base_color = base_color_map[base]
    
    for label in group_labels:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=base_color, 
                   markersize=9, label=label, markeredgecolor='white', markeredgewidth=0.5)
        )

# Legend in multiple columns outside plot
ax.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left', 
          fontsize=7, ncol=3, framealpha=0.9)

ax.grid(True, alpha=0.2)
plt.tight_layout()

output_path = os.path.join(BASE_DIR, 'tsne_plot.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\nt-SNE plot saved: {output_path}")

# Also save class distribution info
dist_path = os.path.join(BASE_DIR, 'tsne_class_distribution.txt')
with open(dist_path, 'w') as f:
    f.write("t-SNE Class Distribution\n")
    f.write("="*40 + "\n\n")
    f.write(f"Total samples: {len(features)}\n")
    f.write(f"Total genes: {len(unique_bases)}\n")
    f.write(f"Total variants: {len(unique_labels)}\n\n")
    f.write("Genes grouped by family:\n")
    for base in sorted(unique_bases):
        variants = sorted(base_groups[base], key=get_variant_number)
        f.write(f"\n  {base}:\n")
        for v in variants:
            count = sum(labels == v)
            f.write(f"    {v}: {count} samples\n")

print(f"Class distribution saved: {dist_path}")
