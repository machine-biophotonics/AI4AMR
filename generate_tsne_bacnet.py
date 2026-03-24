"""
generate_tsne_bacnet.py - Generate t-SNE visualization for BacNet model
Creates interactive web-based t-SNE plot using Plotly
"""

import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from PIL import Image
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

SEBLOCK_REDUCTION = 4

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excite(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=True, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pointwise(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.use_residual:
            if self.downsample is not None:
                identity = self.downsample(identity)
            out = out + identity
        out = self.relu(out)
        return out


class BacNet(nn.Module):
    def __init__(self, num_classes=85, dropout=0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 64, stride=2, use_se=True),
            ResidualBlock(64, 64, stride=1, use_se=True)
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, use_se=True),
            ResidualBlock(128, 128, stride=1, use_se=True)
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, use_se=True),
            ResidualBlock(256, 256, stride=1, use_se=True)
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2, use_se=True),
            ResidualBlock(512, 512, stride=1, use_se=True)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

checkpoint_path = os.path.join(BASE_DIR, 'best_model_bacnet.pth')
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

label_to_idx = checkpoint['label_to_idx']
idx_to_label = checkpoint['idx_to_label']
num_classes = checkpoint['num_classes']
config = checkpoint.get('config', {})

print(f"Loaded BacNet with {num_classes} classes")

model = BacNet(num_classes=num_classes, dropout=config.get('dropout', 0.4))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

feature_extractor = nn.Sequential(
    model.stem,
    model.stage1,
    model.stage2,
    model.stage3,
    model.stage4,
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten()
)
feature_extractor.eval()

norm_mean = config.get('norm_mean', [0.485, 0.456, 0.406])
norm_std = config.get('norm_std', [0.229, 0.224, 0.225])

def get_transform():
    from torchvision.transforms import Compose, ToTensor, Normalize
    return Compose([
        ToTensor(),
        Normalize(mean=norm_mean, std=norm_std)
    ])

transform = get_transform()

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
    dirname = os.path.basename(os.path.dirname(img_path))
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if dirname in plate_maps and well in plate_maps[dirname]:
        return plate_maps[dirname][well]
    return None


class CropDataset(Dataset):
    def __init__(self, image_paths, transform, grid_size=12, crop_size=224, n_crops=10):
        self.image_paths = image_paths
        self.transform = transform
        self.grid_size = grid_size
        self.crop_size = crop_size
        self.n_crops = n_crops
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        self.step_w = (w - crop_size) / (grid_size - 1) if grid_size > 1 else 0
        self.step_h = (h - crop_size) / (grid_size - 1) if grid_size > 1 else 0
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        crops = []
        positions = [(0, 0), (self.grid_size//2, self.grid_size//2), 
                     (self.grid_size-1, self.grid_size-1),
                     (0, self.grid_size-1), (self.grid_size-1, 0)]
        
        for i, j in positions[:self.n_crops]:
            top = int(i * self.step_h)
            left = int(j * self.step_w)
            crop = image.crop((left, top, left + self.crop_size, top + self.crop_size))
            crops.append(self.transform(crop))
        
        while len(crops) < self.n_crops:
            i, j = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
            top = int(i * self.step_h)
            left = int(j * self.step_w)
            crop = image.crop((left, top, left + self.crop_size, top + self.crop_size))
            crops.append(self.transform(crop))
        
        images = torch.stack(crops)
        return images, img_path


print("Loading test images (P6)...")
test_paths = glob.glob(os.path.join(BASE_DIR, 'P6', '*.tif'))
print(f"Found {len(test_paths)} test images")

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
    sampled = paths[:30]
    sampled_paths.extend(sampled)
    sampled_labels.extend([label] * len(sampled))

print(f"Sampled {len(sampled_paths)} images from {len(label_paths)} classes")

N_CROPS = 10
dataset = CropDataset(sampled_paths, transform=transform, n_crops=N_CROPS)
loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

print(f"Extracting features with {N_CROPS} crops per image...")
features = []
labels = []

with torch.no_grad():
    for images, paths in tqdm(loader, desc="Extracting features"):
        batch_size, n_crops, C, H, W = images.shape
        images = images.view(-1, C, H, W).to(device)
        feat = feature_extractor(images)
        feat = feat.view(batch_size, n_crops, -1)
        feat_avg = feat.mean(dim=1)
        features.append(feat_avg.cpu().numpy())
        
        for path in paths:
            label = get_label_from_path(path)
            labels.append(label)

features = np.vstack(features)
labels = np.array(labels)
print(f"Features shape: {features.shape}")

unique_labels = sorted(set(labels))
unique_bases = sorted(set(get_base_name(l) for l in unique_labels))

print(f"Found {len(unique_bases)} unique genes")

print("Running t-SNE...")
perplexity = min(50, max(5, int(np.sqrt(len(features)))))
n_iter = 2000

tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    n_iter=n_iter,
    learning_rate='auto',
    init='pca',
    random_state=42
)

features_2d = tsne.fit_transform(features)
print(f"t-SNE completed. KL divergence: {tsne.kl_divergence_:.4f}")

base_groups = {}
for label in unique_labels:
    base = get_base_name(label)
    if base not in base_groups:
        base_groups[base] = []
    base_groups[base].append(label)

n_bases = len(unique_bases)
colors_list = plt.colormaps['tab20'](np.linspace(0, 1, min(20, n_bases)))
base_color_map = {}
for i, base in enumerate(unique_bases[:20]):
    base_color_map[base] = colors_list[i]
if n_bases > 20:
    colors2 = plt.colormaps['Set3'](np.linspace(0, 1, n_bases - 20))
    for i, base in enumerate(unique_bases[20:]):
        base_color_map[base] = colors2[i]

print("Creating interactive HTML plot with Plotly...")

df_data = []
for i, (x, y) in enumerate(features_2d):
    label = labels[i]
    base = get_base_name(label)
    df_data.append({
        'x': x,
        'y': y,
        'label': label,
        'gene_family': base,
        'variant': get_variant_number(label)
    })

import pandas as pd
df = pd.DataFrame(df_data)

fig = px.scatter(
    df, x='x', y='y', color='gene_family',
    hover_data=['label', 'variant'],
    title=f'BacNet t-SNE Visualization (Test Set P6)<br>{len(unique_bases)} gene families | {len(features)} samples',
    labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
    color_discrete_sequence=px.colors.qualitative.Set1
)

fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
fig.update_layout(
    height=900,
    width=1200,
    legend=dict(
        title="Gene Family",
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.02,
        font=dict(size=10)
    ),
    hoverlabel=dict(
        bgcolor="white",
        font_size=12
    )
)

html_path = os.path.join(BASE_DIR, 'tsne_bacnet_interactive.html')
fig.write_html(html_path)
print(f"Saved interactive t-SNE: {html_path}")

fig_static, ax = plt.subplots(figsize=(20, 16))

for base in unique_bases:
    group_labels = base_groups[base]
    base_color = base_color_map.get(base, 'gray')
    
    for label in sorted(group_labels, key=get_variant_number):
        mask = labels == label
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[base_color],
            label=f"{label}",
            alpha=0.7,
            s=40,
            edgecolors='white',
            linewidths=0.3
        )

ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
ax.set_title(
    f'BacNet t-SNE Visualization (Test Set P6)\n'
    f'{len(unique_bases)} gene families | {len(features)} samples',
    fontsize=14
)

legend_elements = []
for base in sorted(unique_bases):
    group_labels = sorted(base_groups[base], key=get_variant_number)
    base_color = base_color_map.get(base, 'gray')
    
    for label in group_labels:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=base_color, 
                   markersize=9, label=label, markeredgecolor='white', markeredgewidth=0.5)
        )

ax.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left', 
          fontsize=7, ncol=3, framealpha=0.9)

ax.grid(True, alpha=0.2)
plt.tight_layout()

png_path = os.path.join(BASE_DIR, 'tsne_bacnet.png')
plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved static t-SNE: {png_path}")

print("\n" + "="*60)
print("DONE!")
print("="*60)
print(f"Interactive (web): {html_path}")
print(f"Static (image): {png_path}")
print("="*60)
