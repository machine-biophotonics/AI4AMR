"""
Interactive UMAP and t-SNE HTML plots using Plotly.
Uses the same color coding as the Jupyter notebook (GENE_COLORS).
"""

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import torch
from torch import nn
import torchvision.models as models
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import glob
import json
import re
import random
from tqdm import tqdm
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

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
    if label and '_' in label and label != 'WT':
        parts = label.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in ['1', '2', '3']:
            return parts[0], parts[1]
    return label, None

def label_sort_key(label):
    gene, subgroup = parse_gene_subgroup(label)
    subgroup_order = {'1': 0, '2': 1, '3': 2}
    sub_idx = subgroup_order.get(subgroup, 3) if subgroup else 0
    return (0 if gene == 'WT' else 1, gene, sub_idx)

def get_base_name(label):
    if not label:
        return None
    if label.startswith('WT') or label.startswith('NC'):
        return label.split('_')[0]
    parts = label.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return label

def get_variant_number(label):
    if not label:
        return 1
    if label.startswith('WT') or label.startswith('NC'):
        return 1
    parts = label.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return 1


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


def extract_well_from_filename(filename):
    match = re.search(r'Well(\w)(\d+)_', filename)
    if match:
        return f"{match.group(1)}{int(match.group(2)):02d}"
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


def create_interactive_plot(df, x_col, y_col, title, filename):
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color='gene_family',
        symbol='subgroup',
        hover_data={'label': True, x_col: False, y_col: False},
        title=title,
        color_discrete_map=GENE_COLORS,
        opacity=0.7
    )
    
    symbol_map = {'1': 'circle', '2': 'square', '3': 'triangle-up'}
    for trace in fig.data:
        if trace.name in symbol_map:
            trace.marker.symbol = symbol_map[trace.name]
    
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        font=dict(size=12),
        width=1400,
        height=1000,
        legend_title_text='Gene (symbol = variant)',
        legend=dict(itemsizing='constant', font=dict(size=10)),
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    
    fig.write_html(os.path.join(BASE_DIR, filename))
    print(f"  Saved: {BASE_DIR}/{filename}")
    return fig


def main():
    print("Loading model...")
    checkpoint = torch.load(os.path.join(BASE_DIR, 'best_model.pth'), map_location=device)
    
    label_to_idx = checkpoint['label_to_idx']
    idx_to_label = checkpoint['idx_to_label']
    num_classes = checkpoint['num_classes']
    
    print(f"Loaded model with {num_classes} classes")
    
    print("Loading backbone from checkpoint...")

    state_dict = checkpoint['model_state_dict']

    # Extract backbone weights only
    backbone_dict = {
        k.replace("backbone.", ""): v
        for k, v in state_dict.items()
        if k.startswith("backbone.")
    }

    print(f"Loaded {len(backbone_dict)} backbone layers")

    model = models.efficientnet_b0(weights=None)
    model.load_state_dict(backbone_dict, strict=False)

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
    unique_genes = sorted(set(get_base_name(l) for l in unique_labels if get_base_name(l)))
    
    print(f"Found {len(unique_genes)} unique genes")
    
    df = pd.DataFrame({
        'label': labels,
        'gene_family': [get_base_name(l) for l in labels],
        'subgroup': [parse_gene_subgroup(l)[1] if parse_gene_subgroup(l)[1] else ('1' if get_base_name(l) != 'WT' else 'WT') for l in labels]
    })
    
    print("\n1. Creating UMAP visualization...")
    if HAS_UMAP:
        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.1,
            n_components=2,
            metric='euclidean',
            random_state=42
        )
        features_2d = reducer.fit_transform(features)
        df['UMAP_1'] = features_2d[:, 0]
        df['UMAP_2'] = features_2d[:, 1]
        
        create_interactive_plot(
            df, 'UMAP_1', 'UMAP_2',
            f'UMAP Visualization - CNN Features<br><sup>{len(unique_genes)} gene families | {len(features)} samples | Colors from notebook</sup>',
            'umap_interactive.html'
        )
    
    print("\n2. Creating t-SNE visualization...")
    perplexity = min(50, max(5, int(np.sqrt(len(features)))))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=2000,
        learning_rate='auto',
        init='pca',
        random_state=42,
        method='barnes_hut'
    )
    
    features_tsne = tsne.fit_transform(features)
    df['tSNE_1'] = features_tsne[:, 0]
    df['tSNE_2'] = features_tsne[:, 1]
    
    create_interactive_plot(
        df, 'tSNE_1', 'tSNE_2',
        f't-SNE Visualization - CNN Features<br><sup>{len(unique_genes)} gene families | {len(features)} samples | Colors from notebook</sup>',
        'tsne_interactive.html'
    )
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"Open these HTML files in a browser:")
    print(f"  1. {BASE_DIR}/umap_interactive.html")
    print(f"  2. {BASE_DIR}/tsne_interactive.html")


if __name__ == "__main__":
    main()
