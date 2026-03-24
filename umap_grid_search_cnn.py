"""
CNN Feature-based UMAP with Grid Search - GPU Accelerated
Uses EfficientNet-B0 trained model to extract features, then performs UMAP grid search
with Calinski-Harabasz score evaluation.
Generates interactive HTML plots with Plotly.
"""

import os
import sys
import json
import re
import glob
import random
import warnings
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tqdm import tqdm
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import RobustScaler, QuantileTransformer

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

RANDOM_STATE = 42

SUBGROUP_MARKERS = {
    '1': 'circle',
    '2': 'square',
    '3': 'triangle-up'
}

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

try:
    import cuml
    from cuml.UMAP import UMAP as cuUMAP
    HAS_CUML = True
    print("✓ cuML (GPU UMAP) available")
except ImportError:
    HAS_CUML = False
    print("✗ cuML not available, using CPU UMAP")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Installing umap-learn...")
    os.system("pip install umap-learn")
    import umap
    HAS_UMAP = True


def parse_gene_subgroup(label):
    if '_' in label and label != 'WT':
        parts = label.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in ['1', '2', '3']:
            return parts[0], parts[1]
    return label, None


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


def worker_init_fn(worker_id):
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def preprocess_for_umap(X, n_quantiles=1000):
    scaler = RobustScaler()
    X_robust = scaler.fit_transform(X)
    quantile = QuantileTransformer(
        n_quantiles=min(n_quantiles, len(X)),
        output_distribution='normal',
        random_state=RANDOM_STATE
    )
    X_scaled = quantile.fit_transform(X_robust)
    return X_scaled.astype(np.float32)


def label_sort_key(label):
    gene, subgroup = parse_gene_subgroup(label)
    subgroup_order = {'1': 0, '2': 1, '3': 2}
    sub_idx = subgroup_order.get(subgroup, 3) if subgroup else 0
    return (0 if gene == 'WT' else 1, gene, sub_idx)


def run_grid_search(features, labels, gene_labels, subgroup_labels, grid_folder):
    print("\n" + "="*80)
    print("UMAP GRID SEARCH WITH CNN FEATURES")
    print("="*80)
    
    n_neighbors_grid = [15, 30, 50, 100]
    min_dist_grid = [0.0, 0.1, 0.5]
    spread_grid = [0.5, 1.0]
    
    total_combos = len(n_neighbors_grid) * len(min_dist_grid) * len(spread_grid)
    
    print(f"\nFeatures shape: {features.shape}")
    print(f"Number of samples: {len(labels)}")
    print(f"Unique genes: {len(set(gene_labels))}")
    print(f"\nGrid parameters:")
    print(f"  n_neighbors: {n_neighbors_grid}")
    print(f"  min_dist: {min_dist_grid}")
    print(f"  spread: {spread_grid}")
    print(f"Total combinations: {total_combos}")
    print(f"Using GPU: {HAS_CUML}")
    
    X_preprocessed = preprocess_for_umap(features)
    print(f"Preprocessing: RobustScaler → QuantileTransformer")
    print(f"Input dimensions: {X_preprocessed.shape[1]}")
    
    if HAS_CUML:
        X_gpu = cuml.common.client._default_client().client.upload(X_preprocessed)
    
    results = []
    pbar = tqdm(total=total_combos, desc="Grid Search", unit="combo")
    
    for n_neighbors in n_neighbors_grid:
        for min_dist in min_dist_grid:
            for spread in spread_grid:
                if min_dist > spread:
                    pbar.update(1)
                    continue
                
                try:
                    if HAS_CUML:
                        reducer = cuUMAP(
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            spread=spread,
                            metric='euclidean',
                            random_state=RANDOM_STATE,
                            n_epochs=200
                        )
                        embedding = reducer.fit_transform(X_gpu).to_numpy()
                    else:
                        reducer = umap.UMAP(
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            spread=spread,
                            metric='euclidean',
                            random_state=RANDOM_STATE,
                            n_jobs=-1
                        )
                        embedding = reducer.fit_transform(X_preprocessed)
                    
                    ch_score = calinski_harabasz_score(embedding, gene_labels)
                    
                    try:
                        db_score = davies_bouldin_score(embedding, gene_labels)
                    except:
                        db_score = np.nan
                    
                    try:
                        sil_score = silhouette_score(embedding, gene_labels, sample_size=min(5000, len(embedding)))
                    except:
                        sil_score = np.nan
                    
                    results.append({
                        'n_neighbors': n_neighbors,
                        'min_dist': min_dist,
                        'spread': spread,
                        'calinski_harabasz_score': ch_score,
                        'davies_bouldin_score': db_score,
                        'silhouette_score': sil_score,
                        'embedding': embedding
                    })
                    
                    pbar.set_postfix({'n_neigh': n_neighbors, 'm_d': min_dist, 'CH': f'{ch_score:.0f}'})
                    
                except Exception as e:
                    results.append({
                        'n_neighbors': n_neighbors,
                        'min_dist': min_dist,
                        'spread': spread,
                        'calinski_harabasz_score': np.nan,
                        'davies_bouldin_score': np.nan,
                        'silhouette_score': np.nan,
                        'embedding': None
                    })
                
                pbar.update(1)
    
    pbar.close()
    
    results_df = pd.DataFrame(results)
    results_df_valid = results_df.dropna(subset=['calinski_harabasz_score'])
    results_df_valid = results_df_valid.sort_values('calinski_harabasz_score', ascending=False)
    
    print("\n" + "="*80)
    print("GRID SEARCH RESULTS (Top 5 by Calinski-Harabasz)")
    print("="*80)
    print(results_df_valid[['n_neighbors', 'min_dist', 'spread', 'calinski_harabasz_score', 'davies_bouldin_score', 'silhouette_score']].head().to_string())
    
    best_result = results_df_valid.iloc[0]
    best_params = {
        'n_neighbors': int(best_result['n_neighbors']),
        'min_dist': best_result['min_dist'],
        'spread': best_result['spread'],
        'calinski_harabasz_score': best_result['calinski_harabasz_score']
    }
    
    print(f"\nBest parameters:")
    print(f"  n_neighbors = {best_params['n_neighbors']}")
    print(f"  min_dist = {best_params['min_dist']}")
    print(f"  spread = {best_params['spread']}")
    print(f"  Calinski-Harabasz Score = {best_params['calinski_harabasz_score']:.2f}")
    
    return results_df, best_params


def generate_heatmaps(results_df, grid_folder, best_params):
    print("\nGenerating heatmaps...")
    results_valid = results_df.dropna(subset=['calinski_harabasz_score'])
    
    spread_values = sorted(results_valid['spread'].unique())
    n_spreads = len(spread_values)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'spread = {s}' for s in spread_values[:4]]
    )
    
    for idx, spread in enumerate(spread_values[:4]):
        row = idx // 2 + 1
        col = idx % 2 + 1
        subset = results_valid[results_valid['spread'] == spread]
        pivot = subset.groupby(['min_dist', 'n_neighbors'])['calinski_harabasz_score'].mean().unstack()
        fig.add_trace(
            go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale='Viridis', colorbar=dict(title='CH')),
            row=row, col=col
        )
    
    fig.update_layout(title='UMAP Grid Search Heatmaps', height=800, width=900)
    fig.write_html(os.path.join(grid_folder, 'heatmap_grid_search.html'))
    print(f"  Saved: {grid_folder}/heatmap_grid_search.html")


def generate_interactive_umap(embedding, labels, gene_labels, subgroup_labels, best_params, output_path):
    print(f"\nGenerating interactive UMAP...")
    
    df = pd.DataFrame({
        'UMAP_1': embedding[:, 0],
        'UMAP_2': embedding[:, 1],
        'label': labels,
        'gene': gene_labels,
        'subgroup': subgroup_labels
    })
    
    fig = px.scatter(
        df, x='UMAP_1', y='UMAP_2', color='gene', symbol='subgroup',
        hover_data=['label'],
        title=f'UMAP Visualization - CNN Features<br><sup>n_neighbors={best_params["n_neighbors"]}, min_dist={best_params["min_dist"]}, spread={best_params["spread"]}, CH={best_params["calinski_harabasz_score"]:.2f}</sup>',
        color_discrete_map=GENE_COLORS, opacity=0.7
    )
    
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(width=1400, height=1000, legend_title_text='Gene', hoverlabel=dict(bgcolor="white"))
    fig.write_html(output_path)
    print(f"  Saved: {output_path}")
    return fig


def main():
    print("="*80)
    print("CNN FEATURE-BASED UMAP WITH GRID SEARCH (GPU Accelerated)")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_folder = os.path.join(BASE_DIR, f"UMAP_GridSearch_CNN_{timestamp}")
    os.makedirs(analysis_folder, exist_ok=True)
    print(f"Analysis folder: {analysis_folder}")
    
    print("\n[1/5] Loading model...")
    checkpoint = torch.load(os.path.join(BASE_DIR, 'best_model.pth'), map_location=device)
    label_to_idx = checkpoint['label_to_idx']
    num_classes = checkpoint['num_classes']
    print(f"Model classes: {num_classes}")
    
    model = models.efficientnet_b0(weights=None)
    in_features = 1280
    model.classifier[1] = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, num_classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    
    plate_maps = load_plate_data()
    
    print("\n[2/5] Collecting images...")
    test_paths = glob.glob(os.path.join(BASE_DIR, 'P6', '*.tif'))
    print(f"Found {len(test_paths)} images")
    
    label_paths = defaultdict(list)
    for path in test_paths:
        label = get_label_from_path(path, plate_maps)
        if label and label in label_to_idx:
            label_paths[label].append(path)
    
    sampled_paths = []
    for label, paths in label_paths.items():
        sampled_paths.extend(paths[:30])
    print(f"Sampled {len(sampled_paths)} images from {len(label_paths)} classes")
    
    transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    N_PATCHES = 10
    
    print("\n[3/5] Extracting features...")
    dataset = FeatureDataset(sampled_paths, transform=transform, n_patches=N_PATCHES)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    
    features = []
    labels = []
    gene_labels = []
    subgroup_labels = []
    
    with torch.no_grad():
        for images, paths in tqdm(loader, desc="Extracting", unit="batch"):
            batch_size, n_patches, C, H, W = images.shape
            images = images.view(-1, C, H, W).to(device)
            feat = feature_extractor(images).view(batch_size, n_patches, -1).mean(dim=1)
            features.append(feat.cpu().numpy())
            
            for path in paths:
                label = get_label_from_path(path, plate_maps)
                gene, subgroup = parse_gene_subgroup(label)
                labels.append(label)
                gene_labels.append(gene)
                subgroup_labels.append(subgroup if subgroup else ('1' if gene != 'WT' else 'WT'))
    
    features = np.vstack(features)
    labels = np.array(labels)
    gene_labels = np.array(gene_labels)
    subgroup_labels = np.array(subgroup_labels)
    
    print(f"\nFeatures: {features.shape}, Genes: {len(set(gene_labels))}")
    
    print("\n[4/5] Running grid search...")
    results_df, best_params = run_grid_search(features, labels, gene_labels, subgroup_labels, analysis_folder)
    
    print("\n[5/5] Saving results...")
    results_df.to_csv(os.path.join(analysis_folder, 'grid_search_results.csv'), index=False)
    generate_heatmaps(results_df, analysis_folder, best_params)
    
    best_row = results_df.dropna(subset=['calinski_harabasz_score']).iloc[0]
    best_embedding = best_row['embedding']
    
    generate_interactive_umap(
        best_embedding, labels, gene_labels, subgroup_labels, best_params,
        os.path.join(analysis_folder, 'best_umap_interactive.html')
    )
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"\nOutput: {analysis_folder}")
    print(f"  - grid_search_results.csv")
    print(f"  - heatmap_grid_search.html")
    print(f"  - best_umap_interactive.html")


if __name__ == "__main__":
    main()
