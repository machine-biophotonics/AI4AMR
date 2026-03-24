#!/usr/bin/env python3
"""
Generate t-SNE and UMAP visualizations for test set
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import umap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMB_TRAIN = os.path.join(BASE_DIR, "dinov3_embeddings_train_c512.npz")
EMB_TEST = os.path.join(BASE_DIR, "dinov3_embeddings_test_c512.npz")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "dinov3")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pathway colors
PATHWAY_COLORS = {
    'Cell wall': '#E57373',
    'LPS': '#4DB6AC',
    'DNA': '#5C6BC0',
    'Transcription': '#81C784',
    'Translation': '#FFF176',
    'Metabolism': '#AED581',
    'Export': '#80CBC4',
    'Cell division': '#F06292',
    'WT': '#424242',
    'Other': '#888888'
}

FAMILY_GROUPS = {}
for family, variants in {
    'dnaB': ['dnaB_1', 'dnaB_2', 'dnaB_3'],
    'dnaE': ['dnaE_1', 'dnaE_2', 'dnaE_3'],
    'folA': ['folA_1', 'folA_2', 'folA_3'],
    'folP': ['folP_1', 'folP_2', 'folP_3'],
    'ftsI': ['ftsI_1', 'ftsI_2', 'ftsI_3'],
    'ftsZ': ['ftsZ_1', 'ftsZ_2', 'ftsZ_3'],
    'gyrA': ['gyrA_1', 'gyrA_2', 'gyrA_3'],
    'gyrB': ['gyrB_1', 'gyrB_2', 'gyrB_3'],
    'lptA': ['lptA_1', 'lptA_2', 'lptA_3'],
    'lptC': ['lptC_1', 'lptC_2', 'lptC_3'],
    'lpxA': ['lpxA_1', 'lpxA_2', 'lpxA_3'],
    'lpxC': ['lpxC_1', 'lpxC_2', 'lpxC_3'],
    'mrcA': ['mrcA_1', 'mrcA_2', 'mrcA_3'],
    'mrcB': ['mrcB_1', 'mrcB_2', 'mrcB_3'],
    'mrdA': ['mrdA_1', 'mrdA_2', 'mrdA_3'],
    'msbA': ['msbA_1', 'msbA_2', 'msbA_3'],
    'murA': ['murA_1', 'murA_2', 'murA_3'],
    'murC': ['murC_1', 'murC_2', 'murC_3'],
    'parC': ['parC_1', 'parC_2', 'parC_3'],
    'parE': ['parE_1', 'parE_2', 'parE_3'],
    'rplA': ['rplA_1', 'rplA_2', 'rplA_3'],
    'rplC': ['rplC_1', 'rplC_2', 'rplC_3'],
    'rpoA': ['rpoA_1', 'rpoA_2', 'rpoA_3'],
    'rpoB': ['rpoB_1', 'rpoB_2', 'rpoB_3'],
    'rpsA': ['rpsA_1', 'rpsA_2', 'rpsA_3'],
    'rpsL': ['rpsL_1', 'rpsL_2', 'rpsL_3'],
    'secA': ['secA_1', 'secA_2', 'secA_3'],
    'secY': ['secY_1', 'secY_2', 'secY_3'],
}.items():
    for v in variants:
        FAMILY_GROUPS[v] = family
FAMILY_GROUPS['WT'] = 'WT'

PATHWAY_GROUPS = {
    'Cell wall': ['mrcA', 'mrcB', 'mrdA', 'ftsI', 'murA', 'murC'],
    'LPS': ['lpxA', 'lpxC', 'lptA', 'lptC', 'msbA'],
    'DNA': ['gyrA', 'gyrB', 'parC', 'parE', 'dnaE', 'dnaB'],
    'Transcription': ['rpoA', 'rpoB'],
    'Translation': ['rplA', 'rplC', 'rpsA', 'rpsL'],
    'Metabolism': ['folA', 'folP'],
    'Export': ['secA', 'secY'],
    'Cell division': ['ftsZ'],
    'WT': ['WT']
}

def get_family(label):
    return FAMILY_GROUPS.get(label, label)

def get_pathway(label):
    family = get_family(label)
    for pw, genes in PATHWAY_GROUPS.items():
        if family in genes:
            return pw
    return 'Other'

# Load data
print("Loading data...")
test_data = np.load(EMB_TEST)
X_test = test_data['embeddings']
y_test = test_data['labels']

with open(os.path.join(BASE_DIR, 'plate_well_id_path.json')) as f:
    plate_data = json.load(f)

all_labels = sorted(set(
    plate_data[p][r][c]['id'] 
    for p in plate_data for r in plate_data[p] 
    for c in plate_data[p][r]
))
label_to_idx = {l: i for i, l in enumerate(all_labels)}
idx_to_label = {i: l for l, i in label_to_idx.items()}

test_labels = [idx_to_label[i] for i in y_test]
test_pathways = [get_pathway(l) for l in test_labels]
test_families = [get_family(l) for l in test_labels]

# Normalize
X_test_norm = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-8)

print(f"Test set: {X_test_norm.shape}")

# ============== t-SNE ==============
print("\nGenerating t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_test_norm)

# Plot t-SNE by pathway
print("Plotting t-SNE by pathway...")
fig, ax = plt.subplots(figsize=(14, 10))

for pathway in sorted(set(test_pathways)):
    color = PATHWAY_COLORS.get(pathway, '#888888')
    mask = [p == pathway for p in test_pathways]
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
              c=color, label=pathway, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

ax.set_xlabel('t-SNE 1', fontsize=12)
ax.set_ylabel('t-SNE 2', fontsize=12)
ax.set_title('t-SNE of DINOv3 Embeddings (Test Set, n=2016)\nColored by Pathway', fontsize=14)
ax.legend(title='Pathway', loc='best', framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_test_pathway.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: tsne_test_pathway.png")

# ============== UMAP ==============
print("\nGenerating UMAP...")
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
X_umap = reducer.fit_transform(X_test_norm)

# Plot UMAP by pathway
print("Plotting UMAP by pathway...")
fig, ax = plt.subplots(figsize=(14, 10))

for pathway in sorted(set(test_pathways)):
    color = PATHWAY_COLORS.get(pathway, '#888888')
    mask = [p == pathway for p in test_pathways]
    ax.scatter(X_umap[mask, 0], X_umap[mask, 1], 
              c=color, label=pathway, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

ax.set_xlabel('UMAP 1', fontsize=12)
ax.set_ylabel('UMAP 2', fontsize=12)
ax.set_title('UMAP of DINOv3 Embeddings (Test Set, n=2016)\nColored by Pathway', fontsize=14)
ax.legend(title='Pathway', loc='best', framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'umap_test_pathway.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: umap_test_pathway.png")

# ============== t-SNE by family ==============
print("\nGenerating t-SNE by family...")
unique_families = sorted(set(test_families))
family_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_families)))
family_to_color = {f: family_colors[i] for i, f in enumerate(unique_families)}

fig, ax = plt.subplots(figsize=(16, 12))

for family in unique_families:
    color = family_to_color[family]
    mask = [f == family for f in test_families]
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
              c=[color], label=family, alpha=0.7, s=40, edgecolors='white', linewidth=0.3)

ax.set_xlabel('t-SNE 1', fontsize=12)
ax.set_ylabel('t-SNE 2', fontsize=12)
ax.set_title('t-SNE of DINOv3 Embeddings (Test Set)\nColored by Gene Family', fontsize=14)
ax.legend(title='Gene Family', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, ncol=1)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_test_family.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: tsne_test_family.png")

print("\nDone! Generated:")
print("  - tsne_test_pathway.png")
print("  - umap_test_pathway.png")
print("  - tsne_test_family.png")
