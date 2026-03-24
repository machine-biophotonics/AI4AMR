#!/usr/bin/env python3
"""
t-SNE visualization for DINOv3 embeddings with pathway-based colors
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMB_TRAIN_PATH = os.path.join(BASE_DIR, "dinov3_embeddings_train_c512.npz")
EMB_VAL_PATH = os.path.join(BASE_DIR, "dinov3_embeddings_val_c512.npz")
EMB_TEST_PATH = os.path.join(BASE_DIR, "dinov3_embeddings_test_c512.npz")

# Pathway-based color scheme
PATHWAY_COLORS = {
    # Cell wall synthesis (warm reds → oranges)
    'mrcA': '#E57373',
    'mrcB': '#EF5350',
    'mrdA': '#F06292',
    'ftsI': '#EC407A',
    'mreB': '#FF8A65',
    'murA': '#FFB74D',
    'murC': '#FFA726',
    
    # LPS synthesis (teal → cyan → turquoise)
    'lpxA': '#4DB6AC',
    'lpxC': '#26A69A',
    'lptA': '#4DD0E1',
    'lptC': '#26C6DA',
    'msbA': '#80DEEA',
    
    # DNA metabolism (indigo → violet → lavender)
    'gyrA': '#5C6BC0',
    'gyrB': '#3F51B5',
    'parC': '#7986CB',
    'parE': '#9FA8DA',
    'dnaE': '#9575CD',
    'dnaB': '#B39DDB',
    
    # Transcription & translation (greens → golds)
    'rpoA': '#81C784',
    'rpoB': '#66BB6A',
    'rpsA': '#FFF176',
    'rpsL': '#FFEE58',
    'rplA': '#FFD54F',
    'rplC': '#FFCA28',
    
    # Metabolism & protein export (lime → mint)
    'folA': '#AED581',
    'folP': '#9CCC65',
    'secY': '#80CBC4',
    'secA': '#4DB6AC',
    
    # Cell division (magenta → rose)
    'ftsZ': '#F06292',
    'minC': '#F48FB1',
    
    # Control
    'WT': '#424242'
}

def get_gene_family(label):
    """Extract gene family from label (e.g., 'dnaB_1' -> 'dnaB')"""
    if label == 'WT':
        return 'WT'
    if '_' in label:
        return label.rsplit('_', 1)[0]
    return label

def get_color_for_label(label):
    """Get color for a label based on gene family"""
    family = get_gene_family(label)
    return PATHWAY_COLORS.get(family, '#888888')

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
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

print("Loading embeddings...")
train_data = np.load(EMB_TRAIN_PATH)
val_data = np.load(EMB_VAL_PATH)
test_data = np.load(EMB_TEST_PATH)

print(f"Train: {train_data['embeddings'].shape}")
print(f"Val: {val_data['embeddings'].shape}")
print(f"Test: {test_data['embeddings'].shape}")

X = np.vstack([train_data['embeddings'], val_data['embeddings'], test_data['embeddings']])
y = np.concatenate([train_data['labels'], val_data['labels'], test_data['labels']])

split_labels = ['train'] * len(train_data['labels']) + ['val'] * len(val_data['labels']) + ['test'] * len(test_data['labels'])

print(f"Total samples: {len(X)}")

# Check if t-SNE already exists
tsne_path = os.path.join(BASE_DIR, 'results', 'dinov3', 'tsne_coordinates.npz')
if os.path.exists(tsne_path):
    print("Loading existing t-SNE coordinates...")
    tsne_data = np.load(tsne_path)
    X_tsne = tsne_data['coordinates']
else:
    print("Running t-SNE (this may take a few minutes)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    np.savez(tsne_path, coordinates=X_tsne)
    print(f"Saved t-SNE coordinates to {tsne_path}")

print("Creating visualization...")

fig, ax = plt.subplots(figsize=(16, 12))

# Group labels by gene family
unique_labels = sorted(set(idx_to_label[label] for label in y))

# Plot by pathway/gene family
for family in sorted(set(get_gene_family(label) for label in unique_labels)):
    color = get_color_for_label(family)
    family_labels = [label for label in unique_labels if get_gene_family(label) == family]
    
    for split_name, marker, alpha in [('train', 'o', 0.5), ('val', 's', 0.6), ('test', '^', 0.6)]:
        mask = [s == split_name for s in split_labels]
        
        for label in family_labels:
            label_idx = label_to_idx[label]
            label_mask = [y[i] == label_idx and mask[i] for i in range(len(y))]
            if any(label_mask):
                indices = [i for i, m in enumerate(label_mask) if m]
                ax.scatter(X_tsne[indices, 0], X_tsne[indices, 1], 
                          c=color, label=f"{label} ({split_name})",
                          marker=marker, alpha=alpha, s=30)

ax.set_xlabel('t-SNE 1', fontsize=12)
ax.set_ylabel('t-SNE 2', fontsize=12)
ax.set_title('DINOv3 Embeddings t-SNE Visualization (Colored by Pathway)', fontsize=14)

# Create legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5), fontsize=5, ncol=1)

plt.tight_layout()
output_path = os.path.join(BASE_DIR, 'results', 'dinov3', 'tsne_dinov3_pathway.png')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved to {output_path}")
