#!/usr/bin/env python3
"""
t-SNE Visualization of Embeddings
WT NC: Green triangles
NC: Black squares
Other genes: Different colors
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")


def well_to_row_col(well_id: str):
    match = well_id.replace("Well", "")
    return match[0], int(match[1:])


def load_embeddings_with_labels():
    """Load embeddings with gene labels"""
    with open(os.path.join(BASE_DIR, "plate_well_id_path.json")) as f:
        gene_mapping = json.load(f)['P1']
    
    embeddings = {}
    labels = {}
    
    # Load all embeddings
    for data_type in ["Mutants_P1", "Drugs_P1"]:
        data_dir = os.path.join(EMBEDDINGS_DIR, data_type)
        for well_folder in glob.glob(os.path.join(data_dir, "Well*")):
            well_id = os.path.basename(well_id)
            row, col = well_to_row_col(well_id)
            gene_id = gene_mapping[row][str(col)]['id']
            
            well_name = well_id.replace("Well", "")
            
            for npy_file in glob.glob(os.path.join(well_folder, "*.npy")):
                emb = np.load(npy_file)
                img_name = os.path.basename(npy_file).replace(".npy", "")
                
                key = f"{data_type}_{well_id}_{img_name}"
                embeddings[key] = emb
                labels[key] = gene_id
    
    return embeddings, labels


def get_marker_and_color(gene_id):
    """Determine marker and color for each gene"""
    base = gene_id.rsplit('_', 1)[0]
    
    # WT NC - green triangles
    if base == "WT NC":
        return 'triangle', 'green', 'WT NC'
    
    # NC - black squares
    if base == "NC":
        return 'square', 'black', 'NC'
    
    # Other genes - different colors based on first letter
    return None, None, base


def main():
    print("Loading embeddings and labels...")
    embeddings, labels = load_embeddings_with_labels()
    
    X = np.array(list(embeddings.values()))
    keys = list(embeddings.keys())
    gene_labels = [labels[k] for k in keys]
    
    print(f"Total embeddings: {len(X)}")
    
    # Run t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Group by gene
    gene_points = defaultdict(list)
    for i, (key, gene) in enumerate(zip(keys, gene_labels)):
        gene_points[gene].append((X_tsne[i, 0], X_tsne[i, 1]))
    
    # Plot non-WT/NC genes first (as background)
    for gene, points in gene_points.items():
        marker, color, label = get_marker_and_color(gene)
        if marker is None:
            # Regular genes - use color based on first letter
            first_letter = gene[0].upper()
            colors = plt.cm.tab20.colors
            color_idx = (ord(first_letter) - ord('A')) % 20
            ax.scatter([p[0] for p in points], [p[1] for p in points], 
                      c=[colors[color_idx]], s=30, alpha=0.6, label=gene)
    
    # Plot NC - black squares
    if 'NC' in gene_points:
        points = gene_points['NC']
        ax.scatter([p[0] for p in points], [p[1] for p in points],
                  c='black', marker='s', s=100, alpha=0.8, label='NC', edgecolors='black')
    
    # Plot WT NC - green triangles
    if 'WT NC' in gene_points:
        points = gene_points['WT NC']
        ax.scatter([p[0] for p in points], [p[1] for p in points],
                  c='green', marker='^', s=150, alpha=0.9, label='WT NC', 
                  edgecolors='darkgreen', linewidths=2)
    
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('t-SNE of DINOv3 Embeddings\nWT NC = Green Triangles, NC = Black Squares', 
                 fontsize=14, fontweight='bold')
    
    # Legend
    ax.legend(loc='upper right', fontsize=8, ncol=2, bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "tsne_visualization.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: tsne_visualization.png")


if __name__ == '__main__':
    main()