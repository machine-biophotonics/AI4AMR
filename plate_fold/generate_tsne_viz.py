#!/usr/bin/env python3
"""
Generate t-SNE and UMAP visualizations for plate_fold.
Uses predictions.csv to get actual image names - 2016 images per fold.
"""

import os
import argparse
import numpy as np
import pandas as pd
import ast
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


GROUP_COLORS = {
    'cell division': '#E53935',
    'cell wall organization': '#FB8C00',
    'cell wall repair': '#FDD835',
    'lipid A biosynthetic process': '#43A047',
    'lipid translocation': '#8BC34A',
    'Gram-negative-bacterium-type cell outer membrane assembly': '#00BCD4',
    'DNA topological change': '#3F51B5',
    'DNA-templated DNA replication': '#5C6BC0',
    'DNA replication': '#673AB7',
    'chromosome segregation': '#9C27B0',
    'chromosome organization': '#E91E63',
    'bacterial-type flagellum assembly': '#F44336',
    'intracellular protein transmembrane transport': '#009688',
    '10-formyltetrahydrofolate biosynthetic process': '#CDDC39',
    'folic acid biosynthetic process': '#FFEB3B',
}

HIERARCHY = {
    'ftsZ': 'cell division', 'ftsI': 'cell division', 'murA': 'cell division', 'murC': 'cell division',
    'rpsA': 'cytoplasmic translation', 'rpsL': 'cytoplasmic translation',
    'rplA': 'cytoplasmic translation', 'rplC': 'cytoplasmic translation',
    'mrdA': 'cell wall organization', 'mrcA': 'cell wall organization', 'mrcB': 'cell wall repair',
    'lpxA': 'lipid A biosynthetic process', 'lpxC': 'lipid A biosynthetic process',
    'lptA': 'Gram-negative-bacterium-type cell outer membrane assembly',
    'lptC': 'Gram-negative-bacterium-type cell outer membrane assembly',
    'gyrA': 'DNA topological change', 'gyrB': 'DNA topological change',
    'rpoA': 'bacterial-type flagellum assembly', 'rpoB': 'bacterial-type flagellum assembly',
    'secA': 'intracellular protein transmembrane transport', 'secY': 'intracellular protein transmembrane transport',
    'msbA': 'lipid translocation',
    'folA': '10-formyltetrahydrofolate biosynthetic process', 'folP': 'folic acid biosynthetic process',
    'dnaE': 'DNA-templated DNA replication', 'dnaB': 'DNA replication',
    'parC': 'chromosome segregation', 'parE': 'chromosome organization',
}


def get_base_gene(label):
    if not label or label == 'nan':
        return 'Unknown'
    if '_' in str(label):
        return str(label).rsplit('_', 1)[0]
    return str(label)


def get_pathway(label):
    base = get_base_gene(label)
    if 'WT' in str(base).upper() or 'NC' in str(base).upper():
        return 'WT NC'
    if base in HIERARCHY:
        return HIERARCHY[base]
    return 'Unknown'


def extract_image_embeddings(fold_dir):
    """Extract image-level embeddings from predictions.csv - 2016 images."""
    csv_path = os.path.join(fold_dir, 'predictions.csv')
    if not os.path.exists(csv_path):
        print(f"  ERROR: predictions.csv not found in {fold_dir}")
        return None, None
    
    print(f"  Loading predictions.csv...")
    df = pd.read_csv(csv_path, usecols=['image_name', 'ground_truth_label', 'embedding'])
    print(f"  Total crops: {len(df)}")
    
    # Group by image_name - each image has 144 crops
    image_embeddings = []
    image_labels = []
    
    for img_name, group in df.groupby('image_name'):
        emb_list = []
        for emb_str in group['embedding'].values:
            emb = np.array(ast.literal_eval(emb_str))
            emb_list.append(emb)
        
        # Average all crops in this image
        img_emb = np.mean(emb_list, axis=0)
        true_label = group['ground_truth_label'].iloc[0]
        
        image_embeddings.append(img_emb)
        image_labels.append(true_label)
    
    print(f"  Number of images: {len(image_embeddings)}")
    return np.array(image_embeddings), np.array(image_labels)


def plot_tsne(embeddings, labels, title, output_path):
    """Create t-SNE plot."""
    print(f"    Running t-SNE on {len(embeddings)} samples...")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(embeddings)
    
    pathway_labels = [get_pathway(l) for l in labels]
    unique_pathways = sorted(set(pathway_labels))
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for pathway in unique_pathways:
        mask = [p == pathway for p in pathway_labels]
        color = GROUP_COLORS.get(pathway, '#888888')
        ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                   c=color, label=pathway, alpha=0.7, s=50)
    
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def plot_umap(embeddings, labels, title, output_path):
    """Create UMAP plot."""
    if not HAS_UMAP:
        print(f"    UMAP not available, skipping")
        return
    
    print(f"    Running UMAP on {len(embeddings)} samples...")
    
    reducer = umap.UMAP(random_state=42)
    umap_result = reducer.fit_transform(embeddings)
    
    pathway_labels = [get_pathway(l) for l in labels]
    unique_pathways = sorted(set(pathway_labels))
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for pathway in unique_pathways:
        mask = [p == pathway for p in pathway_labels]
        color = GROUP_COLORS.get(pathway, '#888888')
        ax.scatter(umap_result[mask, 0], umap_result[mask, 1],
                   c=color, label=pathway, alpha=0.7, s=50)
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate t-SNE/UMAP visualizations')
    parser.add_argument('--fold', type=str, default='P2', help='Fold to process (P1-P6)')
    args = parser.parse_args()
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    fold_dir = os.path.join(SCRIPT_DIR, f'fold_{args.fold}')
    
    print(f"\n{'='*60}")
    print(f"Processing fold: {args.fold}")
    print(f"{'='*60}")
    
    # Extract image embeddings - 2016 images
    embeddings, labels = extract_image_embeddings(fold_dir)
    
    if embeddings is None:
        return
    
    print(f"  Data shape: {embeddings.shape}")
    print(f"  Unique labels: {len(set(labels))}")
    
    # Save image embeddings
    output_dir = os.path.join(fold_dir, 'confusion')
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'image_embeddings.npy'), embeddings)
    np.save(os.path.join(output_dir, 'image_labels.npy'), labels)
    print(f"  Saved embeddings to {output_dir}")
    
    # Generate t-SNE and UMAP
    tsne_dir = os.path.join(fold_dir, 'tsne')
    os.makedirs(tsne_dir, exist_ok=True)
    
    print(f"\n  Creating t-SNE...")
    plot_tsne(embeddings, labels,
              f't-SNE: {args.fold} - 2016 Images',
              os.path.join(tsne_dir, f'tsne_image_{args.fold}.png'))
    
    print(f"\n  Creating UMAP...")
    plot_umap(embeddings, labels,
               f'UMAP: {args.fold} - 2016 Images',
               os.path.join(tsne_dir, f'umap_image_{args.fold}.png'))
    
    print(f"\nSaved to: {tsne_dir}")
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()