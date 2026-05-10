#!/usr/bin/env python3
"""
Visualize DANN embeddings with t-SNE colored by domain and class.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='dann_embeddings.csv', help='Input CSV')
    parser.add_argument('--class_mapping', type=str, default='class_mapping.json', help='Class mapping')
    parser.add_argument('--output', type=str, default='dann_tsne.png', help='Output PNG')
    parser.add_argument('--perplexity', type=int, default=30, help='t-SNE perplexity')
    
    args = parser.parse_args()
    
    output_dir = os.path.join(SCRIPT_DIR, 'dann_output')
    input_path = os.path.join(output_dir, args.input)
    mapping_path = os.path.join(output_dir, args.class_mapping)
    output_path = os.path.join(output_dir, args.output)
    
    print(f"Loading embeddings from: {input_path}")
    df = pd.read_csv(input_path)
    
    data_types = df['data_type'].values
    
    embed_cols = [c for c in df.columns if c.startswith('emb_')]
    embeddings = df[embed_cols].values
    
    print(f"Shape: {embeddings.shape}")
    print(f"Drugs: {(data_types == 'drug').sum()}, Mutants: {(data_types == 'mutant').sum()}")
    
    # Standardize
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # t-SNE
    print(f"Running t-SNE (perplexity={args.perplexity})...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, max_iter=1000, random_state=42, init='pca')
    embeddings_2d = tsne.fit_transform(embeddings_scaled)
    
    # Plot 1: Colored by domain
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    drug_mask = data_types == 'drug'
    mutant_mask = data_types == 'mutant'
    
    axes[0].scatter(embeddings_2d[drug_mask, 0], embeddings_2d[drug_mask, 1], 
                    c='red', alpha=0.5, label=f'Drug (n={drug_mask.sum()})', s=20)
    axes[0].scatter(embeddings_2d[mutant_mask, 0], embeddings_2d[mutant_mask, 1], 
                    c='blue', alpha=0.5, label=f'Mutant (n={mutant_mask.sum()})', s=20)
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    axes[0].set_title('DANN Embeddings - Colored by Domain')
    axes[0].legend()
    
    # Plot 2: Scatter by both domains but show class
    # For drugs, get antibiotic class; for mutants, get gene class
    with open(mapping_path, 'r') as f:
        class_to_idx = json.load(f)
    
    # Map indices back to class names (simplified - show first 10 classes)
    unique_classes = sorted(class_to_idx.keys())[:10]
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    
    for i, cls in enumerate(unique_classes):
        mask = (df[embed_cols[0]] == embeddings_array[0]).any()  # Simplified
    
    # Simple plot: show all points
    axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                    c=['red' if dt == 'drug' else 'blue' for dt in data_types],
                    alpha=0.3, s=15)
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].set_title('DANN Embeddings - All Samples')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()