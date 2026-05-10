#!/usr/bin/env python3
"""
Generate t-SNE plot for self-supervised embeddings.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='embeddings_P1.csv', help='Input CSV')
    parser.add_argument('--output', type=str, default='tsne_plot.png', help='Output PNG')
    parser.add_argument('--perplexity', type=int, default=30, help='t-SNE perplexity')
    parser.add_argument('--n_iter', type=int, default=1000, help='t-SNE iterations')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, 'self_supervised_trial', args.input)
    output_path = os.path.join(script_dir, 'self_supervised_trial', args.output)
    
    print(f"Loading embeddings from: {input_path}")
    df = pd.read_csv(input_path)
    
    # Get data type column
    data_types = df['data_type'].values
    
    # Get embedding columns
    embed_cols = [c for c in df.columns if c.startswith('emb_')]
    embeddings = df[embed_cols].values
    
    print(f"Shape: {embeddings.shape}")
    print(f"Drugs: {(data_types == 'drug').sum()}, Mutants: {(data_types == 'mutant').sum()}")
    
    # Standardize embeddings
    print("Standardizing embeddings...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Run t-SNE
    print(f"Running t-SNE (perplexity={args.perplexity}, n_iter={args.n_iter})...")
    tsne = TSNE(
        n_components=2, 
        perplexity=args.perplexity, 
        max_iter=args.n_iter,
        random_state=args.random_state,
        init='pca'
    )
    embeddings_2d = tsne.fit_transform(embeddings_scaled)
    
    # Plot
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Mask for drugs and mutants
    drug_mask = data_types == 'drug'
    mutant_mask = data_types == 'mutant'
    
    ax.scatter(embeddings_2d[mutant_mask, 0], embeddings_2d[mutant_mask, 1], 
               c='blue', alpha=0.5, label=f'Mutant (n={mutant_mask.sum()})', s=20)
    ax.scatter(embeddings_2d[drug_mask, 0], embeddings_2d[drug_mask, 1], 
               c='red', alpha=0.5, label=f'Drug (n={drug_mask.sum()})', s=20)
    
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title(f't-SNE of Self-Supervised Embeddings (Perplexity={args.perplexity})', fontsize=14)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()