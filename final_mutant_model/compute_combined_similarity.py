#!/usr/bin/env python3
"""
Combined 185x185 Similarity Matrix with boxes like dino embeddings:
- 96 Mutants (with guides) + 89 Drugs (with concentrations) = 185 items
- Red boxes around groups of same gene/antibiotic
- White divider line at position 96
- Viridis colormap vmin=0.5, vmax=1.0
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_mean_embeddings(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    unique_labels = sorted(set(labels))
    mean_emb = {}
    for label in unique_labels:
        mask = np.array([l == label for l in labels])
        mean_emb[label] = embeddings[mask].mean(axis=0)
    return mean_emb


def main():
    print("="*60)
    print("Combined 185x185 Similarity Matrix (with boxes)")
    print("="*60)
    
    fold_key = "Plate_1"
    
    # Load embeddings
    print("\nLoading embeddings...")
    mutant_npz = np.load(os.path.join(SCRIPT_DIR, "mutant", f"fold_{fold_key}", f"embeddings_{fold_key}_mil_n3.npz"))
    mutant_emb = mutant_npz['embeddings']
    mutant_labels = mutant_npz['labels']
    
    drug_npz = np.load(os.path.join(SCRIPT_DIR, "drug", f"fold_{fold_key}", f"embeddings_{fold_key}_mil_n3.npz"))
    drug_emb = drug_npz['embeddings']
    drug_labels = drug_npz['labels']
    
    print(f"  Mutants: {len(mutant_emb)}, Drugs: {len(drug_emb)}")
    
    # Compute mean embeddings
    print("Computing mean embeddings...")
    mutant_mean = compute_mean_embeddings(mutant_emb, mutant_labels)
    drug_mean = compute_mean_embeddings(drug_emb, drug_labels)
    
    unique_mutants = sorted(mutant_mean.keys())
    unique_drugs = sorted(drug_mean.keys())
    
    print(f"  Unique mutants: {len(unique_mutants)}")
    print(f"  Unique drugs: {len(unique_drugs)}")
    
    # Combined list (185 items)
    all_items = unique_mutants + unique_drugs
    n_items = len(all_items)
    print(f"  Combined: {n_items}")
    
    # Build embedding matrix
    print("Building embedding matrix...")
    emb_matrix = np.zeros((n_items, next(iter(mutant_mean.values())).shape[0]))
    
    for item in all_items:
        idx = all_items.index(item)
        if item in mutant_mean:
            emb_matrix[idx] = mutant_mean[item]
        else:
            emb_matrix[idx] = drug_mean[item]
    
    # Compute similarity matrix
    print("Computing 185x185 similarity...")
    sim_matrix = np.zeros((n_items, n_items))
    
    for i in tqdm(range(n_items), desc="Similarity"):
        for j in range(n_items):
            if i == j:
                sim_matrix[i, j] = 1.0
            elif i < j:
                sim_matrix[i, j] = cosine_similarity(emb_matrix[i], emb_matrix[j])
                sim_matrix[j, i] = sim_matrix[i, j]
    
    print(f"  Min: {sim_matrix.min():.4f}, Max: {sim_matrix.max():.4f}, Mean: {sim_matrix.mean():.4f}")
    
    # Create DataFrame
    df = pd.DataFrame(sim_matrix, index=all_items, columns=all_items)
    
    # Save CSV
    csv_path = os.path.join(SCRIPT_DIR, "similarity_combined_185x185.csv")
    df.to_csv(csv_path)
    print(f"Saved CSV: {csv_path}")
    
    # Create heatmap with boxes
    print("\nCreating heatmap with boxes...")
    fig, ax = plt.subplots(figsize=(40, 36))
    
    # Viridis colormap - auto scale based on actual data
    sns.heatmap(df, 
                cmap='viridis',
                xticklabels=True, 
                yticklabels=True,
                cbar_kws={'label': 'Cosine Similarity', 'shrink': 0.5, 'pad': 0.02},
                ax=ax,
                linewidths=0,
                linecolor='white')
    
    ax.set_title('Combined Similarity Matrix: 96 Mutants + 89 Drugs (185 × 185)', 
                 fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Entities', fontsize=14)
    ax.set_ylabel('Entities', fontsize=14)
    
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=4, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=4)
    
    # White divider line at position 96 (boundary between mutants and drugs)
    ax.axhline(y=len(unique_mutants)-0.5, color='white', linewidth=3)
    ax.axvline(x=len(unique_mutants)-0.5, color='white', linewidth=3)
    
    # Red boxes around groups of same mutant gene (multiple guides)
    gene_positions = {}
    for i, gene in enumerate(unique_mutants):
        base = gene.rsplit('_', 1)[0] if '_' in gene else gene
        if base not in gene_positions:
            gene_positions[base] = []
        gene_positions[base].append(i)
    
    for base, positions in gene_positions.items():
        if len(positions) > 1:
            min_pos = min(positions)
            max_pos = max(positions)
            if max_pos > min_pos:
                rect = Rectangle((min_pos, min_pos), max_pos - min_pos + 1, max_pos - min_pos + 1,
                                linewidth=2.5, edgecolor='red', facecolor='none', linestyle='-')
                ax.add_patch(rect)
    
    # Red boxes around groups of same antibiotic (different concentrations)
    ab_positions = {}
    for i, ab in enumerate(unique_drugs):
        base = ab.rsplit('_', 1)[0] if '_' in ab else ab
        if base not in ab_positions:
            ab_positions[base] = []
        ab_positions[base].append(i)
    
    for base, positions in ab_positions.items():
        if len(positions) > 1:
            # Offset by number of mutants
            offset = len(unique_mutants)
            start = positions[0] + offset
            end = positions[-1] + offset
            if end > start:
                rect = Rectangle((start, start), end - start + 1, end - start + 1,
                                linewidth=2.5, edgecolor='red', facecolor='none', linestyle='-')
                ax.add_patch(rect)
    
    plt.tight_layout()
    
    # Save PNG
    png_path = os.path.join(SCRIPT_DIR, "similarity_combined_185x185.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PNG: {png_path}")
    
    print("\nDONE!")


if __name__ == '__main__':
    main()