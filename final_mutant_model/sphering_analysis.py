#!/usr/bin/env python3
"""
Control-based sphering to align drug and mutant Plate 1 embeddings.
1. Fit ZCA whitening on pooled control wells (drug 'control' + mutant 'NC'/'WT NC')
2. Apply to all embeddings
3. tSNE before vs after (drug vs mutant overlap)
4. Genes × Drugs cosine similarity heatmap after sphering
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def zca_whiten(X, eps=1e-6):
    """Compute ZCA whitening matrix and apply."""
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = Xc.T @ Xc / (Xc.shape[0] - 1)
    U, S, Vt = np.linalg.svd(cov)
    W = (U @ np.diag(1.0 / np.sqrt(S + eps))) @ U.T
    return W, mean


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def main():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    drug_path = os.path.join(SCRIPT_DIR, 'drug/fold_Plate_1/embeddings_Plate_1_mil_n3.npz')
    mut_path = os.path.join(SCRIPT_DIR, 'mutant/fold_Plate_1/embeddings_Plate_1_mil_n3.npz')

    drug_data = np.load(drug_path)
    mut_data = np.load(mut_path)

    drug_emb, drug_labels = drug_data['embeddings'], drug_data['labels']
    mut_emb, mut_labels = mut_data['embeddings'], mut_data['labels']

    print(f"Drug:  {drug_emb.shape}, {len(set(drug_labels))} classes")
    print(f"Mutant: {mut_emb.shape}, {len(set(mut_labels))} classes")

    # ── 2. Identify control samples ────────────────────────────────────────────
    drug_ctrl_mask = np.array([str(lbl) == 'control' for lbl in drug_labels])
    mut_ctrl_mask = np.array([str(lbl).startswith('NC') or str(lbl).startswith('WT NC') for lbl in mut_labels])

    ctrl_emb = np.vstack([drug_emb[drug_ctrl_mask], mut_emb[mut_ctrl_mask]])
    print(f"Control embeddings for sphering: {ctrl_emb.shape}")

    # ── 3. Fit ZCA on controls, apply to all ──────────────────────────────────
    W, ctrl_mean = zca_whiten(ctrl_emb)

    drug_emb_sphered = (drug_emb - ctrl_mean) @ W
    mut_emb_sphered = (mut_emb - ctrl_mean) @ W

    # Combine for tSNE
    combined_raw = np.vstack([drug_emb, mut_emb])
    combined_sphered = np.vstack([drug_emb_sphered, mut_emb_sphered])
    combined_labels_raw = ['drug'] * len(drug_emb) + ['mutant'] * len(mut_emb)
    combined_labels = list(drug_labels) + list(mut_labels)

    # ── 4. tSNE before vs after ───────────────────────────────────────────────
    print("Running tSNE (raw)...")
    tsne_raw = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000, learning_rate='auto').fit_transform(PCA(n_components=50).fit_transform(combined_raw))
    print("Running tSNE (sphered)...")
    tsne_sphered = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000, learning_rate='auto').fit_transform(PCA(n_components=50).fit_transform(combined_sphered))

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    for i, (tsne, title) in enumerate([(tsne_raw, 'Before Sphering'), (tsne_sphered, 'After Sphering')]):
        # Color by domain (drug vs mutant)
        for domain, color, marker in [('drug', '#1f77b4', 'o'), ('mutant', '#d62728', 'x')]:
            mask = [l == domain for l in combined_labels_raw]
            axes[0, i].scatter(tsne[mask, 0], tsne[mask, 1], c=color, marker=marker,
                               label=domain, s=8, alpha=0.5, linewidths=0)
        axes[0, i].set_title(f't-SNE — {title} (drug vs mutant)', fontsize=13, fontweight='bold')
        axes[0, i].legend(fontsize=11)
        axes[0, i].set_xlabel('t-SNE 1', fontsize=11)
        axes[0, i].set_ylabel('t-SNE 2', fontsize=11)

        # Color by class (genes + drugs)
        unique_classes = sorted(set(combined_labels))
        label_to_int = {l: i for i, l in enumerate(unique_classes)}
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
        if len(unique_classes) > 20:
            colors = plt.cm.gist_ncar(np.linspace(0, 1, len(unique_classes)))
        int_labels = [label_to_int[l] for l in combined_labels]
        sc = axes[1, i].scatter(tsne[:, 0], tsne[:, 1], c=int_labels, cmap='tab20' if len(unique_classes) <= 20 else 'gist_ncar',
                                s=6, alpha=0.6, linewidths=0)
        axes[1, i].set_title(f't-SNE — {title} (by class)', fontsize=13, fontweight='bold')
        axes[1, i].set_xlabel('t-SNE 1', fontsize=11)
        axes[1, i].set_ylabel('t-SNE 2', fontsize=11)

    plt.tight_layout()
    out_tsne = os.path.join(SCRIPT_DIR, 'tsne_drug_mutant_sphering.png')
    plt.savefig(out_tsne, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_tsne}")

    # ── 5. Genes × Drugs cosine similarity heatmap ────────────────────────────
    def get_gene(lbl):
        g = str(lbl).rsplit('_', 1)[0]
        return g if g not in ('NC', 'WT NC') else None

    def get_drug(lbl):
        s = str(lbl)
        return s if s != 'control' else None

    def get_drug_name(lbl):
        return str(lbl).rsplit('_', 1)[0]

    # Compute per-gene centroids (post-sphering)
    gene_centroids = {}
    for lbl in set(mut_labels):
        gene = get_gene(lbl)
        if gene is None:
            continue
        mask = mut_labels == lbl
        if gene not in gene_centroids:
            gene_centroids[gene] = []
        gene_centroids[gene].append(mut_emb_sphered[mask].mean(axis=0))
    gene_centroids = {g: np.mean(vecs, axis=0) for g, vecs in gene_centroids.items()}
    genes = sorted(gene_centroids.keys())
    print(f"Genes: {len(genes)}")

    # Compute per-drug centroids (post-sphering), pooling across concentrations
    drug_centroids = {}
    for lbl in set(drug_labels):
        drug = get_drug(lbl)
        if drug is None:
            continue
        drug_name = get_drug_name(lbl)
        mask = drug_labels == lbl
        if drug_name not in drug_centroids:
            drug_centroids[drug_name] = []
        drug_centroids[drug_name].append(drug_emb_sphered[mask].mean(axis=0))
    # Average across concentrations for each drug
    drug_centroids = {d: np.mean(vecs, axis=0) for d, vecs in drug_centroids.items()}
    drug_names = sorted(drug_centroids.keys())
    print(f"Drugs: {len(drug_names)}")

    # Compute similarity matrix (genes × drugs)
    sim = np.zeros((len(genes), len(drug_names)))
    for i, g in enumerate(genes):
        for j, d in enumerate(drug_names):
            sim[i, j] = cosine_sim(gene_centroids[g], drug_centroids[d])

    print(f"Similarity matrix: {sim.shape}")
    print(f"  Range: [{sim.min():.4f}, {sim.max():.4f}]")

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(max(10, len(drug_names) * 0.45), max(8, len(genes) * 0.45)))
    cmap = plt.cm.RdYlBu_r
    im = ax.imshow(sim, cmap=cmap, vmin=sim.min(), vmax=sim.max(), aspect='auto')

    ax.set_xticks(range(len(drug_names)))
    ax.set_yticks(range(len(genes)))
    ax.set_xticklabels(drug_names, rotation=90, fontsize=7)
    ax.set_yticklabels(genes, fontsize=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity', fontsize=10)

    ax.set_title('Drug × Gene Cosine Similarity (post-sphering)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_heatmap = os.path.join(SCRIPT_DIR, 'genes_x_drugs_cosine_sphering.png')
    plt.savefig(out_heatmap, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_heatmap}")

    print("\nDone!")


if __name__ == '__main__':
    main()
