#!/usr/bin/env python3
"""
Control-based sphering on BOTH-domain embeddings (fold Plate_1).
1. Loads combined drug+mutant embeddings from both/fold_Plate_1/
2. Separates drug vs mutant by file path
3. Fits ZCA whitening on pooled controls (NC / WT NC from both domains)
4. tSNE before vs after sphering
5. Genes × Drugs cosine similarity heatmap after sphering
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
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = Xc.T @ Xc / (Xc.shape[0] - 1)
    U, S, Vt = np.linalg.svd(cov)
    W = (U @ np.diag(1.0 / np.sqrt(S + eps))) @ U.T
    return W, mean


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def get_gene(lbl):
    g = str(lbl).rsplit('_', 1)[0]
    return g if g not in ('NC', 'WT NC') else None


def main():
    npz_path = os.path.join(SCRIPT_DIR, 'both/fold_Plate_1/embeddings_Plate_1_mil_n3.npz')
    data = np.load(npz_path, allow_pickle=True)

    emb_all = data['embeddings']
    labels_all = np.array([str(l) for l in data['labels']])
    paths_all = data['paths']

    # ── 1. Separate by domain ──────────────────────────────────────────────────
    drug_mask = np.array(['Drugs_Data' in str(p) for p in paths_all])
    mut_mask = np.array(['Mutants_Data' in str(p) for p in paths_all])

    drug_emb = emb_all[drug_mask]
    drug_labels = labels_all[drug_mask]
    mut_emb = emb_all[mut_mask]
    mut_labels = labels_all[mut_mask]

    print(f"Drug:  {drug_emb.shape}, {len(set(drug_labels))} classes")
    print(f"Mutant: {mut_emb.shape}, {len(set(mut_labels))} classes")

    # ── 2. Controls for sphering ──────────────────────────────────────────────
    def is_ctrl(lbl):
        s = str(lbl)
        return s == 'control' or s.startswith('NC') or s.startswith('WT NC')

    drug_ctrl = drug_emb[np.array([is_ctrl(l) for l in drug_labels])]
    mut_ctrl = mut_emb[np.array([is_ctrl(l) for l in mut_labels])]
    ctrl_emb = np.vstack([drug_ctrl, mut_ctrl])
    print(f"Control embeddings: {ctrl_emb.shape} ({len(drug_ctrl)} drug + {len(mut_ctrl)} mutant)")

    # ── 3. Fit ZCA ────────────────────────────────────────────────────────────
    W, ctrl_mean = zca_whiten(ctrl_emb)
    drug_emb_s = (drug_emb - ctrl_mean) @ W
    mut_emb_s = (mut_emb - ctrl_mean) @ W
    all_emb_s = np.vstack([drug_emb_s, mut_emb_s])

    # ── 4. tSNE before vs after ────────────────────────────────────────────────
    combined_raw = np.vstack([drug_emb, mut_emb])
    combined_sphered = all_emb_s
    domain_labels = ['drug'] * len(drug_emb) + ['mutant'] * len(mut_emb)
    class_labels_all = list(drug_labels) + list(mut_labels)

    print("Running tSNE...")
    tsne_raw = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000).fit_transform(
        PCA(n_components=50).fit_transform(combined_raw))
    tsne_sph = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000).fit_transform(
        PCA(n_components=50).fit_transform(combined_sphered))

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    for i, (tsne, title) in enumerate([(tsne_raw, 'Before Sphering'), (tsne_sph, 'After Sphering')]):
        for dom, color, marker in [('drug', '#1f77b4', 'o'), ('mutant', '#d62728', 'x')]:
            mask = [d == dom for d in domain_labels]
            axes[0, i].scatter(tsne[mask, 0], tsne[mask, 1], c=color, marker=marker,
                               label=dom, s=8, alpha=0.5, linewidths=0)
        axes[0, i].set_title(f't-SNE — {title} (drug vs mutant)', fontsize=13, fontweight='bold')
        axes[0, i].legend(fontsize=11)
        axes[0, i].set_xlabel('t-SNE 1', fontsize=11)
        axes[0, i].set_ylabel('t-SNE 2', fontsize=11)

        # by class
        unique_cls = sorted(set(class_labels_all))
        int_map = {c: j for j, c in enumerate(unique_cls)}
        int_lbl = [int_map[c] for c in class_labels_all]
        cmap_name = 'tab20' if len(unique_cls) <= 20 else 'gist_ncar'
        sc = axes[1, i].scatter(tsne[:, 0], tsne[:, 1], c=int_lbl, cmap=cmap_name,
                                s=6, alpha=0.6, linewidths=0)
        axes[1, i].set_title(f't-SNE — {title} (by class)', fontsize=13, fontweight='bold')
        axes[1, i].set_xlabel('t-SNE 1', fontsize=11)
        axes[1, i].set_ylabel('t-SNE 2', fontsize=11)

    plt.tight_layout()
    out_tsne = os.path.join(SCRIPT_DIR, 'tsne_both_sphering.png')
    plt.savefig(out_tsne, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_tsne}")

    # ── 5. Genes × Drugs cosine similarity ────────────────────────────────────
    # Gene centroids from MUTANT domain
    mut_gene_centroids = {}
    for lbl in sorted(set(mut_labels)):
        gene = get_gene(lbl)
        if gene is None:
            continue
        mask = mut_labels == lbl
        vec = mut_emb_s[mask].mean(axis=0)
        mut_gene_centroids.setdefault(gene, []).append(vec)
    mut_gene_centroids = {g: np.mean(vs, axis=0) for g, vs in mut_gene_centroids.items()}
    genes = sorted(mut_gene_centroids.keys())
    print(f"Mutant genes: {len(genes)}")

    # Drug centroids from DRUG domain, average across concentrations
    drug_drug_centroids = {}
    for lbl in sorted(set(drug_labels)):
        lbl_str = str(lbl)
        parts = lbl_str.rsplit('_', 1)
        if len(parts) < 2:
            continue
        name, num = parts
        # Skip controls
        if name in ('NC', 'WT NC'):
            continue
        mask = drug_labels == lbl_str
        vec = drug_emb_s[mask].mean(axis=0)
        drug_drug_centroids.setdefault(name, []).append(vec)
    drug_drug_centroids = {d: np.mean(vs, axis=0) for d, vs in drug_drug_centroids.items()}
    drug_names = sorted(drug_drug_centroids.keys())
    print(f"Drugs: {len(drug_names)}")

    # Build matrix
    sim = np.zeros((len(genes), len(drug_names)))
    for i, g in enumerate(genes):
        for j, d in enumerate(drug_names):
            sim[i, j] = cosine_sim(mut_gene_centroids[g], drug_drug_centroids[d])
    print(f"Similarity matrix: {sim.shape}, range [{sim.min():.4f}, {sim.max():.4f}]")

    fig, ax = plt.subplots(figsize=(max(10, len(drug_names) * 0.45), max(8, len(genes) * 0.45)))
    im = ax.imshow(sim, cmap=plt.cm.RdYlBu_r, vmin=sim.min(), vmax=sim.max(), aspect='auto')
    ax.set_xticks(range(len(drug_names)))
    ax.set_yticks(range(len(genes)))
    ax.set_xticklabels(drug_names, rotation=90, fontsize=7)
    ax.set_yticklabels(genes, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Cosine Similarity', fontsize=10)
    ax.set_title('Drug × Gene Cosine Similarity (both-domain, post-sphering)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_hm = os.path.join(SCRIPT_DIR, 'genes_x_drugs_cosine_sphering_both.png')
    plt.savefig(out_hm, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_hm}")

    print("\nDone!")


if __name__ == '__main__':
    main()
