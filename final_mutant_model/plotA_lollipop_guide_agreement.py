#!/usr/bin/env python3
"""
Plot A – Lollipop: Per-gene Guide Agreement
Each stem = mean intra-guide cosine similarity for one gene, sorted ascending.
Red dashed line = inter-gene mean (null baseline).
Blue stems = high concordance; Red stems = low agreement (possible off-target/weak knockdown).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def main():
    fold_key = "Plate_1"
    npz_path = os.path.join(SCRIPT_DIR, "mutant", f"fold_{fold_key}", f"embeddings_{fold_key}_mil_n3.npz")

    print(f"Loading: {npz_path}")
    data = np.load(npz_path)
    emb = data['embeddings']  # (N, 1280)
    labels = data['labels']   # e.g. "dnaB_1", "dnaB_2", ...

    gene_map = {}
    for i, lbl in enumerate(labels):
        gene = lbl.rsplit('_', 1)[0] if '_' in lbl else lbl
        if gene not in gene_map:
            gene_map[gene] = []
        gene_map[gene].append(i)

    genes = sorted(gene_map.keys())
    print(f"Genes: {len(genes)}, samples: {len(labels)}")
    for g in genes:
        print(f"  {g}: {len(gene_map[g])} guides -> indices {gene_map[g]}")

    # ── Intra-guide cosine similarity ──────────────────────────────────────────
    intra_sims = {}
    for gene, indices in gene_map.items():
        n = len(indices)
        if n < 2:
            intra_sims[gene] = np.nan
            continue
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                s = cosine_sim(emb[indices[i]], emb[indices[j]])
                sims.append(s)
        intra_sims[gene] = np.mean(sims)

    # ── Inter-gene mean (null baseline) ───────────────────────────────────────
    gene_centroids = {}
    for gene, indices in gene_map.items():
        gene_centroids[gene] = np.mean(emb[indices], axis=0)

    inter_pairs = []
    gene_list = list(gene_centroids.keys())
    for i in range(len(gene_list)):
        for j in range(i + 1, len(gene_list)):
            s = cosine_sim(gene_centroids[gene_list[i]], gene_centroids[gene_list[j]])
            inter_pairs.append(s)

    inter_gene_mean = np.mean(inter_pairs)
    print(f"\nInter-gene mean similarity: {inter_gene_mean:.4f}")

    # ── Sort by intra-guide similarity ─────────────────────────────────────────
    sorted_genes = sorted(genes, key=lambda g: intra_sims[g])
    y_vals = [intra_sims[g] for g in sorted_genes]
    x_vals = list(range(len(sorted_genes)))

    # Color: blue if above baseline, red if below
    colors = ['#1f77b4' if v >= inter_gene_mean else '#d62728' for v in y_vals]

    # ── Plot ────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, len(sorted_genes) * 0.35), 6))

    ax.hlines(y=inter_gene_mean, xmin=-1, xmax=len(sorted_genes),
              color='red', linewidth=1.5, linestyle='--', zorder=1,
              label=f'Inter-gene mean = {inter_gene_mean:.3f}')

    ax.scatter(x_vals, y_vals, color=colors, s=60, zorder=2, clip_on=False)
    ax.vlines(x_vals, ymin=inter_gene_mean, ymax=y_vals,
              color=colors, linewidth=1.2, alpha=0.7, zorder=1)

    ax.set_xticks(x_vals)
    ax.set_xticklabels(sorted_genes, rotation=90, fontsize=10)
    ax.set_xlabel('Gene (sorted by intra-guide similarity)', fontsize=13)
    ax.set_ylabel('Mean Intra-Guide Cosine Similarity', fontsize=13)
    ax.set_title('Per-Gene-Guide Agreement', fontsize=15, fontweight='bold')
    ax.set_xlim(-1, len(sorted_genes))
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    blue_patch = mpatches.Patch(color='#1f77b4', label='High concordance (reproducible)')
    ax.legend(handles=[blue_patch,
                       plt.Line2D([0], [0], color='red', linewidth=1.5,
                                  linestyle='--', label=f'Inter-gene mean = {inter_gene_mean:.3f}')],
              fontsize=11, loc='upper left')

    plt.tight_layout()
    out_png = os.path.join(SCRIPT_DIR, "plotA_lollipop_guide_agreement.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")

    # ── Flag genes below baseline ──────────────────────────────────────────────
    print("\n── Flagged genes (below inter-gene baseline) ──")
    for gene in sorted_genes:
        if intra_sims[gene] < inter_gene_mean:
            print(f"  {gene}: {intra_sims[gene]:.3f} < {inter_gene_mean:.3f}")
    print("Done.")


if __name__ == '__main__':
    main()