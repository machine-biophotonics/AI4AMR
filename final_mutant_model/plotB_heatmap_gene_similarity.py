#!/usr/bin/env python3
"""
Plot B – Heatmap: Gene × Gene Similarity (Diagonal = Intra-gene)
Generates both:
  1. 28x28 heatmap (28 mutant genes, excludes NC/WT NC)
  2. 30x30 heatmap (all 30 genes including NC/WT NC)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist, squareform

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def cosine_dist(a, b):
    return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def compute_matrix(gene_indices, embeddings):
    genes = sorted(gene_indices.keys())
    n = len(genes)

    centroids = {g: np.mean(embeddings[gene_indices[g]], axis=0) for g in genes}
    gene_list = list(centroids.keys())

    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                idx = gene_indices[gene_list[i]]
                if len(idx) < 2:
                    sim[i, j] = np.nan
                else:
                    s = [np.dot(embeddings[idx[p]], embeddings[idx[q]]) /
                         (np.linalg.norm(embeddings[idx[p]]) * np.linalg.norm(embeddings[idx[q]]) + 1e-8)
                         for p in range(len(idx)) for q in range(p + 1, len(idx))]
                    sim[i, j] = np.mean(s)
            else:
                sim[i, j] = np.dot(centroids[gene_list[i]], centroids[gene_list[j]]) / \
                            (np.linalg.norm(centroids[gene_list[i]]) *
                             np.linalg.norm(centroids[gene_list[j]]) + 1e-8)
    return sim, gene_list


def plot_heatmap(sim, gene_list, title, out_path, include_diag=True):
    n = len(gene_list)

    # Hierarchical clustering on off-diagonal cosine distances
    centroids = np.array([np.mean(embeddings[gene_indices[g]], axis=0) for g in gene_list])
    dist_mat = squareform(pdist(centroids, metric='cosine'))
    dist_vec = pdist(centroids, metric='cosine')
    Z = linkage(dist_vec, method='average')
    order = leaves_list(Z)

    sim_reorder = sim[order][:, order]
    genes_ordered = [gene_list[i] for i in order]

    # Scalar stats
    if include_diag:
        diag_vals = np.diag(sim_reorder)
        off_diag_upper = sim_reorder[np.triu_indices(n, k=1)]
        mu_intra = np.nanmean(diag_vals)
        mu_inter = np.nanmean(off_diag_upper)
        sep = mu_intra / mu_inter if mu_inter > 0 else np.nan
    else:
        mu_inter = np.nanmean(sim_reorder)
        mu_intra = np.nan
        sep = np.nan

    # ── Figure setup ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(max(9, n * 0.55), max(7, n * 0.55)))
    fig.suptitle('Gene x Gene Similarity', fontsize=16, fontweight='bold', y=0.97)

    # Heatmap
    ax_heat = fig.add_axes([0.12, 0.12, 0.80, 0.76])

    vmax = max(abs(sim_reorder[~np.isnan(sim_reorder)].min()),
               abs(sim_reorder[~np.isnan(sim_reorder)].max()))
    # Use a perceptually uniform diverging colormap centered at 0
    cmap = plt.cm.RdYlBu_r  # warm = high similarity, cool = low

    im = ax_heat.imshow(sim_reorder, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    # Cell annotations (subset to avoid clutter)
    for i in range(n):
        for j in range(n):
            v = sim_reorder[i, j]
            if not np.isnan(v):
                txt = ax_heat.text(j, i, f'{v:.2f}',
                                   ha='center', va='center', fontsize=7,
                                   color='white' if v > 0.6 else 'black')
    ax_heat.set_xticks(range(n))
    ax_heat.set_yticks(range(n))
    ax_heat.set_xticklabels(genes_ordered, rotation=90, fontsize=10)
    ax_heat.set_yticklabels(genes_ordered, fontsize=10)
    ax_heat.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)

    # Colorbar
    cb_ax = fig.add_axes([0.93, 0.12, 0.015, 0.76])
    cbar = plt.colorbar(im, cax=cb_ax)
    cbar.set_label('Cosine Similarity', fontsize=12)

    # Stats box
    ax_stats = fig.add_axes([0.12, 0.01, 0.75, 0.09])
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)
    ax_stats.axis('off')

    if include_diag:
        stats_text = (
            f'  µ intra-gene = {mu_intra:.4f}        '
            f'  µ inter-gene = {mu_inter:.4f}        '
            f'  Separation ratio = {sep:.2f}'
        )
    else:
        stats_text = f'  µ inter-gene = {mu_inter:.4f}'

    ax_stats.text(0.5, 0.55, stats_text, transform=ax_stats.transAxes,
                  ha='center', va='center', fontsize=12,
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                            edgecolor='gray', alpha=0.9))

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ── Load data ──────────────────────────────────────────────────────────────────
npz_path = os.path.join(SCRIPT_DIR, "mutant", "fold_Plate_1", "embeddings_Plate_1_mil_n3.npz")
data = np.load(npz_path, allow_pickle=True)
embeddings = data['embeddings']
labels = data['labels']

gene_indices = {}
for i, lbl in enumerate(labels):
    gene = lbl.rsplit('_', 1)[0] if '_' in lbl else lbl
    gene_indices.setdefault(gene, []).append(i)

print(f"Total samples: {len(labels)}, genes: {len(gene_indices)}")

# ── 28x28: exclude NC and WT NC ───────────────────────────────────────────────
genes_28 = sorted([g for g in gene_indices if g not in ('NC', 'WT NC')])
gene_idx_28 = {g: gene_indices[g] for g in genes_28}

print(f"\n28-gene matrix (excludes NC, WT NC)")
sim_28, gl_28 = compute_matrix(gene_idx_28, embeddings)
plot_heatmap(sim_28, gl_28,
             'Plot B – Gene × Gene Similarity (28 Mutant Genes, Hierarchical Clustering)',
             os.path.join(SCRIPT_DIR, "plotB_heatmap_28.png"))

# ── 30x30: all genes including NC and WT NC ───────────────────────────────────
genes_30 = sorted(gene_indices.keys())
gene_idx_30 = {g: gene_indices[g] for g in genes_30}

print(f"\n30-gene matrix (all genes)")
sim_30, gl_30 = compute_matrix(gene_idx_30, embeddings)
plot_heatmap(sim_30, gl_30,
             'Plot B – Gene × Gene Similarity (30 Genes, Hierarchical Clustering)',
             os.path.join(SCRIPT_DIR, "plotB_heatmap_30.png"))

print("\nDone.")