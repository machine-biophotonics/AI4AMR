#!/usr/bin/env python3
"""
Cross-plate gene similarity: for each plate pair, compute N×N gene cosine similarity
and plot heatmap with red 3x3 intra-gene boxes.
"""

import os, csv, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMB_DIR = os.path.join(SCRIPT_DIR, 'multi_head_mutant')
OUT_DIR = os.path.join(EMB_DIR, 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)

all_plates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']


def load_plate_embeddings(plate):
    emb_path = os.path.join(EMB_DIR, f'embeddings_{plate}.npy')
    meta_path = os.path.join(EMB_DIR, f'metadata_{plate}.csv')
    if not os.path.exists(emb_path) or not os.path.exists(meta_path):
        return None, None
    embs = np.load(emb_path)
    df = pd.read_csv(meta_path)
    return embs, df


def get_centroids(embeddings, df, label_col='gene'):
    """Compute mean embedding per unique label (guide-level)."""
    groups = df.groupby(label_col)
    centroids = {}
    for name, idx in groups.indices.items():
        centroids[name] = embeddings[idx].mean(axis=0)
    return centroids


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def sort_gene_key(lbl):
    gene = lbl.rsplit('_', 1)[0]
    guide = int(lbl.rsplit('_', 1)[1])
    if gene == 'NC':
        return (0, 0, guide)
    elif gene == 'WT NC':
        return (0, 1, guide)
    else:
        return (1, gene, guide)


def plot_cross_plate_matrix(centroids_a, centroids_b, label_a, label_b, out_path):
    """N×N cosine similarity heatmap: genes from plate A (rows) vs plate B (cols)."""
    ordered = sorted(centroids_a.keys(), key=sort_gene_key)
    n = len(ordered)
    if n == 0:
        return

    sim = np.zeros((n, n))
    for i, ga in enumerate(ordered):
        for j, gb in enumerate(ordered):
            if ga in centroids_a and gb in centroids_b:
                sim[i, j] = cosine_sim(centroids_a[ga], centroids_b[gb])
            else:
                sim[i, j] = np.nan

    diag = np.diag(sim)
    off_diag = sim[np.triu_indices(n, k=1)]
    print(f"  {label_a} vs {label_b}: mean same-gene sim={np.nanmean(diag):.4f}, "
          f"mean diff-gene sim={np.nanmean(off_diag):.4f}")

    figsize = max(10, n * 0.35)
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    cmap = plt.cm.RdYlBu_r
    im = ax.imshow(sim, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    tick_labels = [lbl for lbl in ordered]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=5)
    ax.set_yticklabels(tick_labels, fontsize=5)

    ax.set_xlabel(f'{label_b} (columns)', fontsize=10)
    ax.set_ylabel(f'{label_a} (rows)', fontsize=10)

    # Red boxes around 3x3 intra-gene blocks
    gene_map = {}
    for idx, lbl in enumerate(ordered):
        gene = lbl.rsplit('_', 1)[0]
        if gene in ('NC', 'WT NC'):
            continue
        gene_map.setdefault(gene, []).append(idx)
    for gene, indices in gene_map.items():
        if len(indices) == 3:
            start = indices[0]
            rect = Rectangle((start - 0.5, start - 0.5), 3, 3,
                             linewidth=2.0, edgecolor='red', facecolor='none', zorder=5)
            ax.add_patch(rect)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity', fontsize=10)
    ax.set_title(f'Gene Similarity: {label_a} vs {label_b}', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    print("Loading embeddings...")
    plate_data = {}
    for p in all_plates:
        embs, df = load_plate_embeddings(p)
        if embs is not None:
            plate_data[p] = (embs, df)
            print(f"  {p}: {len(embs)} images, {df['gene'].nunique()} unique genes")

    if len(plate_data) < 2:
        print("Need at least 2 plates for cross-plate comparison. Run extraction on more plates first.")
        return

    # Compute centroids per plate
    centroids_by_plate = {}
    for p, (embs, df) in plate_data.items():
        centroids_by_plate[p] = get_centroids(embs, df, label_col='gene')

    # Define plate pairs (user-requested + all combinations)
    plate_pairs = [
        ('P1', 'P2'), ('P2', 'P3'), ('P3', 'P4'), ('P4', 'P5'), ('P5', 'P6'),
        ('P1', 'P3'), ('P1', 'P4'), ('P2', 'P4'),
    ]
    existing_pairs = [(a, b) for a, b in plate_pairs if a in centroids_by_plate and b in centroids_by_plate]

    for pa, pb in existing_pairs:
        out_path = os.path.join(OUT_DIR, f'cosine_sim_{pa}_vs_{pb}.png')
        plot_cross_plate_matrix(centroids_by_plate[pa], centroids_by_plate[pb], pa, pb, out_path)

    # Also save summary CSV
    rows = []
    for pa, pb in existing_pairs:
        ca = centroids_by_plate[pa]
        cb = centroids_by_plate[pb]
        common_genes = [g for g in ca if g in cb]
        if not common_genes:
            continue
        sims = [cosine_sim(ca[g], cb[g]) for g in common_genes]
        rows.append({
            'plate_A': pa, 'plate_B': pb,
            'n_genes_shared': len(common_genes),
            'mean_same_gene_sim': np.mean(sims),
            'std_same_gene_sim': np.std(sims),
        })
    if rows:
        summary = pd.DataFrame(rows)
        summary_path = os.path.join(EMB_DIR, 'cross_plate_summary.csv')
        summary.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")
        print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
