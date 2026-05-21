#!/usr/bin/env python3
"""
96x96 Cosine Similarity Heatmap – All guide-level classes.
Controls (NC, WT NC) in top-left corner.
Mutant genes sorted alphabetically, guides 1-3 grouped together.
Red boxes around 3x3 intra-gene blocks.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def main():
    fold_key = "Plate_1"
    npz_path = os.path.join(SCRIPT_DIR, "mutant", f"fold_{fold_key}", f"embeddings_{fold_key}_mil_n3.npz")

    print(f"Loading: {npz_path}")
    data = np.load(npz_path)
    embeddings = data['embeddings']
    labels = data['labels']

    # Group by unique guide label (e.g. 'dnaB_1')
    guide_groups = {}
    for i, lbl in enumerate(labels):
        guide_groups.setdefault(lbl, []).append(i)

    print(f"Unique guide classes: {len(guide_groups)}")

    # Build ordered list: controls first, then mutant genes alphabetically
    def sort_key(lbl):
        gene = lbl.rsplit('_', 1)[0]
        guide = int(lbl.rsplit('_', 1)[1])
        if gene == 'NC':
            return (0, 0, guide)
        elif gene == 'WT NC':
            return (0, 1, guide)
        else:
            return (1, gene, guide)

    ordered_labels = sorted(guide_groups.keys(), key=sort_key)
    n = len(ordered_labels)

    # Compute centroids per guide
    centroids = {lbl: np.mean(embeddings[guide_groups[lbl]], axis=0) for lbl in ordered_labels}

    # Build 96x96 similarity matrix
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                idx = guide_groups[ordered_labels[i]]
                if len(idx) < 2:
                    sim[i, j] = np.nan
                else:
                    s = [cosine_sim(embeddings[idx[p]], embeddings[idx[q]])
                         for p in range(len(idx)) for q in range(p + 1, len(idx))]
                    sim[i, j] = np.mean(s)
            else:
                sim[i, j] = cosine_sim(centroids[ordered_labels[i]], centroids[ordered_labels[j]])

    # Print stats
    diag = np.diag(sim)
    off_diag = sim[np.triu_indices(n, k=1)]
    print(f"Mean intra-guide similarity: {np.nanmean(diag):.4f}")
    print(f"Mean inter-guide similarity: {np.nanmean(off_diag):.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    figsize = max(10, n * 0.35)
    fig, ax = plt.subplots(figsize=(figsize, figsize))

    cmap = plt.cm.RdYlBu_r
    im = ax.imshow(sim, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    # Tick labels
    tick_labels = [lbl.rsplit('_', 1)[0] + '_' + lbl.rsplit('_', 1)[1] for lbl in ordered_labels]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=5)
    ax.set_yticklabels(tick_labels, fontsize=5)

    # Red boxes around 3x3 intra-gene blocks (skip controls)
    gene_map = {}
    for idx, lbl in enumerate(ordered_labels):
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

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity', fontsize=10)

    ax.set_title('Cosine Similarity (CRISPRi)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, "cosine_similarity_96x96.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
