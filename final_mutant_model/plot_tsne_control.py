#!/usr/bin/env python3
"""
Load saved control embeddings .npz and plot t-SNE colored by 7 groups.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

GROUP_ORDER = ['ACE-1 -ATC', 'ACE-1 +ATC/NC', 'MG1655 -ATC', 'MG1655 +ATC/NC',
               'NC', 'WT NC', 'drug_control']
GROUP_COLORS = {
    'ACE-1 -ATC':     '#1f77b4',
    'ACE-1 +ATC/NC':  '#17becf',
    'MG1655 -ATC':    '#2ca02c',
    'MG1655 +ATC/NC': '#bcbd22',
    'NC':             '#ff7f0e',
    'WT NC':          '#d62728',
    'drug_control':   '#9467bd',
}

if __name__ == '__main__':
    npz_path = sys.argv[1] if len(sys.argv) > 1 else \
        'control/fold_P6/control_embeddings.npz'
    outdir = os.path.dirname(npz_path)

    data = np.load(npz_path, allow_pickle=True)
    embeddings = data['embeddings']
    labels = data['labels']
    groups = data['groups']
    plates = data['plates']

    print(f"Loaded: {npz_path}")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Labels: {len(set(labels))} unique")
    print(f"  Groups: {len(set(groups))} unique: {sorted(set(groups))}")

    tsne = TSNE(n_components=2, perplexity=50, random_state=42, max_iter=1000, init='pca')
    tsne_result = tsne.fit_transform(embeddings)

    # ── 7-group t-SNE ──
    fig, ax = plt.subplots(figsize=(12, 10))
    for group_name in GROUP_ORDER:
        mask = groups == group_name
        if mask.sum() == 0:
            continue
        ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                   c=GROUP_COLORS[group_name], label=group_name,
                   alpha=0.7, s=20, edgecolors='none')
    plate_tag = str(plates[0]) if len(set(plates)) == 1 else 'all'
    ax.set_title(f't-SNE of Center MIL Embeddings — {plate_tag}\n'
                 f'{len(embeddings)} images, 1280-dim → 2D (perp=50)',
                 fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=10, markerscale=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(outdir, 'tsne_7groups.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")

    # ── 41-class t-SNE ──
    fig2, ax2 = plt.subplots(figsize=(14, 12))
    unique_labels = sorted(set(labels))
    label_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax2.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                    c=[label_colors[i]], label=label,
                    alpha=0.6, s=15, edgecolors='none')
    ax2.set_title(f't-SNE of Center MIL Embeddings — {plate_tag} (41 classes)', fontsize=14)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.legend(fontsize=5, markerscale=2, loc='upper left', ncol=2)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path2 = os.path.join(outdir, 'tsne_41classes.png')
    fig2.savefig(out_path2, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"Saved: {out_path2}")

    # ── UMAP 7-group ──
    print("\nRunning UMAP...")
    reducer = umap.UMAP(random_state=42, n_neighbors=30, min_dist=0.3)
    umap_result = reducer.fit_transform(embeddings)

    fig3, ax3 = plt.subplots(figsize=(12, 10))
    for group_name in GROUP_ORDER:
        mask = groups == group_name
        if mask.sum() == 0:
            continue
        ax3.scatter(umap_result[mask, 0], umap_result[mask, 1],
                    c=GROUP_COLORS[group_name], label=group_name,
                    alpha=0.7, s=20, edgecolors='none')
    ax3.set_title(f'UMAP of Center MIL Embeddings — {plate_tag}\n'
                  f'{len(embeddings)} images, 1280-dim → 2D',
                  fontsize=14)
    ax3.set_xlabel('UMAP 1')
    ax3.set_ylabel('UMAP 2')
    ax3.legend(fontsize=10, markerscale=2)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path3 = os.path.join(outdir, 'umap_7groups.png')
    fig3.savefig(out_path3, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig3)
    print(f"Saved: {out_path3}")

    # ── UMAP 41-class ──
    fig4, ax4 = plt.subplots(figsize=(14, 12))
    unique_labels = sorted(set(labels))
    label_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax4.scatter(umap_result[mask, 0], umap_result[mask, 1],
                    c=[label_colors[i]], label=label,
                    alpha=0.6, s=15, edgecolors='none')
    ax4.set_title(f'UMAP of Center MIL Embeddings — {plate_tag} (41 classes)', fontsize=14)
    ax4.set_xlabel('UMAP 1')
    ax4.set_ylabel('UMAP 2')
    ax4.legend(fontsize=5, markerscale=2, loc='upper left', ncol=2)
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path4 = os.path.join(outdir, 'umap_41classes.png')
    fig4.savefig(out_path4, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig4)
    print(f"Saved: {out_path4}")
