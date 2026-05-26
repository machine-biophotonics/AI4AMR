#!/usr/bin/env python3
"""Compute PaCMAP on ALL test latents (no subsampling) and save the embedding."""

import os, sys, argparse, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import torch
import pacmap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latents', type=str, required=True,
                        help='Path to .pt file from extract_test_latents.py')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--n_neighbors', type=int, default=10)
    parser.add_argument('--mn_ratio', type=float, default=0.5)
    parser.add_argument('--fp_ratio', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data = torch.load(args.latents, map_location='cpu', weights_only=False)
    records = data['records']
    P = records[0]['latents'].shape[0]

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.latents))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {len(records)} records × {P} positions = {len(records) * P} vectors ...")

    all_z, src_types, class_labels = [], [], []
    for rec in records:
        for pos_z in rec['latents']:
            all_z.append(pos_z)
            src_types.append(rec['source'])
            class_labels.append(rec['true_label'])
    Z = np.stack(all_z)
    src_types = np.array(src_types)
    class_labels = np.array(class_labels)
    N = Z.shape[0]
    print(f"  Shape: {Z.shape}, Drug: {(src_types=='drug').sum()}, Mutant: {(src_types=='mutant').sum()}")

    print(f"Running PaCMAP on all {N} points (this may take a while)...")
    t0 = time.time()
    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        MN_ratio=args.mn_ratio,
        FP_ratio=args.fp_ratio,
        random_state=args.seed,
        verbose=True,
    )
    emb = reducer.fit_transform(Z)
    elapsed = time.time() - t0
    print(f"PaCMAP done in {elapsed:.1f}s ({N/elapsed:.0f} pts/s)")

    # Save embedding
    save_path = os.path.join(output_dir, 'pacmap_embedding_all.pt')
    torch.save({
        'embedding': emb,
        'src_types': src_types,
        'class_labels': class_labels,
        'records': records,
    }, save_path)
    print(f"Saved embedding to: {save_path}")

    # Quick plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for src, c, lbl in [('drug', '#E41A1C', 'Drug'), ('mutant', '#4DAF4A', 'Mutant')]:
        mask = src_types == src
        ax.scatter(emb[mask, 0], emb[mask, 1], c=c, s=1, alpha=0.3, label=lbl, rasterized=True)
    ax.set_xlabel('PaCMAP 1', fontsize=12)
    ax.set_ylabel('PaCMAP 2', fontsize=12)
    ax.set_title(f'PaCMAP all {N} points (32-dim latents)', fontsize=13)
    ax.legend(markerscale=20, fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'pacmap_all_drug_vs_mutant.png')
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
