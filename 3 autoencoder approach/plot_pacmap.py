#!/usr/bin/env python3
"""PaCMAP of all test latents from extract_test_latents.py — drug=red, mutant=green."""

import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pacmap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latents', type=str, required=True,
                        help='Path to .pt file from extract_test_latents.py')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--n_neighbors', type=int, default=10,
                        help='PaCMAP n_neighbors')
    parser.add_argument('--mn_ratio', type=float, default=0.5,
                        help='PaCMAP MN_ratio')
    parser.add_argument('--fp_ratio', type=float, default=2.0,
                        help='PaCMAP FP_ratio')
    parser.add_argument('--max_points', type=int, default=200000,
                        help='Max points to plot (subsample if larger)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data = torch.load(args.latents, map_location='cpu', weights_only=False)
    records = data['records']

    print(f"Loaded {len(records)} records with {records[0]['latents'].shape[0]} positions each")

    all_vecs, all_colors, all_labels = [], [], []
    for rec in records:
        src = rec['source']
        color = '#E41A1C' if src == 'drug' else '#4DAF4A'
        for pos_z in rec['latents']:
            all_vecs.append(pos_z)
            all_colors.append(color)
            all_labels.append(src)

    X = np.stack(all_vecs)
    colors = np.array(all_colors)
    labels = np.array(all_labels)

    print(f"Total vectors: {X.shape[0]} × {X.shape[1]} dims  ({np.sum(labels=='drug')} drug, {np.sum(labels=='mutant')} mutant)")

    if X.shape[0] > args.max_points:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(X.shape[0], args.max_points, replace=False)
        X = X[idx]
        colors = colors[idx]
        labels = labels[idx]
        print(f"Subsampled to {args.max_points} points")

    print("Running PaCMAP ...")
    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        MN_ratio=args.mn_ratio,
        FP_ratio=args.fp_ratio,
        random_state=args.seed,
        verbose=True,
    )
    emb = reducer.fit_transform(X)
    print("PaCMAP done.")

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.latents))
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for src in ['drug', 'mutant']:
        mask = labels == src
        c = '#E41A1C' if src == 'drug' else '#4DAF4A'
        label = 'Drug' if src == 'drug' else 'Mutant'
        ax.scatter(emb[mask, 0], emb[mask, 1], c=c, s=1, alpha=0.3, label=label, rasterized=True)

    ax.set_xlabel('PaCMAP 1', fontsize=12)
    ax.set_ylabel('PaCMAP 2', fontsize=12)
    ax.set_title(f'PaCMAP of all test latents ({X.shape[0]} points, 32-dim latents)', fontsize=13)
    ax.legend(markerscale=20, fontsize=12)
    ax.tick_params(labelsize=9)
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'pacmap_drug_vs_mutant.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
