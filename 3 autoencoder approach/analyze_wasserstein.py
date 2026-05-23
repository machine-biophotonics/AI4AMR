#!/usr/bin/env python3
"""
Analyze mutant-drug matching using t-SNE + KDE contour plots.
Part A: Gyrase dose-response visualization — Ciprofloxacin at 4 concentrations
overlaid with gyrB_1 mutant.

Usage:
    python3 analyze_wasserstein.py --latents mil_vae_both/fold_P1/test_latents_P1_*.pt
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde


def compute_kde_on_grid(x, y, grid_size=200, bw_method=None):
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=bw_method)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xpad = (xmax - xmin) * 0.1
    ypad = (ymax - ymin) * 0.1
    xi = np.linspace(xmin - xpad, xmax + xpad, grid_size)
    yi = np.linspace(ymin - ypad, ymax + ypad, grid_size)
    XX, YY = np.meshgrid(xi, yi)
    positions = np.vstack([XX.ravel(), YY.ravel()])
    Z = kde(positions).reshape(grid_size, grid_size)
    return XX, YY, Z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latents', type=str, required=True,
                        help='Path to .pt file from extract_test_latents.py')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as latents file)')
    parser.add_argument('--drug_name', type=str, default='Ciprofloxacin',
                        help='Antibiotic name to plot')
    parser.add_argument('--mutant_name', type=str, default='gyrB_1',
                        help='Mutant ID to compare against')
    parser.add_argument('--perplexity', type=int, default=40,
                        help='t-SNE perplexity')
    parser.add_argument('--tsne_seed', type=int, default=42,
                        help='Random seed for t-SNE')
    parser.add_argument('--grid_size', type=int, default=150,
                        help='KDE contour grid resolution')
    parser.add_argument('--max_points', type=int, default=5000,
                        help='Max points per condition for t-SNE (subsample if larger)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading latents from: {args.latents}")
    data = torch.load(args.latents, map_location='cpu', weights_only=False)
    records = data['records']
    idx_to_class = data.get('idx_to_class', {})
    if isinstance(idx_to_class, dict):
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}

    print(f"Loaded {len(records)} image records")
    print(f"  args: test_plate={data.get('args', {}).get('test_plate', '?')}, "
          f"num_positions={data.get('num_positions', '?')}")

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.latents))
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # Group latents by composite_key (well-level)
    # Store both composite_key and true_label for matching
    # ---------------------------------------------------------------
    well_groups = defaultdict(lambda: {'latents': [], 'true_label': None})
    for rec in records:
        key = rec['composite_key']
        well_groups[key]['latents'].append(rec['latents'])
        if well_groups[key]['true_label'] is None:
            well_groups[key]['true_label'] = rec['true_label']

    drug_wells = {}
    mutant_wells = {}
    for key, info in well_groups.items():
        source = key.split('_')[0]
        entry = {'label': info['true_label'], 'latents_list': info['latents']}
        if source == 'drug':
            drug_wells[key] = entry
        else:
            mutant_wells[key] = entry

    # Find target drug wells by true_label
    drug_name_lower = args.drug_name.lower()
    target_drug_wells = []
    for key, info in drug_wells.items():
        lbl = info['label'].lower()
        if drug_name_lower in lbl:
            parts = info['label'].split('_')
            conc = parts[-1] if len(parts) > 1 else 'unknown'
            target_drug_wells.append((conc, key, info))

    # Find target mutant wells by true_label
    mutant_name_lower = args.mutant_name.lower()
    target_mutant_wells = []
    for key, info in mutant_wells.items():
        lbl = info['label'].lower()
        if mutant_name_lower in lbl:
            target_mutant_wells.append((key, info))

    if len(target_drug_wells) == 0:
        print(f"ERROR: No wells found for drug '{args.drug_name}'")
        print(f"  Available drug labels: {sorted(set(d['label'] for d in drug_wells.values()))}")
        return
    if len(target_mutant_wells) == 0:
        print(f"ERROR: No wells found for mutant '{args.mutant_name}'")
        print(f"  Available mutant labels: {sorted(set(d['label'] for d in mutant_wells.values()))}")
        return

    # Sort drug wells by concentration order
    def conc_key(c):
        c = c.lower().replace('x', '')
        try:
            return float(c)
        except:
            return 999
    target_drug_wells.sort(key=lambda x: conc_key(x[0]))

    print(f"\nFound {len(target_drug_wells)} drug concentrations:")
    for conc, key, info in target_drug_wells:
        n_images = len(info['latents_list'])
        n_pos = info['latents_list'][0].shape[0] if n_images > 0 else 0
        n_total = n_images * n_pos
        print(f"  {conc:>8}  ({key})  {n_images} images × {n_pos} positions = {n_total} vectors")

    print(f"\nFound {len(target_mutant_wells)} mutant wells:")
    for key, info in target_mutant_wells:
        n_images = len(info['latents_list'])
        n_pos = info['latents_list'][0].shape[0] if n_images > 0 else 0
        n_total = n_images * n_pos
        print(f"  {info['label']}  ({key})  {n_images} images × {n_pos} positions = {n_total} vectors")

    # ---------------------------------------------------------------
    # Concatenate all vectors and subsample if needed
    # ---------------------------------------------------------------
    all_vectors = []
    all_labels = []
    all_groups = []

    mutant_latents_all = []
    for key, info in target_mutant_wells:
        for lat in info['latents_list']:
            mutant_latents_all.append(lat)
    if len(mutant_latents_all) > 0:
        mutant_vectors = np.concatenate(mutant_latents_all, axis=0)
        if mutant_vectors.shape[0] > args.max_points:
            idx = np.random.RandomState(args.tsne_seed).choice(
                mutant_vectors.shape[0], args.max_points, replace=False)
            mutant_vectors = mutant_vectors[idx]
        all_vectors.append(mutant_vectors)
        all_labels.extend([f'gyrB_1'] * mutant_vectors.shape[0])
        all_groups.extend(['mutant'] * mutant_vectors.shape[0])

    drug_vectors_by_conc = {}
    for conc, key, info in target_drug_wells:
        vecs = np.concatenate(info['latents_list'], axis=0)
        if vecs.shape[0] > args.max_points:
            idx = np.random.RandomState(args.tsne_seed).choice(
                vecs.shape[0], args.max_points, replace=False)
            vecs = vecs[idx]
        drug_vectors_by_conc[conc] = vecs
        all_vectors.append(vecs)
        all_labels.extend([f'Cipro {conc}'] * vecs.shape[0])
        all_groups.extend(['drug'] * vecs.shape[0])

    X = np.concatenate(all_vectors, axis=0)
    print(f"\nCombined vectors for t-SNE: {X.shape[0]} points × {X.shape[1]} dims")

    # ---------------------------------------------------------------
    # Run t-SNE
    # ---------------------------------------------------------------
    print("Running t-SNE (this may take a while)...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(args.perplexity, X.shape[0] // 3 - 1),
        random_state=args.tsne_seed,
        max_iter=1000,
        verbose=1,
    )
    emb = tsne.fit_transform(X)
    print("t-SNE done.")

    # ---------------------------------------------------------------
    # Build label-to-embedding mapping for plotting
    # ---------------------------------------------------------------
    label_to_emb = {}
    start = 0
    for i, label in enumerate(set(all_labels)):
        # Actually, reconstruct from the order
        pass

    # Reconstruct by group
    group_ranges = {}
    start = 0
    group_ranges['mutant'] = (start, start + mutant_vectors.shape[0])
    start += mutant_vectors.shape[0]
    for conc, vecs in drug_vectors_by_conc.items():
        group_ranges[f'drug_{conc}'] = (start, start + vecs.shape[0])
        start += vecs.shape[0]

    labels_unique = ['gyrB_1'] + [f'Cipro {c}' for c in drug_vectors_by_conc.keys()]
    conc_labels = list(drug_vectors_by_conc.keys())

    # ---------------------------------------------------------------
    # Create 2x2 t-SNE contour plot
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    mutant_start, mutant_end = group_ranges['mutant']
    emb_mutant = emb[mutant_start:mutant_end]

    # Contour color map
    drug_colors = ['#2166AC', '#4393C3', '#D6604D', '#B2182B']
    drug_fill_colors = ['#E0EFFF', '#C8E0F5', '#FDE0D0', '#F5C8C8']

    for idx, (conc, vecs) in enumerate(drug_vectors_by_conc.items()):
        ax = axes[idx]
        start = group_ranges[f'drug_{conc}'][0]
        end = group_ranges[f'drug_{conc}'][1]
        emb_drug = emb[start:end]

        # Compute KDE for drug
        XX_d, YY_d, Z_d = compute_kde_on_grid(
            emb_drug[:, 0], emb_drug[:, 1],
            grid_size=args.grid_size
        )

        # Compute KDE for mutant
        XX_m, YY_m, Z_m = compute_kde_on_grid(
            emb_mutant[:, 0], emb_mutant[:, 1],
            grid_size=args.grid_size
        )

        # Normalize both to same scale for comparison
        Z_d = Z_d / Z_d.max()
        Z_m = Z_m / Z_m.max()

        # Plot drug filled contour
        ax.contourf(XX_d, YY_d, Z_d, levels=10, cmap='Blues', alpha=0.35)
        drug_contour = ax.contour(
            XX_d, YY_d, Z_d, levels=5,
            colors=[drug_colors[idx]], linewidths=1.5, alpha=0.9
        )

        # Plot mutant filled contour
        ax.contourf(XX_m, YY_m, Z_m, levels=10, cmap='Reds', alpha=0.2)
        mutant_contour = ax.contour(
            XX_m, YY_m, Z_m, levels=5,
            colors=['#E41A1C'], linewidths=1.8, alpha=0.85, linestyles='dashed'
        )

        # Scatter a subset of points for texture
        n_scatter = min(500, emb_drug.shape[0])
        rng = np.random.RandomState(42 + idx)
        scatter_idx = rng.choice(emb_drug.shape[0], n_scatter, replace=False)
        ax.scatter(
            emb_drug[scatter_idx, 0], emb_drug[scatter_idx, 1],
            c=[drug_colors[idx]], s=8, alpha=0.25, edgecolors='none'
        )

        n_mut_scatter = min(500, emb_mutant.shape[0])
        mut_scatter_idx = rng.choice(emb_mutant.shape[0], n_mut_scatter, replace=False)
        ax.scatter(
            emb_mutant[mut_scatter_idx, 0], emb_mutant[mut_scatter_idx, 1],
            c=['#E41A1C'], s=8, alpha=0.15, edgecolors='none', marker='s'
        )

        # Labels and title
        ax.set_xlabel('t-SNE 1', fontsize=10)
        ax.set_ylabel('t-SNE 2', fontsize=10)
        ax.set_title(f'Ciprofloxacin {conc}  vs  gyrB_1', fontsize=13, fontweight='bold')
        ax.tick_params(labelsize=8)
        ax.set_aspect('equal')

        # Custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=drug_colors[idx], lw=2, label=f'Cipro {conc}'),
            Line2D([0], [0], color='#E41A1C', lw=2, linestyle='dashed', label='gyrB_1'),
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=9, framealpha=0.8)

    plt.suptitle(
        f't-SNE of Ciprofloxacin dose-response vs gyrB_1 mutant\n'
        f'(perplexity={args.perplexity}, {X.shape[0]} points)',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'tsne_gyrase_dose_response.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    plt.close(fig)

    # ---------------------------------------------------------------
    # Also save: overlay of all concentrations in one plot
    # ---------------------------------------------------------------
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    cmap = plt.cm.Blues
    norm = plt.Normalize(0.25, 2.0)

    mutant_contour_plotted = False
    for idx, (conc, vecs) in enumerate(drug_vectors_by_conc.items()):
        start = group_ranges[f'drug_{conc}'][0]
        end = group_ranges[f'drug_{conc}'][1]
        emb_drug = emb[start:end]

        XX_d, YY_d, Z_d = compute_kde_on_grid(
            emb_drug[:, 0], emb_drug[:, 1],
            grid_size=args.grid_size
        )
        Z_d = Z_d / Z_d.max()

        conc_val = float(conc.replace('x', ''))
        color = cmap(norm(conc_val))

        ax2.contour(
            XX_d, YY_d, Z_d, levels=[0.5],
            colors=[color], linewidths=2.0, alpha=0.85,
        )
        ax2.contourf(
            XX_d, YY_d, Z_d, levels=[0.0, 0.5],
            colors=[color], alpha=0.15,
        )
        ax2.plot([], [], color=color, lw=2, label=f'Cipro {conc}')

    # Add mutant
    XX_m, YY_m, Z_m = compute_kde_on_grid(
        emb_mutant[:, 0], emb_mutant[:, 1],
        grid_size=args.grid_size
    )
    Z_m = Z_m / Z_m.max()
    ax2.contour(
        XX_m, YY_m, Z_m, levels=[0.5],
        colors=['#E41A1C'], linewidths=2.5, alpha=0.9, linestyles='dashed'
    )
    ax2.contourf(
        XX_m, YY_m, Z_m, levels=[0.0, 0.5],
        colors=['#E41A1C'], alpha=0.1,
    )
    ax2.plot([], [], color='#E41A1C', lw=2.5, linestyle='dashed', label='gyrB_1')

    ax2.set_xlabel('t-SNE 1', fontsize=11)
    ax2.set_ylabel('t-SNE 2', fontsize=11)
    ax2.set_title('Ciprofloxacin dose gradient vs gyrB_1', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10, framealpha=0.8)
    ax2.set_aspect('equal')
    ax2.tick_params(labelsize=9)

    plt.tight_layout()
    overlay_path = os.path.join(output_dir, 'tsne_gyrase_overlay.png')
    fig2.savefig(overlay_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {overlay_path}")
    plt.close(fig2)

    print("\nDone!")


if __name__ == '__main__':
    main()
