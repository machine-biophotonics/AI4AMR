#!/usr/bin/env python3
"""PaCMAP + Wasserstein-2 analysis of all test latents.

Usage:
    python3 plot_pacmap_and_wasserstein.py --latents path/to/test_latents_P1_*.pt
"""

import os, sys, argparse, warnings, re
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import pacmap
from scipy.stats import gaussian_kde
import ot

SEED = 42
np.random.seed(SEED)


def compute_wasserstein_1d(proj_drug, proj_mut, n_bins=50):
    bin_edges = np.linspace(min(proj_drug.min(), proj_mut.min()),
                             max(proj_drug.max(), proj_mut.max()), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    p_d, _ = np.histogram(proj_drug, bins=bin_edges, density=True)
    p_m, _ = np.histogram(proj_mut, bins=bin_edges, density=True)
    p_d /= p_d.sum()
    p_m /= p_m.sum()

    C = (bin_centers[:, None] - bin_centers[None, :]) ** 2
    wd = np.sqrt(ot.emd2(p_d, p_m, C))
    return wd, bin_centers, p_d, p_m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latents', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_points', type=int, default=200000)
    parser.add_argument('--pacmap_neighbors', type=int, default=10)
    args = parser.parse_args()

    data = torch.load(args.latents, map_location='cpu', weights_only=False)
    records = data['records']
    P = records[0]['latents'].shape[0]
    print(f"Loaded {len(records)} records × {P} positions = {len(records) * P} vectors")

    all_z, src_types, all_labels = [], [], []
    for rec in records:
        for pos_z in rec['latents']:
            all_z.append(pos_z)
            src_types.append(rec['source'])
            all_labels.append(rec['true_label'])
    Z = np.stack(all_z)
    src_types = np.array(src_types)
    all_labels = np.array(all_labels)

    N = Z.shape[0]
    is_drug = src_types == 'drug'
    is_mutant = src_types == 'mutant'
    print(f"Drug: {is_drug.sum()}, Mutant: {is_mutant.sum()}, Total: {N}")

    # Subsample if needed
    if N > args.max_points:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(N, args.max_points, replace=False)
        Z = Z[idx]
        is_drug = is_drug[idx]
        is_mutant = is_mutant[idx]
        all_labels = all_labels[idx]
        print(f"Subsampled to {args.max_points}")

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.latents))
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # 1. PaCMAP — drug=red, mutant=green
    # ============================================================
    print("Running PaCMAP ...")
    reducer = pacmap.PaCMAP(
        n_components=2, n_neighbors=args.pacmap_neighbors,
        MN_ratio=0.5, FP_ratio=2.0, random_state=SEED, verbose=False,
    )
    emb = reducer.fit_transform(Z)
    print("PaCMAP done.")

    colors = np.where(is_drug, '#E41A1C', '#4DAF4A')

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for src, c, lbl in [('drug', '#E41A1C', 'Drug'), ('mutant', '#4DAF4A', 'Mutant')]:
        mask = (src == 'drug') == is_drug
        ax.scatter(emb[mask, 0], emb[mask, 1], c=c, s=1, alpha=0.3, label=lbl, rasterized=True)
    ax.set_xlabel('PaCMAP 1', fontsize=12)
    ax.set_ylabel('PaCMAP 2', fontsize=12)
    ax.set_title(f'PaCMAP of all test latents (32-dim z, {Z.shape[0]} points)', fontsize=13)
    ax.legend(markerscale=20, fontsize=12)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    pacmap_path = os.path.join(output_dir, 'pacmap_drug_vs_mutant.png')
    fig.savefig(pacmap_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {pacmap_path}")
    plt.close(fig)

    # ============================================================
    # 2. Wasserstein-2 in PaCMAP space (1D projection on drug→mutant axis)
    # ============================================================
    drug_cent = emb[is_drug].mean(axis=0)
    mut_cent = emb[is_mutant].mean(axis=0)
    axis_vec = mut_cent - drug_cent
    axis_vec = axis_vec / np.linalg.norm(axis_vec)
    proj = emb @ axis_vec

    wd, bc, p_d, p_m = compute_wasserstein_1d(proj[is_drug], proj[is_mutant])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    bw = bc[1] - bc[0]
    ax.bar(bc, p_d, width=bw, alpha=0.6, color='#E41A1C', label=f'Drug')
    ax.bar(bc, p_m, width=bw, alpha=0.6, color='#4DAF4A', label=f'Mutant')
    ax.set_xlabel('Projection onto drug → mutant axis (PaCMAP)', fontsize=12)
    ax.set_ylabel('Probability mass', fontsize=12)
    ax.set_title(f'1D Wasserstein-2 = {wd:.4f}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    plt.tight_layout()
    wd_path = os.path.join(output_dir, 'wasserstein_1d_drug_vs_mutant.png')
    fig.savefig(wd_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {wd_path}")
    plt.close(fig)

    # ============================================================
    # 3. Ciprofloxacin dose-response vs gyrB_1 in PaCMAP space
    # ============================================================
    cipro_mask = np.array([('ciprofloxacin' in lbl.lower()) for lbl in all_labels])
    gyrb_mask = np.array([('gyrb_1' in lbl.lower()) for lbl in all_labels])

    if cipro_mask.sum() > 0 and gyrb_mask.sum() > 0:
        # Group by concentration
        conc_groups = {}
        for i in range(len(all_labels)):
            if cipro_mask[i]:
                lbl = all_labels[i]
                parts = lbl.split('_')
                conc = parts[-1]
                conc_groups.setdefault(conc, []).append(proj[i])
        concs = sorted(conc_groups.keys(), key=lambda c: float(c.replace('x', '')))
        proj_gyrb = proj[gyrb_mask]

        colors_cipro = ['#2166AC', '#4393C3', '#D6604D', '#B2182B']

        fig, axes = plt.subplots(1, len(concs), figsize=(5 * len(concs), 4), squeeze=False)
        for idx, conc in enumerate(concs):
            ax = axes[0, idx]
            proj_c = np.array(conc_groups[conc])

            # KDE for Cipro
            kde_c = gaussian_kde(proj_c)
            x_grid = np.linspace(proj.min(), proj.max(), 200)
            ax.plot(x_grid, kde_c(x_grid), color=colors_cipro[idx % len(colors_cipro)],
                    linewidth=2, label=f'Cipro {conc}')

            # KDE for gyrB_1
            kde_g = gaussian_kde(proj_gyrb)
            ax.plot(x_grid, kde_g(x_grid), color='#E41A1C', linewidth=2,
                    linestyle='--', label='gyrB_1')

            wd_c = float(
                compute_wasserstein_1d(proj_c, proj_gyrb)[0]
            )
            ax.set_title(f'Cipro {conc}\nWD={wd_c:.3f}', fontsize=11)
            ax.set_xlabel('1D projection')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)

        plt.suptitle('Ciprofloxacin dose-response vs gyrB_1 (PaCMAP 1D)', fontsize=13, y=1.02)
        plt.tight_layout()
        cipro_path = os.path.join(output_dir, 'wasserstein_cipro_vs_gyrb.png')
        fig.savefig(cipro_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {cipro_path}")
        plt.close(fig)

        # Print WD values
        print("\nCiprofloxacin vs gyrB_1 Wasserstein-2 distances:")
        for idx, conc in enumerate(concs):
            wd_c = float(compute_wasserstein_1d(np.array(conc_groups[conc]), proj_gyrb)[0])
            print(f"  Cipro {conc} vs gyrB_1: WD = {wd_c:.4f}")
    else:
        print(f"Cipro mask: {cipro_mask.sum()}, gyrB mask: {gyrb_mask.sum()} — skipping dose-response")

    print("\nAll plots saved to:", output_dir)


if __name__ == '__main__':
    main()
