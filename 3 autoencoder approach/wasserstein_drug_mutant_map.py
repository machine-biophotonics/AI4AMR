#!/usr/bin/env python3
"""OT alignment + Wasserstein drug-mutant matching.

Aligns drug → mutant space via Sinkhorn barycentric projection,
then computes 1D Wasserstein distances between aligned drug classes
and mutant classes. Produces alignment visualizations + match heatmap.

Usage:
    python3 wasserstein_drug_mutant_map.py --pacmap fold_P1/pacmap_embedding_all.pt
"""

import os, sys, argparse, warnings, re
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import ot
from collections import defaultdict
import csv

SEED = 42
np.random.seed(SEED)

MOA_GROUPS = {
    'Fluoroquinolones': {'drugs': ['ciprofloxacin', 'levofloxacin', 'norfloxacin'],
                         'targets': ['gyra', 'gyrb', 'parc', 'pare']},
    'Rifamycins': {'drugs': ['rifampicin'], 'targets': ['rpoa', 'rpob']},
    'Folate_inhibitors': {'drugs': ['trimethoprim'], 'targets': ['fola', 'folp']},
    'Ribosome_50S': {'drugs': ['chloramphenicol', 'clarithromycin'], 'targets': ['rpla', 'rplc']},
    'Ribosome_30S': {'drugs': ['doxicyclin', 'kanamycin'], 'targets': ['rpsa', 'rpsl']},
    'Penems': {'drugs': ['penicillin', 'mecillinam', 'meropenem'],
               'targets': ['mrca', 'mrcb', 'mrda', 'ftsi']},
    'Cephalosporins': {'drugs': ['cefepim', 'cefsulodin', 'ceftriaxone'],
                       'targets': ['ftsi', 'mrca', 'mrcb', 'mrda']},
    'Polymyxins': {'drugs': ['polymyxin_b', 'colistin'],
                   'targets': ['lpxa', 'lpxc', 'lpta', 'lptc', 'msba']},
}


def get_drug_base(name):
    m = re.match(r'^(.+)_(\d+(?:\.\d+)?x)$', name)
    return m.group(1).lower() if m else name.lower()

def is_drug_class(name):
    return bool(re.match(r'.+_\d+(?:\.\d+)?x$', name)) or name == 'control'

def get_concentration(name):
    m = re.search(r'_(\d+(?:\.\d+)?x)$', name)
    return m.group(1) if m else None

def short_drug(lbl):
    if lbl == 'control': return 'control'
    base = get_drug_base(lbl).replace('_', ' ')
    conc = get_concentration(lbl) or ''
    return f"{base} {conc}"

def short_mutant(lbl):
    return lbl.replace('_', ' ')

def wasserstein_1d(a, b):
    """1D Wasserstein-2 via quantile interpolation. Fast: O(N log N)."""
    a_s = np.sort(a)
    b_s = np.sort(b)
    n = max(len(a_s), len(b_s))
    t = np.linspace(0, 1, n)
    q_a = np.interp(t, np.linspace(0, 1, len(a_s)), a_s)
    q_b = np.interp(t, np.linspace(0, 1, len(b_s)), b_s)
    return float(np.sqrt(np.mean((q_a - q_b) ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pacmap', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--grid', type=int, default=30, help='2D histogram grid')
    parser.add_argument('--align_reg', type=float, default=0.05,
                        help='Sinkhorn reg for alignment')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.pacmap))
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("OT Domain Alignment: Drug → Mutant")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 1. Load
    # ---------------------------------------------------------------
    print("\n[1/5] Loading PaCMAP embedding ...")
    pt = torch.load(args.pacmap, map_location='cpu', weights_only=False)
    emb = pt['embedding'].astype(np.float64)
    src_types = pt['src_types']
    class_labels = pt['class_labels']
    N = emb.shape[0]
    drug_mask = src_types == 'drug'
    mut_mask = src_types == 'mutant'
    print(f"  {N} points ({drug_mask.sum()} drug, {mut_mask.sum()} mutant)")

    # ---------------------------------------------------------------
    # 2. Build 2D histograms + Sinkhorn transport plan
    # ---------------------------------------------------------------
    print("\n[2/5] Computing Sinkhorn transport plan drug → mutant ...")
    G = args.grid
    xmin, xmax = emb[:, 0].min(), emb[:, 0].max()
    ymin, ymax = emb[:, 1].min(), emb[:, 1].max()
    xpad = (xmax - xmin) * 0.05
    ypad = (ymax - ymin) * 0.05
    x_edges = np.linspace(xmin - xpad, xmax + xpad, G + 1)
    y_edges = np.linspace(ymin - ypad, ymax + ypad, G + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    cost_matrix = ot.dist(grid_pts, grid_pts, metric='sqeuclidean')

    h_drug, _, _ = np.histogram2d(emb[drug_mask, 0], emb[drug_mask, 1],
                                  bins=[x_edges, y_edges], density=True)
    h_mut, _, _ = np.histogram2d(emb[mut_mask, 0], emb[mut_mask, 1],
                                 bins=[x_edges, y_edges], density=True)
    p_d = h_drug.ravel() / h_drug.sum()
    p_m = h_mut.ravel() / h_mut.sum()
    p_d = np.maximum(p_d, 1e-12); p_d /= p_d.sum()
    p_m = np.maximum(p_m, 1e-12); p_m /= p_m.sum()

    gamma = ot.sinkhorn(p_d, p_m, cost_matrix, reg=args.align_reg,
                        numItermax=2000, verbose=False)
    print(f"  Transport plan: {gamma.shape}, mass: {gamma.sum():.4f}")

    # ---------------------------------------------------------------
    # 3. Barycentric projection: align each drug point
    # ---------------------------------------------------------------
    print("\n[3/5] Aligning drug points to mutant space (barycentric projection) ...")

    # Find which grid cell each drug point falls in
    ix = np.digitize(emb[drug_mask, 0], x_edges) - 1
    iy = np.digitize(emb[drug_mask, 1], y_edges) - 1
    ix = np.clip(ix, 0, G - 1)
    iy = np.clip(iy, 0, G - 1)
    flat_idx = iy * G + ix

    # Barycentric: for each grid cell, where does its mass go in mutant space?
    gamma_row_norm = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), 1e-12)
    all_cx = np.tile(x_centers, G)
    all_cy = np.repeat(y_centers, G)
    aligned_x = gamma_row_norm @ all_cx
    aligned_y = gamma_row_norm @ all_cy

    # Build aligned embedding: drug points moved, mutant points unchanged
    emb_aligned = emb.copy()
    emb_aligned[drug_mask, 0] = aligned_x[flat_idx]
    emb_aligned[drug_mask, 1] = aligned_y[flat_idx]

    print(f"  Aligned {drug_mask.sum()} drug points to mutant space")

    # ---------------------------------------------------------------
    # 4. Group by class
    # ---------------------------------------------------------------
    print("\n[4/5] Computing Wasserstein distances drug × mutant ...")

    class_indices = defaultdict(list)
    for i, lbl in enumerate(class_labels):
        class_indices[lbl].append(i)

    drug_classes = sorted([c for c in class_indices if is_drug_class(c)])
    mutant_classes = sorted([c for c in class_indices if not is_drug_class(c)])
    drug_classes_v = [c for c in drug_classes if len(class_indices[c]) >= 10]
    mutant_classes_v = [c for c in mutant_classes if len(class_indices[c]) >= 10]
    N_d, N_m = len(drug_classes_v), len(mutant_classes_v)
    print(f"  {N_d} drugs × {N_m} mutants")

    # Compute pairwise 1D Wasserstein on centroid-connecting axis
    wd_matrix = np.zeros((N_d, N_m), dtype=np.float32)

    # Pre-compute class centroids in aligned space
    drug_centroids = {}
    for lbl in drug_classes_v:
        idx = class_indices[lbl]
        drug_centroids[lbl] = emb_aligned[idx].mean(axis=0)

    mut_centroids = {}
    for lbl in mutant_classes_v:
        idx = class_indices[lbl]
        mut_centroids[lbl] = emb[idx].mean(axis=0)

    for i, d in enumerate(drug_classes_v):
        idx_d = class_indices[d]
        pts_d = emb_aligned[idx_d]
        c_d = drug_centroids[d]
        for j, m in enumerate(mutant_classes_v):
            idx_m = class_indices[m]
            pts_m = emb[idx_m]
            c_m = mut_centroids[m]

            # Axis connecting centroids
            axis = c_m - c_d
            norm = np.linalg.norm(axis)
            if norm < 1e-10:
                wd_matrix[i, j] = 0.0
                continue
            axis = axis / norm

            proj_d = pts_d @ axis
            proj_m = pts_m @ axis
            wd_matrix[i, j] = wasserstein_1d(proj_d, proj_m)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{N_d}]")

    # ---------------------------------------------------------------
    # 5. Visualizations
    # ---------------------------------------------------------------
    print("\n[5/5] Generating outputs ...")

    # --- Alignment visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    ax = axes[0]
    ax.scatter(emb[mut_mask, 0], emb[mut_mask, 1],
               c='#4DAF4A', s=1, alpha=0.3, label='Mutant', rasterized=True)
    ax.scatter(emb[drug_mask, 0], emb[drug_mask, 1],
               c='#E41A1C', s=1, alpha=0.3, label='Drug', rasterized=True)
    ax.set_title(f'Before OT alignment: drug vs mutant\n({drug_mask.sum()} drug, {mut_mask.sum()} mutant)',
                 fontsize=13)
    ax.legend(markerscale=20, fontsize=11)
    ax.set_xlabel('PaCMAP 1')
    ax.set_ylabel('PaCMAP 2')
    ax.set_aspect('equal')

    ax = axes[1]
    ax.scatter(emb_aligned[mut_mask, 0], emb_aligned[mut_mask, 1],
               c='#4DAF4A', s=1, alpha=0.3, label='Mutant', rasterized=True)
    ax.scatter(emb_aligned[drug_mask, 0], emb_aligned[drug_mask, 1],
               c='#377EB8', s=1, alpha=0.3, label='Drug (aligned)', rasterized=True)
    ax.set_title(f'After OT alignment: drug → mutant space\n(Sinkhorn ε={args.align_reg}, {G}×{G} grid)',
                 fontsize=13)
    ax.legend(markerscale=20, fontsize=11)
    ax.set_xlabel('PaCMAP 1')
    ax.set_ylabel('PaCMAP 2')
    ax.set_aspect('equal')

    plt.suptitle('PaCMAP: OT Domain Alignment Drug → Mutant', fontsize=14, y=1.01)
    plt.tight_layout()
    align_path = os.path.join(output_dir, 'pacmap_ot_alignment.png')
    fig.savefig(align_path, dpi=200, bbox_inches='tight')
    print(f"  Alignment viz: {align_path}")
    plt.close(fig)

    # --- Density contours overlay ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Subsample for KDE
    rng = np.random.RandomState(SEED)
    n_plot = min(10000, mut_mask.sum())
    idx_mut = rng.choice(np.where(mut_mask)[0], n_plot, replace=False)
    idx_drug = rng.choice(np.where(drug_mask)[0], n_plot, replace=False)

    from scipy.stats import gaussian_kde

    ax.scatter(emb[idx_mut, 0], emb[idx_mut, 1],
               c='#4DAF4A', s=3, alpha=0.2, label='Mutant', rasterized=True)
    ax.scatter(emb_aligned[idx_drug, 0], emb_aligned[idx_drug, 1],
               c='#377EB8', s=3, alpha=0.2, label='Drug (aligned)', rasterized=True)

    # KDE contours
    for pts, color, lbl in [(emb_aligned[idx_drug], '#377EB8', 'Drug aligned'),
                            (emb[idx_mut], '#4DAF4A', 'Mutant')]:
        if len(pts) < 100: continue
        try:
            kde = gaussian_kde(pts.T, bw_method=0.2)
            xi = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 100)
            yi = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 100)
            XX, YY = np.meshgrid(xi, yi)
            Z = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(100, 100)
            ax.contour(XX, YY, Z, levels=5, colors=[color], alpha=0.6, linewidths=1.5)
        except:
            pass

    ax.set_title('After OT alignment: KDE contours overlaid', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlabel('PaCMAP 1')
    ax.set_ylabel('PaCMAP 2')
    ax.set_aspect('equal')
    plt.tight_layout()
    contour_path = os.path.join(output_dir, 'pacmap_ot_aligned_contours.png')
    fig.savefig(contour_path, dpi=200, bbox_inches='tight')
    print(f"  Contour viz: {contour_path}")
    plt.close(fig)

    # --- Heatmap ---
    drug_labels = [short_drug(c) for c in drug_classes_v]
    mutant_labels = [short_mutant(c) for c in mutant_classes_v]

    drug_base_to_group = {}
    for gname, ginfo in MOA_GROUPS.items():
        for d in ginfo['drugs']:
            drug_base_to_group[d] = gname
    def get_mutant_target(lbl):
        m = re.match(r'^([a-zA-Z]+)', lbl)
        return m.group(1).lower() if m else lbl

    drug_groups = {}
    for c in drug_classes_v:
        base = get_drug_base(c)
        drug_groups[c] = drug_base_to_group.get(base, 'Other')
    mutant_groups = {}
    for c in mutant_classes_v:
        target = get_mutant_target(c)
        mutant_groups[c] = (MOA_GROUPS[g]['targets'] for g in MOA_GROUPS
                            if target in MOA_GROUPS[g]['targets'])
        found = None
        for gname, ginfo in MOA_GROUPS.items():
            if target in ginfo['targets']:
                found = gname
                break
        mutant_groups[c] = found or 'Other'

    all_groups = sorted(set(drug_groups.values()) | set(mutant_groups.values()))
    palette = sns.color_palette('Set2', len(all_groups))
    group_cmap = dict(zip(all_groups, palette))

    row_colors = np.array([group_cmap.get(drug_groups[c], (0.8, 0.8, 0.8))
                           for c in drug_classes_v])
    col_colors = np.array([group_cmap.get(mutant_groups[c], (0.8, 0.8, 0.8))
                           for c in mutant_classes_v])

    g = sns.clustermap(
        wd_matrix, row_cluster=True, col_cluster=True,
        method='ward', metric='euclidean',
        xticklabels=mutant_labels, yticklabels=drug_labels,
        figsize=(max(16, N_m * 0.3), max(12, N_d * 0.25)),
        cmap='viridis_r',
        row_colors=[row_colors], col_colors=[col_colors],
        linewidths=0, rasterized=True,
    )
    g.ax_heatmap.set_xlabel('Mutant classes', fontsize=10)
    g.ax_heatmap.set_ylabel('Drug classes', fontsize=10)
    g.fig.suptitle('Wasserstein-2 (OT-aligned drug → mutant): Drug × Mutant',
                   fontsize=13, y=1.02)
    patches = [mpatches.Patch(color=group_cmap[g], label=g) for g in all_groups]
    g.ax_heatmap.legend(handles=patches, loc='upper left', fontsize=7,
                        framealpha=0.8, bbox_to_anchor=(1.05, 1.15))
    heatmap_path = os.path.join(output_dir, 'wd_drug_mutant_ot_aligned.png')
    g.savefig(heatmap_path, dpi=200, bbox_inches='tight')
    print(f"  Heatmap: {heatmap_path}")
    plt.close(g.fig)

    # --- CSVs ---
    matches_csv = os.path.join(output_dir, 'drug_mutant_top_matches_ot.csv')
    with open(matches_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['drug', 'base', 'conc', 'rank', 'mutant', 'target', 'wd'])
        for i, d in enumerate(drug_classes_v):
            base = get_drug_base(d); conc = get_concentration(d) or ''
            top = np.argsort(wd_matrix[i])[:5]
            for rank, j in enumerate(top, 1):
                m = mutant_classes_v[j]
                target = get_mutant_target(m)
                w.writerow([d, base, conc, rank, m, target, f"{wd_matrix[i, j]:.4f}"])
    print(f"  Top matches: {matches_csv}")

    best_csv = os.path.join(output_dir, 'drug_best_mutant_ot.csv')
    with open(best_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['drug', 'base', 'conc', 'best_mutant', 'wd', 'expected_match'])
        for i, d in enumerate(drug_classes_v):
            j = np.argmin(wd_matrix[i])
            m = mutant_classes_v[j]
            base = get_drug_base(d)
            wd_val = wd_matrix[i, j]
            expected = 'no'
            for gname, ginfo in MOA_GROUPS.items():
                if base in ginfo['drugs']:
                    m_target = get_mutant_target(m)
                    if m_target in ginfo['targets']:
                        expected = f'YES ({gname})'
                        break
            w.writerow([d, base, get_concentration(d) or '', m, f"{wd_val:.4f}", expected])
    print(f"  Best matches: {best_csv}")

    unmatched_csv = os.path.join(output_dir, 'drugs_unmatched_ot.csv')
    min_wds = np.array([wd_matrix[i].min() for i in range(N_d)])
    thresh = np.percentile(min_wds, 80)
    with open(unmatched_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['drug', 'base', 'conc', 'min_wd', 'unmatched'])
        for i, d in enumerate(drug_classes_v):
            wd_val = wd_matrix[i].min()
            w.writerow([d, get_drug_base(d), get_concentration(d) or '',
                        f"{wd_val:.4f}", 'YES' if wd_val > thresh else 'no'])
    print(f"  Unmatched drugs: {unmatched_csv}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY (OT-aligned)")
    print("=" * 60)
    wd_flat = wd_matrix.ravel()
    print(f"WD range: [{wd_flat.min():.4f}, {wd_flat.max():.4f}]")
    print(f"WD mean: {wd_flat.mean():.4f}, median: {np.median(wd_flat):.4f}")

    flat_idx = np.argsort(wd_flat)
    print("\nTop 10 closest drug-mutant pairs:")
    for k in range(min(10, len(flat_idx))):
        idx = flat_idx[k]
        i, j = idx // N_m, idx % N_m
        d = drug_classes_v[i]; m = mutant_classes_v[j]
        print(f"  {d:35s} ↔ {m:15s}  WD = {wd_flat[idx]:.4f}")

    for i, d in enumerate(drug_classes_v):
        if d == 'control':
            ctrl = wd_matrix[i]
            print(f"\nControl (water): mean WD={ctrl.mean():.4f}, "
                  f"min={ctrl.min():.4f}, closest={mutant_classes_v[ctrl.argmin()]}")
            break

    print("\nExpected MoA validation:")
    for gname, ginfo in MOA_GROUPS.items():
        if not ginfo['targets']: continue
        for d_base in ginfo['drugs']:
            for conc in ['0.25x', '0.5x', '1x', '2x']:
                found = [c for c in drug_classes_v
                         if c.lower().startswith(d_base) and c.lower().endswith(conc)]
                if not found: continue
                i = drug_classes_v.index(found[0])
                for mt in mutant_classes_v:
                    mt_target = get_mutant_target(mt)
                    if mt_target in ginfo['targets']:
                        j = mutant_classes_v.index(mt)
                        wd_val = wd_matrix[i, j]
                        mean_wd = wd_matrix[i].mean()
                        rank = np.searchsorted(np.sort(wd_matrix[i]), wd_val)
                        pct = 100 * (1 - rank / N_m)
                        print(f"  {found[0]:35s} ↔ {mt:15s}  WD={wd_val:.3f} "
                              f"(mean={mean_wd:.3f}, better than {pct:.0f}% of mutants)")

    print(f"\nAll outputs in: {output_dir}")


if __name__ == '__main__':
    main()
