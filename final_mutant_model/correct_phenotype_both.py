#!/usr/bin/env python3
"""
Phenotype correction on both-domain embeddings.
Reads both/fold_Plate_1/embeddings_Plate_1_mil_n3.npz, applies correction
(centroid subtraction or ZCA sphering), saves corrected embeddings +
centroids + similarity matrix + t-SNE + heatmap.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

SCRIPT_DIR = Path(__file__).parent.absolute()
NPZ_PATH = SCRIPT_DIR / 'both' / 'fold_Plate_1' / 'embeddings_Plate_1_mil_n3.npz'
OUT_DIR = SCRIPT_DIR / 'both' / 'fold_Plate_1'


def zca_whiten(ctrl_emb, reg=1e-6):
    """Compute ZCA whitening matrix from control embeddings."""
    mean = ctrl_emb.mean(axis=0)
    Xc = ctrl_emb - mean
    cov = Xc.T @ Xc / (Xc.shape[0] - 1)
    U, S, Vt = np.linalg.svd(cov)
    W = (U @ np.diag(1.0 / np.sqrt(S + reg))) @ U.T
    return W, mean


def correct_and_analyze(emb, labels, paths, classes, drug_mask, mut_mask,
                         drug_emb, drug_labels, mut_emb, mut_labels, method,
                         per_concentration=False):
    """Apply correction method and run full analysis pipeline."""
    suffix = f'_{method}'

    # Identify controls
    drug_control_mask = np.array([l == 'control' for l in drug_labels])
    mut_control_mask = np.array(
        [l.startswith('NC') or l.startswith('WT NC') for l in mut_labels]
    )
    all_ctrl = np.vstack([
        drug_emb[drug_control_mask],
        mut_emb[mut_control_mask],
    ])

    drug_ctrl_centroid = drug_emb[drug_control_mask].mean(axis=0)
    mut_ctrl_centroid  = mut_emb[mut_control_mask].mean(axis=0)
    pooled_ctrl_centroid = (
        drug_ctrl_centroid * drug_control_mask.sum() +
        mut_ctrl_centroid * mut_control_mask.sum()
    ) / (drug_control_mask.sum() + mut_control_mask.sum())

    print(f"\n  ── [{method}] Correction ──")
    if method == 'centroid':
        drug_emb_c = drug_emb - pooled_ctrl_centroid
        mut_emb_c  = mut_emb  - pooled_ctrl_centroid
        save_extra = dict(
            drug_ctrl_centroid=drug_ctrl_centroid,
            mut_ctrl_centroid=mut_ctrl_centroid,
            pooled_ctrl_centroid=pooled_ctrl_centroid,
        )
    elif method == 'sphering':
        W, ctrl_mean = zca_whiten(all_ctrl)
        drug_emb_c = (drug_emb - ctrl_mean) @ W
        mut_emb_c  = (mut_emb  - ctrl_mean) @ W
        save_extra = dict(
            sphering_mean=ctrl_mean,
            sphering_W=W,
            drug_ctrl_centroid=drug_ctrl_centroid,
            mut_ctrl_centroid=mut_ctrl_centroid,
        )

    emb_c = np.empty_like(emb)
    emb_c[drug_mask] = drug_emb_c
    emb_c[mut_mask]  = mut_emb_c

    # Save corrected npz
    out_npz = OUT_DIR / f'embeddings_Plate_1_mil_n3_corrected{suffix}.npz'
    np.savez_compressed(out_npz, embeddings=emb_c.astype(np.float32),
                        labels=labels, paths=paths, classes=classes,
                        correction_method=method, **save_extra)
    print(f"  Saved: {out_npz}")

    # Class centroids
    all_lbl = np.concatenate([drug_labels, mut_labels])
    all_emb = np.concatenate([drug_emb_c, mut_emb_c])
    centroids = []
    for cl in sorted(set(all_lbl)):
        m = all_lbl == cl
        centroids.append([cl] + all_emb[m].mean(axis=0).tolist())
    cols = ['class_name'] + [f'dim_{i}' for i in range(emb.shape[1])]
    centroids_df = pd.DataFrame(centroids, columns=cols)
    out_csv = OUT_DIR / f'corrected_centroids{suffix}.csv'
    centroids_df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv} ({len(centroids_df)} rows)")

    # Cosine similarity helpers
    def cosine_sim(a, b):
        an = np.linalg.norm(a, axis=1, keepdims=True)
        bn = np.linalg.norm(b, axis=1, keepdims=True)
        return (a @ b.T) / (an @ bn.T + 1e-8)

    def get_drug_name(lbl):
        p = lbl.rsplit('_', 1)
        return p[0] if len(p) > 1 and p[1].endswith('x') else lbl

    def get_gene_name(lbl):
        p = lbl.rsplit('_', 1)
        return p[0] if len(p) > 1 and p[1].isdigit() else lbl

    # Drug centroids (averaged across concentrations)
    drug_cents = {}
    for lbl in sorted(set(drug_labels)):
        if lbl == 'control':
            continue
        dn = get_drug_name(lbl)
        drug_cents.setdefault(dn, []).append(drug_emb_c[drug_labels == lbl].mean(axis=0))
    drug_names = sorted(drug_cents)
    drug_mat = np.array([np.mean(drug_cents[d], axis=0) for d in drug_names])

    # Mutant centroids (averaged across replicates)
    mut_cents = {}
    for lbl in sorted(set(mut_labels)):
        if lbl.startswith('NC') or lbl.startswith('WT NC'):
            continue
        gn = get_gene_name(lbl)
        mut_cents.setdefault(gn, []).append(mut_emb_c[mut_labels == lbl].mean(axis=0))
    gene_names = sorted(mut_cents)
    mut_mat = np.array([np.mean(mut_cents[g], axis=0) for g in gene_names])

    sim = cosine_sim(drug_mat, mut_mat)
    print(f"  Similarity: {sim.shape}, range [{sim.min():.4f}, {sim.max():.4f}]")

    # Save averaged similarity
    pd.DataFrame(sim, index=drug_names, columns=gene_names).to_csv(
        OUT_DIR / f'corrected_drug_mutant_similarity{suffix}.csv')
    print(f"  Saved similarity csv")

    # Per-concentration similarity if requested
    if per_concentration:
        drug_cents_raw = {}
        for lbl in sorted(set(drug_labels)):
            if lbl == 'control':
                continue
            drug_cents_raw[lbl] = drug_emb_c[drug_labels == lbl].mean(axis=0)
        drug_raw_names = sorted(drug_cents_raw)
        drug_raw_mat = np.array([drug_cents_raw[d] for d in drug_raw_names])
        sim_raw = cosine_sim(drug_raw_mat, mut_mat)
        pd.DataFrame(sim_raw, index=drug_raw_names, columns=gene_names).to_csv(
            OUT_DIR / f'corrected_drug_mutant_similarity_per_conc{suffix}.csv')
        print(f"  Saved per-concentration similarity csv ({sim_raw.shape[0]} drugs)")

    # t-SNE
    out_tsne = OUT_DIR / f'corrected_tsne_comparison{suffix}.png'
    plot_tsne_comparison(emb, emb_c, labels, paths, out_tsne)

    # Threshold analysis
    flat = sim.flatten()
    thresh = flat.mean() + 2 * flat.std()
    thresh_p95 = np.percentile(flat, 95)
    thresh_used = max(thresh, thresh_p95)

    matches = []
    for i, d in enumerate(drug_names):
        for j, g in enumerate(gene_names):
            v = sim[i, j]
            if v > thresh_used:
                matches.append((v, d, g))
    matches.sort(reverse=True)

    print(f"\n  [{method}] Thresholds: mean+2σ={thresh:.4f}, p95={thresh_p95:.4f}")
    print(f"  [{method}] Top matches (>={thresh_used:.4f}):")
    for rank, (v, d, g) in enumerate(matches[:15], 1):
        print(f"    {rank}. {d:25s} × {g:10s} = {v:+.4f}")

    # Heatmap
    vmax = max(abs(flat.min()), abs(flat.max()))
    fig, axes = plt.subplots(1, 2, figsize=(max(20, len(gene_names) * 0.6),
                                              max(8, len(drug_names) * 0.45)))

    ax = axes[0]
    im = ax.imshow(sim, cmap=plt.cm.RdYlBu_r, vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(gene_names)))
    ax.set_yticks(range(len(drug_names)))
    ax.set_xticklabels(gene_names, rotation=90, fontsize=7)
    ax.set_yticklabels(drug_names, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Cosine Similarity')
    ax.set_title(f'All Pairs [{method}]', fontsize=13, fontweight='bold')

    mask = sim > thresh_used
    for i in range(len(drug_names)):
        for j in range(len(gene_names)):
            if mask[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False, edgecolor='lime', linewidth=2.5))

    ax = axes[1]
    masked = np.ma.masked_where(~mask, sim)
    cmap2 = plt.cm.RdYlBu_r.copy()
    cmap2.set_bad(color='#f0f0f0')
    im2 = ax.imshow(masked, cmap=cmap2, vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(gene_names)))
    ax.set_yticks(range(len(drug_names)))
    ax.set_xticklabels(gene_names, rotation=90, fontsize=7)
    ax.set_yticklabels(drug_names, fontsize=8)
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04).set_label('Cosine Similarity')
    ax.set_title(f'Significant Matches >{thresh_used:.3f} [{method}]', fontsize=13, fontweight='bold')

    for rank, (v, d, g) in enumerate(matches[:20], 1):
        axes[1].text(gene_names.index(g), drug_names.index(d), str(rank),
                      ha='center', va='center', fontsize=6, color='black', fontweight='bold')

    plt.suptitle(f'Drug × Gene Cosine Similarity ({method})', fontsize=15, fontweight='bold')
    plt.tight_layout()
    out_hm = OUT_DIR / f'corrected_similarity_heatmap{suffix}.png'
    plt.savefig(out_hm, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_hm}")

    return matches


def main():
    parser = argparse.ArgumentParser(description='Phenotype correction on both-domain embeddings')
    parser.add_argument('--method', type=str, default='sphering',
                        choices=['centroid', 'sphering'],
                        help='Correction method')
    parser.add_argument('--run_all', action='store_true', default=False,
                        help='Run all methods (centroid + sphering)')
    parser.add_argument('--per_concentration', action='store_true', default=False,
                        help='Also output per-concentration similarity matrix')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ────────────────────────────────────────────────────────────────
    print(f"Loading: {NPZ_PATH}")
    data = np.load(NPZ_PATH, allow_pickle=True)
    emb = data['embeddings'].astype(np.float64)
    labels = np.array([str(l) for l in data['labels']])
    paths = np.array([str(p) for p in data['paths']])
    classes = data['classes']
    N = len(emb)
    print(f"  Shape: {emb.shape}, {len(set(labels))} classes")

    # ── 2. Split drug vs mutant ────────────────────────────────────────────────
    drug_mask = np.array(['Drugs_Data' in p for p in paths])
    mut_mask  = np.array(['Mutants_Data' in p for p in paths])
    assert drug_mask.sum() + mut_mask.sum() == N
    drug_emb = emb[drug_mask]
    drug_labels = labels[drug_mask]
    mut_emb = emb[mut_mask]
    mut_labels = labels[mut_mask]
    print(f"  Drug: {len(drug_emb)} wells, {len(set(drug_labels))} classes")
    print(f"  Mutant: {len(mut_emb)} wells, {len(set(mut_labels))} classes")

    # Verify label separation
    overlap = set(drug_labels) & set(mut_labels)
    print(f"  [DEBUG] Label overlap: {'NONE — correct' if not overlap else sorted(overlap)[:10]}")
    drug_has_mutant = [l for l in set(drug_labels) if any(c.islower() for c in l[:3]) and '_' in l and l.rsplit('_',1)[1].isdigit()]
    mut_has_drug = [l for l in set(mut_labels) if any(c.isupper() for c in l[:3]) and '_' in l and l.rsplit('_',1)[1].endswith('x')]
    print(f"  [DEBUG] Drugs looking like mutants: {'NONE' if not drug_has_mutant else drug_has_mutant[:5]}")
    print(f"  [DEBUG] Mutants looking like drugs: {'NONE' if not mut_has_drug else mut_has_drug[:5]}")

    # Run
    if args.run_all:
        methods = ['centroid', 'sphering']
    else:
        methods = [args.method]

    all_matches = {}
    for m in methods:
        all_matches[m] = correct_and_analyze(
            emb, labels, paths, classes,
            drug_mask, mut_mask, drug_emb, drug_labels, mut_emb, mut_labels,
            m, args.per_concentration)

    # ── Comparison summary ──────────────────────────────────────────────────
    if len(methods) > 1:
        print(f"\n{'='*60}")
        print(f"  COMPARISON: centroid vs sphering")
        print(f"{'='*60}")
        for m in methods:
            hits = all_matches[m]
            n_sig = len([x for x in hits if x[0] > 0])
            top_n = 5
            print(f"\n  [{m}] Top {top_n} matches:")
            for v, d, g in hits[:top_n]:
                print(f"    {d:25s} × {g:10s} = {v:+.4f}")
        print(f"\n  Done!")


def plot_tsne_comparison(emb_raw, emb_corrected, labels, paths, out_path, perplexity=40):
    N = len(emb_raw)
    combined = np.vstack([emb_raw, emb_corrected])
    pca_50 = PCA(n_components=50).fit_transform(combined)
    half = N
    raw_pca = pca_50[:half]
    cor_pca = pca_50[half:]

    domain_labels = ['drug' if 'Drugs_Data' in str(p) else 'mutant' for p in paths]
    class_labels = [str(l) for l in labels]
    unique_classes = sorted(set(class_labels))
    class_to_int = {c: i for i, c in enumerate(unique_classes)}

    print("  Running t-SNE (before vs after)...")
    tsne_raw = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                     max_iter=1000).fit_transform(raw_pca)
    tsne_cor = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                     max_iter=1000).fit_transform(cor_pca)

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    for col, (tsne, title) in enumerate([(tsne_raw, 'Before Correction'),
                                          (tsne_cor, 'After Correction')]):
        for dom, color, marker in [('drug', '#1f77b4', 'o'), ('mutant', '#d62728', 'x')]:
            mask = [d == dom for d in domain_labels]
            axes[0, col].scatter(tsne[mask, 0], tsne[mask, 1], c=color, marker=marker,
                                  label=dom, s=8, alpha=0.5, linewidths=0)
        axes[0, col].set_title(f't-SNE — {title} (drug vs mutant)', fontsize=13, fontweight='bold')
        axes[0, col].legend(fontsize=11)

        int_labels = [class_to_int[c] for c in class_labels]
        cmap = 'tab20' if len(unique_classes) <= 20 else 'gist_ncar'
        sc = axes[1, col].scatter(tsne[:, 0], tsne[:, 1], c=int_labels, cmap=cmap,
                                   s=6, alpha=0.6, linewidths=0)
        axes[1, col].set_title(f't-SNE — {title} (by class)', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


if __name__ == '__main__':
    main()
