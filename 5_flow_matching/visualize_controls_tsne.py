#!/usr/bin/env python3
"""Control-centered ZCA whitening + separability check + t-SNE on all 185 classes.

Pipeline (best practice):
  1. Control-center: subtract ALL 13 control classes' mean (drug control + NC_1..6 + WT NC_1..6)
  2. ZCA whiten: decorrelate + sphere, computed from control-centered data's covariance
     (does NOT re-center, so control stays at origin)
  3. Check separability: centroid distances, intra-class spread, Fisher discriminant ratio
  4. t-SNE (init='pca', perp=50, max_iter=5000)

Usage:
    python3 visualize_controls_tsne.py --modes t05_cond
"""
import os, warnings, json
warnings.filterwarnings("ignore")
import numpy as np

from mil_model import load_labels

SEED = 42
np.random.seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CONTROL_NAMES = (['control']
                 + [f'NC_{i}' for i in range(1, 7)]
                 + [f'WT NC_{i}' for i in range(1, 7)])


def zca_whiten(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """ZCA whitening: decorrelate + sphere. Does NOT re-center.
    
    Computes whitening matrix W from covariance of X (assumed already centered),
    then returns X @ W^T, preserving the centering.
    """
    cov = X.T @ X / (X.shape[0] - 1)
    S, V = np.linalg.eigh(cov)
    S = np.maximum(S, 0)
    W = (V / np.sqrt(S + eps)) @ V.T
    return X @ W.T


def centroid_separability(feats: np.ndarray, labels: np.ndarray,
                           drug_set, mutant_set, ctl_set, tag: str):
    """Report class group separability metrics."""
    ctl_mask = np.array([l in ctl_set for l in labels])
    drug_mask = np.array([l in drug_set for l in labels])
    mutant_mask = np.array([l in mutant_set for l in labels])

    ctl_c = feats[ctl_mask].mean(axis=0)
    drug_c = feats[drug_mask].mean(axis=0)
    mutant_c = feats[mutant_mask].mean(axis=0)

    print(f"\n  ── Separability ({tag}) ──")
    print(f"  Control centroid norm : {np.linalg.norm(ctl_c):.4f}  (should be ~0 after centering)")
    print(f"  Drug   centroid norm : {np.linalg.norm(drug_c):.4f}")
    print(f"  Mutant centroid norm : {np.linalg.norm(mutant_c):.4f}")
    print(f"  ||drug - mutant||    : {np.linalg.norm(drug_c - mutant_c):.4f}")
    print(f"  ||drug - control||   : {np.linalg.norm(drug_c - ctl_c):.4f}")
    print(f"  ||mutant - control|| : {np.linalg.norm(mutant_c - ctl_c):.4f}")

    for name, mask, c in [('Control', ctl_mask, ctl_c),
                           ('Drug', drug_mask, drug_c),
                           ('Mutant', mutant_mask, mutant_c)]:
        spread = np.sqrt(((feats[mask] - c) ** 2).sum(axis=1)).mean()
        print(f"  {name:8s} intra spread: {spread:.4f}  (n={mask.sum()})")

    between = np.var(np.stack([ctl_c, drug_c, mutant_c]), axis=0).sum()
    within = 0
    for mask, c in [(ctl_mask, ctl_c), (drug_mask, drug_c), (mutant_mask, mutant_c)]:
        within += ((feats[mask] - c) ** 2).sum()
    within /= len(feats)
    print(f"  Fisher discriminant ratio (between/within): {between / (within + 1e-12):.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', type=str, default=None)
    parser.add_argument('--tsne_perplexity', type=float, default=50.0)
    parser.add_argument('--tsne_iter', type=int, default=5000)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--modes', type=str, default='t05_cond')
    args = parser.parse_args()

    feature_dir = args.feature_dir or os.path.join(SCRIPT_DIR, 'latent_analysis_pacmap')
    output_dir = args.output_dir or os.path.join(SCRIPT_DIR, 'controls_tsne')
    os.makedirs(output_dir, exist_ok=True)

    # ── Load labels & class sets ─────────────────────────────
    print("[1/3] Loading labels & class sets...")
    _, class_names, label_to_idx = load_labels(PROJECT_ROOT, SCRIPT_DIR)
    labels_all = np.load(os.path.join(feature_dir, 'labels.npy'))
    num_classes = len(class_names)
    print(f"  {len(labels_all)} images, {num_classes} classes")

    ctl_set = {label_to_idx[c] for c in CONTROL_NAMES if c in label_to_idx}
    mutant_path = os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')
    mutant_class_names = set()
    if os.path.exists(mutant_path):
        with open(mutant_path) as f:
            md = json.load(f)
        for plate in md.values():
            for cols in plate.values():
                for cinfo in cols.values():
                    if 'id' in cinfo:
                        mutant_class_names.add(cinfo['id'])
    mutant_set = {label_to_idx[c] for c in mutant_class_names
                  if c in label_to_idx and label_to_idx[c] not in ctl_set}
    drug_set = set(range(num_classes)) - mutant_set - ctl_set
    print(f"  Drug: {len(drug_set)}, Mutant: {len(mutant_set)}, Control: {len(ctl_set)}")

    modes = [m.strip() for m in args.modes.split(',')]

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Mode: {mode}")
        print(f"{'='*60}")

        feats = np.load(os.path.join(feature_dir, f'feats_{mode}.npy')).astype(np.float64)
        print(f"  Input: {feats.shape}")

        # ── 1. Control-center (all 13 control classes) ───────
        ctrl_mask_all = np.isin(labels_all, list(ctl_set))
        ctrl_mean = feats[ctrl_mask_all].mean(axis=0, keepdims=True)
        feats_c = feats - ctrl_mean
        print(f"  Centered: subtracted {ctrl_mask_all.sum()} controls' mean ({len(ctl_set)} classes)")

        # ── 2. ZCA whiten (preserves centering) ──────────────
        feats_w = zca_whiten(feats_c)
        cov_diag = np.diag(np.cov(feats_w, rowvar=False))
        print(f"  ZCA whitened: mean={feats_w.mean():.4f}, "
              f"cov diag ~ {cov_diag.mean():.4f}±{cov_diag.std():.4f} "
              f"(should be 1.0±~0.1)")

        np.save(os.path.join(output_dir, f'sphered_{mode}.npy'), feats_w)

        # ── 3. Separability check ────────────────────────────
        centroid_separability(feats_w, labels_all, drug_set, mutant_set, ctl_set, mode)

        # ── 4. t-SNE (init='pca', perp=50) ───────────────────
        print(f"\n  ── t-SNE (perp={args.tsne_perplexity}, iter={args.tsne_iter}) ──")
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity,
                     max_iter=args.tsne_iter, random_state=SEED,
                     init='pca', verbose=1)
        embedding = tsne.fit_transform(feats_w)

        np.save(os.path.join(output_dir, f'tsne_{mode}.npy'), embedding)
        np.save(os.path.join(output_dir, f'labels_{mode}.npy'), labels_all)

        # ── Plot ──────────────────────────────────────────────
        print("  Plotting...")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        ctl_arr = np.array([2 if l in ctl_set
                           else 1 if l in mutant_set
                           else 0 for l in labels_all], dtype=int)

        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        fig.suptitle(f'Control-centered → ZCA → t-SNE  ({mode}, perp={args.tsne_perplexity})',
                     fontsize=13, y=1.02)

        colors = [(1, 0, 0, 0.12), (0, 0.7, 0, 0.12), (0, 0, 1, 0.3)]
        labels_plot = ['Drug', 'Mutant', 'Control']
        for t_val in (0, 1, 2):
            mask = ctl_arr == t_val
            if mask.sum() == 0: continue
            axes[0].scatter(embedding[mask, 0], embedding[mask, 1],
                            c=[colors[t_val]], s=2, alpha=0.3, rasterized=True,
                            label=f'{labels_plot[t_val]} ({mask.sum()})')
        axes[0].set_title('Drug / Mutant / Control', fontsize=11)
        axes[0].legend(fontsize=9, markerscale=8)
        axes[0].set_xticks([]); axes[0].set_yticks([])

        ctrl_overlay = np.isin(labels_all, list(ctl_set))
        non_ctrl = ~ctrl_overlay
        axes[1].scatter(embedding[non_ctrl, 0], embedding[non_ctrl, 1],
                        c='#cccccc', s=1, alpha=0.2, rasterized=True,
                        label=f'Non-control ({non_ctrl.sum()})')
        axes[1].scatter(embedding[ctrl_overlay, 0], embedding[ctrl_overlay, 1],
                        c='black', s=8, alpha=0.6, rasterized=True,
                        label=f'All 13 controls ({ctrl_overlay.sum()})')
        axes[1].set_title('All 13 control classes (centered at origin)', fontsize=11)
        axes[1].legend(fontsize=9, markerscale=4)
        axes[1].set_xticks([]); axes[1].set_yticks([])

        out = os.path.join(output_dir, f'ctrlc_{mode}.png')
        fig.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out}")

    print("\nDone.")


if __name__ == '__main__':
    main()
