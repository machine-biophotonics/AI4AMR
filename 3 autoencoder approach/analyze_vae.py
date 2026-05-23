import os
import sys
import argparse
import json
import re
import glob
import random
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import ot  # Python Optimal Transport

warnings.filterwarnings('ignore')

from mil_model import MultiCropDataset, extract_well_from_filename
from vae_model import MILVAE

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def sliced_wasserstein(X: np.ndarray, Y: np.ndarray, n_projections: int = 1000) -> float:
    """Sliced 1-Wasserstein distance between two point clouds (multivariate)."""
    return ot.sliced_wasserstein_distance(X, Y, n_projections)


def sinkhorn_wasserstein(X: np.ndarray, Y: np.ndarray, reg: float = 0.01) -> float:
    """Entropy-regularized Sinkhorn divergence between two point clouds."""
    n, m = len(X), len(Y)
    a = np.ones(n) / n
    b = np.ones(m) / m
    M = ot.dist(X, Y, metric='sqeuclidean')
    M /= M.max() if M.max() > 0 else 1.0
    try:
        W = ot.sinkhorn2(a, b, M, reg, numItermax=200)
        return float(np.asarray(W).flat[0])
    except Exception:
        return float('nan')


def extract_latent(model, loader, device):
    model.eval()
    all_z, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc='Extracting latents'):
            images = images.to(device)
            bag = model.encode_bag(images)
            mu = model.vae_mu(bag)
            all_z.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())
            all_paths.extend(paths)
    z = np.concatenate(all_z, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return z, labels, all_paths


def extract_bag_embeddings(model, loader, device):
    model.eval()
    all_bags, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc='Extracting bags'):
            images = images.to(device)
            bag = model.encode_bag(images)
            all_bags.append(bag.cpu().numpy())
            all_labels.append(labels.numpy())
            all_paths.extend(paths)
    bags = np.concatenate(all_bags, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return bags, labels, all_paths


def tsne_visualization(z, labels, class_names, output_path, title='VAE Latent Space (t-SNE)'):
    print("Running t-SNE...")
    n_samp = min(len(z), 10000)
    if len(z) > n_samp:
        idx = np.random.choice(len(z), n_samp, replace=False)
        z_s = z[idx]
        lbl_s = labels[idx]
    else:
        z_s, lbl_s = z, labels

    z_scaled = StandardScaler().fit_transform(z_s)
    perplexity = min(30, len(z_s) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=SEED, max_iter=1000)
    z_2d = tsne.fit_transform(z_scaled)

    unique = sorted(np.unique(lbl_s))
    n_colors = max(len(unique), 20)
    cmap = plt.cm.tab20 if n_colors <= 20 else plt.cm.tab40 if n_colors <= 40 else plt.cm.nipy_spectral
    if n_colors > 40:
        colors = cmap(np.linspace(0, 1, n_colors))[:len(unique)]
    else:
        colors = cmap(np.linspace(0, 1, n_colors))[:len(unique)]

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    for idx, cls in enumerate(unique):
        mask = lbl_s == cls
        label_name = class_names[cls] if cls < len(class_names) else str(cls)
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1], c=[colors[idx]],
                   label=label_name, alpha=0.6, s=8, edgecolors='none')

    ax.set_title(title, fontsize=16)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8,
              markerscale=3, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"t-SNE saved: {output_path}")

    np.savez(output_path.replace('.png', '.npz'), z_2d=z_2d, labels=lbl_s, class_names=class_names)
    return z_2d


def compute_wasserstein_matrix(z, labels, class_names, output_dir):
    print("\nComputing pairwise Sinkhorn-Wasserstein distances...")
    unique = sorted(np.unique(labels))
    class_samples = {}
    for cls in unique:
        z_cls = z[labels == cls]
        if len(z_cls) >= 5:
            class_samples[cls] = z_cls

    cls_list = sorted(class_samples.keys())
    n = len(cls_list)
    wd = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            Xi = class_samples[cls_list[i]]
            Xj = class_samples[cls_list[j]]
            d = sinkhorn_wasserstein(Xi, Xj, reg=0.01)
            wd[i, j] = wd[j, i] = d
        if (i + 1) % 10 == 0:
            print(f"  WD progress: {i+1}/{n}")

    np.savez(os.path.join(output_dir, 'wasserstein_matrix.npz'),
             matrix=wd, class_indices=cls_list,
             class_names=[class_names[c] for c in cls_list])

    fig, ax = plt.subplots(figsize=(max(12, n * 0.4), max(10, n * 0.35)))
    vmax = np.nanpercentile(wd, 95)
    im = ax.imshow(wd, aspect='auto', cmap='viridis', vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Sinkhorn-Wasserstein')

    labels_str = [
        class_names[c] if len(class_names[c]) < 25 else class_names[c][:22] + '...'
        for c in cls_list
    ]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels_str, rotation=90, fontsize=6)
    ax.set_yticklabels(labels_str, fontsize=6)
    ax.set_title('Sinkhorn-Wasserstein Distance Between Phenotype Distributions', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wasserstein_heatmap.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Wasserstein heatmap saved")

    with open(os.path.join(output_dir, 'top_similar_classes.txt'), 'w') as f:
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((wd[i, j], i, j))
        pairs.sort(key=lambda x: x[0])

        f.write("Most Similar (lowest WD):\n")
        for rank, (d, i, j) in enumerate(pairs[:20]):
            f.write(f"  {rank+1}. {class_names[cls_list[i]]:<30} vs {class_names[cls_list[j]]:<30}  WD={d:.4f}\n")

        f.write("\nMost Dissimilar (highest WD):\n")
        for rank, (d, i, j) in enumerate(pairs[-20:]):
            f.write(f"  {rank+1}. {class_names[cls_list[i]]:<30} vs {class_names[cls_list[j]]:<30}  WD={d:.4f}\n")

    return wd, cls_list


def drug_vs_mutant_wasserstein(z, labels, class_names, output_dir):
    print("\nCross-domain (Drug vs Mutant) Wasserstein distances...")
    unique = sorted(np.unique(labels))
    drug_classes = []
    mutant_classes = []
    for cls in unique:
        name = class_names[cls] if cls < len(class_names) else str(cls)
        # Drug names end with concentration suffix (e.g. _0.25x, _1x, _2x)
        # 'control' is a no-treatment control → drug
        is_drug = bool(re.search(r'_\d+(\.\d+)?x$', name)) or (name == 'control')
        if is_drug:
            drug_classes.append(cls)
        else:
            mutant_classes.append(cls)

    if len(drug_classes) < 2 or len(mutant_classes) < 2:
        print(f"  Not enough drug ({len(drug_classes)}) or mutant ({len(mutant_classes)}) classes. "
              f"Try --data_mode both.")
        return

    class_samples = {}
    for cls in unique:
        z_cls = z[labels == cls]
        if len(z_cls) >= 5:
            class_samples[cls] = z_cls

    drug_list = [c for c in drug_classes if c in class_samples]
    mut_list = [c for c in mutant_classes if c in class_samples]

    n_d, n_m = len(drug_list), len(mut_list)
    wd_cross = np.zeros((n_d, n_m))

    for i, di in enumerate(drug_list):
        for j, mj in enumerate(mut_list):
            wd_cross[i, j] = sinkhorn_wasserstein(
                class_samples[di], class_samples[mj], reg=0.01
            )
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{n_d}")

    drug_names = [class_names[d] for d in drug_list]
    mut_names = [class_names[m] for m in mut_list]

    np.savez(os.path.join(output_dir, 'drug_mutant_wasserstein.npz'),
             matrix=wd_cross, drug_indices=drug_list, mutant_indices=mut_list,
             drug_names=drug_names, mutant_names=mut_names)

    fig, ax = plt.subplots(figsize=(max(14, n_m * 0.35), max(10, n_d * 0.35)))
    vmax = np.nanpercentile(wd_cross, 90)
    im = ax.imshow(wd_cross, aspect='auto', cmap='viridis', vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Sinkhorn-Wasserstein')
    ax.set_xticks(range(n_m))
    ax.set_yticks(range(n_d))
    ax.set_xticklabels([n[:20] for n in mut_names], rotation=90, fontsize=5)
    ax.set_yticklabels([n[:20] for n in drug_names], fontsize=5)
    ax.set_xlabel('Mutant Classes', fontsize=12)
    ax.set_ylabel('Drug Classes', fontsize=12)
    ax.set_title('Sinkhorn-Wasserstein: Drug vs Mutant Phenotype Distributions', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drug_mutant_wasserstein.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Drug-Mutant heatmap saved")

    for di in drug_list[:5]:
        distances = [(wd_cross[drug_list.index(di), j], mut_names[j]) for j in range(n_m)]
        distances.sort(key=lambda x: x[0])
        print(f"  {class_names[di]:<25} → closest mutants: "
              f"{', '.join(f'{n}({d:.3f})' for d, n in distances[:5])}")

    return wd_cross, drug_names, mut_names


def drug_concentration_analysis(z, labels, class_names, output_dir):
    print("\nConcentration effect analysis...")
    drug_groups = defaultdict(dict)
    for cls in sorted(np.unique(labels)):
        name = class_names[cls]
        parts = name.rsplit('_', 1)
        if len(parts) == 2 and 'x' in parts[1]:
            base_drug = parts[0]
            conc = parts[1]
            drug_groups[base_drug][conc] = cls

    def conc_val(c):
        try:
            return float(c.replace('x', ''))
        except ValueError:
            return 0.0

    drug_groups = {k: v for k, v in drug_groups.items() if len(v) >= 3}

    if not drug_groups:
        print("  No drug classes with ≥3 concentrations found. Skipping.")
        return

    n_plots = min(len(drug_groups), 12)
    cols, rows = 4, (n_plots + 3) // 4
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for plot_idx, (drug_name, concs) in enumerate(
        sorted(drug_groups.items(), key=lambda x: -len(x[1]))[:n_plots]
    ):
        ax = axes[plot_idx]
        conc_order = sorted(concs.keys(), key=conc_val)
        means, all_samples = [], []

        for conc in conc_order:
            cls = concs[conc]
            z_cls = z[labels == cls]
            if len(z_cls) < 3:
                continue
            all_samples.append(z_cls)

        if len(all_samples) < 2:
            ax.text(0.5, 0.5, 'Insufficient samples', transform=ax.transAxes, ha='center')
            continue

        base = all_samples[0]
        wds = [0.0]
        for k in range(1, len(all_samples)):
            wds.append(sinkhorn_wasserstein(base, all_samples[k], reg=0.01))

        x = list(range(len(conc_order)))
        ax.plot(x, wds, 'o-', linewidth=2, markersize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(conc_order, fontsize=8)
        ax.set_xlabel('Concentration', fontsize=10)
        ax.set_ylabel('WD from lowest conc', fontsize=9)
        ax.set_title(f'{drug_name}', fontsize=11)
        ax.grid(True, alpha=0.3)

    for k in range(n_plots, len(axes)):
        axes[k].set_visible(False)

    plt.suptitle('Drug Concentration Effect on Phenotype Distribution', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drug_concentration_effect.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Concentration analysis saved")


def latent_interpolation(model, z, labels, class_names, output_dir, device):
    print("\nLatent interpolation between class centroids...")
    out_dir = os.path.join(output_dir, 'interpolation')
    os.makedirs(out_dir, exist_ok=True)

    unique = sorted(np.unique(labels))
    centroids = {}
    for cls in unique:
        z_cls = z[labels == cls]
        if len(z_cls) >= 5:
            centroids[cls] = z_cls.mean(axis=0)

    keys = list(centroids.keys())
    n_interp = 8
    count = 0

    for i in range(min(6, len(keys))):
        for j in range(i + 1, min(6, len(keys))):
            if count >= 6:
                break
            c1, c2 = keys[i], keys[j]
            z1, z2 = centroids[c1], centroids[c2]
            alphas = np.linspace(0, 1, n_interp)

            fig, axes = plt.subplots(1, n_interp, figsize=(4 * n_interp, 4))
            with torch.no_grad():
                for k, alpha in enumerate(alphas):
                    z_interp = (1 - alpha) * z1 + alpha * z2
                    z_t = torch.from_numpy(z_interp).float().unsqueeze(0).to(device)
                    if model.pixel_decoder is not None:
                        recon = model.decode_img(z_t).cpu().squeeze().numpy()
                        recon = (recon + 1) / 2
                        recon = np.clip(recon, 0, 1)
                        axes[k].imshow(recon, cmap='gray')
                    else:
                        axes[k].text(0.5, 0.5, 'recon', ha='center')
                    axes[k].set_title(f'α={alpha:.2f}', fontsize=10)
                    axes[k].axis('off')

            name1 = class_names[c1][:15] if c1 < len(class_names) else str(c1)
            name2 = class_names[c2][:15] if c2 < len(class_names) else str(c2)
            plt.suptitle(f'{name1} → {name2}', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'interp_{name1}_{name2}.png'), dpi=150, bbox_inches='tight')
            plt.close()
            count += 1

    print(f"Interpolation images saved to {out_dir}")


def latent_traversal(model, z, output_dir, device):
    print("\nLatent traversal...")
    out_dir = os.path.join(output_dir, 'traversal')
    os.makedirs(out_dir, exist_ok=True)

    if model.pixel_decoder is None:
        print("  No pixel decoder available. Skipping.")
        return

    z0 = z.mean(axis=0)
    n_dims = min(z.shape[1], 8)
    n_steps = 9
    delta = 2.5

    for dim in range(n_dims):
        vals = np.linspace(-delta, delta, n_steps)
        fig, axes = plt.subplots(1, n_steps, figsize=(3 * n_steps, 3))
        with torch.no_grad():
            for k, val in enumerate(vals):
                z_traj = z0.copy()
                z_traj[dim] += val
                z_t = torch.from_numpy(z_traj).float().unsqueeze(0).to(device)
                recon = model.decode_img(z_t).cpu().squeeze().numpy()
                recon = (recon + 1) / 2
                recon = np.clip(recon, 0, 1)
                axes[k].imshow(recon, cmap='gray')
                axes[k].set_title(f'z[{dim}]={val:.1f}', fontsize=9)
                axes[k].axis('off')
        plt.suptitle(f'Latent Traversal: Dim {dim}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'traversal_dim{dim}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Traversal images saved to {out_dir}")


def compute_mig(z: np.ndarray, factor_labels: np.ndarray, n_bins: int = 20) -> float:
    from sklearn.feature_selection import mutual_info_classif
    from scipy.stats import entropy

    valid = factor_labels >= 0
    if valid.sum() < 10:
        return float('nan')
    z_f = z[valid]
    labels_f = factor_labels[valid]
    unique = np.unique(labels_f)
    if len(unique) < 3:
        return float('nan')

    n_dims = z_f.shape[1]
    mi_per_dim = np.zeros(n_dims)
    for d in range(n_dims):
        if z_f[:, d].std() < 1e-8:
            mi_per_dim[d] = 0.0
            continue
        try:
            mi = mutual_info_classif(z_f[:, d].reshape(-1, 1), labels_f,
                                     discrete_features=False, random_state=42)
            mi_per_dim[d] = float(mi[0])
        except Exception:
            mi_per_dim[d] = 0.0

    mi_sorted = np.sort(mi_per_dim)
    if len(mi_sorted) < 2:
        return float('nan')
    top1, top2 = mi_sorted[-1], mi_sorted[-2]
    h = entropy(np.bincount(labels_f.astype(int)))
    if h < 1e-10:
        return float('nan')
    return float((top1 - top2) / h)


def compute_dci(z: np.ndarray, drug_labels: np.ndarray, mutant_labels: np.ndarray) -> dict:
    from sklearn.ensemble import GradientBoostingClassifier

    results = {}
    for factor_name, factor_labels in [('drug', drug_labels), ('mutant', mutant_labels)]:
        valid = factor_labels >= 0
        if valid.sum() < 50:
            results[factor_name] = {'disentanglement': float('nan'), 'informativeness': float('nan')}
            continue
        z_f = z[valid]
        labels_f = factor_labels[valid]
        unique = np.unique(labels_f)
        if len(unique) < 2:
            results[factor_name] = {'disentanglement': float('nan'), 'informativeness': float('nan')}
            continue

        try:
            clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, min_samples_leaf=5,
                random_state=42, validation_fraction=0.1, n_iter_no_change=10
            )
            clf.fit(z_f, labels_f)
            acc = clf.score(z_f, labels_f)

            importances = np.array([tree.feature_importances_ for tree in clf.estimators_])
            importance_per_dim = importances.mean(axis=0)

            total_imp = importance_per_dim.sum()
            if total_imp > 0:
                importance_per_dim = importance_per_dim / total_imp

            sorted_imp = np.sort(importance_per_dim)[::-1]
            cumsum = np.cumsum(sorted_imp)
            n_effective = int((cumsum < 0.95).sum()) + 1

            disent = float(1.0 - (n_effective - 1) / max(len(importance_per_dim) - 1, 1))

            results[factor_name] = {
                'disentanglement': round(disent, 4),
                'informativeness': round(acc, 4),
                'n_effective_dims': n_effective,
                'total_dims': len(importance_per_dim),
            }
        except Exception as e:
            results[factor_name] = {'disentanglement': float('nan'), 'informativeness': float('nan'), 'error': str(e)}

    return results


def compute_disentanglement_metrics(
    z: np.ndarray,
    drug_label_ids: np.ndarray,
    mutant_label_ids: np.ndarray,
    output_dir: str,
    drug_class_names: list = None,
    mutant_class_names: list = None,
):
    print("  Computing MIG (Mutual Information Gap)...")
    mig_drug = compute_mig(z, drug_label_ids)
    mig_mutant = compute_mig(z, mutant_label_ids)
    print(f"    MIG (drug factor):   {mig_drug:.6f}")
    print(f"    MIG (mutant factor): {mig_mutant:.6f}")

    print("  Computing DCI (Disentanglement, Completeness, Informativeness)...")
    dci_results = compute_dci(z, drug_label_ids, mutant_label_ids)
    for fname, res in dci_results.items():
        print(f"    {fname}: disentanglement={res.get('disentanglement', 'N/A')}, "
              f"informativeness={res.get('informativeness', 'N/A')}, "
              f"n_effective_dims={res.get('n_effective_dims', 'N/A')}/{res.get('total_dims', 'N/A')}")

    metrics = {
        'mig_drug': mig_drug,
        'mig_mutant': mig_mutant,
        'dci': dci_results,
    }
    with open(os.path.join(output_dir, 'disentanglement_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {os.path.join(output_dir, 'disentanglement_metrics.json')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='best_mil_vae.pth path')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--data_mode', type=str, default='mutant', choices=['drug', 'mutant', 'both'])
    parser.add_argument('--test_plate', type=str, default='Plate_6')
    parser.add_argument('--no_tsne', action='store_true')
    parser.add_argument('--no_wasserstein', action='store_true')
    parser.add_argument('--no_drug_mutant', action='store_true')
    parser.add_argument('--no_concentration', action='store_true')
    parser.add_argument('--no_interpolation', action='store_true')
    parser.add_argument('--no_traversal', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--neighborhood', type=int, default=3)
    parser.add_argument('--grid_size', type=int, default=12)
    parser.add_argument('--compute_disentanglement', action='store_true',
                        help='Compute MIG and DCI disentanglement metrics (requires --data_mode both)')
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    IC50_MAPPING_PATH = os.path.join(PROJECT_ROOT, 'final_mutant_model', 'plate_well_ic50_mapping.json')
    MUTANT_MAPPING_PATH = os.path.join(PROJECT_ROOT, 'final_mutant_model', 'plate_well_id_path.json')

    with open(IC50_MAPPING_PATH) as f:
        ic50_data = json.load(f)
    with open(MUTANT_MAPPING_PATH) as f:
        mutant_data = json.load(f)

    plate_maps = {}
    for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
        plate_maps[plate] = {}
        if args.data_mode in ('drug', 'both') and plate in ic50_data:
            for well, info in ic50_data[plate].items():
                antibiotic = info.get('antibiotic', '')
                ic50_mult = info.get('ic50_multiple', '')
                if antibiotic and ic50_mult:
                    if ic50_mult == 'control':
                        drug_class = 'control'
                    else:
                        ic50_str = ic50_mult if 'x' in ic50_mult else f"{ic50_mult}x"
                        drug_class = f"{antibiotic.replace(' ', '_')}_{ic50_str}"
                    plate_maps[plate][f"drug_{well}"] = drug_class

        if args.data_mode in ('mutant', 'both') and plate in mutant_data:
            for row, cols in mutant_data[plate].items():
                for col, info in cols.items():
                    if 'id' in info:
                        well = f"{row}{int(col):02d}"
                        plate_maps[plate][f"mutant_{well}"] = info['id']

    all_classes = sorted(set(
        label for pm in plate_maps.values() for label in pm.values() if label
    ))
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    num_classes = len(all_classes)

    # Separate drug and mutant class lists for disentanglement
    all_drug_classes = sorted(set(
        v for pm in plate_maps.values()
        for k, v in pm.items() if k.startswith('drug_') and v
    ))
    all_mutant_classes = sorted(set(
        v for pm in plate_maps.values()
        for k, v in pm.items() if k.startswith('mutant_') and v
    ))
    drug_class_to_idx = {c: i for i, c in enumerate(all_drug_classes)}
    mutant_class_to_idx = {c: i for i, c in enumerate(all_mutant_classes)}
    print(f"Classes: {num_classes} total ({len(all_drug_classes)} drug, {len(all_mutant_classes)} mutant)")

    def extract_label(path):
        path_lower = path.lower()
        for pn in range(1, 7):
            if f'/p{pn}/' in path_lower:
                plate_key = f'P{pn}'
                break
        else:
            return None
        well = extract_well_from_filename(os.path.basename(path))
        if well is None:
            return None
        if '/mutants_data/' in path_lower:
            prefix = 'mutant_'
        else:
            prefix = 'drug_'
        cw = f"{prefix}{well}"
        if plate_key in plate_maps and cw in plate_maps[plate_key]:
            return plate_maps[plate_key][cw]
        return None

    def extract_both_labels(path):
        """Return (drug_class_str_or_None, mutant_id_str_or_None) for a path."""
        path_lower = path.lower()
        for pn in range(1, 7):
            if f'/p{pn}/' in path_lower:
                plate_key = f'P{pn}'
                break
        else:
            return None, None
        well = extract_well_from_filename(os.path.basename(path))
        if well is None:
            return None, None
        drug_lbl = plate_maps.get(plate_key, {}).get(f"drug_{well}", None)
        mutant_lbl = plate_maps.get(plate_key, {}).get(f"mutant_{well}", None)
        return drug_lbl, mutant_lbl

    def get_image_paths(plate):
        plate_key = f"P{plate.split('_')[-1]}"
        search_dirs = []
        if args.data_mode in ('drug', 'both'):
            search_dirs.append((os.path.join(PROJECT_ROOT, 'Drugs_Data', plate_key), 'drug'))
        if args.data_mode in ('mutant', 'both'):
            search_dirs.append((os.path.join(PROJECT_ROOT, 'Mutants_Data', plate_key), 'mutant'))
        valid = []
        for plate_dir, _ in search_dirs:
            if not os.path.exists(plate_dir):
                continue
            paths = []
            for p in ['*.tif', '*.tiff']:
                paths.extend(glob.glob(os.path.join(plate_dir, '**', p), recursive=True))
            for path in paths:
                well = extract_well_from_filename(os.path.basename(path))
                if well:
                    valid.append(path)
        return valid

    test_norm = f"Plate_{args.test_plate[-1]}" if 'P' in args.test_plate else args.test_plate
    test_paths, test_labels = [], []

    for path in get_image_paths(test_norm):
        lbl = extract_label(path)
        if lbl in class_to_idx:
            test_paths.append(path)
            test_labels.append(class_to_idx[lbl])

    print(f"Test images: {len(test_paths)}")

    test_dataset = MultiCropDataset(
        test_paths, test_labels, None,
        neighborhood=args.neighborhood, grid_size=args.grid_size,
        augment=False, num_channels=1, extraction_mode='neighborhood'
    )
    test_dataset.set_epoch(0)

    class MILWrapDataset(Dataset):
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            img, lbl = self.base[idx]
            return img, lbl, ""

    test_loader = DataLoader(
        MILWrapDataset(test_dataset), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    model = MILVAE(
        num_classes=num_classes, latent_dim=32, beta=0.1,
        num_heads=4, dropout=0.5, use_contrastive=True,
        feature_decoder=True, pixel_decoder=True,
    ).to(device)

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    print("Model loaded.")

    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, f'mil_vae_{args.data_mode}',
                                       f'fold_{test_norm}', 'analysis')
    os.makedirs(args.output_dir, exist_ok=True)

    z, labels, _ = extract_latent(model, test_loader, device)
    print(f"Latent codes: {z.shape}")

    np.savez(os.path.join(args.output_dir, 'latent_codes.npz'),
             z=z, labels=labels, class_names=all_classes)

    if not args.no_tsne:
        tsne_visualization(z, labels, all_classes,
                           os.path.join(args.output_dir, 'tsne_latent.png'),
                           title=f'VAE Latent - {args.data_mode} (test={test_norm})')

    if not args.no_wasserstein:
        compute_wasserstein_matrix(z, labels, all_classes, args.output_dir)

    if args.data_mode == 'both' and not args.no_drug_mutant:
        drug_vs_mutant_wasserstein(z, labels, all_classes, args.output_dir)

    if not args.no_concentration:
        drug_concentration_analysis(z, labels, all_classes, args.output_dir)

    if not args.no_interpolation:
        latent_interpolation(model, z, labels, all_classes, args.output_dir, device)

    if not args.no_traversal:
        latent_traversal(model, z, args.output_dir, device)

    if args.compute_disentanglement and args.data_mode == 'both':
        print("\n" + "="*60)
        print("DISENTANGLEMENT METRICS")
        print("="*60)
        drug_label_ids = []
        mutant_label_ids = []
        for p in test_paths:
            dl, ml = extract_both_labels(p)
            did = drug_class_to_idx.get(dl, -1) if dl is not None else -1
            mid = mutant_class_to_idx.get(ml, -1) if ml is not None else -1
            drug_label_ids.append(did)
            mutant_label_ids.append(mid)
        drug_label_ids = np.array(drug_label_ids, dtype=np.int32)
        mutant_label_ids = np.array(mutant_label_ids, dtype=np.int32)

        compute_disentanglement_metrics(
            z, drug_label_ids, mutant_label_ids, args.output_dir,
            drug_class_names=all_drug_classes, mutant_class_names=all_mutant_classes
        )

    print(f"\nAll analyses in: {args.output_dir}")


if __name__ == '__main__':
    main()
