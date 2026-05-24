#!/usr/bin/env python3
"""Extract and visualize MAE latent space, then benchmark MoA hit rate.

Usage:
    python3 visualize_latent.py --checkpoint path/to/mae_best.pth --model small
    python3 visualize_latent.py --checkpoint path/to/mae_best.pth --model small --output_dir /path/to/results
    python3 visualize_latent.py --checkpoint path/to/mae_best.pth --model small --n_crops 144
"""
import os, sys, warnings, glob, json, re
warnings.filterwarnings("ignore")
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

from mil_model import MAECropDataset
from mae_model import mae_vit_tiny, mae_vit_small, mae_vit_base

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True, help='Path to MAE checkpoint')
parser.add_argument('--model', type=str, default='small', choices=['tiny', 'small', 'base'])
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--n_crops', type=int, default=144, help='Number of crops per image (default: all 144)')
args = parser.parse_args()

OUTPUT_DIR = args.output_dir or os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f'latent_viz_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print(f"MAE Latent Space Visualization")
print(f"Checkpoint: {args.checkpoint}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 60)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

all_paths = []
for pi in range(1, 7):
    for prefix in ['Drugs_Data', 'Mutants_Data']:
        d = os.path.join(PROJECT_ROOT, prefix, f'P{pi}')
        if os.path.exists(d):
            for ext in ['*.tif', '*.tiff']:
                all_paths.extend(glob.glob(os.path.join(d, '**', ext), recursive=True))
all_paths = sorted(set(all_paths))
print(f"Found {len(all_paths)} images total")

model_map = {'tiny': mae_vit_tiny, 'small': mae_vit_small, 'base': mae_vit_base}
mae = model_map[args.model](in_chans=1, mask_ratio=0.75, norm_pix_loss=True)
ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
mae.load_state_dict(ckpt['model_state_dict'])
mae.to(device)
mae.eval()
print(f"Model loaded ({sum(p.numel() for p in mae.parameters()):,} params)")

plate_map = {}
class_map = {}
drug_to_moa = {}
all_labels = []

for p in all_paths:
    parts = p.split('/')
    plate_prefix = None
    for prefix in ['Drugs_Data', 'Mutants_Data']:
        if prefix in parts:
            plate_prefix = prefix
            break

    plate_idx = None
    condition = None
    for pi in range(1, 7):
        if f'P{pi}' in parts:
            plate_idx = pi
            break

    if plate_prefix == 'Drugs_Data':
        condition_str = parts[parts.index('Drugs_Data') + 2] if len(parts) > parts.index('Drugs_Data') + 2 else 'unknown'
        if '_' in condition_str and condition_str.count('_') >= 1:
            drug_name = condition_str.split('_')[0]
            class_name = condition_str
        else:
            drug_name = condition_str
            class_name = condition_str
    elif plate_prefix == 'Mutants_Data':
        mutant_parts = parts[parts.index('Mutants_Data') + 2] if len(parts) > parts.index('Mutants_Data') + 2 else 'unknown'
        condition_str = f'mutant_{mutant_parts}'
        drug_name = f'mutant_{mutant_parts}'
        class_name = condition_str
    else:
        drug_name = 'unknown'
        class_name = 'unknown'

    plate_map[p] = f'P{plate_idx}' if plate_idx else 'unknown'
    class_map[p] = class_name
    all_labels.append({'path': p, 'drug': drug_name, 'plate': f'P{plate_idx}' if plate_idx else 'unknown',
                        'type': 'drug' if plate_prefix == 'Drugs_Data' else 'mutant'})

moa_map = {}
drug_names_sorted = sorted(set(l['drug'] for l in all_labels))
for i, dn in enumerate(drug_names_sorted):
    moa_map[dn] = f'moa_{i}'
if os.path.exists(os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')):
    with open(os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')) as f:
        ic50 = json.load(f)
    for entry in ic50:
        drug = entry.get('Drug', '')
        moa = entry.get('MOA', '')
        if drug and moa:
            moa_map[f'drug_{drug}'] = moa
            moa_map[drug] = moa
    print(f"Loaded MoA mappings from ic50 file")

for l in all_labels:
    dn = l['drug']
    if dn.startswith('mutant_'):
        l['moa'] = dn.replace('mutant_', '')
    elif dn in moa_map:
        l['moa'] = moa_map[dn]
    else:
        l['moa'] = dn

drug_type = sorted(set(l['drug'] for l in all_labels if l['type'] == 'drug'))
mutant_type = sorted(set(l['drug'] for l in all_labels if l['type'] == 'mutant'))
print(f"Drug conditions: {len(drug_type)}, Mutant conditions: {len(mutant_type)}")

dataset = MAECropDataset(all_paths, augment=False, seed=SEED)
if args.n_crops < 144:
    old_set_epoch = dataset.set_epoch
    def new_set_epoch(epoch):
        old_set_epoch(0)
        num_imgs = len(dataset.image_paths)
        dataset.epoch_centers = {i: dataset.positions[min(i % args.n_crops, len(dataset.positions)-1)] for i in range(num_imgs)}
    dataset.set_epoch = new_set_epoch
    dataset.set_epoch(0)

loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True,
)

print("\nExtracting embeddings...")
all_embeddings = []
with torch.no_grad():
    for imgs in tqdm(loader, desc="Encoding"):
        imgs = imgs.to(device, non_blocking=True)
        emb = mae.encode_pooled(imgs)
        all_embeddings.append(emb.cpu())
embeddings = torch.cat(all_embeddings, dim=0).numpy().astype(np.float64)
print(f"Embeddings shape: {embeddings.shape}")

unique_labels = {}
for i, l in enumerate(all_labels):
    for key in ['drug', 'plate', 'type', 'moa']:
        if key not in unique_labels:
            unique_labels[key] = []
        unique_labels[key].append(l[key])

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
coords_pca = pca.fit_transform(embeddings)

scatter_kwargs = dict(s=3, alpha=0.5, edgecolors='none')

type_colors = {'drug': '#2196F3', 'mutant': '#FF5722'}
type_labels = [unique_labels['type'][i] for i in range(len(embeddings))]
for i, (t, c) in enumerate([('drug', '#2196F3'), ('mutant', '#FF5722')]):
    mask = [x == t for x in type_labels]
    axes[0].scatter(coords_pca[mask, 0], coords_pca[mask, 1], c=c, label=t, **scatter_kwargs)
axes[0].set_title(f'PCA: Drug vs Mutant')
axes[0].legend(markerscale=5)

plate_colors = {f'P{i}': plt.cm.Set1(i-1) for i in range(1, 7)}
plate_labels = [unique_labels['plate'][i] for i in range(len(embeddings))]
for pl in sorted(set(plate_labels)):
    mask = [x == pl for x in plate_labels]
    axes[1].scatter(coords_pca[mask, 0], coords_pca[mask, 1], c=plate_colors.get(pl, 'gray'), label=pl, **scatter_kwargs)
axes[1].set_title(f'PCA: Colored by Plate')
axes[1].legend(markerscale=5)

drug_names_plot = sorted(set(unique_labels['drug']))
n_drugs_plot = len(drug_names_plot)
drug_cmap = plt.cm.tab20
drug_labels_plot = [unique_labels['drug'][i] for i in range(len(embeddings))]
if n_drugs_plot <= 40:
    drug_colors = {d: drug_cmap(i % 20) for i, d in enumerate(drug_names_plot)}
    for d in drug_names_plot:
        mask = [x == d for x in drug_labels_plot]
        axes[2].scatter(coords_pca[mask, 0], coords_pca[mask, 1], c=[drug_colors[d]], label=d, **scatter_kwargs)
    axes[2].set_title(f'PCA: Colored by Drug/Mutant')
    axes[2].legend(markerscale=5, fontsize=4, loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_overview.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved pca_overview.png")

print("\nComputing well-level (bag) embeddings...")
from collections import defaultdict
bag_embeddings = defaultdict(list)
bag_labels = defaultdict(dict)

for i, l in enumerate(all_labels):
    key = (l['drug'], l['plate'])
    bag_embeddings[key].append(embeddings[i])
    bag_labels[key] = l

n_crops_per_well = [len(v) for v in bag_embeddings.values()]
print(f"  Wells: {len(bag_embeddings)}, Crops/well: mean={np.mean(n_crops_per_well):.1f}, min={min(n_crops_per_well)}, max={max(n_crops_per_well)}")

well_emb = np.array([np.mean(v, axis=0) for v in bag_embeddings.values()])
well_keys = list(bag_embeddings.keys())
well_info = [bag_labels[k] for k in well_keys]
print(f"  Well-level embeddings: {well_emb.shape}")

print("Computing cosine similarity matrix...")
n_wells = len(well_emb)
cos_sim = np.zeros((n_wells, n_wells))
for i in range(n_wells):
    ni = np.linalg.norm(well_emb[i])
    for j in range(n_wells):
        nj = np.linalg.norm(well_emb[j])
        cos_sim[i, j] = np.dot(well_emb[i], well_emb[j]) / (ni * nj + 1e-12)

well_names = [f"{info['drug']}_{info['plate']}" for info in well_info]
drug_names_well = [info['drug'] for info in well_info]
type_names_well = [info['type'] for info in well_info]

row_colors = []
for t in type_names_well:
    row_colors.append('#2196F3' if t == 'drug' else '#FF5722')

g = sns.clustermap(cos_sim, row_colors=row_colors, col_colors=row_colors,
                   xticklabels=False, yticklabels=False,
                   figsize=(12, 12), cmap='RdBu_r', vmin=-1, vmax=1,
                   linewidths=0, method='average')
g.fig.suptitle('Cosine Similarity: MAE Well-Level Embeddings', fontsize=14, y=1.02)
drug_patch = mpatches.Patch(color='#2196F3', label='Drug')
mutant_patch = mpatches.Patch(color='#FF5722', label='Mutant')
g.ax_heatmap.legend(handles=[drug_patch, mutant_patch], loc='upper left')
g.savefig(os.path.join(OUTPUT_DIR, 'cosine_similarity_clustered.png'), dpi=150, bbox_inches='tight')
plt.close(g.fig)
print("Saved cosine_similarity_clustered.png")

drug_well_mask = np.array([t == 'drug' for t in type_names_well])
mutant_well_mask = np.array([t == 'mutant' for t in type_names_well])
drug_drug = cos_sim[np.ix_(drug_well_mask, drug_well_mask)]
mutant_mutant = cos_sim[np.ix_(mutant_well_mask, mutant_well_mask)]
drug_mutant = cos_sim[np.ix_(drug_well_mask, mutant_well_mask)]

print(f"\n  Drug-Drug cos sim:   mean={drug_drug.mean():.4f} ± {drug_drug.std():.4f}")
print(f"  Mutant-Mutant cos sim: mean={mutant_mutant.mean():.4f} ± {mutant_mutant.std():.4f}")
print(f"  Drug-Mutant cos sim:  mean={drug_mutant.mean():.4f} ± {drug_mutant.std():.4f}")

well_drug_names = [drug_names_well[i] for i in range(n_wells) if drug_well_mask[i]]
well_mutant_names = [drug_names_well[i] for i in range(n_wells) if mutant_well_mask[i]]

drug_indices = {d: [] for d in set(well_drug_names)}
mutant_indices = {d: [] for d in set(well_mutant_names)}

well_names_arr = np.array(well_names)
for i, d in enumerate(drug_names_well):
    if d in drug_indices:
        drug_indices[d].append(i)
    else:
        mutant_indices[d].append(i)

n_drugs = len(drug_indices)
n_mutants = len(mutant_indices)
print(f"\n  Unique drugs: {n_drugs}, Unique mutants: {n_mutants}")

drug_centroids = {}
for d, idxs in drug_indices.items():
    drug_centroids[d] = well_emb[idxs].mean(axis=0)
mutant_centroids = {}
for m, idxs in mutant_indices.items():
    mutant_centroids[m] = well_emb[idxs].mean(axis=0)

drug_names_cent = list(drug_centroids.keys())
mutant_names_cent = list(mutant_centroids.keys())
drug_vecs = np.array([drug_centroids[d] for d in drug_names_cent])
mutant_vecs = np.array([mutant_centroids[m] for m in mutant_names_cent])

cent_cos = np.zeros((len(drug_names_cent), len(mutant_names_cent)))
for i in range(len(drug_names_cent)):
    for j in range(len(mutant_names_cent)):
        di = drug_vecs[i]
        mj = mutant_vecs[j]
        cent_cos[i, j] = np.dot(di, mj) / (np.linalg.norm(di) * np.linalg.norm(mj) + 1e-12)

drug_moa_list = []
for d in drug_names_cent:
    if d.startswith('drug_'):
        dname = d.replace('drug_', '')
    else:
        dname = d
    dm = moa_map.get(d, dname)
    drug_moa_list.append(dm)

fig, ax = plt.subplots(figsize=(max(10, n_mutants * 0.4), max(8, n_drugs * 0.3)))
im = ax.imshow(cent_cos, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_yticks(range(n_drugs))
ax.set_yticklabels(drug_names_cent, fontsize=4)
ax.set_xticks(range(n_mutants))
ax.set_xticklabels(mutant_names_cent, fontsize=4, rotation=90)
ax.set_xlabel('Mutants')
ax.set_ylabel('Drugs')
ax.set_title('Centroid Cosine Similarity: Drug → Mutant')
plt.colorbar(im, ax=ax, shrink=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'drug_mutant_centroid_cosine.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved drug_mutant_centroid_cosine.png")

print("\nComputing MoA hit rate...")
drug_to_moa_dict = {}
for d, moa_str in zip(drug_names_cent, drug_moa_list):
    drug_to_moa_dict[d] = moa_str

mutant_moa_map = {}
for m in mutant_names_cent:
    mutant_moa_map[m] = m

all_drug_names = drug_names_cent
all_mutant_names = mutant_names_cent

moa_hits = []
for k in [1, 3, 5, 10]:
    correct = 0
    total = 0
    for i, d in enumerate(all_drug_names):
        sims = cent_cos[i]
        top_k_idx = np.argsort(sims)[::-1][:k]
        for idx in top_k_idx:
            matched_mutant = all_mutant_names[idx]
            matched_moa = mutant_moa_map.get(matched_mutant, matched_mutant)
            drug_moa = drug_to_moa_dict.get(d, d)
            if drug_moa == matched_moa:
                correct += 1
                break
        total += 1
    hit_rate = correct / total * 100
    moa_hits.append((k, hit_rate))
    print(f"  Top-{k} MoA hit rate: {hit_rate:.1f}% ({correct}/{total})")

fig, ax = plt.subplots(figsize=(6, 4))
ks, hrs = zip(*moa_hits)
ax.bar([str(k) for k in ks], hrs, color='#4CAF50', alpha=0.7)
ax.set_xlabel('Top-K')
ax.set_ylabel('MoA Hit Rate (%)')
ax.set_title('MAE: Drug→Mutant MoA Hit Rate')
for i, (k, hr) in enumerate(moa_hits):
    ax.text(i, hr + 1, f'{hr:.1f}%', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'moa_hit_rate.png'), dpi=150)
plt.close()
print("Saved moa_hit_rate.png")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
from sklearn.manifold import TSNE
print("\nComputing t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=SEED)
coords_tsne = tsne.fit_transform(embeddings[:5000] if len(embeddings) > 5000 else embeddings)

for i, (t, c) in enumerate([('drug', '#2196F3'), ('mutant', '#FF5722')]):
    mask = [x == t for x in type_labels[:len(coords_tsne)]]
    axes[0].scatter(coords_tsne[mask, 0], coords_tsne[mask, 1], c=c, label=t, s=3, alpha=0.5)
axes[0].set_title('t-SNE: Drug vs Mutant')
axes[0].legend(markerscale=5)

tsne_well = TSNE(n_components=2, perplexity=min(30, len(well_emb)-1), random_state=SEED)
coords_tsne_well = tsne_well.fit_transform(well_emb)

for i, (t, c) in enumerate([('drug', '#2196F3'), ('mutant', '#FF5722')]):
    mask = [x == t for x in type_names_well]
    axes[1].scatter(coords_tsne_well[mask, 0], coords_tsne_well[mask, 1], c=c, label=t, s=20, alpha=0.7, edgecolors='none')
axes[1].set_title('t-SNE: Well-Level')
axes[1].legend(markerscale=3)

top_moa_hits = set()
for i, d in enumerate(all_drug_names[:20]):
    sims = cent_cos[i]
    top1 = all_mutant_names[np.argmax(sims)]
    top_moa = mutant_moa_map.get(top1, top1)
    drug_moa = drug_to_moa_dict.get(d, d)
    if drug_moa == top_moa:
        top_moa_hits.add(d)

if len(set(drug_moa_list)) <= 20:
    moa_names = list(set(drug_moa_list))
    moa_cmap = {m: plt.cm.tab20(i % 20) for i, m in enumerate(moa_names)}
    drug_moa_arr = [drug_moa_list[i] for i in range(len(coords_tsne_well))]
    for m in moa_names:
        mask = [drug_moa_arr[i] if i < len(drug_moa_arr) else None for i in range(len(coords_tsne_well))]
        mask_ok = [x == m for x in drug_moa_arr[:len(coords_tsne_well)]]
        axes[2].scatter(coords_tsne_well[mask_ok, 0], coords_tsne_well[mask_ok, 1], c=[moa_cmap[m]], label=f'{m[:20]}', s=20, alpha=0.7, edgecolors='none')
    axes[2].set_title('t-SNE: Well-Level by MoA')
    axes[2].legend(markerscale=3, fontsize=4, loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_overview.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved tsne_overview.png")

print("\nComputing PCA on well embeddings...")
pca_well = PCA(n_components=min(50, well_emb.shape[0], well_emb.shape[1]))
coords_pca_well = pca_well.fit_transform(well_emb)
var_explained = pca_well.explained_variance_ratio_

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(range(1, len(var_explained)+1), np.cumsum(var_explained) * 100, 'b-', linewidth=2)
axes[0].axhline(y=90, color='r', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Number of PCs')
axes[0].set_ylabel('Cumulative Variance Explained (%)')
axes[0].set_title('PCA Variance Explained')
axes[0].grid(True, alpha=0.3)

for i, (t, c) in enumerate([('drug', '#2196F3'), ('mutant', '#FF5722')]):
    mask = [x == t for x in type_names_well]
    axes[1].scatter(coords_pca_well[mask, 0], coords_pca_well[mask, 1], c=c, label=t, s=20, alpha=0.7, edgecolors='none')
axes[1].set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)')
axes[1].set_title('PCA: Well-Level Embeddings')
axes[1].legend(markerscale=3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_well_level.png'), dpi=150)
plt.close()
print("Saved pca_well_level.png")

pc1_drug = [coords_pca_well[i, 0] for i in range(len(type_names_well)) if type_names_well[i] == 'drug']
pc1_mutant = [coords_pca_well[i, 0] for i in range(len(type_names_well)) if type_names_well[i] == 'mutant']

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(pc1_drug, bins=30, alpha=0.6, color='#2196F3', label='Drug', density=True)
ax.hist(pc1_mutant, bins=30, alpha=0.6, color='#FF5722', label='Mutant', density=True)
ax.set_xlabel('PC1 Score')
ax.set_ylabel('Density')
ax.set_title(f'PC1 Distribution: Drug vs Mutant')
ax.legend()

from sklearn.metrics import roc_auc_score
pc1_all = coords_pca_well[:, 0]
pc1_labels = [1 if t == 'drug' else 0 for t in type_names_well]
auc = roc_auc_score(pc1_labels, -pc1_all)
ax.text(0.98, 0.95, f'AUC = {auc:.3f}', transform=ax.transAxes, ha='right', va='top', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pc1_histogram.png'), dpi=150)
plt.close()
print(f"PC1 AUC for drug-vs-mutant: {auc:.3f}")
print("Saved pc1_histogram.png")

if os.path.exists(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')):
    print("\nLoading plate well ID mapping for per-drug OT analysis...")
    with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')) as f:
        well_id_map = json.load(f)

print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")
print(f"  Embedding dim: {embeddings.shape[1]}")
print(f"  Total crops: {len(embeddings)}")
print(f"  Number of wells: {n_wells}")
print(f"  Drugs: {n_drugs}, Mutants: {n_mutants}")
print(f"  Drug-Drug cos sim: {drug_drug.mean():.4f}")
print(f"  Drug-Mutant cos sim: {drug_mutant.mean():.4f}")
print(f"  PC1 drug-vs-mutant AUC: {auc:.3f}")
print(f"  MoA Top-1: {moa_hits[0][1]:.1f}%")
print(f"  MoA Top-5: {moa_hits[2][1]:.1f}%")
print(f"  MoA Top-10: {moa_hits[3][1]:.1f}%")
print(f"\n  Outputs: {OUTPUT_DIR}")
print(f"{'='*60}")
