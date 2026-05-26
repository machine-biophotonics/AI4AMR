#!/usr/bin/env python3
"""PCA analysis: remove drug-vs-mutant domain axis and recompute MoA hit rate."""
import os, sys, warnings, re
warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.decomposition import PCA

SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/3 autoencoder approach/mil_vae_both/fold_P1'
LATENTS_PATH = os.path.join(OUTPUT_DIR, 'test_latents_P1_20260523_222527.pt')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MOA = {
    'fluoroquinolones': {'drugs': ['ciprofloxacin','levofloxacin','norfloxacin'], 'targets': ['gyra','gyrb','parc','pare']},
    'rifamycins': {'drugs': ['rifampicin'], 'targets': ['rpoa','rpob']},
    'folate_inhibitors': {'drugs': ['trimethoprim'], 'targets': ['fola','folp']},
    'ribosome_50s': {'drugs': ['chloramphenicol','clarithromycin'], 'targets': ['rpla','rplc']},
    'ribosome_30s': {'drugs': ['doxicyclin','kanamycin'], 'targets': ['rpsa','rpsl']},
    'penems': {'drugs': ['penicillin','mecillinam','meropenem'], 'targets': ['mrca','mrcb','mrda','ftsi']},
    'cephalosporins': {'drugs': ['cefepim','cefsulodin','ceftriaxone'], 'targets': ['ftsi','mrca','mrcb','mrda']},
    'polymyxins': {'drugs': ['polymyxin_b','colistin'], 'targets': ['lpxa','lpxc','lpta','lptc','msba']},
}

def drug_base(name):
    m = re.match(r'^(.+)_(\d+(?:\.\d+)?x)$', name)
    return m.group(1).lower() if m else name.lower()
def mutant_gene(lbl):
    m = re.match(r'^([a-zA-Z]+)', lbl)
    return m.group(1).lower() if m else lbl
def is_moa(d, m):
    db = drug_base(d)
    mg = mutant_gene(m)
    if db == 'control': return False
    for g in MOA.values():
        if db in g['drugs'] and mg in g['targets']:
            return True
    return False

print("=" * 60)
print("PCA domain removal analysis")
print("=" * 60)

pt = torch.load(LATENTS_PATH, map_location='cpu', weights_only=False)
records = pt['records']

# Accumulate per class
class_bags = defaultdict(list)
class_mus = defaultdict(list)
class_src = {}
for r in records:
    class_bags[r['true_label']].append(r['bag'].astype(np.float64))
    class_mus[r['true_label']].append(r['mu'].astype(np.float64))
    class_src[r['true_label']] = r['source']

classes = sorted(class_bags.keys())
N = len(classes)

# Mean 1280-dim
mean_bags = np.zeros((N, 1280))
for i, c in enumerate(classes):
    all_b = np.concatenate(class_bags[c], axis=0)
    mean_bags[i] = all_b.mean(axis=0)

src_arr = np.array([class_src[c] for c in classes])
drug_mask = src_arr == 'drug'
mut_mask = src_arr == 'mutant'
label_arr = np.array(['drug' if s == 'drug' else 'mutant' for s in src_arr])

# ---- PCA on mean class vectors ----
print(f"\n[1/4] PCA on {N} mean class vectors (1280-dim) ...")
pca = PCA(n_components=min(50, N))
scores = pca.fit_transform(mean_bags)
var_exp = pca.explained_variance_ratio_
cum_var = np.cumsum(var_exp)
print(f"  PC1: {var_exp[0]*100:.2f}% variance")
print(f"  PC2: {var_exp[1]*100:.2f}%")
print(f"  PC3: {var_exp[2]*100:.2f}%")
print(f"  PC1-5 cumulative: {cum_var[4]*100:.2f}%")
print(f"  PC1-10 cumulative: {cum_var[9]*100:.2f}%")

# How well does each PC separate drug vs mutant?
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
pc_separation = []
for k in range(min(20, N)):
    lr = LogisticRegression(class_weight='balanced')
    lr.fit(scores[:, k:k+1], (src_arr == 'drug').astype(int))
    pred = lr.predict_proba(scores[:, k:k+1])[:, 1]
    auc = roc_auc_score((src_arr == 'drug').astype(int), pred)
    pc_separation.append(auc)
    if auc > 0.8:
        print(f"  PC{k+1}: domain separation AUC = {auc:.4f}")

# Find PCs that separate domains (AUC > threshold)
SEP_THRESH = 0.7
n_pcs = len(pc_separation)
sep_pcs = [k for k in range(n_pcs) if pc_separation[k] > SEP_THRESH]
if not sep_pcs:
    sep_pcs = [0]  # at least PC1
print(f"  PCs separating drug vs mutant (AUC > {SEP_THRESH}): {len(sep_pcs)} PCs")

# ---- Visualize PCA space ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

colors = ['#E41A1C' if s == 'drug' else '#4DAF4A' for s in label_arr]
markers = ['o' if s == 'drug' else '^' for s in label_arr]

for ax, (pcx, pcy) in zip(axes, [(0, 1), (2, 3), (4, 5)]):
    for i in range(N):
        ax.scatter(scores[i, pcx], scores[i, pcy],
                   c=colors[i], marker=markers[i], s=20, alpha=0.6, edgecolors='none')
    ax.set_xlabel(f'PC{pcx+1} ({var_exp[pcx]*100:.1f}%)', fontsize=10)
    ax.set_ylabel(f'PC{pcy+1} ({var_exp[pcy]*100:.1f}%)', fontsize=10)
    ax.axhline(0, color='gray', lw=0.5, alpha=0.3)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.3)

axes[0].legend([plt.Line2D([0],[0], color='#E41A1C', marker='o', linestyle='none', label='Drug'),
                plt.Line2D([0],[0], color='#4DAF4A', marker='^', linestyle='none', label='Mutant')],
               ['Drug', 'Mutant'], fontsize=9)
axes[0].set_title(f'PC1 vs PC2', fontsize=12)
axes[1].set_title(f'PC3 vs PC4', fontsize=12)
axes[2].set_title(f'PC5 vs PC6', fontsize=12)
fig.suptitle('PCA on 185 mean class vectors (1280-dim bag)', fontsize=14, y=1.02)
plt.tight_layout()
pca_path = os.path.join(OUTPUT_DIR, 'pca_domain_separation.png')
fig.savefig(pca_path, dpi=150, bbox_inches='tight')
print(f"\n  PCA scatter: {pca_path}")
plt.close()

# ---- Remove top domain PCs ----
print(f"\n[2/4] Removing {len(sep_pcs)} domain-separating PCs ...")
loadings = pca.components_  # (n_components, 1280)
# Build projection matrix onto null-space of selected PCs
if sep_pcs:
    A = loadings[sep_pcs]  # (k, 1280)
    # Projection onto orthogonal complement: I - A^T (A A^T)^{-1} A
    # Since A is orthonormal (PCA components are orthonormal), P = I - A^T A
    P = np.eye(1280) - A.T @ A
    mean_bags_clean = mean_bags @ P.T
else:
    mean_bags_clean = mean_bags.copy()

# ---- Recompute cosine similarity and MoA ----
def evaluate_moa(mean_vecs, label):
    norms = np.linalg.norm(mean_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    sim = (mean_vecs / norms) @ (mean_vecs / norms).T

    drug_idx = [i for i in range(N) if src_arr[i] == 'drug']
    mut_idx = [i for i in range(N) if src_arr[i] == 'mutant']
    drug_classes = [classes[i] for i in drug_idx]
    mut_classes = [classes[i] for i in mut_idx]

    hits1 = hits3 = hits5 = total = 0
    for di, d in enumerate(drug_classes):
        if d == 'control': continue
        m_sim = sim[drug_idx[di]][mut_idx]
        ranked = np.argsort(-m_sim)
        total += 1
        if any(is_moa(d, mut_classes[r]) for r in ranked[:1]): hits1 += 1
        if any(is_moa(d, mut_classes[r]) for r in ranked[:3]): hits3 += 1
        if any(is_moa(d, mut_classes[r]) for r in ranked[:5]): hits5 += 1

    print(f"  [{label}] Top-1: {hits1}/{total} = {100*hits1/total:.1f}%  "
          f"Top-3: {hits3}/{total} = {100*hits3/total:.1f}%  "
          f"Top-5: {hits5}/{total} = {100*hits5/total:.1f}%")
    return sim, drug_idx, mut_idx, drug_classes, mut_classes

print("\n[3/4] MoA evaluation:")
print("  Before removing domain PCs (original 1280-dim):")
sim_orig, drug_idx, mut_idx, drug_classes, mut_classes = evaluate_moa(mean_bags, 'original')

print("  After removing domain PCs:")
sim_clean, _, _, _, _ = evaluate_moa(mean_bags_clean, 'clean')

# Also try removing just PC1, just PC1-2, etc.
print("\n[4/4] Sweeping number of removed PCs:")
for n_remove in [0, 1, 2, 3, 5, 10, 20]:
    if n_remove == 0:
        vecs = mean_bags
    else:
        top_pcs = loadings[:n_remove]
        P = np.eye(1280) - top_pcs.T @ top_pcs
        vecs = mean_bags @ P.T
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    sim = (vecs / norms) @ (vecs / norms).T
    hits1 = hits5 = total = 0
    for di, d in enumerate(drug_classes):
        if d == 'control': continue
        m_sim = sim[drug_idx[di]][mut_idx]
        ranked = np.argsort(-m_sim)
        total += 1
        if any(is_moa(d, drug_classes_r) for drug_classes_r in [mut_classes[ranked[0]]] if is_moa(d, mut_classes[ranked[0]])):
            hits1 += 1
        if any(is_moa(d, mut_classes[r]) for r in ranked[:5]):
            hits5 += 1
    # recompute properly
    hits1_v = 0
    for di, d in enumerate(drug_classes):
        if d == 'control': continue
        m_sim = sim[drug_idx[di]][mut_idx]
        ranked = np.argsort(-m_sim)
        if is_moa(d, mut_classes[ranked[0]]):
            hits1_v += 1
    hits5_v = 0
    for di, d in enumerate(drug_classes):
        if d == 'control': continue
        m_sim = sim[drug_idx[di]][mut_idx]
        ranked = np.argsort(-m_sim)[:5]
        if any(is_moa(d, mut_classes[r]) for r in ranked):
            hits5_v += 1
    print(f"  Remove top {n_remove:2d} PCs: Top-1={hits1_v}/{total}={100*hits1_v/total:.1f}%  Top-5={hits5_v}/{total}={100*hits5_v/total:.1f}%")

# ---- Show true positive examples after removal ----
print(f"\nTrue positives after removing {len(sep_pcs)} domain PCs (top-1):")
for di, d in enumerate(drug_classes):
    if d == 'control': continue
    m_sim = sim_clean[drug_idx[di]][mut_idx]
    ranked = np.argsort(-m_sim)
    best_m = mut_classes[ranked[0]]
    if is_moa(d, best_m):
        print(f"  {d:35s} → {best_m:15s}  cos={m_sim[ranked[0]]:.4f}")

print(f"\n{'=' * 60}")
print("Done")
print(f"{'=' * 60}")
