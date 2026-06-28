#!/usr/bin/env python3
"""
Cross-domain matching heatmap: drugs at 2x vs mutant guide 1.
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOLD_DIR = os.path.join(SCRIPT_DIR, 'both', 'fold_Plate_6')

PROJ_FILE    = os.path.join(FOLD_DIR, 'proj.npy')
LABELS_FILE  = os.path.join(FOLD_DIR, 'labels.npy')
DOMAINS_FILE = os.path.join(FOLD_DIR, 'domains.npy')

# ── Load embeddings ────────────────────────────────────────────────────────
for f in [PROJ_FILE, LABELS_FILE, DOMAINS_FILE]:
    if not os.path.exists(f):
        print(f"ERROR: {f} not found. Run predict_all_crops.py with --save_embeddings first.")
        sys.exit(1)

proj    = np.load(PROJ_FILE)
labels  = np.load(LABELS_FILE)
domains = np.load(DOMAINS_FILE)
print(f"Loaded {proj.shape[0]} samples, proj_dim={proj.shape[1]}")

# ── Build class name maps ──────────────────────────────────────────────────
with open(os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')) as f:
    IC50_DATA = json.load(f)
with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')) as f:
    MUTANT_DATA = json.load(f)

drug_set = set()
for plate, wells in IC50_DATA.items():
    for well, info in wells.items():
        ab = info.get('antibiotic', '')
        ic50 = info.get('ic50_multiple', '')
        if ab and ic50:
            if ic50 == 'control':
                drug_set.add('control')
            else:
                drug_set.add(f'{ab.replace(" ", "_")}_{ic50}')
drug_classes = sorted(drug_set)
drug_idx_to_name = OrderedDict((i, n) for i, n in enumerate(drug_classes))

mutant_set = set()
for plate, rows in MUTANT_DATA.items():
    for row, cols in rows.items():
        for col, info in cols.items():
            if 'id' in info:
                mutant_set.add(info['id'])
mutant_classes = sorted(mutant_set)
mutant_idx_to_name = OrderedDict((i, n) for i, n in enumerate(mutant_classes))

print(f"Drug class indices: 0..{len(drug_classes)-1}")
print(f"Mutant class indices: 0..{len(mutant_classes)-1}")

# ── Filter: drug 2x only ──────────────────────────────────────────────────
drug_2x_names = set()
for plate, wells in IC50_DATA.items():
    for well, info in wells.items():
        ab = info.get('antibiotic', '')
        ic50 = info.get('ic50_multiple', '')
        if ab and ic50:
            if ic50 == 'control':
                drug_2x_names.add('control')
            elif ic50 == '2x':
                drug_2x_names.add(f'{ab.replace(" ", "_")}_2x')
drug_2x_idx = set()
drug_2x_to_short = {}
for idx, name in drug_idx_to_name.items():
    if name in drug_2x_names:
        drug_2x_idx.add(idx)
        short = name.replace('_2x', '').replace('_', ' ')
        drug_2x_to_short[idx] = short

# ── Filter: mutant guide 1 only ───────────────────────────────────────────
mutant_g1_names = {m for m in mutant_set if m.endswith('_1')}
mutant_g1_idx = set()
mutant_g1_to_short = {}
for idx, name in mutant_idx_to_name.items():
    if name in mutant_g1_names:
        mutant_g1_idx.add(idx)
        short = name.replace('_1', '').replace('_', ' ')
        mutant_g1_to_short[idx] = short

print(f"\nDrug 2x classes: {len(drug_2x_idx)}")
print(f"Mutant guide-1 classes: {len(mutant_g1_idx)}")

# ── Group embeddings by class ──────────────────────────────────────────────
drug_embs = {idx: [] for idx in drug_2x_idx}
mutant_embs = {idx: [] for idx in mutant_g1_idx}

for i in range(len(proj)):
    lbl = int(labels[i])
    if domains[i] == 0 and lbl in drug_embs:
        drug_embs[lbl].append(proj[i])
    elif domains[i] == 1 and lbl in mutant_embs:
        mutant_embs[lbl].append(proj[i])

drug_means = np.array([np.mean(drug_embs[idx], axis=0) for idx in sorted(drug_2x_idx)])
mutant_means = np.array([np.mean(mutant_embs[idx], axis=0) for idx in sorted(mutant_g1_idx)])

# ── Cosine similarity ─────────────────────────────────────────────────────
def cosine_sim(a, b):
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.dot(a_norm, b_norm.T)

sim_matrix = cosine_sim(drug_means, mutant_means)
print(f"Similarity matrix shape: {sim_matrix.shape}")

# ── Labels for plot ───────────────────────────────────────────────────────
drug_labels = [drug_2x_to_short[idx] for idx in sorted(drug_2x_idx)]
mutant_labels = [mutant_g1_to_short[idx] for idx in sorted(mutant_g1_idx)]

# ── Heatmap ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 14))
sns.heatmap(sim_matrix,
            xticklabels=mutant_labels,
            yticklabels=drug_labels,
            cmap='RdBu_r',
            center=0.85, vmin=0.65, vmax=1.0,
            square=False,
            annot=False,
            linewidths=0,
            ax=ax)
ax.set_title('Cross-domain matching: drugs at 2x vs mutant guide 1', fontsize=16)
ax.set_xlabel('Mutant gene (guide 1)', fontsize=13)
ax.set_ylabel('Drug type (2x)', fontsize=13)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()

out_png = os.path.join(FOLD_DIR, 'matching_heatmap_2x_vs_g1.png')
plt.savefig(out_png, dpi=200, bbox_inches='tight')
print(f"Saved heatmap to {out_png}")

# ── Also print best matches ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("BEST MATCHES (drug 2x → mutant guide 1)")
print("=" * 70)
for i, dl in enumerate(drug_labels):
    best_j = np.argmax(sim_matrix[i])
    print(f"  {dl:20s} → {mutant_labels[best_j]:20s}  ({sim_matrix[i,best_j]:.4f})")

print("\n" + "=" * 70)
print("BEST MATCHES (mutant guide 1 → drug 2x)")
print("=" * 70)
for j, ml in enumerate(mutant_labels):
    best_i = np.argmax(sim_matrix[:, j])
    print(f"  {ml:20s} → {drug_labels[best_i]:20s}  ({sim_matrix[best_i,j]:.4f})")

print("\nDone.")
