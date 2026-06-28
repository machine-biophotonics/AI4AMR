#!/usr/bin/env python3
"""
Cross-domain matching: average projected embeddings per drug type / gene,
compute cosine similarity matrix, find best matches.
"""

import os, sys, json, re
import numpy as np
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Config ──────────────────────────────────────────────────────────────────
FOLD_DIR = os.path.join(SCRIPT_DIR, 'both', 'fold_Plate_6')
PROJ_FILE   = os.path.join(FOLD_DIR, 'proj.npy')
LABELS_FILE = os.path.join(FOLD_DIR, 'labels.npy')
PREDS_FILE  = os.path.join(FOLD_DIR, 'preds.npy')
DOMAINS_FILE= os.path.join(FOLD_DIR, 'domains.npy')

# ── Load embeddings ────────────────────────────────────────────────────────
if not os.path.exists(PROJ_FILE):
    print(f"ERROR: {PROJ_FILE} not found. Run predict_all_crops.py with --save_embeddings first.")
    sys.exit(1)

proj   = np.load(PROJ_FILE)       # N × 256
labels = np.load(LABELS_FILE)     # N
preds  = np.load(PREDS_FILE)      # N
domains = np.load(DOMAINS_FILE)   # 0=drug, 1=mutant

print(f"Loaded {proj.shape[0]} samples, proj_dim={proj.shape[1]}")
print(f"  Drug samples: {(domains==0).sum()}, Mutant samples: {(domains==1).sum()}")

# ── Build class name maps (replicating logic from predict_all_crops.py) ───
with open(os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')) as f:
    IC50_DATA = json.load(f)

with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')) as f:
    MUTANT_DATA = json.load(f)

# Drug classes (with concentration)
drug_set = set()
for plate, wells in IC50_DATA.items():
    for well, info in wells.items():
        antibiotic = info.get('antibiotic', '')
        ic50 = info.get('ic50_multiple', '')
        if antibiotic and ic50:
            if ic50 == 'control':
                drug_set.add('control')
            else:
                ic50_str = ic50 if 'x' in str(ic50) else f'{ic50}x'
                drug_set.add(f'{antibiotic.replace(" ", "_")}_{ic50_str}')
drug_classes = sorted(drug_set)
drug_idx_to_name = {i: n for i, n in enumerate(drug_classes)}
print(f"\nDrug classes (indices 0..{len(drug_classes)-1}): {len(drug_classes)} total")

# Mutant classes
mutant_set = set()
for plate, rows in MUTANT_DATA.items():
    for row, cols in rows.items():
        for col, info in cols.items():
            if 'id' in info:
                mutant_set.add(info['id'])
mutant_classes = sorted(mutant_set)
mutant_idx_to_name = {i: n for i, n in enumerate(mutant_classes)}
print(f"Mutant classes (indices 0..{len(mutant_classes)-1}): {len(mutant_classes)} total")

# ── Map class name → higher-level category ─────────────────────────────────
# Drug: "Avibactam_0.25x" → "Avibactam", "control" → "control"
def drug_category(class_name):
    if class_name == 'control':
        return 'control'
    # Strip the _0.25x, _0.5x, _1x, _2x suffix
    m = re.match(r'^(.+)_(0\.25|0\.5|1|2)x$', class_name)
    if m:
        return m.group(1)
    return class_name  # fallback

# Mutant: "dnaB_1" → "dnaB", "NC_1" → "NC", "WT NC_1" → "WT NC"
def mutant_category(class_name):
    if class_name.startswith('WT NC'):
        return 'WT NC'
    if class_name.startswith('NC_'):
        return 'NC'
    # Strip the _1, _2, _3 suffix from gene names
    m = re.match(r'^([a-zA-Z]+)_\d+$', class_name)
    if m:
        return m.group(1)
    return class_name

# ── Group embeddings by category for each domain ──────────────────────────
drug_cat_embs = defaultdict(list)    # drug_category → list of embeddings
mutant_cat_embs = defaultdict(list)  # mutant_category → list of embeddings

for i in range(len(proj)):
    if domains[i] == 0:  # drug
        cls_name = drug_idx_to_name.get(labels[i], None)
        if cls_name is None:
            continue
        cat = drug_category(cls_name)
        drug_cat_embs[cat].append(proj[i])
    else:  # mutant
        cls_name = mutant_idx_to_name.get(labels[i], None)
        if cls_name is None:
            continue
        cat = mutant_category(cls_name)
        mutant_cat_embs[cat].append(proj[i])

# ── Compute mean embedding per category ───────────────────────────────────
def mean_embedding(emb_list):
    return np.mean(emb_list, axis=0)

drug_cats = sorted(drug_cat_embs.keys())
mutant_cats = sorted(mutant_cat_embs.keys())

drug_means = np.array([mean_embedding(drug_cat_embs[c]) for c in drug_cats])
mutant_means = np.array([mean_embedding(mutant_cat_embs[c]) for c in mutant_cats])

print(f"\nDrug categories ({len(drug_cats)}): {drug_cats}")
print(f"Mutant categories ({len(mutant_cats)}): {mutant_cats}")
print(f"\nDrug means shape: {drug_means.shape}")
print(f"Mutant means shape: {mutant_means.shape}")

# ── Cosine similarity matrix ──────────────────────────────────────────────
def cosine_sim(a, b):
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.dot(a_norm, b_norm.T)  # (n_drug, n_mutant)

sim_matrix = cosine_sim(drug_means, mutant_means)

# ── Best match for each drug category ─────────────────────────────────────
print("\n" + "=" * 80)
print("BEST MUTANT MATCH FOR EACH DRUG CATEGORY")
print("=" * 80)
for i, drug_cat in enumerate(drug_cats):
    best_j = np.argmax(sim_matrix[i])
    best_score = sim_matrix[i, best_j]
    best_mutant = mutant_cats[best_j]
    print(f"  {drug_cat:25s} → {best_mutant:25s}  (cosine sim={best_score:.4f})")

# ── Top-3 for each drug ──────────────────────────────────────────────────
print("\n" + "=" * 80)
print("TOP-3 MUTANT MATCHES FOR EACH DRUG CATEGORY")
print("=" * 80)
for i, drug_cat in enumerate(drug_cats):
    top3 = np.argsort(sim_matrix[i])[::-1][:3]
    matches = '; '.join(f'{mutant_cats[j]} ({sim_matrix[i,j]:.4f})' for j in top3)
    print(f"  {drug_cat:25s} → {matches}")

# ── Also: top drug match for each mutant category ─────────────────────────
print("\n" + "=" * 80)
print("BEST DRUG MATCH FOR EACH MUTANT CATEGORY")
print("=" * 80)
for j, mut_cat in enumerate(mutant_cats):
    best_i = np.argmax(sim_matrix[:, j])
    best_score = sim_matrix[best_i, j]
    best_drug = drug_cats[best_i]
    print(f"  {mut_cat:25s} → {best_drug:25s}  (cosine sim={best_score:.4f})")

# ── Save similarity matrix ────────────────────────────────────────────────
out_path = os.path.join(FOLD_DIR, 'cross_domain_matching.npz')
np.savez(out_path,
         drug_categories=drug_cats,
         mutant_categories=mutant_cats,
         similarity_matrix=sim_matrix)
print(f"\nSaved matching matrix to {out_path}")

print("\nDone.")
