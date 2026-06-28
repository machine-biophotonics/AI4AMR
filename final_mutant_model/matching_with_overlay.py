#!/usr/bin/env python3
"""
Cross-domain matching: drugs at 2x (all positions) vs mutant genes (all 3 guides).
Shows cosine similarity heatmap with expected match green boxes overlaid.
"""

import os, sys, json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict, OrderedDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOLD_DIR = os.path.join(SCRIPT_DIR, 'both', 'fold_Plate_6')

PROJ_FILE    = os.path.join(FOLD_DIR, 'proj.npy')
LABELS_FILE  = os.path.join(FOLD_DIR, 'labels.npy')
DOMAINS_FILE = os.path.join(FOLD_DIR, 'domains.npy')

# ── Expected matches (from literature) ─────────────────────────────────────
# Keys: drug short name (as displayed on heatmap y-axis)
# Values: list of mutant gene names
expected = {
    'Cefsulodin':      ['mrcA', 'mrcB'],
    'Penicillin':      ['mrcA', 'mrcB', 'ftsI'],
    'Sulbactam':       [],  # no essential target in WT E. coli (β-lactamase inhibitor)
    'Avibactam':       [],  # no essential target in WT E. coli (β-lactamase inhibitor)
    'Mecillinam':      ['mrdA'],
    'Meropenem':       ['mrdA', 'ftsI', 'mrcA', 'mrcB'],
    'Clavulanic Acid': [],  # no essential target in WT E. coli (β-lactamase inhibitor)
    'Relebactam':      [],  # no essential target in WT E. coli (β-lactamase inhibitor)
    'Aztreonam':       ['ftsI'],
    'Cefepim':         ['ftsI', 'mrcA', 'mrcB', 'mrdA'],
    'Ceftriaxone':     ['ftsI', 'mrcA', 'mrcB'],
    'Chloramphenicol': [],  # rRNA-targeted (23S PTC) — no gene-encoded target in this set
    'Clarithromycin':  [],  # rRNA-targeted (23S exit tunnel) — no gene-encoded target in this set
    'Doxicyclin':      [],  # rRNA-targeted (16S A-site) — no gene-encoded target in this set
    'Kanamycin':       [],  # rRNA-targeted (16S h44) — no gene-encoded target in this set
    'Ciprofloxacin':   ['gyrA', 'gyrB', 'parC', 'parE'],
    'Levofloxacin':    ['gyrA', 'gyrB', 'parC', 'parE'],
    'Norfloxacin':     ['gyrA', 'gyrB', 'parC', 'parE'],
    'Rifampicin':      ['rpoB'],    # rpoA is general RNAP stress, not target-specific
    'Trimethoprim':    ['folA'],    # folP is NOT a direct TMP target (DHFR only)
    'Colistin':        ['lpxA', 'lpxC', 'lptA', 'lptC'],
    'Polymyxin B':     ['lpxA', 'lpxC', 'lptA', 'lptC'],
}

# ── Load embeddings ────────────────────────────────────────────────────────
for f in [PROJ_FILE, LABELS_FILE, DOMAINS_FILE]:
    if not os.path.exists(f):
        print(f"ERROR: {f} not found. Run predict_all_crops.py --save_embeddings first.")
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

# Drug: index → class name (with concentration)
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

# Mutant: index → class name (with guide suffix)
mutant_set = set()
for plate, rows in MUTANT_DATA.items():
    for row, cols in rows.items():
        for col, info in cols.items():
            if 'id' in info:
                mutant_set.add(info['id'])
mutant_classes = sorted(mutant_set)
mutant_idx_to_name = OrderedDict((i, n) for i, n in enumerate(mutant_classes))

# ── Filter to drug 2x only ────────────────────────────────────────────────
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
drug_short = {}
for idx, name in drug_idx_to_name.items():
    if name in drug_2x_names:
        drug_2x_idx.add(idx)
        short = name.replace('_2x', '').replace('_', ' ')
        drug_short[idx] = short

# ── Mutant: group by gene (all guides) ────────────────────────────────────
def gene_name(mutant_class_name):
    if mutant_class_name.startswith('WT NC'):
        return 'WT NC'
    if mutant_class_name.startswith('NC_'):
        return 'NC'
    m = re.match(r'^([a-zA-Z]+)_\d+$', mutant_class_name)
    if m:
        return m.group(1)
    return mutant_class_name

mutant_gene_sets = defaultdict(set)  # gene_name → set of indices
for idx, name in mutant_idx_to_name.items():
    g = gene_name(name)
    mutant_gene_sets[g].add(idx)

mutant_genes_sorted = sorted(mutant_gene_sets.keys())
print(f"\nDrug 2x classes: {len(drug_2x_idx)}")
print(f"Mutant gene groups: {len(mutant_genes_sorted)}")

# ── Group embeddings ───────────────────────────────────────────────────────
drug_embs = {idx: [] for idx in drug_2x_idx}
mutant_embs = {g: [] for g in mutant_genes_sorted}

for i in range(len(proj)):
    lbl = int(labels[i])
    if domains[i] == 0 and lbl in drug_embs:
        drug_embs[lbl].append(proj[i])
    elif domains[i] == 1:
        name = mutant_idx_to_name.get(lbl)
        if name:
            g = gene_name(name)
            if g in mutant_embs:
                mutant_embs[g].append(proj[i])

# ── Mean embeddings ────────────────────────────────────────────────────────
sorted_drug_idxs = sorted(drug_2x_idx)
drug_means = np.array([np.mean(drug_embs[idx], axis=0) for idx in sorted_drug_idxs])
mutant_means = np.array([np.mean(mutant_embs[g], axis=0) for g in mutant_genes_sorted])

# ── Cosine similarity ─────────────────────────────────────────────────────
def cosine_sim(a, b):
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.dot(a_norm, b_norm.T)

sim = cosine_sim(drug_means, mutant_means)
print(f"Similarity matrix: {sim.shape}")

# ── Labels ─────────────────────────────────────────────────────────────────
drug_labels = [drug_short[idx] for idx in sorted_drug_idxs]
mutant_labels = list(mutant_genes_sorted)

# ── Build expected match mask for green boxes ──────────────────────────────
# expected_mask[i, j] = True if mutant_labels[j] is an expected match for drug_labels[i]
expected_mask = np.zeros(sim.shape, dtype=bool)
for i, dl in enumerate(drug_labels):
    if dl in expected:
        for target_gene in expected[dl]:
            if target_gene in mutant_genes_sorted:
                j = mutant_genes_sorted.index(target_gene)
                expected_mask[i, j] = True

# ── Draw heatmap ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 14))
sns.heatmap(sim,
            xticklabels=mutant_labels,
            yticklabels=drug_labels,
            cmap='RdBu_r',
            center=0.85, vmin=0.65, vmax=1.0,
            square=False,
            annot=False,
            linewidths=0.5, linecolor='gray',
            ax=ax)
ax.set_title('Cross-domain matching: drugs at 2x vs mutant genes (all guides)', fontsize=16)
ax.set_xlabel('Mutant gene (all 3 guides averaged)', fontsize=13)
ax.set_ylabel('Drug type (2x)', fontsize=13)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=9)

# ── Green boxes on expected matches ────────────────────────────────────────
for i in range(sim.shape[0]):
    for j in range(sim.shape[1]):
        if expected_mask[i, j]:
            rect = mpatches.Rectangle((j, i), 1, 1,
                                      fill=False, edgecolor='lime', linewidth=3, clip_on=False)
            ax.add_patch(rect)

# ── Legend ──────────────────────────────────────────────────────────────────
green_patch = mpatches.Patch(edgecolor='lime', linewidth=3, fill=False, label='Expected match (literature)')
ax.legend(handles=[green_patch], loc='upper right', fontsize=12)

plt.tight_layout()
out_file = os.path.join(FOLD_DIR, 'matching_overlay_2x_vs_all_guides.png')
plt.savefig(out_file, dpi=200, bbox_inches='tight')
print(f"Saved: {out_file}")

# ── Print best matches with expected check ─────────────────────────────────
print("\n" + "=" * 80)
print("BEST MATCH (drug 2x → mutant gene, all guides)")
print("=" * 80)
hit_count = 0
total_expected = 0
for i, dl in enumerate(drug_labels):
    best_j = np.argmax(sim[i])
    score = sim[i, best_j]
    best_m = mutant_labels[best_j]
    is_expected = "✓" if expected_mask[i, best_j] else "✗"
    print(f"  {dl:20s} → {best_m:20s}  ({score:.4f})  {is_expected}")

# ── Check if expected match is in top-K for each drug ─────────────────────
print("\n" + "=" * 80)
print("TOP-3 WITH EXPECTED MATCH HIGHLIGHT")
print("=" * 80)
for i, dl in enumerate(drug_labels):
    top5 = np.argsort(sim[i])[::-1][:5]
    expected_genes = expected.get(dl, [])
    found_expected = [g for g in expected_genes if g in mutant_labels]
    if not found_expected:
        continue
    rank_strs = []
    for rank, j in enumerate(top5):
        marker = "★" if expected_mask[i, j] else " "
        rank_strs.append(f"{mutant_labels[j]:12s}{marker}({sim[i,j]:.4f})")
    print(f"  {dl:20s}: {' | '.join(rank_strs)}")

# ── Stats ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("STATS")
print("=" * 80)
for i, dl in enumerate(drug_labels):
    expected_genes = expected.get(dl, [])
    found_expected = [g for g in expected_genes if g in mutant_labels]
    if not found_expected:
        continue
    # Find ranks of expected genes in similarity list
    order = np.argsort(sim[i])[::-1]
    ranks = [int(np.where(order == mutant_labels.index(g))[0][0]) + 1 for g in found_expected]
    print(f"  {dl:20s}: expected={found_expected}, ranks={ranks}")

print("\nDone.")
