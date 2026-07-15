#!/usr/bin/env python3
"""
Cross-domain matching: cosine similarity between drug 2x embeddings and mutant guide 1 embeddings.
Saves matching_heatmap.png with expected-match green boxes + prints best matches and hit rate.
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

EXPECTED = {
    'Cefsulodin':      ['mrcA', 'mrcB'],
    'Penicillin':      ['mrcA', 'mrcB', 'ftsI'],
    'Sulbactam':       [],
    'Avibactam':       [],
    'Mecillinam':      ['mrdA'],
    'Meropenem':       ['mrdA', 'ftsI', 'mrcA', 'mrcB'],
    'Clavulanic Acid': [],
    'Relebactam':      [],
    'Aztreonam':       ['ftsI'],
    'Cefepim':         ['ftsI', 'mrcA', 'mrcB', 'mrdA'],
    'Ceftriaxone':     ['ftsI', 'mrcA', 'mrcB'],
    'Chloramphenicol': [],
    'Clarithromycin':  [],
    'Doxicyclin':      [],
    'Kanamycin':       [],
    'Ciprofloxacin':   ['gyrA', 'gyrB', 'parC', 'parE'],
    'Levofloxacin':    ['gyrA', 'gyrB', 'parC', 'parE'],
    'Norfloxacin':     ['gyrA', 'gyrB', 'parC', 'parE'],
    'Rifampicin':      ['rpoB'],
    'Trimethoprim':    ['folA'],
    'Colistin':        ['lpxA', 'lpxC', 'lptA', 'lptC'],
    'Polymyxin B':     ['lpxA', 'lpxC', 'lptA', 'lptC'],
}

for f in [PROJ_FILE, LABELS_FILE, DOMAINS_FILE]:
    if not os.path.exists(f):
        print(f"ERROR: {f} not found. Run predict_all_crops.py --data_mode both --dual_classifier --save_embeddings first.")
        sys.exit(1)

proj    = np.load(PROJ_FILE)
labels  = np.load(LABELS_FILE)
domains = np.load(DOMAINS_FILE)
print(f"Loaded {proj.shape[0]} samples, proj_dim={proj.shape[1]}")

with open(os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')) as f:
    IC50_DATA = json.load(f)
with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')) as f:
    MUTANT_DATA = json.load(f)

drug_set = set()
for plate, wells in IC50_DATA.items():
    for well, info in wells.items():
        ab = info.get('antibiotic', '')
        ic = info.get('ic50_multiple', '')
        if ab and ic:
            if ic == 'control':
                drug_set.add('control')
            else:
                drug_set.add(f'{ab.replace(" ", "_")}_{ic}')
drug_idx_to_name = OrderedDict((i, n) for i, n in enumerate(sorted(drug_set)))

mutant_set = set()
for plate, rows in MUTANT_DATA.items():
    for row, cols in rows.items():
        for col, info in cols.items():
            if 'id' in info:
                mutant_set.add(info['id'])
mutant_idx_to_name = OrderedDict((i, n) for i, n in enumerate(sorted(mutant_set)))

# ── Drug: 2x + control ───────────────────────────────────────────────────
def gene_name(n):
    if n.startswith('WT NC'): return 'WT NC'
    if n.startswith('NC_'): return 'NC'
    m = re.match(r'^([a-zA-Z]+)_\d+$', n)
    return m.group(1) if m else n

drug_idxs = {}
for idx, name in drug_idx_to_name.items():
    if name == 'control':
        drug_idxs[idx] = 'control'
    elif name.endswith('_2x'):
        short = name.replace('_2x', '').replace('_', ' ')
        drug_idxs[idx] = short

# ── Mutant: all guides per gene + controls ───────────────────────────────
mutant_groups = defaultdict(list)
for idx, name in mutant_idx_to_name.items():
    g = gene_name(name)
    mutant_groups[g].append(idx)

available_genes = sorted(mutant_groups.keys())

# ── Average embeddings per class ──────────────────────────────────────────
drug_embs = {idx: [] for idx in drug_idxs}
gene_embs = {g: [] for g in available_genes}

for i in range(len(proj)):
    lbl = int(labels[i])
    if domains[i] == 0 and lbl in drug_embs:
        drug_embs[lbl].append(proj[i])
    elif domains[i] == 1:
        name = mutant_idx_to_name.get(lbl)
        if name:
            g = gene_name(name)
            if g in gene_embs:
                gene_embs[g].append(proj[i])

sorted_drug_idxs = sorted(drug_idxs.keys())
drug_means = np.array([np.mean(drug_embs[idx], axis=0) for idx in sorted_drug_idxs])
gene_means = np.array([np.mean(gene_embs[g], axis=0) for g in available_genes])

drug_labels = [drug_idxs[idx] for idx in sorted_drug_idxs]
gene_labels = list(available_genes)

print(f"\nDrug 2x + control: {len(drug_labels)} classes")
print(f"Mutant genes (all guides) + NC + WT NC: {len(gene_labels)}")

# ── Cosine similarity ─────────────────────────────────────────────────────
def cosine_sim(a, b):
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.dot(an, bn.T)

sim = cosine_sim(drug_means, gene_means)
print(f"Similarity matrix: {sim.shape}, range=[{sim.min():.4f}, {sim.max():.4f}]")

# ── Expected match mask ───────────────────────────────────────────────────
expected_mask = np.zeros(sim.shape, dtype=bool)
for i, dl in enumerate(drug_labels):
    if dl in EXPECTED:
        for tg in EXPECTED[dl]:
            if tg in available_genes:
                expected_mask[i, available_genes.index(tg)] = True

# ── Heatmap ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 14))
sns.heatmap(sim,
            xticklabels=gene_labels,
            yticklabels=drug_labels,
            cmap='RdBu_r',
            center=0.5, vmin=0.0, vmax=1.0,
            annot=True, fmt='.2f',
            annot_kws={'fontsize': 5},
            linewidths=0.3, linecolor='lightgray',
            cbar_kws={'label': 'Cosine similarity', 'shrink': 0.6},
            ax=ax)
ax.set_title('Cross-domain matching: drugs (2x + control) vs mutant genes', fontsize=14, fontweight='bold')
ax.set_xlabel('Mutant gene (all guides + controls)', fontsize=11)
ax.set_ylabel('Antibiotic (2x + control)', fontsize=11)
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=8)

for i in range(sim.shape[0]):
    for j in range(sim.shape[1]):
        if expected_mask[i, j]:
            ax.add_patch(mpatches.Rectangle((j, i), 1, 1,
                          fill=False, edgecolor='lime', linewidth=2.5, clip_on=False))

ax.legend(handles=[mpatches.Patch(edgecolor='lime', linewidth=2.5, fill=False,
                                  label='Expected phenocopy')],
          loc='lower left', fontsize=9)

plt.tight_layout()
out_png = os.path.join(FOLD_DIR, 'matching_heatmap.png')
plt.savefig(out_png, dpi=200, bbox_inches='tight')
print(f"\nSaved heatmap to {out_png}")
plt.close()

# ── Save matrix ───────────────────────────────────────────────────────────
out_npz = os.path.join(FOLD_DIR, 'cross_domain_matching.npz')
np.savez(out_npz,
         drug_categories=drug_labels,
         mutant_categories=gene_labels,
         similarity_matrix=sim)
print(f"Saved matrix to {out_npz}")

# ── Top-5 matches with hit rate ───────────────────────────────────────────
print("\n" + "=" * 90)
print("TOP-5 MATCHES (drug → mutant gene, all guides + NC/WT NC)")
print("=" * 90)
hit = total = 0
for i, dl in enumerate(drug_labels):
    top5 = np.argsort(sim[i])[::-1][:5]
    matches = []
    is_hit = False
    for j in top5:
        marker = " ★" if expected_mask[i, j] else "  "
        matches.append(f"{gene_labels[j]:12s}{marker} ({sim[i,j]:.4f})")
        if expected_mask[i, j]:
            is_hit = True
    exp_genes = EXPECTED.get(dl, [])
    if exp_genes:
        total += 1
        if is_hit:
            hit += 1
    chk = "✓" if is_hit else "✗"
    print(f"  {dl:20s}  {chk}  {' | '.join(matches)}")
    if dl in EXPECTED and exp_genes:
        found = [g for g in exp_genes if g in available_genes]
        if found:
            order = np.argsort(sim[i])[::-1]
            ranks = [int(np.where(order == available_genes.index(g))[0][0]) + 1 for g in found]
            print(f"  {'':20s}     exp={found} ranks={ranks}")

print(f"\n  Hit rate: {hit}/{total} ({hit/max(total,1)*100:.0f}%)")

# ═══════════════════════════════════════════════════════════════════════════
#  MOA × PATHWAY GROUPED MATCHING
# ═══════════════════════════════════════════════════════════════════════════

DRUG_GROUPS = OrderedDict([
    ('PBP 1A/B',    ['Cefsulodin', 'Penicillin', 'Sulbactam']),
    ('PBP 2',       ['Avibactam', 'Mecillinam', 'Meropenem',
                      'Clavulanic Acid', 'Relebactam']),
    ('PBP 3',       ['Aztreonam', 'Cefepim', 'Ceftriaxone']),
    ('Ribosome',    ['Chloramphenicol', 'Clarithromycin', 'Doxicyclin',
                      'Kanamycin']),
    ('DNA gyrase',  ['Ciprofloxacin', 'Levofloxacin', 'Norfloxacin']),
    ('RNA polymerase', ['Rifampicin']),
    ('DNA synthesis',  ['Trimethoprim']),
    ('Membrane integrity', ['Colistin', 'Polymyxin B']),
])

GENE_GROUPS = OrderedDict([
    ('Cell wall\n(PBPs)',      ['mrcA', 'mrcB', 'mrdA', 'ftsI']),
    ('PG biosynthesis',        ['murA', 'murC']),
    ('Cell division',          ['ftsZ']),
    ('DNA replication',        ['dnaB', 'dnaE']),
    ('Topoisomerases',         ['gyrA', 'gyrB', 'parC', 'parE']),
    ('RNA polymerase',         ['rpoA', 'rpoB']),
    ('Ribosome',               ['rpsA', 'rpsL', 'rplA', 'rplC']),
    ('Sec translocon',         ['secA', 'secY']),
    ('Folate biosynthesis',    ['folA', 'folP']),
    ('LPS biosynthesis',       ['lpxA', 'lpxC']),
    ('LPS transport',          ['lptA', 'lptC']),
])

# Build reverse map: drug_short → MOA group name
drug_to_moa = {}
for moa, members in DRUG_GROUPS.items():
    for m in members:
        drug_to_moa[m] = moa

# Collect embeddings per MOA group
moa_embs = defaultdict(list)
for i, dl in enumerate(drug_labels):
    if dl == 'control':
        continue
    if dl in drug_to_moa:
        moa = drug_to_moa[dl]
        idx = sorted_drug_idxs[i]
        moa_embs[moa].extend(drug_embs[idx])

# Build reverse map: gene → pathway
gene_to_pathway = {}
for pathway, members in GENE_GROUPS.items():
    for m in members:
        gene_to_pathway[m] = pathway

# Collect embeddings per pathway
pathway_embs = defaultdict(list)
for j, gl in enumerate(gene_labels):
    if gl in gene_to_pathway:
        pathway_embs[gene_to_pathway[gl]].extend(gene_embs[gl])

moa_names = [m for m in DRUG_GROUPS.keys() if m in moa_embs]
pathway_names = [p for p in GENE_GROUPS.keys() if p in pathway_embs]

moa_means = np.array([np.mean(moa_embs[m], axis=0) for m in moa_names])
pathway_means = np.array([np.mean(pathway_embs[p], axis=0) for p in pathway_names])

sim_moa = cosine_sim(moa_means, pathway_means)
print(f"\nMOA × Pathway matrix: {sim_moa.shape}, range=[{sim_moa.min():.4f}, {sim_moa.max():.4f}]")

# Heatmap
fig2, ax2 = plt.subplots(figsize=(12, 8))
sns.heatmap(sim_moa,
            xticklabels=[p.replace('\n', ' ') for p in pathway_names],
            yticklabels=moa_names,
            cmap='RdBu_r',
            center=0.5, vmin=0.0, vmax=1.0,
            annot=True, fmt='.2f', annot_kws={'fontsize': 9},
            linewidths=1, linecolor='lightgray',
            cbar_kws={'label': 'Cosine similarity', 'shrink': 0.7},
            ax=ax2)
ax2.set_title('Cross-domain matching: drug MOA class × gene pathway', fontsize=13, fontweight='bold')
ax2.set_xlabel('Gene pathway', fontsize=11)
ax2.set_ylabel('Drug MOA class', fontsize=11)
plt.xticks(rotation=30, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
out_moa = os.path.join(FOLD_DIR, 'matching_moa_heatmap.png')
plt.savefig(out_moa, dpi=200, bbox_inches='tight')
print(f"Saved MOA heatmap to {out_moa}")
plt.close(fig2)

# Print best pathway match for each MOA class
print("\n" + "=" * 70)
print("BEST PATHWAY MATCH FOR EACH MOA CLASS")
print("=" * 70)
for i, moa in enumerate(moa_names):
    j = np.argmax(sim_moa[i])
    print(f"  {moa:25s} → {pathway_names[j]:25s}  ({sim_moa[i,j]:.4f})")

# Print best MOA match for each pathway
print("\n" + "=" * 70)
print("BEST MOA MATCH FOR EACH PATHWAY")
print("=" * 70)
for j, pathway in enumerate(pathway_names):
    i = np.argmax(sim_moa[:, j])
    print(f"  {pathway:25s} → {moa_names[i]:25s}  ({sim_moa[i,j]:.4f})")

# ═══════════════════════════════════════════════════════════════════════════
#  PER-GUIDE TOP-5 MATCHES (preserving individual guide numbers)
# ═══════════════════════════════════════════════════════════════════════════

drug_2x_guide = defaultdict(list)
mutant_guide = defaultdict(list)

for i in range(len(proj)):
    lbl = int(labels[i])
    if domains[i] == 0:
        name = drug_idx_to_name.get(lbl)
        if name and name.endswith('_2x'):
            drug_2x_guide[lbl].append(proj[i])
    elif domains[i] == 1:
        name = mutant_idx_to_name.get(lbl)
        if name and re.match(r'^[a-zA-Z]+_\d+$', name):
            mutant_guide[lbl].append(proj[i])

sorted_drug_guide = sorted(drug_2x_guide.keys())
sorted_mutant_guide = sorted(mutant_guide.keys())

drug_guide_means = np.array([np.mean(drug_2x_guide[idx], axis=0) for idx in sorted_drug_guide])
mutant_guide_means = np.array([np.mean(mutant_guide[idx], axis=0) for idx in sorted_mutant_guide])

drug_guide_labels = [drug_idx_to_name[idx].replace('_2x', '').replace('_', ' ') for idx in sorted_drug_guide]
mutant_guide_labels = [mutant_idx_to_name[idx] for idx in sorted_mutant_guide]

sim_guide = cosine_sim(drug_guide_means, mutant_guide_means)
print(f"\nPer-guide matrix: {sim_guide.shape}, range=[{sim_guide.min():.4f}, {sim_guide.max():.4f}]")

print("\n" + "=" * 95)
print("TOP-5 MUTANT GUIDES FOR EACH DRUG (2x)")
print("=" * 95)
for i, dl in enumerate(drug_guide_labels):
    top5 = np.argsort(sim_guide[i])[::-1][:5]
    matches = ' | '.join(f'{mutant_guide_labels[j]:15s}({sim_guide[i,j]:.4f})' for j in top5)
    print(f"  {dl:20s}: {matches}")

print("\n" + "=" * 95)
print("TOP-5 DRUGS (2x) FOR EACH MUTANT GUIDE")
print("=" * 95)
for j, ml in enumerate(mutant_guide_labels):
    top5 = np.argsort(sim_guide[:, j])[::-1][:5]
    matches = ' | '.join(f'{drug_guide_labels[i]:20s}({sim_guide[i,j]:.4f})' for i in top5)
    print(f"  {ml:15s}: {matches}")

print("\nDone.")
