#!/usr/bin/env python3
"""
Grouped cross-domain matching heatmap.
22 drugs at 2x (guide 1 mutants), grouped by drug class and gene pathway.
Plot 1: drug × gene heatmap with group separators.
Plot 2: group-level average (8 drug classes × 11 pathways) simplified heatmap.
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

# ── Groupings ──────────────────────────────────────────────────────────────
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
    ('PG biosynth.',           ['murA', 'murC']),
    ('Cell division',          ['ftsZ']),
    ('DNA replication',        ['dnaB', 'dnaE']),
    ('Topoisomerases',         ['gyrA', 'gyrB', 'parC', 'parE']),
    ('RNA polymerase',         ['rpoA', 'rpoB']),
    ('Ribosome',               ['rpsA', 'rpsL', 'rplA', 'rplC']),
    ('Sec translocon',         ['secA', 'secY']),
    ('Folate biosynth.',       ['folA', 'folP']),
    ('LPS biosynth.',          ['lpxA', 'lpxC']),
    ('LPS transport',          ['lptA', 'lptC']),
])

EXPECTED = {
    'Cefsulodin':      ['mrcA', 'mrcB'],
    'Penicillin':      ['mrcA', 'mrcB', 'ftsI'],
    'Sulbactam':       [], 'Avibactam': [], 'Clavulanic Acid': [], 'Relebactam': [],
    'Mecillinam':      ['mrdA'],
    'Meropenem':       ['mrdA', 'ftsI', 'mrcA', 'mrcB'],
    'Aztreonam':       ['ftsI'],
    'Cefepim':         ['ftsI', 'mrcA', 'mrcB', 'mrdA'],
    'Ceftriaxone':     ['ftsI', 'mrcA', 'mrcB'],
    'Chloramphenicol': [], 'Clarithromycin': [], 'Doxicyclin': [], 'Kanamycin': [],
    'Ciprofloxacin':   ['gyrA', 'gyrB', 'parC', 'parE'],
    'Levofloxacin':    ['gyrA', 'gyrB', 'parC', 'parE'],
    'Norfloxacin':     ['gyrA', 'gyrB', 'parC', 'parE'],
    'Rifampicin':      ['rpoB'],
    'Trimethoprim':    ['folA'],
    'Colistin':        ['lpxA', 'lpxC', 'lptA', 'lptC'],
    'Polymyxin B':     ['lpxA', 'lpxC', 'lptA', 'lptC'],
}

# ── Load ───────────────────────────────────────────────────────────────────
for f in [PROJ_FILE, LABELS_FILE, DOMAINS_FILE]:
    if not os.path.exists(f):
        print(f"ERROR: {f} not found"); sys.exit(1)

proj = np.load(PROJ_FILE); labels = np.load(LABELS_FILE); domains = np.load(DOMAINS_FILE)
print(f"Loaded {proj.shape[0]} samples, dim={proj.shape[1]}")

with open(os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')) as f: IC50 = json.load(f)
with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')) as f: MUT = json.load(f)

# ── Build class maps ───────────────────────────────────────────────────────
drug_set = set()
for plate, wells in IC50.items():
    for well, info in wells.items():
        ab, ic = info.get('antibiotic',''), info.get('ic50_multiple','')
        if ab and ic:
            drug_set.add(f'{ab.replace(" ", "_")}_{ic}' if ic != 'control' else 'control')
drug_idx_to_name = {i: n for i, n in enumerate(sorted(drug_set))}

mutant_set = set()
for plate, rows in MUT.items():
    for row, cols in rows.items():
        for col, info in cols.items():
            if 'id' in info:
                mutant_set.add(info['id'])
mutant_idx_to_name = {i: n for i, n in enumerate(sorted(mutant_set))}

# ── Drug 2x indices in group order ─────────────────────────────────────────
drug_2x_idxs = {}
for idx, name in drug_idx_to_name.items():
    if name.endswith('_2x'):
        short = name.replace('_2x', '').replace('_', ' ')
        drug_2x_idxs[idx] = short

short_to_idx = {v: k for k, v in drug_2x_idxs.items()}
drug_order = []
for gn, members in DRUG_GROUPS.items():
    for m in members:
        if m in short_to_idx:
            drug_order.append(short_to_idx[m])
print(f"Drug 2x: {len(drug_order)}")

# ── Mutant guide-1 genes in group order ────────────────────────────────────
def gene_name(n):
    if n.startswith('WT NC'): return 'WT NC'
    if n.startswith('NC_'): return 'NC'
    m = re.match(r'^([a-zA-Z]+)_\d+$', n)
    return m.group(1) if m else n

gene_to_idxs = defaultdict(set)
for idx, name in mutant_idx_to_name.items():
    g = gene_name(name)
    if g not in ('NC', 'WT NC') and name.endswith('_1'):
        gene_to_idxs[g].add(idx)

available_genes = []
for gn, members in GENE_GROUPS.items():
    for m in members:
        if m in gene_to_idxs and m not in available_genes:
            available_genes.append(m)
print(f"Mutant genes: {len(available_genes)}")

# ── Group embeddings ───────────────────────────────────────────────────────
drug_emb = {idx: [] for idx in drug_order}
gene_emb = defaultdict(list)
for i in range(len(proj)):
    lbl = int(labels[i])
    if domains[i] == 0 and lbl in drug_emb:
        drug_emb[lbl].append(proj[i])
    elif domains[i] == 1:
        name = mutant_idx_to_name.get(lbl)
        if name:
            g = gene_name(name)
            if g in available_genes:
                gene_emb[g].append(proj[i])

drug_means = np.array([np.mean(drug_emb[idx], axis=0) for idx in drug_order])
gene_means = np.array([np.mean(gene_emb[g], axis=0) for g in available_genes])

def cosine_sim(a, b):
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.dot(an, bn.T)

sim = cosine_sim(drug_means, gene_means)
print(f"Matrix: {sim.shape}, range=[{sim.min():.4f}, {sim.max():.4f}]")

# ── Group info for separators ──────────────────────────────────────────────
idx_to_short = {k: v for k, v in drug_2x_idxs.items()}
drug_group_of = {}
for i, di in enumerate(drug_order):
    short = idx_to_short[di]
    for gname, members in DRUG_GROUPS.items():
        if short in members:
            drug_group_of[i] = gname
            break

gene_group_of = {}
for j, gn in enumerate(available_genes):
    for gname, members in GENE_GROUPS.items():
        if gn in members:
            gene_group_of[j] = gname
            break

# ── Expected mask ──────────────────────────────────────────────────────────
expected_mask = np.zeros(sim.shape, dtype=bool)
for i, di in enumerate(drug_order):
    ds = idx_to_short[di]
    if ds in EXPECTED:
        for tg in EXPECTED[ds]:
            if tg in available_genes:
                expected_mask[i, available_genes.index(tg)] = True

# ── Helper: find group boundaries ─────────────────────────────────────────
def group_boundaries(assignments, n):
    boundaries = []
    for i in range(n):
        if i == 0 or assignments[i] != assignments[i-1]:
            boundaries.append(i)
    boundaries.append(n)
    return boundaries

drug_boundaries = group_boundaries([drug_group_of[i] for i in range(sim.shape[0])], sim.shape[0])
gene_boundaries = group_boundaries([gene_group_of[j] for j in range(sim.shape[1])], sim.shape[1])

# =========================================================================
# PLOT 1: Detailed heatmap with group separators
# =========================================================================
drug_labels = [idx_to_short[d] for d in drug_order]
gene_labels = list(available_genes)

fig1, ax1 = plt.subplots(figsize=(20, 14))
sns.heatmap(sim, xticklabels=gene_labels, yticklabels=drug_labels,
            cmap='RdBu_r', center=0.5, vmin=0.0, vmax=1.0,
            annot=True, fmt='.2f', annot_kws={'fontsize': 4.5},
            linewidths=0.3, linecolor='lightgray',
            cbar_kws={'label': 'Cosine similarity', 'shrink': 0.6}, ax=ax1)
ax1.set_title('Cross-domain matching: drugs at 2x vs mutant genes (guide 1)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Mutant gene', fontsize=11)
ax1.set_ylabel('Antibiotic (2x)', fontsize=11)
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=8)

# Group separators (white lines)
for b in drug_boundaries[1:-1]:
    ax1.axhline(b, color='white', linewidth=3)
for b in gene_boundaries[1:-1]:
    ax1.axvline(b, color='white', linewidth=3)

# Add drug group labels on y-axis (using text at margins)
drug_group_names_unique = list(OrderedDict.fromkeys([drug_group_of[i] for i in range(sim.shape[0])]))
for gi, gname in enumerate(drug_group_names_unique):
    start = drug_boundaries[gi]
    end = drug_boundaries[gi+1]
    mid = (start + end) / 2 - 0.5
    ax1.text(-1.8, mid, gname, ha='right', va='center', fontsize=8,
             fontweight='bold', rotation=0, transform=ax1.get_yaxis_transform())

# Add gene group labels on x-axis
gene_group_names_unique = list(OrderedDict.fromkeys([gene_group_of[j] for j in range(sim.shape[1])]))
for gi, gname in enumerate(gene_group_names_unique):
    start = gene_boundaries[gi]
    end = gene_boundaries[gi+1]
    mid = (start + end) / 2 - 0.5
    ax1.text(mid, 1.03, gname, ha='center', va='bottom', fontsize=7,
             fontweight='bold', rotation=30, transform=ax1.get_xaxis_transform())

# Expected match green boxes
for i in range(sim.shape[0]):
    for j in range(sim.shape[1]):
        if expected_mask[i, j]:
            ax1.add_patch(mpatches.Rectangle((j, i), 1, 1, fill=False,
                          edgecolor='lime', linewidth=2, clip_on=False))
ax1.legend(handles=[mpatches.Patch(edgecolor='lime', linewidth=2, fill=False,
                                   label='Expected phenocopy')],
           loc='lower left', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FOLD_DIR, 'matching_grouped.png'), dpi=200, bbox_inches='tight')
print(f"Saved: matching_grouped.png")
plt.close(fig1)

# =========================================================================
# PLOT 2: Simplified group-level heatmap (8 drug classes × 11 pathways)
# =========================================================================
# Average embeddings within each drug group
drug_group_embs = defaultdict(list)
for i, di in enumerate(drug_order):
    gname = drug_group_of[i]
    drug_group_embs[gname].extend(drug_emb[di])

# Average embeddings within each gene pathway
gene_group_embs = defaultdict(list)
for j, gn in enumerate(available_genes):
    gname = gene_group_of[j]
    gene_group_embs[gname].extend(gene_emb[gn])

dg_names = list(DRUG_GROUPS.keys())
gg_names = list(GENE_GROUPS.keys())

dg_means = np.array([np.mean(drug_group_embs[g], axis=0) for g in dg_names])
gg_means = np.array([np.mean(gene_group_embs[g], axis=0) for g in gg_names])

sim2 = cosine_sim(dg_means, gg_means)

fig2, ax2 = plt.subplots(figsize=(12, 8))
sns.heatmap(sim2, xticklabels=gg_names, yticklabels=dg_names,
            cmap='RdBu_r', center=0.5, vmin=0.0, vmax=1.0,
            annot=True, fmt='.2f', annot_kws={'fontsize': 9},
            linewidths=1, linecolor='lightgray',
            cbar_kws={'label': 'Cosine similarity', 'shrink': 0.7}, ax=ax2)
ax2.set_title('Cross-domain matching: drug class × gene pathway (averaged)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Gene pathway', fontsize=11)
ax2.set_ylabel('Antibiotic class', fontsize=11)
plt.xticks(rotation=30, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FOLD_DIR, 'matching_group_level.png'), dpi=200, bbox_inches='tight')
print(f"Saved: matching_group_level.png")
plt.close(fig2)

# =========================================================================
# Print matches
# =========================================================================
print("\n" + "=" * 70)
print("BEST MATCHES (drug → gene)")
hit = total = 0
for i, di in enumerate(drug_order):
    ds = idx_to_short[di]
    j = np.argmax(sim[i])
    score = sim[i, j]
    best = gene_labels[j]
    exp = EXPECTED.get(ds, [])
    if exp:
        total += 1
        if expected_mask[i, j]:
            hit += 1
    chk = "✓" if expected_mask[i, j] else "✗"
    print(f"  {ds:20s} → {best:8s}  ({score:.4f})  {chk}  exp={exp}")
print(f"\n  Hit rate: {hit}/{total} ({hit/max(total,1)*100:.0f}%)")
print("\nDone.")
