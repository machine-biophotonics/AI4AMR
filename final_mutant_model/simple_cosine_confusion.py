import os, json, re, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOLD_DIR = os.path.join(SCRIPT_DIR, 'both', 'fold_Plate_6')

proj = np.load(os.path.join(FOLD_DIR, 'proj.npy'))
labels = np.load(os.path.join(FOLD_DIR, 'labels.npy'))
domains = np.load(os.path.join(FOLD_DIR, 'domains.npy'))

with open(os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')) as f:
    IC50 = json.load(f)
with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')) as f:
    MUT = json.load(f)

drug_set = set()
for plate, wells in IC50.items():
    for well, info in wells.items():
        ab, ic = info.get('antibiotic',''), info.get('ic50_multiple','')
        if ab and ic:
            drug_set.add(f'{ab.replace(" ", "_")}_{ic}' if ic != 'control' else 'control')
drug_classes = sorted(drug_set)
drug_idx_to_name = {i: n for i, n in enumerate(drug_classes)}

mutant_set = set()
for plate, rows in MUT.items():
    for row, cols in rows.items():
        for col, info in cols.items():
            if 'id' in info:
                mutant_set.add(info['id'])
mutant_classes = sorted(mutant_set)
mutant_idx_to_name = {i: n for i, n in enumerate(mutant_classes)}

is_drug = domains == 0
is_mutant = domains == 1

drug_label = labels[is_drug]
mutant_label = labels[is_mutant]

drug_2x_mask = np.array([drug_idx_to_name[l].endswith('_2x') or drug_idx_to_name[l] == 'control' for l in drug_label])
mutant_g1_mask = np.array([mutant_idx_to_name[l].endswith('_1') or mutant_idx_to_name[l].startswith('NC') or mutant_idx_to_name[l].startswith('WT NC') for l in mutant_label])

drug_labels_2x = drug_label[drug_2x_mask]
mutant_labels_g1 = mutant_label[mutant_g1_mask]

drug_emb = proj[is_drug][drug_2x_mask]
mutant_emb = proj[is_mutant][mutant_g1_mask]

drug_means = {}
for lbl in np.unique(drug_labels_2x):
    mask = drug_labels_2x == lbl
    drug_means[drug_idx_to_name[lbl]] = drug_emb[mask].mean(axis=0)

mutant_means = {}
for lbl in np.unique(mutant_labels_g1):
    mask = mutant_labels_g1 == lbl
    mutant_means[mutant_idx_to_name[lbl]] = mutant_emb[mask].mean(axis=0)

def cat_name(n):
    if n == 'control' or n.startswith('NC') or n.startswith('WT NC'):
        return n
    m = re.match(r'^(.+)_(?:0\.25|0\.5|1|2)x$', n)
    return m.group(1) if m else n

def gene_name(n):
    if n.startswith('WT NC'): return 'WT NC'
    if n.startswith('NC_'): return 'NC'
    m = re.match(r'^([a-zA-Z]+)_\d+$', n)
    return m.group(1) if m else n

drug_cats = sorted(set(cat_name(n) for n in drug_means))
mutant_cats = sorted(set(gene_name(n) for n in mutant_means))

drug_cat_mean = {}
for dc in drug_cats:
    vals = [v for n, v in drug_means.items() if cat_name(n) == dc]
    drug_cat_mean[dc] = np.mean(vals, axis=0)

mutant_cat_mean = {}
for mc in mutant_cats:
    vals = [v for n, v in mutant_means.items() if gene_name(n) == mc]
    mutant_cat_mean[mc] = np.mean(vals, axis=0)

d_list = [drug_cat_mean[k] for k in drug_cats]
m_list = [mutant_cat_mean[k] for k in mutant_cats]
sim = np.dot(np.array(d_list), np.array(m_list).T)
d_norm = np.linalg.norm(np.array(d_list), axis=1, keepdims=True)
m_norm = np.linalg.norm(np.array(m_list), axis=1, keepdims=True)
sim = sim / (d_norm @ m_norm.T)

THRESH = 0.9
cm = (sim >= THRESH).astype(int)

fig, ax = plt.subplots(figsize=(10, 7))
ax.imshow(cm, cmap='gray_r', aspect='auto')
ax.set_xticks(range(len(mutant_cats)))
ax.set_yticks(range(len(drug_cats)))
ax.set_xticklabels(mutant_cats, rotation=90, fontsize=8)
ax.set_yticklabels(drug_cats, fontsize=8)
ax.set_xlabel('Mutant gene (guide 1)')
ax.set_ylabel('Drug type (2x)')
ax.set_title(f'Cosine similarity >= {THRESH} (drug 2x, mutant guide 1)')

plt.tight_layout()
out = os.path.join(FOLD_DIR, 'cosine_confusion_2x_g1.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved: {out} (threshold={THRESH})')
for i, d in enumerate(drug_cats):
    hits = [mutant_cats[j] for j in range(len(mutant_cats)) if cm[i, j]]
    if hits:
        print(f'  {d:20s} -> {", ".join(hits)}')
plt.close()
