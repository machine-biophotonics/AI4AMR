#!/usr/bin/env python3
"""Wasserstein distance between each drug antibiotic and each mutant gene."""
import numpy as np, json, re, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import ot  # POT

BASE = os.path.dirname(os.path.abspath(__file__))
FOLD = "Plate_1"
DATA = np.load(f"{BASE}/both/fold_{FOLD}/embeddings_{FOLD}_mil_n3.npz", allow_pickle=True)
paths = DATA["paths"]
embeddings = DATA["embeddings"]
IC50 = json.load(open(f"{BASE}/plate_well_ic50_mapping.json"))
MUT = json.load(open(f"{BASE}/plate_well_id_path.json"))

def fix_label(img_path):
    pl = img_path.lower()
    if "/drugs_data/" in pl: src = "drug"
    elif "/mutants_data/" in pl: src = "mutant"
    else: return None, None
    m = re.search(r"Well(\w\d+)_", os.path.basename(img_path))
    well = m.group(1) if m else None
    if not well: return None, None
    pk = None
    for pn in range(1,7):
        if f"/p{pn}/" in pl: pk = f"P{pn}"; break
    if not pk: return None, None
    if src == "drug":
        if pk in IC50 and well in IC50[pk]:
            info = IC50[pk][well]
            ab = info.get("antibiotic","")
            ic = info.get("ic50_multiple","")
            if ab and ic:
                if ic == "control": return None, None
                return "drug", ab
    else:
        row, col_raw = well[0], well[1:].lstrip("0") or "0"
        try:
            if pk in MUT and row in MUT[pk] and col_raw in MUT[pk][row]:
                label = MUT[pk][row][col_raw].get("id","unknown")
                if label in ("unknown", "NC", "WT NC"): return None, None
                return "mutant", label.rsplit("_",1)[0] if "_" in label else label
        except: pass
    return None, None

# Collect embeddings per group
drug_groups = {}
mutant_groups = {}
for emb, path in zip(embeddings, paths):
    src, name = fix_label(path)
    if src == "drug":
        drug_groups.setdefault(name, []).append(emb)
    elif src == "mutant":
        mutant_groups.setdefault(name, []).append(emb)

print(f"Drug groups: {len(drug_groups)} -> {sorted(drug_groups.keys())}")
print(f"Mutant groups: {len(mutant_groups)} -> {sorted(mutant_groups.keys())}")
for k, v in drug_groups.items():
    print(f"  {k}: {len(v)} samples")
for k, v in mutant_groups.items():
    print(f"  {k}: {len(v)} samples")

drug_names = sorted(drug_groups.keys())
mutant_names = sorted(mutant_groups.keys())
n_d = len(drug_names)
n_m = len(mutant_names)

# ---- 1. Sliced Wasserstein Distance ----
print("\nComputing sliced Wasserstein distance (500 projections)...")
swd = np.zeros((n_d, n_m))
for i, dname in enumerate(drug_names):
    darr = np.array(drug_groups[dname])  # (N_d, 1280)
    for j, mname in enumerate(mutant_names):
        marr = np.array(mutant_groups[mname])  # (N_m, 1280)
        swd[i,j] = ot.sliced_wasserstein_distance(darr, marr, n_projections=500, seed=42)
    print(f"  {dname}: {i+1}/{n_d} done")
print(f"SWD range: {swd.min():.3f} - {swd.max():.3f}")

# ---- 2. Centroid Euclidean Distance ----
print("\nComputing centroid Euclidean distance...")
centroid_euc = np.zeros((n_d, n_m))
for i, dname in enumerate(drug_names):
    dcent = np.mean(drug_groups[dname], axis=0)
    for j, mname in enumerate(mutant_names):
        mcent = np.mean(mutant_groups[mname], axis=0)
        centroid_euc[i,j] = np.linalg.norm(dcent - mcent)
print(f"Euclidean range: {centroid_euc.min():.3f} - {centroid_euc.max():.3f}")

outdir = f"{BASE}/both/fold_{FOLD}"
np.savez(f"{outdir}/wasserstein_distances.npz",
         drug_names=drug_names, mutant_names=mutant_names,
         swd=swd, centroid_euclidean=centroid_euc)

# ---- Heatmap ----
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

# SWD heatmap
ax = axes[0]
vmin, vmax = swd.min(), swd.max()
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
im = ax.imshow(swd, aspect='auto', cmap='viridis_r', norm=norm)
ax.set_xticks(range(n_m))
ax.set_xticklabels(mutant_names, rotation=90, fontsize=7)
ax.set_yticks(range(n_d))
ax.set_yticklabels(drug_names, fontsize=8)
ax.set_title("Sliced Wasserstein Distance", fontsize=14)
ax.set_xlabel("Mutant Gene", fontsize=12)
ax.set_ylabel("Drug Antibiotic", fontsize=12)
fig.colorbar(im, ax=ax, shrink=0.8)

# Clustered heatmap with seaborn
ax = axes[1]
g = sns.clustermap(swd, xticklabels=mutant_names, yticklabels=drug_names,
                   cmap='viridis_r', figsize=(16, 10), linewidths=0.5,
                   linecolor='gray', method='average', dendrogram_ratio=(0.15, 0.15),
                   cbar_kws={'label': 'Sliced Wasserstein Distance'})
g.ax_heatmap.set_xlabel("Mutant Gene", fontsize=12)
g.ax_heatmap.set_ylabel("Drug Antibiotic", fontsize=12)
g.fig.suptitle("Clustered SWD: Drug Antibiotics × Mutant Genes", fontsize=14, y=1.02)
g.savefig(f"{outdir}/wasserstein_drug_mutant_clustered.png", dpi=150, bbox_inches='tight')
plt.close('all')

# Bar plot: for each drug, which mutant is closest?
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(n_d)
width = 0.35
best_mutant_idx = np.argmin(swd, axis=1)
best_mutant_dist = swd[np.arange(n_d), best_mutant_idx]
colors = [plt.cm.tab10(i % 10) for i in best_mutant_idx]
ax.bar(x, best_mutant_dist, width, color=colors)
ax.set_xticks(x)
ax.set_xticklabels(drug_names, rotation=45, ha='right', fontsize=9)
ax.set_ylabel("Minimum SWD to any mutant", fontsize=12)
ax.set_title("Closest mutant gene for each antibiotic", fontsize=14)
for i, idx in enumerate(best_mutant_idx):
    ax.text(i, best_mutant_dist[i]+0.1, mutant_names[idx], ha='center', va='bottom', fontsize=7, rotation=45)
plt.tight_layout()
plt.savefig(f"{outdir}/wasserstein_drug_closest_mutant.png", dpi=150, bbox_inches='tight')
plt.close()

# Also save CSV
import csv
with open(f"{outdir}/wasserstein_drug_mutant_matrix.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Drug\\Mutant"] + mutant_names)
    for i, dname in enumerate(drug_names):
        w.writerow([dname] + [f"{swd[i,j]:.4f}" for j in range(n_m)])
print("\nAll saved to", outdir)

