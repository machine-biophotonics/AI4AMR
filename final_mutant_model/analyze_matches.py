#!/usr/bin/env python3
"""Analyze Wasserstein matches: threshold, known ground truth, what it means."""
import numpy as np, json, re, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.abspath(__file__))
FOLD = "Plate_1"
outdir = f"{BASE}/both/fold_{FOLD}"
npz = np.load(f"{outdir}/wasserstein_distances.npz", allow_pickle=True)
swd = npz["swd"]  # (22, 30)
drug_names = list(npz["drug_names"])
mutant_names = list(npz["mutant_names"])
n_d, n_m = swd.shape

# ---- Ground truth drug-target pairs ----
GROUND_TRUTH = {
    "Ciprofloxacin": ["gyrA","gyrB","parC","parE"],
    "Norfloxacin":   ["gyrA","gyrB","parC","parE"],
    "Levofloxacin":  ["gyrA","gyrB","parC","parE"],
    "Rifampicin":    ["rpoB","rpoA"],
    "Chloramphenicol": ["rplC","rplA","rpsA","rpsL"],
    "Doxicyclin":    ["rplC","rplA","rpsA","rpsL"],
    "Clarithromycin":["rplC","rplA","rpsA","rpsL"],
    "Kanamycin":     ["rpsA","rpsL"],
    "Trimethoprim":  ["folA"],
    "Penicillin":    ["mrcA","mrcB","mrdA","ftsI"],
    "Meropenem":     ["mrcA","mrcB","mrdA","ftsI"],
    "Cefepim":       ["mrcA","mrcB","mrdA","ftsI"],
    "Ceftriaxone":   ["mrcA","mrcB","mrdA","ftsI"],
    "Cefsulodin":    ["mrcA","mrcB","mrdA","ftsI"],
    "Mecillinam":    ["mrcA","mrcB","mrdA","ftsI"],
    "Aztreonam":     ["mrcA","mrcB","mrdA","ftsI"],
    "Colistin":      ["lpxC","lpxA","msbA"],
    "Polymyxin B":   ["lpxC","lpxA","msbA"],
    "Avibactam":     [],  # beta-lactamase inhibitor, no direct target
    "Sulbactam":     [],
    "Relebactam":    [],
    "Clavulanic Acid": [],
}

# ---- Distribution of all SWD values ----
all_vals = swd.flatten()
print(f"All SWD: mean={all_vals.mean():.4f}, std={all_vals.std():.4f}")
print(f"  min={all_vals.min():.4f} (drug={drug_names[swd.argmin()//n_m]}, mut={mutant_names[swd.argmin()%n_m]})")
print(f"  max={all_vals.max():.4f}")

# Percentiles
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th percentile = {np.percentile(all_vals, p):.4f}")

# ---- Method 1: Best mutant for each drug (lowest SWD) ----
print("\n=== Best mutant match for each drug ===")
for i, d in enumerate(drug_names):
    best_j = np.argmin(swd[i])
    best_val = swd[i, best_j]
    gt = GROUND_TRUTH.get(d, [])
    hit = " ✓" if mutant_names[best_j] in gt else ""
    print(f"  {d:20s} → {mutant_names[best_j]:8s} (SWD={best_val:.4f}){hit}")

# ---- Method 2: Top-3 for each drug ----
print("\n=== Top-3 mutants for each drug ===")
for i, d in enumerate(drug_names):
    top3 = np.argsort(swd[i])[:3]
    names = [f"{mutant_names[j]}({swd[i,j]:.3f})" for j in top3]
    gt = GROUND_TRUTH.get(d, [])
    hits = [mutant_names[j] for j in top3 if mutant_names[j] in gt]
    hit_str = f" <- HIT: {hits}" if hits else ""
    print(f"  {d:20s}: {', '.join(names)}{hit_str}")

# ---- Method 3: Z-score based thresholding ----
# For each drug, compute z-scores across all mutants
print("\n=== Z-score based matches (z < -1.5: significantly close) ===")
z_thresh = -1.5
n_found = 0
n_expected = 0
for i, d in enumerate(drug_names):
    mu = swd[i].mean()
    sd = swd[i].std()
    zs = (swd[i] - mu) / sd
    matches = [mutant_names[j] for j in range(n_m) if zs[j] < z_thresh]
    gt = GROUND_TRUTH.get(d, [])
    true_pos = [m for m in matches if m in gt]
    false_neg = [m for m in gt if m not in matches and m in mutant_names]
    n_found += len(true_pos)
    n_expected += len([g for g in gt if g in mutant_names])
    if matches:
        print(f"  {d:20s}: {matches}")
        if true_pos:
            print(f"    ✓ hits: {true_pos}")
        if false_neg:
            print(f"    ✗ missed: {false_neg}")
print(f"\nTotal known targets in set: {n_expected}, found by z-score: {n_found} (recall={n_found/n_expected:.1%})")

# ---- Method 4: Global percentile threshold ----
print("\n=== Global percentile threshold (bottom 10%) ===")
thresh = np.percentile(all_vals, 10)
matches = np.argwhere(swd < thresh)
print(f"Threshold (10th percentile) = {thresh:.4f}")
print(f"Total pair matches: {len(matches)}")
for i, j in matches:
    d = drug_names[i]; m = mutant_names[j]
    gt = GROUND_TRUTH.get(d, [])
    hit = " ✓" if m in gt else ""
    print(f"  {d:20s} ↔ {m:8s} (SWD={swd[i,j]:.4f}){hit}")

# ---- Figures ----
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Histogram of all SWD values
ax = axes[0,0]
ax.hist(all_vals, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
ax.axvline(np.percentile(all_vals, 10), color='red', ls='--', label='10th pctl')
ax.axvline(np.percentile(all_vals, 5), color='darkred', ls='--', label='5th pctl')
ax.axvline(all_vals.mean(), color='black', ls='-', label='mean')
ax.set_xlabel("Sliced Wasserstein Distance")
ax.set_ylabel("Count")
ax.set_title("Distribution of 660 drug-mutant SWD values")
ax.legend(fontsize=8)

# 2. Per-drug z-scores as heatmap
ax = axes[0,1]
z_all = (swd - swd.mean(axis=1, keepdims=True)) / swd.std(axis=1, keepdims=True)
im = ax.imshow(z_all, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
ax.set_xticks(range(n_m)); ax.set_xticklabels(mutant_names, rotation=90, fontsize=6)
ax.set_yticks(range(n_d)); ax.set_yticklabels(drug_names, fontsize=7)
ax.set_title("Z-score per drug row (blue=close)", fontsize=11)
fig.colorbar(im, ax=ax, shrink=0.8)

# 3. Recall@k curve
ax = axes[1,0]
k_vals = range(1, n_m+1)
recalls = []
for k in k_vals:
    hits = 0
    total_known = 0
    for i, d in enumerate(drug_names):
        gt = [g for g in GROUND_TRUTH.get(d, []) if g in mutant_names]
        total_known += len(gt)
        topk = set(mutant_names[j] for j in np.argsort(swd[i])[:k])
        hits += sum(1 for g in gt if g in topk)
    recalls.append(hits / max(total_known, 1))
ax.plot(k_vals, recalls, 'o-', color='darkgreen', markersize=3)
ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel("Top-K mutants considered per drug")
ax.set_ylabel("Recall")
ax.set_title("Known target recovery: Recall@K")
ax.set_xscale('log')

# 4. ROC-style: average rank of correct target
ax = axes[1,1]
ranks = []
for i, d in enumerate(drug_names):
    gt = GROUND_TRUTH.get(d, [])
    gt_in_set = [g for g in gt if g in mutant_names]
    if not gt_in_set: continue
    sorted_idx = np.argsort(swd[i])
    for g in gt_in_set:
        rank = np.where(sorted_idx == mutant_names.index(g))[0][0] + 1
        ranks.append(rank)
ax.hist(ranks, bins=30, alpha=0.7, color='purple', edgecolor='white')
ax.axvline(np.median(ranks), color='red', ls='--', label=f'median={np.median(ranks):.0f}')
ax.set_xlabel("Rank of known target among 30 mutants")
ax.set_ylabel("Count")
ax.set_title(f"Known target ranks (n={len(ranks)})")
ax.legend()

plt.tight_layout()
plt.savefig(f"{outdir}/wasserstein_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {outdir}/wasserstein_analysis.png")

