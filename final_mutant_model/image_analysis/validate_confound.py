#!/usr/bin/env python3
"""Rigorous validation of drug-vs-mutant confound in pixel-level features.
   Loads pre-saved CSV from save_features_all_plates.py, runs:

   1. 5-fold stratified cross-validated AUC (per plate + pooled)
   2. Permutation test (H0: label ↔ features independent)
   3. Plate-ID prediction (can we tell which plate an image comes from?)
   4. Per-treatment scatter (is it biological or global?)
   5. Well-position effect (edge wells different?)
   6. Region comparison (full vs center1128 vs center224)
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='all_plates_features.csv',
                    help='Input CSV from save_features_all_plates.py')
parser.add_argument('--output', default='validation',
                    help='Output directory (relative to output_all_plates/)')
parser.add_argument('--permutations', type=int, default=1000,
                    help='Number of permutations for test')
parser.add_argument('--folds', type=int, default=5,
                    help='Cross-validation folds')
parser.add_argument('--feat_prefix', default='center1128_mp_',
                    help='Feature prefix (full_mp_, center1128_mp_, center224_mp_)')
args = parser.parse_args()

import numpy as np; np.random.seed(42)
import os, csv, re, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, 'output_all_plates', args.output)
os.makedirs(OUT, exist_ok=True)

FP = args.feat_prefix  # shorthand

# === Load data ===
inpath = os.path.join(SCRIPT_DIR, 'output_all_plates', args.input)
if not os.path.exists(inpath):
    raise FileNotFoundError(f"Run save_features_all_plates.py first — missing {inpath}")

rows = []
with open(inpath) as f:
    reader = csv.DictReader(f)
    feat_names = [k for k in reader.fieldnames if k.startswith(FP)]
    if not feat_names:
        raise ValueError(f"No features found with prefix '{FP}'. Available: {[k for k in reader.fieldnames if 'mp_' in k]}")
    for r in reader:
        for k in feat_names:
            r[k] = float(r[k])
        rows.append(r)

mp_feats = [f'{FP}mean', f'{FP}std', f'{FP}snr', f'{FP}entropy']
FEAT_NAMES_SHORT = ['mean','std','snr','entropy']

print(f"Loaded {len(rows)} rows from {inpath}")
print(f"  Feature prefix: {FP}")
print(f"  Features: {mp_feats}")
print(f"  Plates: {sorted(set(r['plate'] for r in rows))}")
print(f"  Mutant: {sum(1 for r in rows if r['type']=='mutant')}, "
      f"Drug: {sum(1 for r in rows if r['type']=='drug')}")

# =====================================================
# 1. CROSS-VALIDATED AUC per plate
# =====================================================
print(f"\n{'='*80}")
print("1. CROSS-VALIDATED AUC (5-fold stratified, drug vs mutant)")
print(f"{'='*80}")

all_cv_results = []
for plate in sorted(set(r['plate'] for r in rows)):
    pr = [r for r in rows if r['plate'] == plate]
    X = np.array([[r[f] for f in mp_feats] for r in pr])
    y = np.array([0 if r['type']=='mutant' else 1 for r in pr])

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    cv_aucs = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_tr_s, y_tr)
        cv_aucs.append(roc_auc_score(y_va, lr.predict_proba(X_va_s)[:,1]))

    mean_auc = np.mean(cv_aucs)
    std_auc = np.std(cv_aucs)
    all_cv_results.append((plate, mean_auc, std_auc, cv_aucs))
    print(f"  {plate}: CV AUC = {mean_auc:.4f} ± {std_auc:.4f}")

# Pooled across all plates
X_all = np.array([[r[f] for f in mp_feats] for r in rows])
y_all = np.array([0 if r['type']=='mutant' else 1 for r in rows])
skf_all = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
cv_all = []
for tr, va in skf_all.split(X_all, y_all):
    s = StandardScaler().fit(X_all[tr])
    lr = LogisticRegression(max_iter=1000).fit(s.transform(X_all[tr]), y_all[tr])
    cv_all.append(roc_auc_score(y_all[va], lr.predict_proba(s.transform(X_all[va]))[:,1]))
print(f"  ALL:  CV AUC = {np.mean(cv_all):.4f} ± {np.std(cv_all):.4f}")
print(f"  (vs training AUC ~0.88-0.94 — cross-validation removes optimism)")

# Plot
fig, ax = plt.subplots(figsize=(10,5))
plates = [x[0] for x in all_cv_results]
means = [x[1] for x in all_cv_results]
stds = [x[2] for x in all_cv_results]
ax.bar(plates, means, yerr=stds, capsize=5, color='steelblue', alpha=0.7)
ax.axhline(0.5, color='gray', ls='--', lw=1, label='Chance')
ax.set_ylabel('5-fold CV AUC'); ax.set_title(f'Drug vs Mutant: CV AUC ({FP.replace("_mp_","")} region)')
ax.set_ylim(0.4, 1.0); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT,'cv_auc_per_plate.png'), dpi=150); plt.close()

# =====================================================
# 2. PERMUTATION TEST
# =====================================================
print(f"\n{'='*80}")
print(f"2. PERMUTATION TEST ({args.permutations} shuffles)")
print(f"{'='*80}")

scaler = StandardScaler().fit(X_all)
X_all_s = scaler.transform(X_all)
lr_obs = LogisticRegression(max_iter=1000).fit(X_all_s, y_all)
auc_obs = roc_auc_score(y_all, lr_obs.predict_proba(X_all_s)[:,1])

perm_aucs = []
for _ in tqdm(range(args.permutations), desc='Permuting'):
    y_shuff = y_all.copy()
    np.random.shuffle(y_shuff)
    lr_p = LogisticRegression(max_iter=1000).fit(X_all_s, y_shuff)
    perm_aucs.append(roc_auc_score(y_shuff, lr_p.predict_proba(X_all_s)[:,1]))
perm_aucs = np.array(perm_aucs)

perm_p_val = (np.sum(perm_aucs >= auc_obs) + 1) / (args.permutations + 1)
print(f"  Observed AUC: {auc_obs:.4f}")
print(f"  Permutation null: μ={np.mean(perm_aucs):.4f}, σ={np.std(perm_aucs):.4f}")
print(f"  p-value: {perm_p_val:.4f} ({'SIGNIFICANT' if perm_p_val < 0.05 else 'NOT SIGNIFICANT'})")

fig, ax = plt.subplots(figsize=(8,5))
ax.hist(perm_aucs, bins=50, alpha=0.7, color='gray', label=f'Null ({args.permutations} perms)')
ax.axvline(auc_obs, color='red', ls='--', lw=2, label=f'Observed AUC={auc_obs:.3f}')
ax.set_xlabel('AUC'); ax.set_ylabel('Frequency')
ax.set_title(f'Permutation test: p={perm_p_val:.4f}')
ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT,'permutation_test.png'), dpi=150); plt.close()

# =====================================================
# 3. PLATE-ID PREDICTION (batch effect test)
# =====================================================
print(f"\n{'='*80}")
print("3. PLATE-ID PREDICTION (batch effect test)")
print(f"{'='*80}")

plate_ids = sorted(set(r['plate'] for r in rows))
plate_aucs = {}
for plate in plate_ids:
    y_plate = np.array([1 if r['plate'] == plate else 0 for r in rows])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, va in skf.split(X_all, y_plate):
        s = StandardScaler().fit(X_all[tr])
        lr = LogisticRegression(max_iter=1000).fit(s.transform(X_all[tr]), y_plate[tr])
        aucs.append(roc_auc_score(y_plate[va], lr.predict_proba(s.transform(X_all[va]))[:,1]))
    plate_aucs[plate] = (np.mean(aucs), np.std(aucs))

for plate, (m, s) in sorted(plate_aucs.items()):
    tag = "STRONG batch effect" if m > 0.8 else "MODERATE" if m > 0.7 else "WEAK"
    print(f"  {plate} vs rest: CV AUC = {m:.3f} ± {s:.3f} — {tag}")

from sklearn.multiclass import OneVsRestClassifier
y_plate_mc = np.array([plate_ids.index(r['plate']) for r in rows])
mc_aucs = []
for tr, va in skf_all.split(X_all, y_plate_mc):
    s = StandardScaler().fit(X_all[tr])
    lr = OneVsRestClassifier(LogisticRegression(max_iter=1000)).fit(s.transform(X_all[tr]), y_plate_mc[tr])
    y_score = lr.predict_proba(s.transform(X_all[va]))
    mc_aucs.append(roc_auc_score(y_plate_mc[va], y_score, multi_class='ovr'))
print(f"  Multi-class (6 plates): CV AUC (macro) = {np.mean(mc_aucs):.3f} ± {np.std(mc_aucs):.3f}")

fig, ax = plt.subplots(figsize=(10,5))
p_n = [x[0] for x in sorted(plate_aucs.items())]
p_m = [x[1][0] for x in sorted(plate_aucs.items())]
p_s = [x[1][1] for x in sorted(plate_aucs.items())]
ax.bar(p_n, p_m, yerr=p_s, capsize=5, color='coral', alpha=0.7)
ax.axhline(0.5, color='gray', ls='--', lw=1, label='Chance')
ax.axhline(0.8, color='red', ls=':', lw=1, label='Strong batch effect')
ax.set_ylabel('CV AUC'); ax.set_title('Plate-ID prediction (batch effect)')
ax.set_ylim(0.4, 1.0); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT,'plate_id_prediction.png'), dpi=150); plt.close()

# =====================================================
# 4. PER-TREATMENT SCATTER
# =====================================================
print(f"\n{'='*80}")
print("4. PER-WELL PAIRED ANALYSIS (matched wells across drug/mutant)")
print(f"{'='*80}")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for idx, plate in enumerate(sorted(set(r['plate'] for r in rows))):
    ax = axes[idx // 3][idx % 3]
    pr = [r for r in rows if r['plate'] == plate]
    mut_by_well = {r['well']: r for r in pr if r['type'] == 'mutant'}
    drug_by_well = {r['well']: r for r in pr if r['type'] == 'drug'}
    common_wells = sorted(set(mut_by_well.keys()) & set(drug_by_well.keys()))
    if common_wells:
        ent_feat = f'{FP}entropy'
        m_vals = [mut_by_well[w][ent_feat] for w in common_wells]
        d_vals = [drug_by_well[w][ent_feat] for w in common_wells]
        ax.scatter(m_vals, d_vals, alpha=0.7, s=30)
        lo = min(min(m_vals), min(d_vals)) - 0.1
        hi = max(max(m_vals), max(d_vals)) + 0.1
        ax.plot([lo, hi], [lo, hi], 'gray', ls='--', alpha=0.5)
        ax.set_xlabel('Mutant entropy'); ax.set_ylabel('Drug entropy')
        ax.set_title(f'{plate}: {len(common_wells)} matched wells')
        r_val, r_pval = pearsonr(m_vals, d_vals)
        ax.text(0.05, 0.95, f'r={r_val:.2f}, p={r_pval:.3f}',
                transform=ax.transAxes, va='top', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No matched wells', ha='center', va='center', transform=ax.transAxes)
plt.suptitle(f'Per-well entropy: drug vs mutant same well position ({FP.replace("_mp_","")} region)')
plt.tight_layout(); plt.savefig(os.path.join(OUT,'per_well_entropy.png'), dpi=150); plt.close()

below = 0; total = 0
for plate in sorted(set(r['plate'] for r in rows)):
    pr = [r for r in rows if r['plate'] == plate]
    muts = {r['well']: r for r in pr if r['type']=='mutant'}
    drugs = {r['well']: r for r in pr if r['type']=='drug'}
    ent_feat = f'{FP}entropy'
    for w in set(muts.keys()) & set(drugs.keys()):
        total += 1
        if drugs[w][ent_feat] < muts[w][ent_feat]:
            below += 1
if total > 0:
    pct = below / total * 100
    print(f"  Drug entropy < Mutant entropy: {below}/{total} = {pct:.0f}%")
    if 30 < pct < 70:
        print("  → No consistent direction: likely not a systematic artifact")
    else:
        print("  → Consistent direction: suggestive of a batch/plate artifact")

# =====================================================
# 5. WELL POSITION EFFECT
# =====================================================
print(f"\n{'='*80}")
print("5. WELL POSITION ANALYSIS (edge effects)")
print(f"{'='*80}")

for plate in sorted(set(r['plate'] for r in rows)):
    pr = [r for r in rows if r['plate'] == plate]
    row_letters = []
    for r in pr:
        m = re.match(r'([A-Z])', r['well'])
        row_letters.append(m.group(1) if m else '?')
    edge = [r for r, l in zip(pr, row_letters) if l in ('A','H')]
    inner = [r for r, l in zip(pr, row_letters) if l not in ('A','H')]
    if edge and inner:
        e_std = np.mean([r[f'{FP}std'] for r in edge])
        i_std = np.mean([r[f'{FP}std'] for r in inner])
        print(f"  {plate}: edge std={e_std:.4f}, inner std={i_std:.4f} "
              f"(diff={e_std - i_std:+.4f})")

# =====================================================
# 6. REGION COMPARISON
# =====================================================
print(f"\n{'='*80}")
print("6. REGION COMPARISON")
print(f"{'='*80}")
available_regions = sorted(set(
    k.replace('_mp_mean', '') for k in rows[0].keys()
    if k.endswith('_mp_mean')
))
print(f"  Available regions: {available_regions}")
for region in available_regions:
    prefix = f'{region}_mp_'
    feats = [f'{prefix}{m}' for m in ['mean','std','snr','entropy']]
    if all(f in rows[0] for f in feats):
        X_r = np.array([[r[f] for f in feats] for r in rows])
        aucs_r = []
        for tr, va in skf_all.split(X_r, y_all):
            s = StandardScaler().fit(X_r[tr])
            lr = LogisticRegression(max_iter=1000).fit(s.transform(X_r[tr]), y_all[tr])
            aucs_r.append(roc_auc_score(y_all[va], lr.predict_proba(s.transform(X_r[va]))[:,1]))
        print(f"  {region:48s}: CV AUC = {np.mean(aucs_r):.4f} ± {np.std(aucs_r):.4f}")
    else:
        print(f"  {region:48s}: SKIP (missing features in CSV)")

# =====================================================
# SUMMARY
# =====================================================
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"  Region:                    {FP.replace('_mp_','')}")
print(f"  Cross-validated AUC (pooled): {np.mean(cv_all):.4f} ± {np.std(cv_all):.4f}")
print(f"  Permutation test p-value:     {perm_p_val:.4f}")
print(f"  Plate-ID prediction AUC:      {np.mean(mc_aucs):.3f}")
if np.mean(cv_all) > 0.75:
    print(f"\n  CONCLUSION: Pixel-level stats are a STRONG confound (CV AUC > 0.75).")
    print(f"  The MIL model's ~55% accuracy could be explained by texture/contrast shortcuts.")
elif np.mean(cv_all) > 0.6:
    print(f"\n  CONCLUSION: Pixel-level stats are a MODERATE confound.")
else:
    print(f"\n  CONCLUSION: Pixel-level stats are a WEAK confound (CV AUC < 0.6).")
    print(f"  The model likely learns morphology beyond pixel statistics.")

if np.mean(mc_aucs) > 0.7:
    print(f"  WARNING: Strong plate-level batch effects detected.")
    print(f"  Images from different plates are distinguishable by pixel stats alone.")

print(f"\nAll outputs saved to {OUT}/")

