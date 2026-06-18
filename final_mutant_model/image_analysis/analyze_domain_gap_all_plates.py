#!/usr/bin/env python3
"""Analyze brightness domain gap between Drug and Mutant across all plates.
   Uses the model's actual preprocessing pipeline (not z-score)."""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--plates', default='P1,P2,P3,P4,P5,P6',
                    help='Comma-separated plate names')
args = parser.parse_args()
PLATES = args.plates.split(',')

import numpy as np; np.random.seed(42)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json, re, csv, random, sys, warnings, copy
from PIL import Image
from scipy.stats import ks_2samp, entropy as sp_entropy, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
warnings.filterwarnings('ignore')
random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
BASE = os.path.dirname(PROJECT_DIR)
MUTANT_BASE = os.path.join(BASE, 'Mutants_Data')
DRUG_BASE = os.path.join(BASE, 'Drugs_Data')
sys.path.insert(0, PROJECT_DIR)
OUT = os.path.join(SCRIPT_DIR, 'output_all_plates'); os.makedirs(OUT, exist_ok=True)

with open(os.path.join(PROJECT_DIR, 'plate_well_id_path.json')) as f:
    mutant_map = json.load(f)
with open(os.path.join(PROJECT_DIR, 'plate_well_ic50_mapping.json')) as f:
    drug_map = json.load(f)

def load_gray(path):
    img = np.array(Image.open(path))
    if len(img.shape) == 3: img = img[:,:,0]
    return img.astype(np.float32)

def extract_well(fname):
    m = re.search(r'Well([A-Z]\d+)_', fname)
    return m.group(1) if m else None

def bhattacharyya(h1, h2):
    h1 = h1/h1.sum(); h2 = h2/h2.sum()
    return np.sqrt(1 - np.sum(np.sqrt(h1*h2)))

def cohens_d(a, b):
    return (np.mean(a) - np.mean(b)) / np.sqrt((np.var(a) + np.var(b)) / 2)

def point_biserial_r(continuous, binary):
    return pearsonr(continuous, binary)[0]

def compute_stats(img, range_min=0, range_max=65535):
    h, _ = np.histogram(img, bins=256, range=(range_min, range_max))
    p = h / h.sum(); p = p[p > 0]
    return {
        'mean': float(img.mean()),
        'std': float(img.std()),
        'snr': float(img.mean() / (img.std() + 1e-8)),
        'entropy': float(sp_entropy(p)),
        'p1': float(np.percentile(img, 1)),
        'p99': float(np.percentile(img, 99)),
        'median': float(np.median(img)),
        'hist': h.tolist(),
    }

def process_plate(plate, base_dir, mapping, datatype):
    plate_dir = os.path.join(base_dir, plate)
    if not os.path.exists(plate_dir): return []
    wells = {}
    for root, _, files in os.walk(plate_dir):
        for fname in files:
            if not fname.endswith(('.tif','.tiff','.png')): continue
            well = extract_well(fname)
            if not well: continue
            wells.setdefault(well, []).append(os.path.join(root, fname))
    results = []
    for well, paths in tqdm(wells.items(), desc=f'{datatype} {plate}'):
        path = random.choice(paths)
        img = load_gray(path)
        raw = compute_stats(img)
        mp = compute_stats(model_preprocess(img), range_min=-1, range_max=1)
        label = (mapping.get(plate,{}).get(well,{}).get('id','unknown')
                 if datatype=='mutant'
                 else f"{drug_map.get(plate,{}).get(well,{}).get('antibiotic','unknown')}_{drug_map.get(plate,{}).get(well,{}).get('ic50_multiple','')}")
        entry = {'plate': plate, 'well': well, 'label': label, 'type': datatype, 'path': path}
        for k in raw: entry[f'raw_{k}'] = raw[k]
        for k in mp:  entry[f'mp_{k}']  = mp[k]
        results.append(entry)
    return results

def model_preprocess(img):
    """Replicates model pipeline: uint16 -> /65535 -> [0,1] -> *255 -> uint8 -> /255 -> (x-0.5)/0.5"""
    img_01 = img.astype(np.float32) / 65535.0
    img_255 = (img_01 * 255).astype(np.uint8)
    img_norm = (img_255.astype(np.float32) / 255.0 - 0.5) / 0.5
    return img_norm

def analyze_plate(plate):
    print(f"\n{'='*70}")
    print(f"PLATE {plate}")
    print(f"{'='*70}")

    mutant = process_plate(plate, MUTANT_BASE, mutant_map, 'mutant')
    drug   = process_plate(plate, DRUG_BASE,   drug_map,   'drug')
    results = mutant + drug
    print(f"  Mutant: {len(mutant)}, Drug: {len(drug)}")

    if not mutant or not drug:
        print(f"  Skipping {plate}: missing data")
        return None

    # Save CSV (raw only, model-preprocessed is derived)
    with open(os.path.join(OUT,f'{plate}_domain_gap.csv'),'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=['plate','well','label','type','path'] +
                           [f'raw_{k}' for k in ['mean','std','snr','entropy','p1','p99','median']])
        w.writeheader()
        for r in results: w.writerow({k:r[k] for k in w.fieldnames})

    y_binary = np.array([0]*len(mutant) + [1]*len(drug))

    # ---- RAW (16-bit) stats ----
    metrics = ['mean','std','snr','entropy','p1','p99','median']
    print(f"\n--- RAW 16-BIT ---")
    print(f"{'Metric':>12} {'Mutant μ±σ':>18} {'Drug μ±σ':>18} {'KS p':>10} {'Cohen d':>8} {'r_pb':>7}")
    print("-"*70)
    for m in metrics:
        m_vals = np.array([r[f'raw_{m}'] for r in mutant])
        d_vals = np.array([r[f'raw_{m}'] for r in drug])
        all_vals = np.concatenate([m_vals, d_vals])
        ks = ks_2samp(m_vals, d_vals)
        d = cohens_d(m_vals, d_vals)
        r_pb = point_biserial_r(all_vals, y_binary)
        print(f"{m:>12} {np.mean(m_vals):>8.0f}±{np.std(m_vals):>5.0f}"
              f" {np.mean(d_vals):>8.0f}±{np.std(d_vals):>5.0f}"
              f" {ks.pvalue:>10.2e} {d:>8.2f} {r_pb:>7.3f}")

    h_mutant = np.sum([r['raw_hist'] for r in mutant], axis=0)
    h_drug   = np.sum([r['raw_hist'] for r in drug],   axis=0)
    bd = bhattacharyya(h_mutant, h_drug)
    print(f"\nBhattacharyya (16-bit raw): {bd:.4f}")

    X = np.array([r['raw_mean'] for r in results]).reshape(-1, 1)
    lr = LogisticRegression()
    lr.fit(X, y_binary)
    y_prob = lr.predict_proba(X)[:, 1]
    auc = roc_auc_score(y_binary, y_prob)
    print(f"Brightness AUC (16-bit raw): {auc:.4f} (coeff={lr.coef_[0][0]:.4f})")
    if auc > 0.8: print("  → Strong confound")
    elif auc > 0.7: print("  → Moderate confound")
    else: print("  → Weak confound")

    fpr, tpr, _ = roc_curve(y_binary, y_prob)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr, tpr, label=f'AUC={auc:.3f}')
    ax.plot([0,1],[0,1],'k--',alpha=0.3)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title(f'{plate}: Brightness → Drug vs Mutant (raw)')
    ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT,f'{plate}_brightness_roc_raw.png'), dpi=150); plt.close()

    # ---- Model-preprocessed stats (computed during sampling, single pass) ----
    mp_metrics = ['mean','std','snr','entropy']
    print(f"\n--- MODEL PREPROCESSED (raw/65535 → [0,1], then (x-0.5)/0.5 → [-1,1]) ---")
    print(f"{'Metric':>12} {'Mutant μ±σ':>18} {'Drug μ±σ':>18} {'KS p':>10} {'Cohen d':>8} {'r_pb':>7}")
    print("-"*70)
    for m in mp_metrics:
        m_vals = np.array([r[f'mp_{m}'] for r in mutant])
        d_vals = np.array([r[f'mp_{m}'] for r in drug])
        all_vals = np.concatenate([m_vals, d_vals])
        ks = ks_2samp(m_vals, d_vals)
        d = cohens_d(m_vals, d_vals)
        r_pb = point_biserial_r(all_vals, y_binary)
        print(f"{m:>12} {np.mean(m_vals):>8.4f}±{np.std(m_vals):>7.4f}"
              f" {np.mean(d_vals):>8.4f}±{np.std(d_vals):>7.4f}"
              f" {ks.pvalue:>10.2e} {d:>8.2f} {r_pb:>7.3f}")

    hm_mutant = np.sum([r['mp_hist'] for r in mutant], axis=0)
    hm_drug   = np.sum([r['mp_hist'] for r in drug],   axis=0)
    bd_mp = bhattacharyya(hm_mutant, hm_drug)
    print(f"\nBhattacharyya (model preprocessed): {bd_mp:.4f}")

    X_mp = np.array([r['mp_mean'] for r in results]).reshape(-1, 1)
    lr_mp = LogisticRegression()
    lr_mp.fit(X_mp, y_binary)
    y_prob_mp = lr_mp.predict_proba(X_mp)[:, 1]
    auc_mp = roc_auc_score(y_binary, y_prob_mp)
    print(f"Brightness AUC (model preprocessed): {auc_mp:.4f} (coeff={lr_mp.coef_[0][0]:.4f})")
    if auc_mp > 0.8: print("  → Strong confound")
    elif auc_mp > 0.7: print("  → Moderate confound")
    else: print("  → Weak confound")

    fpr_mp, tpr_mp, _ = roc_curve(y_binary, y_prob_mp)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr_mp, tpr_mp, label=f'AUC={auc_mp:.3f}')
    ax.plot([0,1],[0,1],'k--',alpha=0.3)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title(f'{plate}: Brightness → Drug vs Mutant (model preproc)')
    ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT,f'{plate}_brightness_roc_mp.png'), dpi=150); plt.close()

    # ---- Multi-feature logistic regression (all pixel stats) ----
    mp_feature_names = ['mp_mean','mp_std','mp_snr','mp_entropy']
    X_all = np.array([[r[f] for f in mp_feature_names] for r in results])
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)
    lr_all = LogisticRegression(max_iter=1000)
    lr_all.fit(X_all_scaled, y_binary)
    y_prob_all = lr_all.predict_proba(X_all_scaled)[:, 1]
    auc_all = roc_auc_score(y_binary, y_prob_all)

    print(f"\n--- MULTI-FEATURE LOGISTIC REGRESSION ({', '.join(mp_feature_names)}) ---")
    print(f"  AUC (all 4 mp features): {auc_all:.4f}")
    for name, coef in zip(['mean','std','snr','entropy'], lr_all.coef_[0]):
        print(f"    {name:>10}: {coef:>+8.4f}")
    improv = auc_all - auc_mp
    print(f"  Improvement over brightness-only: {improv:+.4f}")
    if auc_all > 0.8:
        print("  → STRONG confound: pixel stats alone distinguish drug from mutant")
    elif auc_all > 0.7:
        print("  → MODERATE confound: model may use texture/contrast shortcuts")
    else:
        print("  → WEAK confound: pixel stats insufficient to explain model performance")

    # ROC multi vs brightness
    fpr_all, tpr_all, _ = roc_curve(y_binary, y_prob_all)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr, tpr, label=f'Brightness-only AUC={auc:.3f}', ls='--')
    ax.plot(fpr_all, tpr_all, label=f'All 4 features AUC={auc_all:.3f}')
    ax.plot([0,1],[0,1],'k--',alpha=0.3)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title(f'{plate}: Brightness vs all pixel stats')
    ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT,f'{plate}_roc_comparison.png'), dpi=150); plt.close()

    # ---- Plots ----
    fig, ax = plt.subplots(figsize=(8,5))
    bin_edges = np.linspace(0,65535,257)
    ax.stairs(h_mutant/h_mutant.sum(), bin_edges, fill=True, alpha=0.5, color='C0', label='Mutant')
    ax.stairs(h_drug/h_drug.sum(), bin_edges, fill=True, alpha=0.5, color='C1', label='Drug')
    ax.set_xlabel('Pixel intensity (16-bit)'); ax.set_ylabel('Density'); ax.legend()
    ax.set_title(f'{plate}: Bhattacharyya={bd:.4f}')
    plt.tight_layout(); plt.savefig(os.path.join(OUT,f'{plate}_histogram_raw.png'), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8,5))
    bin_edges_mp = np.linspace(-1,1,257)
    ax.stairs(hm_mutant/hm_mutant.sum(), bin_edges_mp, fill=True, alpha=0.5, color='C0', label='Mutant')
    ax.stairs(hm_drug/hm_drug.sum(), bin_edges_mp, fill=True, alpha=0.5, color='C1', label='Drug')
    ax.set_xlabel('Model input ([-1, 1])'); ax.set_ylabel('Density'); ax.legend()
    ax.set_title(f'{plate} model preproc: Bhattacharyya={bd_mp:.4f}')
    plt.tight_layout(); plt.savefig(os.path.join(OUT,f'{plate}_histogram_mp.png'), dpi=150); plt.close()

    # Find top discriminating feature (largest |Cohen's d| on mp stats)
    mp_best = max(['mean','std','snr','entropy'], key=lambda m: abs(cohens_d(
        np.array([r[f'mp_{m}'] for r in mutant]),
        np.array([r[f'mp_{m}'] for r in drug]))))

    return {
        'plate': plate,
        'n_mutant': len(mutant), 'n_drug': len(drug),
        'raw_mutant_mean': float(np.mean([r['raw_mean'] for r in mutant])),
        'raw_drug_mean': float(np.mean([r['raw_mean'] for r in drug])),
        'cohens_d_brightness_raw': float(cohens_d(
            np.array([r['raw_mean'] for r in mutant]),
            np.array([r['raw_mean'] for r in drug]))),
        'bhattacharyya_raw': float(bd),
        'auc_brightness_raw': float(auc),
        'mp_mutant_mean': float(np.mean([r['mp_mean'] for r in mutant])),
        'mp_drug_mean': float(np.mean([r['mp_mean'] for r in drug])),
        'cohens_d_brightness_mp': float(cohens_d(
            np.array([r['mp_mean'] for r in mutant]),
            np.array([r['mp_mean'] for r in drug]))),
        'bhattacharyya_mp': float(bd_mp),
        'auc_brightness_mp': float(auc_mp),
        'auc_all_features': float(auc_all),
        'auc_improvement': float(improv),
        'best_feature': mp_best,
        'cohens_d_std_mp': float(cohens_d(
            np.array([r['mp_std'] for r in mutant]),
            np.array([r['mp_std'] for r in drug]))),
        'cohens_d_entropy_mp': float(cohens_d(
            np.array([r['mp_entropy'] for r in mutant]),
            np.array([r['mp_entropy'] for r in drug]))),
        'cohens_d_snr_mp': float(cohens_d(
            np.array([r['mp_snr'] for r in mutant]),
            np.array([r['mp_snr'] for r in drug]))),
    }

# === Run all plates ===
all_results = []
for plate in PLATES:
    res = analyze_plate(plate)
    if res: all_results.append(res)

# === Cross-plate summary ===
print(f"\n{'='*130}")
print("CROSS-PLATE SUMMARY")
print(f"{'='*130}")
print(f"{'Plate':>6} {'d_mean':>7} {'d_std':>7} {'d_SNR':>7} {'d_entr':>7}"
      f" {'AUC_b':>6} {'AUC_all':>7} {'best':>8} {'Bhatt':>7} {'Bhatt_mp':>7}")
print("-"*130)
for r in all_results:
    print(f"{r['plate']:>6} {r['cohens_d_brightness_mp']:>7.2f} {r['cohens_d_std_mp']:>7.2f}"
          f" {r['cohens_d_snr_mp']:>7.2f} {r['cohens_d_entropy_mp']:>7.2f}"
          f" {r['auc_brightness_mp']:>6.3f} {r['auc_all_features']:>7.3f}"
          f" {r['best_feature']:>8} {r['bhattacharyya_raw']:>7.4f} {r['bhattacharyya_mp']:>7.4f}")

# Save summary CSV
with open(os.path.join(OUT,'summary_all_plates.csv'),'w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=all_results[0].keys())
    w.writeheader(); w.writerows(all_results)

# Summary plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
plates = [r['plate'] for r in all_results]

ax1.axhline(0, color='gray', lw=0.5)
ax1.axhline(0.8, color='gray', ls='--', lw=0.5, label='Large')
ax1.axhline(-0.8, color='gray', ls='--', lw=0.5)
ax1.bar(plates, [r['cohens_d_brightness_raw'] for r in all_results],
        color='steelblue', alpha=0.7)
ax1.set_ylabel("Cohen's d (raw 16-bit)"); ax1.set_title('Cohen\'s d: raw 16-bit'); ax1.legend()

ax2.axhline(0, color='gray', lw=0.5)
ax2.axhline(0.8, color='gray', ls='--', lw=0.5, label='Large')
ax2.axhline(-0.8, color='gray', ls='--', lw=0.5)
ax2.bar(plates, [r['cohens_d_brightness_mp'] for r in all_results],
        color='coral', alpha=0.7)
ax2.set_ylabel("Cohen's d (model preproc)"); ax2.set_title('Cohen\'s d: model preprocessed'); ax2.legend()

ax3.axhline(0.5, color='gray', ls='--', lw=0.5, label='Chance')
ax3.bar(plates, [r['auc_brightness_raw'] for r in all_results],
        color='steelblue', alpha=0.7)
ax3.set_ylabel('AUC (raw)'); ax3.set_title('AUC: raw 16-bit'); ax3.legend()

ax4.axhline(0.5, color='gray', ls='--', lw=0.5, label='Chance')
ax4.bar(plates, [r['auc_brightness_mp'] for r in all_results],
        color='coral', alpha=0.7)
ax4.set_ylabel('AUC (model preproc)'); ax4.set_title('AUC: model preprocessed'); ax4.legend()

plt.tight_layout(); plt.savefig(os.path.join(OUT,'summary_all_plates.png'), dpi=150); plt.close()

print(f"\nAll outputs saved to {OUT}/")
