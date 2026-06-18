#!/usr/bin/env python3
"""Rigorous domain gap analysis: Drug P1 vs Mutant P1.
   Includes Cohen's d, z-score normalization, logistic regression, model features."""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--only_model', action='store_true', help='Skip all phases except model feature extraction')
args = parser.parse_args()

import numpy as np; np.random.seed(42)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json, re, csv, random, sys, warnings
from PIL import Image
from scipy.stats import ks_2samp, wasserstein_distance, entropy as sp_entropy, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
warnings.filterwarnings('ignore')
random.seed(42)

# === Paths ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
BASE = os.path.dirname(PROJECT_DIR)
MUTANT_BASE = os.path.join(BASE, 'Mutants_Data')
DRUG_BASE = os.path.join(BASE, 'Drugs_Data')

sys.path.insert(0, PROJECT_DIR)

OUT = os.path.join(SCRIPT_DIR, 'output'); os.makedirs(OUT, exist_ok=True)

with open(os.path.join(PROJECT_DIR, 'plate_well_id_path.json')) as f:
    mutant_map = json.load(f)
with open(os.path.join(PROJECT_DIR, 'plate_well_ic50_mapping.json')) as f:
    drug_map = json.load(f)

# === Helpers ===
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
    """Point-biserial correlation. binary in {0,1}."""
    return pearsonr(continuous, binary)[0]

def compute_stats(img):
    h, _ = np.histogram(img, bins=256, range=(0, 65535))
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

# === Sampling ===
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
        stats = compute_stats(img)
        label = (mapping.get(plate,{}).get(well,{}).get('id','unknown')
                 if datatype=='mutant'
                 else f"{drug_map.get(plate,{}).get(well,{}).get('antibiotic','unknown')}_{drug_map.get(plate,{}).get(well,{}).get('ic50_multiple','')}")
        stats['plate'] = plate; stats['well'] = well
        stats['label'] = label; stats['type'] = datatype
        stats['path'] = path; stats['_img'] = img
        results.append(stats)
    return results

if args.only_model:
    # Load existing CSV for paths/labels
    mutant = []; drug = []
    with open(os.path.join(OUT,'p1_domain_gap.csv')) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['_img'] = None
            if row['type'] == 'mutant':
                mutant.append(row)
            else:
                drug.append(row)
    results = mutant + drug
    print(f"Loaded {len(mutant)} mutant, {len(drug)} drug from CSV")
else:
    print("Phase 1: Sampling 1 image/well from P1...")
    mutant = process_plate('P1', MUTANT_BASE, mutant_map, 'mutant')
    drug   = process_plate('P1', DRUG_BASE,   drug_map,   'drug')
    results = mutant + drug
    print(f"  Mutant: {len(mutant)}, Drug: {len(drug)}")

    # --- CSV ---
    with open(os.path.join(OUT,'p1_domain_gap.csv'),'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=['plate','well','label','type','mean','std','snr','entropy','p1','p99','median'])
        w.writeheader()
        for r in results: w.writerow({k:r[k] for k in w.fieldnames})

y_binary = np.array([0]*len([r for r in results if r['type']=='mutant']) + [1]*len([r for r in results if r['type']=='drug']))

if not args.only_model:
    print("\n" + "="*90)
    print("DOMAIN GAP ANALYSIS — P1 Drug vs P1 Mutant")
    print("="*90)
    metrics = ['mean','std','snr','entropy','p1','p99','median']
    print(f"{'Metric':>12} {'Mutant μ±σ':>18} {'Drug μ±σ':>18} {'KS p':>10} {'Cohen d':>8} {'r_pb':>7}")
    print("-"*90)
    for m in metrics:
        m_vals = np.array([r[m] for r in mutant])
        d_vals = np.array([r[m] for r in drug])
        all_vals = np.concatenate([m_vals, d_vals])
        ks = ks_2samp(m_vals, d_vals)
        d = cohens_d(m_vals, d_vals)
        r = point_biserial_r(all_vals, y_binary)
        print(f"{m:>12} {np.mean(m_vals):>8.0f}±{np.std(m_vals):>5.0f}"
              f" {np.mean(d_vals):>8.0f}±{np.std(d_vals):>5.0f}"
              f" {ks.pvalue:>10.2e} {d:>8.2f} {r:>7.3f}")

    print("\nPhase 2: Building histograms...")
    h_mutant = np.sum([r['hist'] for r in mutant], axis=0)
    h_drug   = np.sum([r['hist'] for r in drug],   axis=0)
    bd = bhattacharyya(h_mutant, h_drug)
    print(f"Bhattacharyya distance (0=identical, 1=completely different): {bd:.4f}")

    print("\nPhase 3: Logistic regression — brightness alone predicts drug/mutant...")
    X = np.array([r['mean'] for r in results]).reshape(-1, 1)
    y = y_binary
    lr = LogisticRegression()
    lr.fit(X, y)
    y_prob = lr.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)
    print(f"  AUC (brightness→drug/mutant): {auc:.4f}")
    print(f"  Coefficient: {lr.coef_[0][0]:.4f} (positive = brighter → drug)")
    print(f"  Interpretation: {'Strong confound' if auc > 0.8 else 'Moderate confound' if auc > 0.7 else 'Weak confound'}")

    fpr, tpr, _ = roc_curve(y, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    ax.plot([0,1], [0,1], 'k--', alpha=0.3)
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
    ax.set_title('ROC: Brightness alone → Drug vs Mutant')
    ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT,'p1_brightness_roc.png'), dpi=150); plt.close()

if not args.only_model:
    # === Z-score normalization comparison ===
    print("\nPhase 4: Z-score normalization — re-running stats after per-image normalization...")
    z_mutant = []
    z_drug = []
    for r in tqdm(mutant, desc='Z-score mutant'):
        img = load_gray(r['path'])
        img_z = (img - img.mean()) / (img.std() + 1e-8)
        s = compute_stats(img_z)
        s['type'] = 'mutant'; s['label'] = r['label']
        z_mutant.append(s)
    for r in tqdm(drug, desc='Z-score drug'):
        img = load_gray(r['path'])
        img_z = (img - img.mean()) / (img.std() + 1e-8)
        s = compute_stats(img_z)
        s['type'] = 'drug'; s['label'] = r['label']
        z_drug.append(s)

    print(f"\n{'='*90}")
    print("AFTER Z-SCORE NORMALIZATION (per-image)")
    print(f"{'='*90}")
    print(f"{'Metric':>12} {'Mutant μ±σ':>18} {'Drug μ±σ':>18} {'KS p':>10} {'Cohen d':>8} {'r_pb':>7}")
    print("-"*90)
    z_metrics = ['mean','std','snr','entropy']
    for m in z_metrics:
        m_vals = np.array([r[m] for r in z_mutant])
        d_vals = np.array([r[m] for r in z_drug])
        all_vals = np.concatenate([m_vals, d_vals])
        ks = ks_2samp(m_vals, d_vals)
        d = cohens_d(m_vals, d_vals)
        r = point_biserial_r(all_vals, y_binary)
        print(f"{m:>12} {np.mean(m_vals):>8.2f}±{np.std(m_vals):>5.2f}"
              f" {np.mean(d_vals):>8.2f}±{np.std(d_vals):>5.2f}"
              f" {ks.pvalue:>10.2e} {d:>8.2f} {r:>7.3f}")

    print("Building z-score histograms...")
    hz_mutant = np.sum([r['hist'] for r in z_mutant], axis=0)
    hz_drug   = np.sum([r['hist'] for r in z_drug],   axis=0)
    bd_z = bhattacharyya(hz_mutant, hz_drug)
    print(f"Bhattacharyya (z-scored): {bd_z:.4f}")

# === Model feature extraction + PCA (optional) ===
print("\nPhase 5: Model feature extraction...")
try:
    import torch
    from mil_model import AttentionMILModel
    ckpt_dir = os.path.join(PROJECT_DIR, 'mutant_guide_1', 'fold_Plate_1')
    ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, 'best_model_acc.pth')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AttentionMILModel(num_classes=30, pooling='attention',
                              backbone='efficientnet_b0', num_heads=4,
                              num_channels=1, pretrained=False)
    state = torch.load(ckpt_path, map_location='cpu')
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    backbone = model.backbone.to(device)
    backbone.eval()
    
    features = []
    brightnesses = []
    labels = []
    for r in tqdm(mutant + drug, desc='Extracting features'):
        img = load_gray(r['path'])
        brightnesses.append(r['mean'])
        labels.append(r['type'])
        # Preprocess: same as training pipeline
        img_norm = img.astype(np.float32) / 65535.0
        img_norm = (img_norm - 0.5) / 0.5
        tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = backbone(tensor).cpu().numpy().flatten()
        features.append(feat)
    features = np.array(features)
    
    # PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(StandardScaler().fit_transform(features))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, color_by, cmap, title in [
        (axes[0], brightnesses, 'viridis', 'By brightness'),
        (axes[1], [0 if l=='mutant' else 1 for l in labels], 'coolwarm', 'By type')]:
        sc = ax.scatter(coords[:,0], coords[:,1], c=color_by, cmap=cmap, alpha=0.7, s=30)
        plt.colorbar(sc, ax=ax)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title(title)
    plt.tight_layout(); plt.savefig(os.path.join(OUT,'p1_model_features_pca.png'), dpi=150); plt.close()
    
    # Correlation: PC1 vs brightness
    r_pc1, p_pc1 = pearsonr(coords[:,0], brightnesses)
    print(f"  PC1 vs brightness: r={r_pc1:.3f}, p={p_pc1:.2e}")
    print(f"  PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")
    if abs(r_pc1) > 0.7:
        print("  → Strong evidence: model features correlate with brightness (confound)")
    elif abs(r_pc1) > 0.4:
        print("  → Moderate evidence: some brightness signal in features")
    else:
        print("  → Weak/no evidence: features not dominated by brightness")
    
    # AUROC on model features
    X_feat = StandardScaler().fit_transform(features)
    lr_feat = LogisticRegression(max_iter=1000)
    lr_feat.fit(X_feat, y_binary)
    auc_feat = roc_auc_score(y_binary, lr_feat.predict_proba(X_feat)[:,1])
    print(f"  Logistic regression AUC on 1280-dim features: {auc_feat:.4f}")
    auc_brightness = locals().get('auc', None)
    if auc_brightness is not None:
        print(f"  Logistic regression AUC on brightness alone: {auc_brightness:.4f}")
        if auc_feat > auc_brightness + 0.05:
            print("  → Model uses features BEYOND brightness (morphology matters)")
        else:
            print("  → Model performance ≈ brightness alone (confound dominates)")
        
except Exception as e:
    print(f"  Model extraction skipped ({type(e).__name__}: {e})")

if not args.only_model:
    # === Plots ===
    print("\nPhase 6: Generating plots...")

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(18,5))
    for ax, labs, means, title, color in [
        (axes[0], [r['label'] for r in mutant], [r['mean'] for r in mutant], 'Mutant genes (P1)', 'C0'),
        (axes[1], [r['label'] for r in drug],   [r['mean'] for r in drug],   'Drug treatments (P1)', 'C1')]:
        idx = np.argsort(means)
        labs_sorted = [labs[i] for i in idx]
        means_sorted = [means[i] for i in idx]
        ax.bar(range(len(labs_sorted)), means_sorted, color=color, alpha=0.7)
        ax.set_xticks(range(len(labs_sorted)))
        ax.set_xticklabels(labs_sorted, rotation=90, fontsize=5)
        ax.set_ylabel('Mean brightness')
        ax.set_title(title)
    plt.tight_layout(); plt.savefig(os.path.join(OUT,'p1_per_well_brightness.png'), dpi=150); plt.close()

    for ykey, ylabel, fname in [('std','Std (contrast)','p1_brightness_vs_contrast'),
                                 ('snr','SNR','p1_snr_vs_brightness'),
                                 ('entropy','Entropy','p1_entropy_vs_brightness')]:
        fig, ax = plt.subplots(figsize=(8,6))
        for data, color, label in [(mutant,'C0','Mutant'), (drug,'C1','Drug')]:
            ax.scatter([r['mean'] for r in data], [r[ykey] for r in data],
                       c=color, alpha=0.6, s=30, label=label)
        ax.set_xlabel('Mean brightness'); ax.set_ylabel(ylabel); ax.legend()
        ax.set_title(f'P1: {ylabel} vs Brightness')
        plt.tight_layout(); plt.savefig(os.path.join(OUT,f'{fname}.png'), dpi=150); plt.close()

    # Histograms (raw)
    fig, ax = plt.subplots(figsize=(8, 5))
    bin_edges = np.linspace(0, 65535, 257)
    ax.stairs(h_mutant / h_mutant.sum(), bin_edges, fill=True, alpha=0.5, color='C0', label='Mutant')
    ax.stairs(h_drug   / h_drug.sum(),   bin_edges, fill=True, alpha=0.5, color='C1', label='Drug')
    ax.set_xlabel('Pixel intensity'); ax.set_ylabel('Density'); ax.legend()
    ax.set_title('P1: Pixel intensity distribution')
    plt.tight_layout(); plt.savefig(os.path.join(OUT,'p1_pixel_histogram_overlay.png'), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(h_mutant / h_mutant.sum(), label='Mutant avg', color='C0')
    ax.plot(h_drug   / h_drug.sum(),   label='Drug avg',   color='C1')
    ax.set_xlabel('Intensity bin'); ax.set_ylabel('Normalized frequency'); ax.legend()
    ax.set_title(f'P1: Average intensity histogram — Bhattacharyya={bd:.4f}')
    plt.tight_layout(); plt.savefig(os.path.join(OUT,'p1_avg_histogram_comparison.png'), dpi=150); plt.close()

    # Histograms (z-scored)
    bin_edges_z = np.linspace(-5, 5, 257)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stairs(hz_mutant / hz_mutant.sum(), bin_edges_z, fill=True, alpha=0.5, color='C0', label='Mutant')
    ax.stairs(hz_drug   / hz_drug.sum(),   bin_edges_z, fill=True, alpha=0.5, color='C1', label='Drug')
    ax.set_xlabel('Z-score'); ax.set_ylabel('Density'); ax.legend()
    ax.set_title('P1: Z-scored pixel distribution')
    plt.tight_layout(); plt.savefig(os.path.join(OUT,'p1_zscore_histogram.png'), dpi=150); plt.close()

    # Cohen's d comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    before_d = [cohens_d(np.array([r[m] for r in mutant]), np.array([r[m] for r in drug])) for m in z_metrics]
    after_d  = [cohens_d(np.array([r[m] for r in z_mutant]), np.array([r[m] for r in z_drug])) for m in z_metrics]
    x = np.arange(len(z_metrics))
    w = 0.35
    ax.bar(x - w/2, before_d, w, label='Raw', color='C0', alpha=0.7)
    ax.bar(x + w/2, after_d,  w, label='Z-scored', color='C3', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(z_metrics)
    ax.set_ylabel("Cohen's d"); ax.axhline(0, color='gray', lw=0.5)
    ax.axhline(0.8, color='gray', ls='--', lw=0.5, label='Large effect')
    ax.set_title("Cohen's d: Raw vs Z-scored")
    ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT,'p1_cohens_d_comparison.png'), dpi=150); plt.close()

    # === Summary ===
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    print(f"  Brightness AUC (logistic regression):          {auc:.4f}")
    print(f"  Bhattacharyya (raw pixels):                    {bd:.4f}")
    print(f"  Bhattacharyya (z-scored):                      {bd_z:.4f}")
    print(f"  Mean brightness mutant:                        {np.mean([r['mean'] for r in mutant]):.0f}")
    print(f"  Mean brightness drug:                          {np.mean([r['mean'] for r in drug]):.0f}")
    print(f"  Cohen's d (brightness):                        {cohens_d(np.array([r['mean'] for r in mutant]), np.array([r['mean'] for r in drug])):.2f}")
    print(f"  Cohen's d after z-score:                       {cohens_d(np.array([r['mean'] for r in z_mutant]), np.array([r['mean'] for r in z_drug])):.2f}")
    print(f"\nAll plots saved to {OUT}/")
