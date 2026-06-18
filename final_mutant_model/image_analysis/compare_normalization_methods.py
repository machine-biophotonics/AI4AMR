#!/usr/bin/env python3
"""Compare normalization techniques for removing pixel confound.
   Shows: Original, Z-score, CLAHE, Z-score+CLAHE, Multi-Image HM, Single-Ref HM
"""
import numpy as np; np.random.seed(42)
import os, json, re, random, warnings
from PIL import Image
from skimage.exposure import match_histograms, equalize_adapthist
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
warnings.filterwarnings('ignore')
random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(PROJECT_DIR)
MUTANT_BASE = os.path.join(BASE_DIR, 'Mutants_Data')
DRUG_BASE = os.path.join(BASE_DIR, 'Drugs_Data')
OUT = os.path.join(SCRIPT_DIR, 'output_all_plates')
os.makedirs(OUT, exist_ok=True)

CROP_SIZE = 224

def load_gray(path):
    img = np.array(Image.open(path))
    if len(img.shape) == 3: img = img[:,:,0]
    return img.astype(np.float32)

def extract_well(fname):
    m = re.search(r'Well([A-Z]\d+)_', fname)
    return m.group(1) if m else None

def center_crop(img, size):
    h, w = img.shape
    half = size // 2
    cy, cx = h // 2, w // 2
    return img[cy-half:cy+half, cx-half:cx+half]

def collect_wells(plate, base_dir):
    plate_dir = os.path.join(base_dir, plate)
    if not os.path.exists(plate_dir): return {}
    wells = {}
    for root, _, files in os.walk(plate_dir):
        for fname in files:
            if not fname.endswith(('.tif','.tiff','.png')): continue
            well = extract_well(fname)
            if not well: continue
            if well not in wells:
                wells[well] = []
            wells[well].append(os.path.join(root, fname))
    return wells

def get_image_path(plate, well, base_dir):
    wells = collect_wells(plate, base_dir)
    if well not in wells: return None
    return random.choice(wells[well])

# --- Normalization techniques ---
def zscore_norm(img):
    """Per-image z-score: (x - mean) / std"""
    return (img - img.mean()) / (img.std() + 1e-8)

def clahe_norm(img):
    """CLAHE: adaptive histogram equalization, clip_limit=0.03"""
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_uint8 = (img_norm * 255).astype(np.uint8)
    result = equalize_adapthist(img_uint8, kernel_size=None, clip_limit=0.03, nbins=256)
    return result.astype(np.float32)

def zscore_clahe_norm(img):
    """Z-score then CLAHE"""
    z = zscore_norm(img)
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    z_uint8 = (z_norm * 255).astype(np.uint8)
    result = equalize_adapthist(z_uint8, kernel_size=None, clip_limit=0.03, nbins=256)
    return result.astype(np.float32)

def multi_ref_hm(img, ref_crops):
    """Multi-image reference HM (current approach)"""
    return match_histograms(img, ref_crops.astype(np.float32))

def single_ref_hm(img, ref_single):
    """Single-image reference HM"""
    return match_histograms(img, ref_single.astype(np.float32))

def model_preprocess_vis(img):
    """EfficientNet preprocessing for display"""
    if img.max() > 1.0 or img.min() < 0:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

# --- Build references ---
print("Building references...")
PLATES = ['P1','P2','P3','P4','P5','P6']

# Multi-image reference (all mutant crops)
multi_ref_crops = []
for plate in tqdm(PLATES, desc='Multi-image ref'):
    wells = collect_wells(plate, MUTANT_BASE)
    for well, paths in wells.items():
        path = random.choice(paths)
        crop = center_crop(load_gray(path), CROP_SIZE)
        multi_ref_crops.append(crop)
multi_ref = np.vstack(multi_ref_crops)

# Single-image reference (one clean mutant crop from P1)
m1 = collect_wells('P1', MUTANT_BASE)
single_ref_well = random.choice(list(m1.keys()))
single_ref_path = random.choice(m1[single_ref_well])
single_ref = center_crop(load_gray(single_ref_path), CROP_SIZE)

# --- Select wells ---
print("Selecting wells...")
shared_P2 = sorted(set(collect_wells('P2', MUTANT_BASE)) & set(collect_wells('P2', DRUG_BASE)))
shared_P5 = sorted(set(collect_wells('P5', MUTANT_BASE)) & set(collect_wells('P5', DRUG_BASE)))

samples = []
for plate, shared in [('P2', shared_P2), ('P5', shared_P5)]:
    w = shared[0] if shared else 'B2'
    for base_dir, datatype in [(MUTANT_BASE, 'mutant'), (DRUG_BASE, 'drug')]:
        path = get_image_path(plate, w, base_dir)
        if path:
            samples.append((plate, w, datatype, path))

# --- Compute and display ---
methods = [
    ('Original', lambda im: im, True),
    ('Z-score', lambda im: zscore_norm(im), False),
    ('CLAHE', lambda im: clahe_norm(im), False),
    ('Z-score + CLAHE', lambda im: zscore_clahe_norm(im), False),
    ('HM (multi-ref)', lambda im: multi_ref_hm(im, multi_ref), False),
    ('HM (single-ref)', lambda im: single_ref_hm(im, single_ref), False),
]

n_rows = len(samples)
n_cols = len(methods) + 1  # +1 for labels column

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows + 1))
fig.suptitle('Normalization Methods Comparison (224×224 center crops)', fontsize=14, y=0.98)

for row_idx, (plate, well, datatype, path) in enumerate(samples):
    img = load_gray(path)
    orig = center_crop(img, CROP_SIZE)

    # Col 0: Sample label
    ax = axes[row_idx, 0]
    ax.text(0.5, 0.5, f'{plate}\n{well}\n{datatype}',
            ha='center', va='center', fontsize=10, fontweight='bold',
            transform=ax.transAxes)
    ax.axis('off')

    for col_idx, (name, func, _) in enumerate(methods):
        ax = axes[row_idx, col_idx + 1]
        result = func(orig.copy())
        # Normalize to [0,1] for display
        if result.max() > 1 or result.min() < 0:
            if result.max() == result.min():
                display = np.zeros_like(result)
            else:
                display = (result - result.min()) / (result.max() - result.min() + 1e-8)
        else:
            display = result

        im = ax.imshow(display, cmap='gray')
        if row_idx == 0:
            ax.set_title(name, fontsize=10, fontweight='bold')
        ax.axis('off')

# Row per method + stats text
for row_idx, (plate, well, datatype, path) in enumerate(samples):
    img = load_gray(path)
    orig = center_crop(img, CROP_SIZE)
    
    stats_text = []
    for col_idx, (name, func, show_stats) in enumerate(methods):
        result = func(orig.copy())
        if show_stats:
            stats_text.append(f'{name}: mean={orig.mean():.0f}, std={orig.std():.0f}')
        else:
            if name == 'Z-score':
                stats_text.append(f'{name}: mean={result.mean():.2f}, std={result.std():.2f}')
            elif name == 'CLAHE':
                stats_text.append(f'{name}: mean={result.mean():.3f}, std={result.std():.3f}')
            elif name == 'Z-score + CLAHE':
                stats_text.append(f'{name}: mean={result.mean():.3f}, std={result.std():.3f}')
            elif 'HM' in name:
                stats_text.append(f'{name}: mean={result.mean():.0f}, std={result.std():.0f}')

plt.tight_layout(rect=[0, 0, 1, 0.95])
outpath = os.path.join(OUT, 'compare_normalization_methods.png')
plt.savefig(outpath, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {outpath}")

# --- Second figure: Pixel stats scatter plot ---
# Show how each method affects the drug-mutant separability
print("\nComputing per-method confound AUC for pooled data...")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def compute_stats_4(img):
    """mean, std, snr, entropy"""
    h, _ = np.histogram(img, bins=256, range=(0, 65535))
    p = h / h.sum(); p = p[p > 0]
    from scipy.stats import entropy
    return np.array([img.mean(), img.std(), img.mean()/(img.std()+1e-8), entropy(p)])

# For each plate and type, sample one image per well
method_names = [m[0] for m in methods]
method_stats = {m: [] for m in method_names}
method_labels = []

for plate in tqdm(PLATES, desc='Sampling plates'):
    for base_dir, datatype in [(MUTANT_BASE, 'mutant'), (DRUG_BASE, 'drug')]:
        wells = collect_wells(plate, base_dir)
        for well, paths in wells.items():
            path = random.choice(paths)
            img = load_gray(path)
            orig = center_crop(img, CROP_SIZE)
            for name, func, _ in methods:
                result = func(orig.copy())
                stats = compute_stats_4(result)
                method_stats[name].append(stats)
            method_labels.append(0 if datatype == 'mutant' else 1)

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
fig2.suptitle('Confound AUC by Normalization Method (per plate, 5-fold CV)', fontsize=14)

for idx, (name, func, _) in enumerate(methods):
    ax = axes2[idx // 3][idx % 3]
    X = np.array(method_stats[name])
    y = np.array(method_labels)
    
    plate_map = []
    for plate in PLATES:
        for base_dir, datatype in [(MUTANT_BASE, 'mutant'), (DRUG_BASE, 'drug')]:
            wells = collect_wells(plate, base_dir)
            for well in wells:
                plate_map.append(plate)
    
    plate_aucs = {}
    for plate in sorted(set(plate_map)):
        idxs = [i for i, p in enumerate(plate_map) if p == plate]
        if len(set(y[idxs])) < 2: continue
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        for tr, va in skf.split(X[idxs], y[idxs]):
            s = StandardScaler().fit(X[idxs][tr])
            lr = LogisticRegression(max_iter=1000).fit(s.transform(X[idxs][tr]), y[idxs][tr])
            aucs.append(roc_auc_score(y[idxs][va], lr.predict_proba(s.transform(X[idxs][va]))[:,1]))
        plate_aucs[plate] = (np.mean(aucs), np.std(aucs))
    
    plates = [p for p in sorted(plate_aucs.keys())]
    means = [plate_aucs[p][0] for p in plates]
    stds = [plate_aucs[p][1] for p in plates]
    ax.bar(plates, means, yerr=stds, capsize=4, alpha=0.7, color='steelblue')
    ax.axhline(0.5, color='gray', ls='--', lw=1)
    ax.set_ylim(0.3, 1.0)
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.set_ylabel('CV AUC')
    
    # Pooled
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs_p = []
    for tr, va in skf.split(X, y):
        s = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=1000).fit(s.transform(X[tr]), y[tr])
        aucs_p.append(roc_auc_score(y[va], lr.predict_proba(s.transform(X[va]))[:,1]))
    ax.text(0.5, 0.05, f'Pooled AUC = {np.mean(aucs_p):.3f}', ha='center', va='bottom',
            transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout(rect=[0, 0, 1, 0.95])
outpath2 = os.path.join(OUT, 'compare_normalization_auc.png')
plt.savefig(outpath2, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {outpath2}")

# Print table summary
print(f"\n{'Method':25s} {'Pooled AUC':>10s}")
print('-' * 37)
for name, func, _ in methods:
    X = np.array(method_stats[name])
    y = np.array(method_labels)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs_p = []
    for tr, va in skf.split(X, y):
        s = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=1000).fit(s.transform(X[tr]), y[tr])
        aucs_p.append(roc_auc_score(y[va], lr.predict_proba(s.transform(X[va]))[:,1]))
    print(f"{name:25s} {np.mean(aucs_p):.4f} ± {np.std(aucs_p):.4f}")
