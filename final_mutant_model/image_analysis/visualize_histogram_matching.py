#!/usr/bin/env python3
"""Visualize histogram matching: original vs matched for center224 and center1128.
Shows images + histograms side by side for both drug and mutant examples.
"""
import numpy as np; np.random.seed(42)
import os, json, re, random, warnings
from PIL import Image
from scipy.stats import entropy as sp_entropy
from skimage.exposure import match_histograms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore'); random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
BASE = os.path.dirname(PROJECT_DIR)
MUTANT_BASE = os.path.join(BASE, 'Mutants_Data')
DRUG_BASE = os.path.join(BASE, 'Drugs_Data')
OUT = os.path.join(SCRIPT_DIR, 'output_all_plates'); os.makedirs(OUT, exist_ok=True)

STRIDE = (2720 - 224) // 11
NEIGHBORHOOD_SPAN = 4 * STRIDE + 224  # 1128
CROP_SIZE = 224

def load_gray(path):
    img = np.array(Image.open(path))
    return img[:,:,0].astype(np.float32) if len(img.shape)==3 else img.astype(np.float32)

def extract_well(fname):
    m = re.search(r'Well([A-Z]\d+)_', fname)
    return m.group(1) if m else None

def center_crop(img, size):
    h, w = img.shape; half = size // 2
    return img[h//2-half:h//2+half, w//2-half:w//2+half]

def collect_wells(plate, base_dir):
    pd_ = os.path.join(base_dir, plate)
    if not os.path.exists(pd_): return {}
    wells = {}
    for root, _, files in os.walk(pd_):
        for fname in files:
            if not fname.endswith(('.tif','.tiff','.png')): continue
            well = extract_well(fname)
            if well:
                wells.setdefault(well, []).append(os.path.join(root, fname))
    return wells

# =============================================
# BUILD REFERENCES (same as save_features_hm_full.py)
# =============================================
PLATES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
print("Building references...")

ref_224_crops = []
ref_1128_crops = []

for plate in PLATES:
    wells = collect_wells(plate, MUTANT_BASE)
    for wi, (well, paths) in enumerate(wells.items()):
        path = random.choice(paths)
        img = load_gray(path)
        ref_224_crops.append(center_crop(img, CROP_SIZE))
        if wi < 8:
            ref_1128_crops.append(center_crop(img, NEIGHBORHOOD_SPAN))

ref_img_224 = np.vstack(ref_224_crops)
ref_img_1128 = np.vstack(ref_1128_crops)
print(f"  224 reference: {ref_img_224.shape}")
print(f"  1128 reference: {ref_img_1128.shape}")

# =============================================
# SAMPLE IMAGES (1 drug + 1 mutant from P1)
# =============================================
print("\nSampling images...")
examples = []

for datatype, base_dir in [('Drug', DRUG_BASE), ('Mutant', MUTANT_BASE)]:
    wells = collect_wells('P1', base_dir)
    # Pick first well
    well = list(wells.keys())[0]
    path = random.choice(wells[well])
    img = load_gray(path)
    examples.append((datatype, path, img))

print(f"  Drug:   {examples[0][1]}")
print(f"  Mutant: {examples[1][1]}")

# =============================================
# PLOT: 2 rows (drug, mutant) x 3 columns per region
# =============================================
fig, axes = plt.subplots(4, 6, figsize=(24, 16))
regions = [
    ('Center 224×224', CROP_SIZE, ref_img_224),
    ('Center 1128×1128', NEIGHBORHOOD_SPAN, ref_img_1128),
]

col_idx = 0
for region_label, region_size, ref_img in regions:
    for row_idx, (datatype, path, full_img) in enumerate(examples):
        crop = center_crop(full_img, region_size)
        matched = match_histograms(crop, ref_img)
        
        # Row 1: Original image
        ax_img = axes[row_idx * 2, col_idx]
        im = ax_img.imshow(crop, cmap='gray', vmin=crop.min(), vmax=crop.max())
        ax_img.set_title(f'{datatype} — {region_label}\nOriginal', fontsize=10)
        ax_img.axis('off')
        
        # Row 1: Original histogram
        ax_hist = axes[row_idx * 2, col_idx + 1]
        ax_hist.hist(crop.ravel(), bins=256, color='steelblue', alpha=0.7, density=True)
        ax_hist.set_title(f'{datatype} — Original Histogram', fontsize=9)
        ax_hist.set_xlabel('Pixel intensity')
        ax_hist.set_ylabel('Density')
        ax_hist.tick_params(labelsize=7)
        
        # Row 2: Matched image
        ax_img2 = axes[row_idx * 2 + 1, col_idx]
        im2 = ax_img2.imshow(matched, cmap='gray', vmin=matched.min(), vmax=matched.max())
        ax_img2.set_title(f'{datatype} — {region_label}\nHistogram Matched', fontsize=10)
        ax_img2.axis('off')
        
        # Row 2: Matched histogram
        ax_hist2 = axes[row_idx * 2 + 1, col_idx + 1]
        ax_hist2.hist(matched.ravel(), bins=256, color='coral', alpha=0.7, density=True)
        ax_hist2.set_title(f'{datatype} — Matched Histogram', fontsize=9)
        ax_hist2.set_xlabel('Pixel intensity')
        ax_hist2.set_ylabel('Density')
        ax_hist2.tick_params(labelsize=7)
        
        # Row 2: Overlay histogram
        ax_overlay = axes[row_idx * 2 + 1, col_idx + 2]
        ax_overlay.hist(crop.ravel(), bins=256, color='steelblue', alpha=0.5, density=True, label='Original')
        ax_overlay.hist(matched.ravel(), bins=256, color='coral', alpha=0.5, density=True, label='Matched')
        # Also show reference histogram
        ax_overlay.hist(ref_img.ravel(), bins=256, color='green', alpha=0.2, density=True, label='Reference')
        ax_overlay.set_title(f'{datatype} — Overlay', fontsize=9)
        ax_overlay.set_xlabel('Pixel intensity')
        ax_overlay.legend(fontsize=7)
        ax_overlay.tick_params(labelsize=7)
    
    col_idx += 3

# Reference histograms in last column
for ri, (region_label, _, ref_img) in enumerate(regions):
    ax_ref = axes[ri * 2, 5]
    ax_ref.hist(ref_img.ravel(), bins=256, color='green', alpha=0.7, density=True)
    ax_ref.set_title(f'Reference Histogram\n{region_label}', fontsize=9)
    ax_ref.set_xlabel('Pixel intensity')
    ax_ref.tick_params(labelsize=7)
    
    # Empty the bottom-right cells
    axes[ri * 2 + 1, 5].axis('off')

plt.suptitle('Histogram Matching Visualization: Original vs Matched Images and Pixel Distributions\n'
             '(Reference: aggregate of all mutant center crops across 6 plates)',
             fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
outpath = os.path.join(OUT, 'histogram_matching_visualization.png')
fig.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved {outpath}")
