#!/usr/bin/env python3
"""Visual comparison: original vs rank-normalized at 3 levels + histograms."""
import argparse, numpy as np, os, json, re, random, warnings
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore'); random.seed(42); np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--plate', default='P2')
parser.add_argument('--output', default='ranknorm_visualization.png')
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
BASE = os.path.dirname(PROJECT_DIR)
MUTANT_BASE = os.path.join(BASE, 'Mutants_Data')
DRUG_BASE = os.path.join(BASE, 'Drugs_Data')
OUT = os.path.join(SCRIPT_DIR, 'output_all_plates'); os.makedirs(OUT, exist_ok=True)

with open(os.path.join(PROJECT_DIR, 'plate_well_id_path.json')) as f: mutant_map = json.load(f)
with open(os.path.join(PROJECT_DIR, 'plate_well_ic50_mapping.json')) as f: drug_map = json.load(f)

STRIDE = (2720 - 224) // 11; NEIGHBORHOOD_SPAN = 4 * STRIDE + 224; CROP_SIZE = 224

def load_gray(path):
    img = np.array(Image.open(path))
    return img[:,:,0].astype(np.float32) if len(img.shape)==3 else img.astype(np.float32)

def center_crop(img, size):
    half=size//2; h,w=img.shape; return img[h//2-half:h//2+half, w//2-half:w//2+half]

def rank_normalize(img):
    flat = img.flatten()
    ranks = np.argsort(np.argsort(flat)).astype(np.float32)
    return (ranks / (len(ranks) - 1)).reshape(img.shape)

def pick_one(plate, base_dir, mapping, datatype):
    pd_ = os.path.join(base_dir, plate)
    wells = {}
    for root,_,files in os.walk(pd_):
        for f in files:
            if not f.endswith(('.tif','.tiff','.png')): continue
            m = re.search(r'Well([A-Z]\d+)_', f)
            if m: wells.setdefault(m.group(1), []).append(os.path.join(root, f))
    if not wells: return None, 'unknown'
    well = random.choice(list(wells.keys()))
    path = random.choice(wells[well])
    label = (mapping.get(plate,{}).get(well,{}).get('id','unknown') if datatype=='mutant'
             else f"{drug_map.get(plate,{}).get(well,{}).get('antibiotic','unknown')}")
    return path, label

drug_path, drug_label = pick_one(args.plate, DRUG_BASE, drug_map, 'drug')
mut_path, mut_label = pick_one(args.plate, MUTANT_BASE, mutant_map, 'mutant')
print(f"Drug: {drug_label} @ {drug_path}")
print(f"Mutant: {mut_label} @ {mut_path}")

drug_img = load_gray(drug_path)
mut_img = load_gray(mut_path)

region_names = ['full', 'center1128', 'center224']
region_labels = {'full': 'Full 2720×2720', 'center1128': 'Center 1128×1128', 'center224': 'Center 224×224'}
crop_sizes = {'full': 2720, 'center1128': NEIGHBORHOOD_SPAN, 'center224': CROP_SIZE}

# Crop images
d_crops = {}
m_crops = {}
for rn in region_names:
    d_crops[rn] = center_crop(drug_img, crop_sizes[rn]) if rn != 'full' else drug_img
    m_crops[rn] = center_crop(mut_img, crop_sizes[rn]) if rn != 'full' else mut_img

# Also save a debug histogram
fig_debug, ax_debug = plt.subplots(1, 1, figsize=(8, 4))
d224 = rank_normalize(d_crops['center224'])
m224 = rank_normalize(m_crops['center224'])
ax_debug.hist(d224.flatten(), bins=80, alpha=0.6, color='red', label='Drug', density=True)
ax_debug.hist(m224.flatten(), bins=80, alpha=0.6, color='blue', label='Mutant', density=True)
ax_debug.axhline(1.0, color='gray', ls='--', alpha=0.5)
ax_debug.legend()
ax_debug.set_title('Debug: Rank-norm center224 histograms')
ax_debug.set_xlabel('Pixel value'); ax_debug.set_ylabel('Density')
print(f"d224 ranknorm stats: min={d224.min():.4f}, max={d224.max():.4f}, mean={d224.mean():.4f}, std={d224.std():.4f}")
print(f"m224 ranknorm stats: min={m224.min():.4f}, max={m224.max():.4f}, mean={m224.mean():.4f}, std={m224.std():.4f}")
debug_out = os.path.join(OUT, 'debug_hist.png')
fig_debug.savefig(debug_out, dpi=100); plt.close(fig_debug)
print(f"Saved debug {debug_out}")

# Main figure: 5 rows x 3 cols
fig = plt.figure(figsize=(20, 24))
fig.suptitle(f'Original vs Rank Normalization — Plate {args.plate}\nDrug: {drug_label}   |   Mutant: {mut_label}', fontsize=16, y=0.97)

colors = {'Drug': '#e41a1c', 'Mutant': '#377eb8'}
rn_colors = {'Drug': '#ff9999', 'Mutant': '#99ccff'}

for ci, rn in enumerate(region_names):
    # === Row 0: Original drug ===
    ax = plt.subplot2grid((5, 3), (0, ci))
    ax.imshow(d_crops[rn], cmap='gray')
    ax.axis('off')
    if ci == 0: ax.set_ylabel('Original Drug', fontsize=11, fontweight='bold')
    ax.set_title(region_labels[rn], fontsize=12, fontweight='bold')
    
    # === Row 1: Original mutant ===
    ax = plt.subplot2grid((5, 3), (1, ci))
    ax.imshow(m_crops[rn], cmap='gray')
    ax.axis('off')
    if ci == 0: ax.set_ylabel('Original Mutant', fontsize=11, fontweight='bold')
    
    # === Row 2: Rank-normalized (drug | mutant) ===
    ax = plt.subplot2grid((5, 3), (2, ci))
    d_rn = rank_normalize(d_crops[rn])
    m_rn = rank_normalize(m_crops[rn])
    gap = np.ones((d_rn.shape[0], 16))
    strip = np.concatenate([d_rn, gap, m_rn], axis=1)
    ax.imshow(strip, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    y_pos = -d_rn.shape[0] * 0.04
    ax.text(d_rn.shape[1]/2, y_pos, 'Drug', ha='center', fontsize=9, color='red', fontweight='bold')
    ax.text(d_rn.shape[1]+16+d_rn.shape[1]/2, y_pos, 'Mut', ha='center', fontsize=9, color='blue', fontweight='bold')
    if ci == 0: ax.set_ylabel('Rank-Norm', fontsize=11, fontweight='bold')
    
    # === Row 3: Original histograms ===
    ax = plt.subplot2grid((5, 3), (3, ci))
    bins = 80
    ax.hist(d_crops[rn].flatten(), bins=bins, alpha=0.6, color=colors['Drug'], label='Drug', density=True)
    ax.hist(m_crops[rn].flatten(), bins=bins, alpha=0.6, color=colors['Mutant'], label='Mut', density=True)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    if ci == 0: ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title('Original hist', fontsize=10)
    
    # === Row 4: Rank-norm histograms ===
    ax = plt.subplot2grid((5, 3), (4, ci))
    ax.hist(d_rn.flatten(), bins=bins, alpha=0.6, color=rn_colors['Drug'], label='Drug RN', density=True)
    ax.hist(m_rn.flatten(), bins=bins, alpha=0.6, color=rn_colors['Mutant'], label='Mut RN', density=True)
    ax.set_xlim(0, 1)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5, lw=1)
    if ci == 0: ax.set_ylabel('Density', fontsize=10)
    ax.set_xlabel('Pixel value', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title('Rank-norm hist', fontsize=10)

plt.tight_layout()
outpath = os.path.join(OUT, args.output)
fig.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved {outpath}")
