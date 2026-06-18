#!/usr/bin/env python3
"""Compare original vs histogram-matched images for drug and mutant wells.
   Generates 4x4 panel: original crop, matched crop, diff map, histograms.
"""
import numpy as np; np.random.seed(42)
import os, json, re, random, warnings
from PIL import Image
from skimage.exposure import match_histograms
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

with open(os.path.join(PROJECT_DIR, 'plate_well_id_path.json')) as f:
    mutant_map = json.load(f)
with open(os.path.join(PROJECT_DIR, 'plate_well_ic50_mapping.json')) as f:
    drug_map = json.load(f)

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

# Build reference from all mutant crops across all plates
print("Building reference histogram from mutant crops...")
PLATES = ['P1','P2','P3','P4','P5','P6']
ref_crops = []
for plate in tqdm(PLATES):
    wells = collect_wells(plate, MUTANT_BASE)
    for well, paths in wells.items():
        path = random.choice(paths)
        crop = center_crop(load_gray(path), CROP_SIZE)
        ref_crops.append(crop)
ref_img = np.vstack(ref_crops)
print(f"  Reference: {ref_img.shape} ({len(ref_crops)} crops)")

def get_image_path(plate, well, base_dir):
    wells = collect_wells(plate, base_dir)
    if well not in wells: return None
    return random.choice(wells[well])

# Select wells: P2 (HM worked well) and P5 (HM struggled)
shared_P2 = sorted(set(collect_wells('P2', MUTANT_BASE)) & set(collect_wells('P2', DRUG_BASE)))
shared_P5 = sorted(set(collect_wells('P5', MUTANT_BASE)) & set(collect_wells('P5', DRUG_BASE)))

target_wells = []
for plate, shared, note in [('P2', shared_P2, 'HM worked (AUC 0.53 → chance)'),
                              ('P5', shared_P5, 'HM struggled (AUC 0.85)')]:
    w = shared[0] if shared else 'B2'
    for base_dir, datatype in [(MUTANT_BASE, 'mutant'), (DRUG_BASE, 'drug')]:
        path = get_image_path(plate, w, base_dir)
        if path:
            target_wells.append((plate, w, datatype, path, note))

print(f"Selected wells:")
for pw, datatype, note in [(w[0]+' '+w[1], w[2], w[4]) for w in target_wells]:
    print(f"  {pw} ({datatype}) — {note}")

# Build figure
n_rows = len(target_wells)
fig, axes = plt.subplots(n_rows, 4, figsize=(24, 4 * n_rows))
fig.suptitle('Original vs Histogram-Matched Center Crops (224×224)', fontsize=16, y=0.98)

for row_idx, (plate, well, datatype, path, note) in enumerate(target_wells):
    img = load_gray(path)
    orig = center_crop(img, CROP_SIZE)
    matched = match_histograms(orig, ref_img.astype(np.float32))
    diff = np.abs(orig.astype(np.float32) - matched.astype(np.float32))

    # Col 0: Original
    ax = axes[row_idx, 0]
    vmin, vmax = orig.min(), orig.max()
    im = ax.imshow(orig, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(f'{plate} {well} {datatype} — Original\n{note}', fontsize=9)
    ax.axis('off')

    # Col 1: Matched
    ax = axes[row_idx, 1]
    im = ax.imshow(matched, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(f'Histogram-Matched', fontsize=9)
    ax.axis('off')

    # Col 2: Diff map
    ax = axes[row_idx, 2]
    diff_norm = diff / diff.max() if diff.max() > 0 else diff
    im = ax.imshow(diff_norm, cmap='inferno', vmin=0, vmax=0.3)
    ax.set_title(f'|Orig - Matched|\nmax diff={diff.max()/65535*100:.1f}%', fontsize=9)
    ax.axis('off')

    # Col 3: Histogram overlay
    ax = axes[row_idx, 3]
    bins = np.linspace(0, 65535, 256)
    ax.hist(orig.ravel(), bins=bins, alpha=0.5, density=True,
            label=f'Original (μ={orig.mean():.0f})', color='steelblue')
    ax.hist(matched.ravel(), bins=bins, alpha=0.5, density=True,
            label=f'Matched (μ={matched.mean():.0f})', color='darkorange')
    ax.hist(ref_img.ravel(), bins=bins, alpha=0.3, density=True,
            label=f'Ref (μ={ref_img.mean():.0f})', color='green', ls='--')
    ax.set_xlabel('Pixel intensity'); ax.set_ylabel('Density')
    ax.set_title('Pixel intensity distribution', fontsize=9)
    ax.legend(fontsize=7)
    ax.set_yscale('log')

plt.tight_layout(rect=[0, 0, 1, 0.96])
outpath = os.path.join(OUT, 'compare_original_vs_matched.png')
plt.savefig(outpath, dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved: {outpath}")
