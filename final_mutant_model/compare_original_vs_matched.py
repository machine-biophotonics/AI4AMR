#!/usr/bin/env python3
"""Compare original vs histogram-matched images for drug and mutant wells.
   Generates a 2x2 panel per well: original crop, matched crop, diff map, histograms.
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
BASE = os.path.dirname(PROJECT_DIR)
MUTANT_BASE = os.path.join(BASE, 'Mutants_Data')
DRUG_BASE = os.path.join(BASE, 'Drugs_Data')
OUT = os.path.join(SCRIPT_DIR, 'image_analysis', 'output_all_plates')
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

# Select wells to visualize: one where HM worked well (P2) and one where it didn't (P5)
# Pick a shared well present in both mutant and drug for those plates
def get_random_well(plate, base_dir):
    wells = collect_wells(plate, base_dir)
    if not wells: return None
    return random.choice(list(wells.keys()))

def get_image_path(plate, well, base_dir):
    wells = collect_wells(plate, base_dir)
    if well not in wells: return None
    return random.choice(wells[well])

rows = []
shared_wells_P2 = set(collect_wells('P2', MUTANT_BASE).keys()) & set(collect_wells('P2', DRUG_BASE).keys())
shared_wells_P5 = set(collect_wells('P5', MUTANT_BASE).keys()) & set(collect_wells('P5', DRUG_BASE).keys())

target_wells = [
    ('P2', list(shared_wells_P2)[0] if shared_wells_P2 else 'B2', 'HM worked well (P2, AUC 0.53 → chance)'),
    ('P5', list(shared_wells_P5)[0] if shared_wells_P5 else 'B5', 'HM struggled (P5, AUC 0.85 → still confounded)'),
]

fig, axes = plt.subplots(4, 4, figsize=(20, 20))
fig.suptitle('Original vs Histogram-Matched Center Crops (224×224)', fontsize=16, y=0.98)

for col_idx, (plate, well, subtitle) in enumerate(target_wells):
    for row_idx, (base_dir, datatype) in enumerate([
        (MUTANT_BASE, 'mutant'), (DRUG_BASE, 'drug')
    ]):
        path = get_image_path(plate, well, base_dir)
        if not path:
            print(f"  WARNING: No image for {plate} {well} {datatype}")
            continue
        img = load_gray(path)
        orig = center_crop(img, CROP_SIZE)
        matched = match_histograms(orig, ref_img.astype(np.float32))
        diff = np.abs(orig - matched)

        row_offset = row_idx * 2

        # Row 1: Original
        ax = axes[row_offset, col_idx]
        im = ax.imshow(orig, cmap='gray', vmin=0, vmax=65535)
        ax.set_title(f'{plate} {well} {datatype} — Original', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Row 2: Matched
        ax = axes[row_offset + 1, col_idx]
        im = ax.imshow(matched, cmap='gray', vmin=0, vmax=65535)
        ax.set_title(f'{plate} {well} {datatype} — Matched', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

# Bottom row: difference maps and histograms for first (mutant P2) and second (mutant P5)
# Let me reorganize — better to show diff + histograms for the mutant wells

# Actually, let me redo the layout more clearly
plt.close(fig)

# New layout: 2 rows per plate (one mutant, one drug), 4 columns
# Col 0: original, Col 1: matched, Col 2: diff map, Col 3: histogram overlay
n_wells = len(target_wells) * 2  # 4 total (P2 mut, P2 drug, P5 mut, P5 drug)
fig, axes = plt.subplots(n_wells, 4, figsize=(24, 4 * n_wells))
fig.suptitle('Original vs Histogram-Matched Center Crops (224×224)', fontsize=16, y=0.98)

row_idx = 0
for plate, well, subtitle in target_wells:
    for base_dir, datatype in [(MUTANT_BASE, 'mutant'), (DRUG_BASE, 'drug')]:
        path = get_image_path(plate, well, base_dir)
        if not path:
            print(f"  WARNING: No image for {plate} {well} {datatype}")
            continue
        img = load_gray(path)
        orig = center_crop(img, CROP_SIZE)
        matched = match_histograms(orig, ref_img.astype(np.float32))
        diff = np.abs(orig - matched)
        diff_norm = diff / diff.max() if diff.max() > 0 else diff

        # Col 0: Original
        ax = axes[row_idx, 0]
        im = ax.imshow(orig, cmap='gray', vmin=0, vmax=65535)
        ax.set_title(f'{plate} {well} {datatype}\nOriginal', fontsize=10)
        ax.axis('off')

        # Col 1: Matched
        ax = axes[row_idx, 1]
        im = ax.imshow(matched, cmap='gray', vmin=0, vmax=65535)
        ax.set_title(f'Histogram-Matched', fontsize=10)
        ax.axis('off')

        # Col 2: Diff map
        ax = axes[row_idx, 2]
        im = ax.imshow(diff_norm, cmap='inferno', vmin=0, vmax=0.3)
        ax.set_title(f'|Orig - Matched|\nmax diff={diff.max()/65535*100:.1f}%', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, label='Normalized diff')

        # Col 3: Histogram overlay
        ax = axes[row_idx, 3]
        bins = np.linspace(0, 65535, 256)
        ax.hist(orig.ravel(), bins=bins, alpha=0.5, density=True, label='Original', color='steelblue')
        ax.hist(matched.ravel(), bins=bins, alpha=0.5, density=True, label='Matched', color='darkorange')
        ax.hist(ref_img.ravel(), bins=bins, alpha=0.3, density=True, label='Reference', color='green', ls='--')
        ax.set_xlabel('Pixel intensity'); ax.set_ylabel('Density')
        ax.set_title('Pixel intensity distribution', fontsize=10)
        ax.legend(fontsize=8)
        ax.set_yscale('log')

        row_idx += 1

# Add a text row for the subtitle
plt.tight_layout(rect=[0, 0, 1, 0.96])
outpath = os.path.join(OUT, 'compare_original_vs_matched.png')
plt.savefig(outpath, dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved: {outpath}")
