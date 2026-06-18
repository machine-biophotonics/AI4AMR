#!/usr/bin/env python3
"""Sample 1 image/well per plate, compute raw + model-preprocessed features
   on THREE spatial regions, save to a single CSV.

   Regions:
     full       = entire 2720×2720 image
     center1128 = center 1128×1128 (bounding box of 5×5 neighborhood of 224×224 crops)
     center224  = center 224×224 (single model crop)
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--plates', default='P1,P2,P3,P4,P5,P6',
                    help='Comma-separated plate names')
parser.add_argument('--output', default='all_plates_features.csv',
                    help='Output CSV filename')
args = parser.parse_args()
PLATES = args.plates.split(',')

import numpy as np; np.random.seed(42)
import os, json, re, csv, random, warnings
from PIL import Image
from scipy.stats import entropy as sp_entropy
from tqdm import tqdm
warnings.filterwarnings('ignore')
random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
BASE = os.path.dirname(PROJECT_DIR)
MUTANT_BASE = os.path.join(BASE, 'Mutants_Data')
DRUG_BASE = os.path.join(BASE, 'Drugs_Data')
OUT = os.path.join(SCRIPT_DIR, 'output_all_plates'); os.makedirs(OUT, exist_ok=True)

with open(os.path.join(PROJECT_DIR, 'plate_well_id_path.json')) as f:
    mutant_map = json.load(f)
with open(os.path.join(PROJECT_DIR, 'plate_well_ic50_mapping.json')) as f:
    drug_map = json.load(f)

# --- Model crop parameters ---
# From train_mil.py defaults: grid_size=12, crop_size=224, neighborhood=5
# stride = (2720 - 224) // (12 - 1) = 226
# 5×5 neighborhood crops span: (half_n=2) → offset range 0..4, width = 4*stride + crop_size
STRIDE = (2720 - 224) // 11  # = 226
HALF_N = 2
NEIGHBORHOOD_SPAN = 4 * STRIDE + 224  # = 1128
CENTER = 2720 // 2  # = 1360
CROP_SIZE = 224

def load_gray(path):
    img = np.array(Image.open(path))
    if len(img.shape) == 3: img = img[:,:,0]
    return img.astype(np.float32)

def extract_well(fname):
    m = re.search(r'Well([A-Z]\d+)_', fname)
    return m.group(1) if m else None

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
    }

def model_preprocess(img):
    img_01 = img.astype(np.float32) / 65535.0
    img_255 = (img_01 * 255).astype(np.uint8)
    return (img_255.astype(np.float32) / 255.0 - 0.5) / 0.5

def center_crop(img, size):
    h, w = img.shape
    half = size // 2
    cy, cx = h // 2, w // 2
    return img[cy-half:cy+half, cx-half:cx+half]

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
        img_1128 = center_crop(img, NEIGHBORHOOD_SPAN)
        img_224  = center_crop(img, CROP_SIZE)

        label = (mapping.get(plate,{}).get(well,{}).get('id','unknown')
                 if datatype=='mutant'
                 else f"{drug_map.get(plate,{}).get(well,{}).get('antibiotic','unknown')}_{drug_map.get(plate,{}).get(well,{}).get('ic50_multiple','')}")
        entry = {'plate': plate, 'well': well, 'label': label, 'type': datatype, 'path': path}

        # Stats on three regions: full, center1128, center224
        for region_name, region_img in [('full', img), ('center1128', img_1128), ('center224', img_224)]:
            region_raw = compute_stats(region_img)
            region_mp  = compute_stats(model_preprocess(region_img), range_min=-1, range_max=1)
            for k, v in region_raw.items(): entry[f'{region_name}_raw_{k}'] = v
            for k, v in region_mp.items():  entry[f'{region_name}_mp_{k}'] = v

        results.append(entry)
    return results

# === Sample all plates ===
all_rows = []
for plate in PLATES:
    print(f"\nSampling {plate}...")
    for base_dir, mapping, datatype in [(MUTANT_BASE, mutant_map, 'mutant'),
                                         (DRUG_BASE, drug_map, 'drug')]:
        rows = process_plate(plate, base_dir, mapping, datatype)
        all_rows.extend(rows)

# === Save to CSV ===
raw_fields = ['mean','std','snr','entropy','p1','p99','median']
mp_fields  = ['mean','std','snr','entropy','p1','p99','median']
regions = ['full', 'center1128', 'center224']
fieldnames = ['plate','well','label','type','path']
for reg in regions:
    fieldnames += [f'{reg}_raw_{k}' for k in raw_fields]
    fieldnames += [f'{reg}_mp_{k}' for k in mp_fields]

outpath = os.path.join(OUT, args.output)
with open(outpath, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(all_rows)

print(f"\nSaved {len(all_rows)} rows to {outpath}")
print(f"  Plates: {sorted(set(r['plate'] for r in all_rows))}")
print(f"  Types: {sorted(set(r['type'] for r in all_rows))}")
print(f"  Regions: {regions}")
