#!/usr/bin/env python3
"""Histogram matching to eliminate drug-mutant pixel confound.
   Efficient: matches only center crops (~0.6 sec/image), not full 2720×2720.

   Usage:
     python3 histogram_matching_analysis.py
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--plates', default='P1,P2,P3,P4,P5,P6', help='Comma-separated plate names')
parser.add_argument('--output', default='all_plates_features_matched.csv', help='Output CSV')
parser.add_argument('--force', action='store_true', default=False,
                    help='Re-run from scratch even if output CSV exists')
parser.add_argument('--skip-validation', action='store_true', default=False,
                    help='Skip Phase 4 validation step')
args = parser.parse_args()
PLATES = args.plates.split(',')

import numpy as np; np.random.seed(42)
import os, json, re, csv, random, warnings, subprocess, sys
from PIL import Image
from scipy.stats import entropy as sp_entropy
from skimage.exposure import match_histograms
from tqdm import tqdm
warnings.filterwarnings('ignore')
random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATE = os.path.join(SCRIPT_DIR, 'validate_confound.py')
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
BASE = os.path.dirname(PROJECT_DIR)
MUTANT_BASE = os.path.join(BASE, 'Mutants_Data')
DRUG_BASE = os.path.join(BASE, 'Drugs_Data')
OUT = os.path.join(SCRIPT_DIR, 'output_all_plates'); os.makedirs(OUT, exist_ok=True)

with open(os.path.join(PROJECT_DIR, 'plate_well_id_path.json')) as f:
    mutant_map = json.load(f)
with open(os.path.join(PROJECT_DIR, 'plate_well_ic50_mapping.json')) as f:
    drug_map = json.load(f)

STRIDE = (2720 - 224) // 11
NEIGHBORHOOD_SPAN = 4 * STRIDE + 224  # 1128
CROP_SIZE = 224
CROP_1128_HALF = NEIGHBORHOOD_SPAN // 2  # 564

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

# =====================================================
# PHASES 1-3: Build reference, match histograms, save CSV
# =====================================================
outpath = os.path.join(OUT, args.output)

if os.path.exists(outpath) and not args.force:
    print("=" * 80)
    print("PHASES 1-3: Data already processed")
    print("=" * 80)
    print(f"  CSV exists: {outpath}")
    print(f"  Use --force to re-process, --skip-validation to skip Phase 4.\n")
else:
    # =====================================================
    # PHASE 1: Building reference histograms
    # =====================================================
    print("=" * 80)
    print("PHASE 1: Building reference histograms")
    print("=" * 80)

    # 224 reference: stack all mutant center 224 crops
    ref_224_crops = []
    for plate in tqdm(PLATES, desc='Collecting 224 mutant crops'):
        wells = collect_wells(plate, MUTANT_BASE)
        for well, paths in wells.items():
            path = random.choice(paths)
            crop = center_crop(load_gray(path), CROP_SIZE)
            ref_224_crops.append(crop)
    ref_img_224 = np.vstack(ref_224_crops)
    print(f"  224 reference: {ref_img_224.shape} ({len(ref_224_crops)} crops)")

    # 1128 reference: stack first 48 mutant center 1128 crops
    ref_1128_crops = []
    for plate in tqdm(PLATES, desc='Collecting 1128 mutant crops'):
        wells = collect_wells(plate, MUTANT_BASE)
        for well, paths in list(wells.items())[:8]:
            path = random.choice(paths)
            crop = center_crop(load_gray(path), NEIGHBORHOOD_SPAN)
            ref_1128_crops.append(crop)
    ref_img_1128 = np.vstack(ref_1128_crops)
    print(f"  1128 reference: {ref_img_1128.shape} ({len(ref_1128_crops)} crops)")

    # =====================================================
    # PHASE 2: Matching histograms on CROPS only (fast)
    # =====================================================
    print("\n" + "=" * 80)
    print("PHASE 2: Matching histograms on center crops + computing stats")
    print("=" * 80)

    def process_plate(plate, base_dir, mapping, datatype):
        plate_dir = os.path.join(base_dir, plate)
        if not os.path.exists(plate_dir): return []
        wells = collect_wells(plate, base_dir)
        results = []
        for well, paths in tqdm(wells.items(), desc=f'{datatype} {plate}'):
            path = random.choice(paths)
            img = load_gray(path)

            orig_224   = center_crop(img, CROP_SIZE)
            orig_1128  = center_crop(img, NEIGHBORHOOD_SPAN)

            matched_224   = match_histograms(orig_224,   ref_img_224.astype(np.float32))
            matched_1128  = match_histograms(orig_1128,  ref_img_1128.astype(np.float32))

            label = (mapping.get(plate,{}).get(well,{}).get('id','unknown')
                     if datatype=='mutant'
                     else f"{drug_map.get(plate,{}).get(well,{}).get('antibiotic','unknown')}_{drug_map.get(plate,{}).get(well,{}).get('ic50_multiple','')}")
            entry = {'plate': plate, 'well': well, 'label': label, 'type': datatype, 'path': path}

            for prefix, rim in [('orig_center224', orig_224), ('orig_center1128', orig_1128)]:
                raw = compute_stats(rim)
                mp  = compute_stats(model_preprocess(rim), range_min=-1, range_max=1)
                for k, v in raw.items(): entry[f'{prefix}_raw_{k}'] = v
                for k, v in mp.items():  entry[f'{prefix}_mp_{k}'] = v

            for prefix, rim in [('hm224_center224', matched_224), ('hm224_center1128', matched_1128)]:
                raw = compute_stats(rim)
                mp  = compute_stats(model_preprocess(rim), range_min=-1, range_max=1)
                for k, v in raw.items(): entry[f'{prefix}_raw_{k}'] = v
                for k, v in mp.items():  entry[f'{prefix}_mp_{k}'] = v

            results.append(entry)
        return results

    all_rows = []
    for plate in PLATES:
        print(f"\n  Plate {plate}...")
        for base_dir, mapping, datatype in [(MUTANT_BASE, mutant_map, 'mutant'),
                                             (DRUG_BASE, drug_map, 'drug')]:
            rows = process_plate(plate, base_dir, mapping, datatype)
            all_rows.extend(rows)

    # =====================================================
    # PHASE 3: Save CSV
    # =====================================================
    all_fields = ['mean','std','snr','entropy','p1','p99','median']
    mp_fields  = ['mean','std','snr','entropy','p1','p99','median']

    fieldnames = ['plate','well','label','type','path']
    for prefix in ['orig_center224','orig_center1128','hm224_center224','hm224_center1128']:
        for k in all_fields: fieldnames.append(f'{prefix}_raw_{k}')
        for k in mp_fields:  fieldnames.append(f'{prefix}_mp_{k}')

    with open(outpath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {outpath}")
    print(f"Phases 1-3 complete.")

# =====================================================
# PHASE 4: Run validation
# =====================================================
print("\n" + "=" * 80)
print("PHASE 4: Running validation on original vs matched features")
print("=" * 80)

if args.skip_validation:
    print("  Skipping validation (--skip-validation set).")
elif os.path.exists(VALIDATE):
    for label, prefix in [("ORIGINAL (no matching)", 'orig_center224_mp_'),
                           ("MATCHED (224 ref, center224)", 'hm224_center224_mp_'),
                           ("MATCHED (224 ref, center1128)", 'hm224_center1128_mp_')]:
        print(f"\n--- {label} ---")
        out_dir = 'validation_' + prefix.replace('_mp_','').replace('_','')
        subprocess.run([
            sys.executable, VALIDATE,
            '--input', args.output,
            '--output', out_dir,
            '--feat_prefix', prefix,
            '--permutations', '500'
        ], cwd=SCRIPT_DIR)
else:
    print(f"  validate_confound.py not found at {VALIDATE}")

