#!/usr/bin/env python3
"""Histogram matching for all 3 regions (full, center1128, center224)
   using mutant reference images. Extends the original matched CSV
   with full-image features for complete ROC curve analysis.
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--plates', default='P1,P2,P3,P4,P5,P6')
parser.add_argument('--output', default='all_plates_features_hm_full.csv')
args = parser.parse_args()
PLATES = args.plates.split(',')

import numpy as np; np.random.seed(42)
import os, json, re, csv, random, warnings
from PIL import Image
from scipy.stats import entropy as sp_entropy
from skimage.exposure import match_histograms
from tqdm import tqdm
warnings.filterwarnings('ignore'); random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
BASE = os.path.dirname(PROJECT_DIR)
MUTANT_BASE = os.path.join(BASE, 'Mutants_Data')
DRUG_BASE = os.path.join(BASE, 'Drugs_Data')
OUT = os.path.join(SCRIPT_DIR, 'output_all_plates'); os.makedirs(OUT, exist_ok=True)

with open(os.path.join(PROJECT_DIR, 'plate_well_id_path.json')) as f: mutant_map = json.load(f)
with open(os.path.join(PROJECT_DIR, 'plate_well_ic50_mapping.json')) as f: drug_map = json.load(f)

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

def compute_stats(img):
    h, _ = np.histogram(img, bins=256, range=(float(img.min()), float(img.max()) + 1e-8))
    p = h / h.sum(); p = p[p > 0]
    return {
        'mean': float(img.mean()), 'std': float(img.std()),
        'snr': float(img.mean() / (img.std() + 1e-8)),
        'entropy': float(sp_entropy(p)),
        'p1': float(np.percentile(img, 1)),
        'p99': float(np.percentile(img, 99)),
        'median': float(np.median(img)),
    }

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
# BUILD REFERENCES (from mutant images)
# =============================================
print("=" * 60)
print("Building reference histograms from mutant images")
print("=" * 60)

# Full image reference: stack 6 mutant full images (1 per plate)
ref_full = []
# 224 reference: stack all mutant center 224 crops
ref_224_crops = []
# 1128 reference: stack first 48 mutant center 1128 crops
ref_1128_crops = []

for plate in tqdm(PLATES, desc='Collecting mutant references'):
    wells = collect_wells(plate, MUTANT_BASE)
    for wi, (well, paths) in enumerate(wells.items()):
        path = random.choice(paths)
        img = load_gray(path)
        if wi == 0:  # 1 full image per plate
            ref_full.append(img)
        ref_224_crops.append(center_crop(img, CROP_SIZE))
        if wi < 8:
            ref_1128_crops.append(center_crop(img, NEIGHBORHOOD_SPAN))

ref_img_full = np.vstack(ref_full) if ref_full else None
ref_img_224 = np.vstack(ref_224_crops)
ref_img_1128 = np.vstack(ref_1128_crops)
print(f"  Full reference: {ref_img_full.shape if ref_img_full is not None else 'N/A'}")
print(f"  224 reference:  {ref_img_224.shape} ({len(ref_224_crops)} crops)")
print(f"  1128 reference: {ref_img_1128.shape} ({len(ref_1128_crops)} crops)")

# =============================================
# PROCESS ALL PLATES
# =============================================
print("\n" + "=" * 60)
print("Processing all plates (histogram matching + stats)")
print("=" * 60)

all_rows = []
for plate in PLATES:
    for datatype, base_dir, mapping in [('mutant', MUTANT_BASE, mutant_map), ('drug', DRUG_BASE, drug_map)]:
        wells = collect_wells(plate, base_dir)
        for well, paths in tqdm(wells.items(), desc=f'{datatype} {plate}'):
            path = random.choice(paths)
            img = load_gray(path)
            
            # Histogram match each region to its reference
            img_full = match_histograms(img, ref_img_full) if ref_img_full is not None else img
            img_1128 = match_histograms(center_crop(img, NEIGHBORHOOD_SPAN), ref_img_1128)
            img_224 = match_histograms(center_crop(img, CROP_SIZE), ref_img_224)
            
            label = (mapping.get(plate,{}).get(well,{}).get('id','unknown') if datatype=='mutant'
                     else f"{drug_map.get(plate,{}).get(well,{}).get('antibiotic','unknown')}_{drug_map.get(plate,{}).get(well,{}).get('ic50_multiple','')}")
            
            entry = {'plate': plate, 'well': well, 'label': label, 'type': datatype, 'path': path}
            for rn, ri in [('full', img_full), ('center1128', img_1128), ('center224', img_224)]:
                for k, v in compute_stats(ri).items():
                    entry[f'hm_{rn}_raw_{k}'] = v
                    entry[f'hm_{rn}_mp_{k}'] = v
            all_rows.append(entry)

# =============================================
# SAVE CSV
# =============================================
fields = ['mean','std','snr','entropy','p1','p99','median']
regions = ['full','center1128','center224']
fieldnames = ['plate','well','label','type','path']
for reg in regions:
    fieldnames += [f'hm_{reg}_raw_{k}' for k in fields] + [f'hm_{reg}_mp_{k}' for k in fields]

outpath = os.path.join(OUT, args.output)
with open(outpath, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(all_rows)
print(f"\nSaved {len(all_rows)} rows to {outpath}")
print(f"  Regions: full, center1128, center224 (all histogram-matched to mutant reference)")
