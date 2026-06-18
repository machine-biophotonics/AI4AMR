#!/usr/bin/env python3
"""Fourier Phase-Preserving Normalization.
FFT -> keep phase (preserves ALL spatial structure: edges, textures, morphology)
     -> match magnitude spectrum to reference (global intensity distribution)
     -> inverse FFT

Reference: aggregate magnitude spectrum from all mutant images across plates.
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--plates', default='P1,P2,P3,P4,P5,P6')
parser.add_argument('--output', default='all_plates_features_fourier.csv')
args = parser.parse_args()
PLATES = args.plates.split(',')

import numpy as np; np.random.seed(42)
import os, json, re, csv, random, warnings
from PIL import Image
from scipy.stats import entropy as sp_entropy
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
NEIGHBORHOOD_SPAN = 4 * STRIDE + 224
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

def fourier_normalize(img, ref_magnitude):
    """Phase-preserving Fourier normalization.
    Args:
        img: (H, W) float32 array
        ref_magnitude: (H, W) reference magnitude spectrum (from FFT)
    Returns:
        normalized image with same phase as input, ref-matched magnitude
    """
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)
    
    # Replace magnitude with reference magnitude
    # Scale reference to match input's overall energy level
    scale = magnitude.sum() / (ref_magnitude.sum() + 1e-8)
    new_magnitude = ref_magnitude * scale
    
    # Reconstruct with original phase + new magnitude
    fshift_new = new_magnitude * np.exp(1j * phase)
    f_new = np.fft.ifftshift(fshift_new)
    img_new = np.fft.ifft2(f_new).real
    
    return img_new.astype(np.float32)

def build_ref_magnitude(images):
    """Build reference magnitude spectrum from list of images.
    Average magnitude spectrum across all images.
    """
    magnitudes = []
    for img in images:
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        magnitudes.append(magnitude)
    return np.mean(magnitudes, axis=0)

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
# BUILD REFERENCE MAGNITUDE SPECTRA
# =============================================
print("=" * 60)
print("Building reference magnitude spectra from mutant images")
print("=" * 60)

ref_full_imgs = []
ref_224_imgs = []
ref_1128_imgs = []

for plate in tqdm(PLATES, desc='Collecting mutant references'):
    wells = collect_wells(plate, MUTANT_BASE)
    for wi, (well, paths) in enumerate(wells.items()):
        path = random.choice(paths)
        img = load_gray(path)
        if wi == 0:
            ref_full_imgs.append(img)
        ref_224_imgs.append(center_crop(img, CROP_SIZE))
        if wi < 8:
            ref_1128_imgs.append(center_crop(img, NEIGHBORHOOD_SPAN))

if ref_full_imgs:
    ref_mag_full = build_ref_magnitude(ref_full_imgs)
else:
    ref_mag_full = None
ref_mag_224 = build_ref_magnitude(ref_224_imgs)
ref_mag_1128 = build_ref_magnitude(ref_1128_imgs)
print(f"  Full reference magnitude:  {'built from ' + str(len(ref_full_imgs)) + ' images' if ref_mag_full is not None else 'N/A'}")
print(f"  224 reference magnitude:   built from {len(ref_224_imgs)} crops")
print(f"  1128 reference magnitude:  built from {len(ref_1128_imgs)} crops")

# =============================================
# PROCESS ALL PLATES
# =============================================
print("\n" + "=" * 60)
print("Processing all plates (Fourier normalization + stats)")
print("=" * 60)

all_rows = []
for plate in PLATES:
    for datatype, base_dir, mapping in [('mutant', MUTANT_BASE, mutant_map), ('drug', DRUG_BASE, drug_map)]:
        wells = collect_wells(plate, base_dir)
        for well, paths in tqdm(wells.items(), desc=f'{datatype} {plate}'):
            path = random.choice(paths)
            img = load_gray(path)
            
            # Fourier normalize each region
            img_full = fourier_normalize(img, ref_mag_full) if ref_mag_full is not None else img
            img_1128 = fourier_normalize(center_crop(img, NEIGHBORHOOD_SPAN), ref_mag_1128)
            img_224 = fourier_normalize(center_crop(img, CROP_SIZE), ref_mag_224)
            
            label = (mapping.get(plate,{}).get(well,{}).get('id','unknown') if datatype=='mutant'
                     else f"{drug_map.get(plate,{}).get(well,{}).get('antibiotic','unknown')}_{drug_map.get(plate,{}).get(well,{}).get('ic50_multiple','')}")
            
            entry = {'plate': plate, 'well': well, 'label': label, 'type': datatype, 'path': path}
            for rn, ri in [('full', img_full), ('center1128', img_1128), ('center224', img_224)]:
                for k, v in compute_stats(ri).items():
                    entry[f'fourier_{rn}_raw_{k}'] = v
                    entry[f'fourier_{rn}_mp_{k}'] = v
            all_rows.append(entry)

# =============================================
# SAVE CSV
# =============================================
fields = ['mean','std','snr','entropy','p1','p99','median']
regions = ['full','center1128','center224']
fieldnames = ['plate','well','label','type','path']
for reg in regions:
    fieldnames += [f'fourier_{reg}_raw_{k}' for k in fields] + [f'fourier_{reg}_mp_{k}' for k in fields]

outpath = os.path.join(OUT, args.output)
with open(outpath, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(all_rows)
print(f"\nSaved {len(all_rows)} rows to {outpath}")
print(f"  Regions: {', '.join(regions)} (all Fourier phase-preserving normalized)")
