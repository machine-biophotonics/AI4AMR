#!/usr/bin/env python3
"""Per-crop histogram matching to standard normal (Gaussian) distribution.
   Preserves spatial structure perfectly while making every crop have
   the same Gaussian histogram → eliminates all pixel-stat confounds.

   Column names identical to original CSV for drop-in plotting.
"""
import argparse, numpy as np, os, json, re, csv, random, warnings
from PIL import Image; from scipy.stats import entropy as sp_entropy, norm
from tqdm import tqdm
warnings.filterwarnings('ignore'); random.seed(42); np.random.seed(42)
parser = argparse.ArgumentParser()
parser.add_argument('--output', default='all_plates_features_gaussian_hm.csv')
args = parser.parse_args()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR); BASE = os.path.dirname(PROJECT_DIR)
MUTANT_BASE = os.path.join(BASE, 'Mutants_Data'); DRUG_BASE = os.path.join(BASE, 'Drugs_Data')
OUT = os.path.join(SCRIPT_DIR, 'output_all_plates'); os.makedirs(OUT, exist_ok=True)
with open(os.path.join(PROJECT_DIR, 'plate_well_id_path.json')) as f: mutant_map = json.load(f)
with open(os.path.join(PROJECT_DIR, 'plate_well_ic50_mapping.json')) as f: drug_map = json.load(f)
STRIDE = (2720 - 224) // 11; NEIGHBORHOOD_SPAN = 4 * STRIDE + 224; CROP_SIZE = 224

def load_gray(path):
    img = np.array(Image.open(path))
    return img[:,:,0].astype(np.float32) if len(img.shape)==3 else img.astype(np.float32)

def compute_stats(img):
    h, _ = np.histogram(img, bins=256, range=(-4, 4))
    p = h/h.sum(); p = p[p>0]
    return {'mean':float(img.mean()),'std':float(img.std()),'snr':float(img.mean()/(img.std()+1e-8)),
            'entropy':float(sp_entropy(p)),'p1':float(np.percentile(img,1)),
            'p99':float(np.percentile(img,99)),'median':float(np.median(img))}

def center_crop(img, size):
    half=size//2; h,w=img.shape; return img[h//2-half:h//2+half, w//2-half:w//2+half]

def histogram_match_gaussian(img, n_bins=1024):
    """Histogram match img to N(0,1) distribution.
    Returns image with Gaussian histogram but same spatial structure.
    Uses 1024 bins for accurate mapping.
    """
    # Get image histogram and CDF
    img_flat = img.flatten()
    hist, bin_edges = np.histogram(img_flat, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    cdf_img = np.cumsum(hist * np.diff(bin_edges))
    cdf_img = cdf_img / cdf_img[-1]  # normalize to [0, 1]
    
    # Target: standard normal CDF evaluated at sample points
    # We'll use the same number of points as the image has unique values
    n_pixels = len(img_flat)
    # Target quantiles from N(0,1), range [-4, 4] covers >99.99%
    target_values = np.linspace(-4, 4, n_bins)
    cdf_target = norm.cdf(target_values)
    
    # Map: for each pixel, find its CDF value in original, then find target value with same CDF
    # Use interpolation for continuous mapping
    # Step 1: for each pixel value, find its CDF value
    pixel_cdf = np.interp(img_flat, bin_centers, cdf_img, left=0, right=1)
    # Step 2: for each CDF value, find the target Gaussian value
    matched = np.interp(pixel_cdf, cdf_target, target_values, left=-4, right=4)
    
    return matched.reshape(img.shape).astype(np.float32)

def process_plate(plate, base_dir, mapping, datatype):
    pd_ = os.path.join(base_dir, plate)
    if not os.path.exists(pd_): return []
    wells = {}
    for root,_,files in os.walk(pd_):
        for f in files:
            if not f.endswith(('.tif','.tiff','.png')): continue
            m = re.search(r'Well([A-Z]\d+)_', f)
            if m: wells.setdefault(m.group(1), []).append(os.path.join(root, f))
    results = []
    for well, paths in tqdm(wells.items(), desc=f'{datatype} {plate}'):
        path = random.choice(paths); img = load_gray(path)
        imgs = {'full': histogram_match_gaussian(img),
                'center1128': histogram_match_gaussian(center_crop(img, NEIGHBORHOOD_SPAN)),
                'center224': histogram_match_gaussian(center_crop(img, CROP_SIZE))}
        label = (mapping.get(plate,{}).get(well,{}).get('id','unknown') if datatype=='mutant'
                 else f"{drug_map.get(plate,{}).get(well,{}).get('antibiotic','unknown')}_{drug_map.get(plate,{}).get(well,{}).get('ic50_multiple','')}")
        entry = {'plate':plate,'well':well,'label':label,'type':datatype,'path':path}
        for rn, ri in imgs.items():
            for k,v in compute_stats(ri).items():
                entry[f'{rn}_raw_{k}'] = v; entry[f'{rn}_mp_{k}'] = v
        results.append(entry)
    return results

all_rows = []
for plate in ['P1','P2','P3','P4','P5','P6']:
    print(f"\nSampling {plate}...")
    all_rows.extend(process_plate(plate, MUTANT_BASE, mutant_map, 'mutant'))
    all_rows.extend(process_plate(plate, DRUG_BASE, drug_map, 'drug'))
fields = ['mean','std','snr','entropy','p1','p99','median']
regions = ['full','center1128','center224']
fieldnames = ['plate','well','label','type','path']
for reg in regions:
    fieldnames += [f'{reg}_raw_{k}' for k in fields] + [f'{reg}_mp_{k}' for k in fields]
outpath = os.path.join(OUT, args.output)
with open(outpath, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(all_rows)
print(f"\nSaved {len(all_rows)} rows to {outpath}")
print(f"  Features computed AFTER per-crop histogram matching to N(0,1)")
