#!/usr/bin/env python3
"""Sample 1 image/well per plate, compute Canny edge density and Sobel
gradient magnitude statistics on THREE spatial regions, save to CSV.
Uses multiprocessing for speed.

Regions:
  full       = entire 2720x2720 image
  center1128 = center 1128x1128 (bounding box of 5x5 neighborhood)
  center224  = center 224x224 (single model crop)
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--plates', default='P1,P2,P3,P4,P5,P6',
                    help='Comma-separated plate names')
parser.add_argument('--output', default='all_plates_features_edge.csv',
                    help='Output CSV filename')
parser.add_argument('--canny_sigmas', default='1,2,3,5',
                    help='Comma-separated Canny Gaussian sigma values')
parser.add_argument('--canny_low', type=float, default=None,
                    help='Canny low threshold (auto if None)')
parser.add_argument('--canny_high', type=float, default=None,
                    help='Canny high threshold (auto if None)')
parser.add_argument('--sobel', action='store_true', default=True,
                    help='Compute Sobel features (default: True)')
parser.add_argument('--no_sobel', action='store_true', dest='no_sobel',
                    help='Skip Sobel feature computation')
parser.add_argument('--workers', type=int, default=None,
                    help='Number of parallel workers (default: all cores)')
parser.add_argument('--no_vis', action='store_true',
                    help='Skip sample visualization')
args = parser.parse_args()
PLATES = args.plates.split(',')
CANNY_SIGMAS = [float(s) for s in args.canny_sigmas.split(',')]

import numpy as np; np.random.seed(42)
import os, json, re, csv, random, warnings, sys
from PIL import Image
from scipy.stats import entropy as sp_entropy
from skimage import feature, filters
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
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
STRIDE = (2720 - 224) // 11  # = 226
NEIGHBORHOOD_SPAN = 4 * STRIDE + 224  # = 1128
CROP_SIZE = 224

def load_gray(path):
    img = np.array(Image.open(path))
    if len(img.shape) == 3: img = img[:,:,0]
    return img.astype(np.float32)

def extract_well(fname):
    m = re.search(r'Well([A-Z]\d+)_', fname)
    return m.group(1) if m else None

def model_preprocess(img):
    img_01 = img.astype(np.float32) / 65535.0
    img_255 = (img_01 * 255).astype(np.uint8)
    return (img_255.astype(np.float32) / 255.0 - 0.5) / 0.5

def compute_stats(img, range_min=-1, range_max=1):
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

def center_crop(img, size):
    h, w = img.shape
    half = size // 2
    cy, cx = h // 2, w // 2
    return img[cy-half:cy+half, cx-half:cx+half]

def process_one(args):
    """Process a single well. args is a tuple to support multiprocessing."""
    plate, well, path, datatype, label, canny_sigmas, canny_low, canny_high, do_sobel = args

    try:
        img = load_gray(path)
    except Exception as e:
        return {'error': str(e), 'plate': plate, 'well': well, 'path': path}

    img_1128 = center_crop(img, NEIGHBORHOOD_SPAN)
    img_224  = center_crop(img, CROP_SIZE)

    entry = {'plate': plate, 'well': well, 'label': label, 'type': datatype, 'path': path}

    for region_name, region_img in [('full', img), ('center1128', img_1128), ('center224', img_224)]:
        proc = model_preprocess(region_img)

        for sigma in canny_sigmas:
            edges = feature.canny(
                proc.astype(np.float32),
                sigma=sigma,
                low_threshold=canny_low,
                high_threshold=canny_high,
            )
            canny_stats = compute_stats(edges.astype(np.float32), range_min=0, range_max=1)
            for k, v in canny_stats.items():
                entry[f'{region_name}_canny_sigma{sigma}_mp_{k}'] = v

        if do_sobel:
            grad = filters.sobel(proc.astype(np.float32))
            gm = grad.max()
            sobel_stats = compute_stats(grad, range_min=0, range_max=gm if gm > 0 else 1)
            for k, v in sobel_stats.items():
                entry[f'{region_name}_sobel_mp_{k}'] = v

    return entry

# === Collect all tasks ===
tasks = []
for plate in PLATES:
    for base_dir, mapping, datatype in [(MUTANT_BASE, mutant_map, 'mutant'),
                                         (DRUG_BASE, drug_map, 'drug')]:
        plate_dir = os.path.join(base_dir, plate)
        if not os.path.exists(plate_dir):
            continue
        wells = {}
        for root, _, files in os.walk(plate_dir):
            for fname in files:
                if not fname.endswith(('.tif','.tiff','.png')): continue
                well = extract_well(fname)
                if not well: continue
                wells.setdefault(well, []).append(os.path.join(root, fname))
        for well, paths in wells.items():
            path = random.choice(paths)
            label = (mapping.get(plate,{}).get(well,{}).get('id','unknown')
                     if datatype=='mutant'
                     else f"{drug_map.get(plate,{}).get(well,{}).get('antibiotic','unknown')}_{drug_map.get(plate,{}).get(well,{}).get('ic50_multiple','')}")
            tasks.append((plate, well, path, datatype, label,
                          CANNY_SIGMAS, args.canny_low, args.canny_high, not args.no_sobel))

print(f"Total tasks: {len(tasks)}")

# === Process in parallel ===
n_workers = args.workers if args.workers else cpu_count()
print(f"Workers: {n_workers}")

with Pool(n_workers) as pool:
    results = list(tqdm(pool.imap_unordered(process_one, tasks),
                        total=len(tasks), desc='Processing'))

# Check for errors
errors = [r for r in results if 'error' in r]
if errors:
    print(f"\nWARNING: {len(errors)} images failed:")
    for e in errors[:5]:
        print(f"  {e['plate']}/{e['well']}: {e['error']}")
    results = [r for r in results if 'error' not in r]

# === Save to CSV ===
stats_fields = ['mean','std','snr','entropy','p1','p99','median']
regions = ['full', 'center1128', 'center224']
fieldnames = ['plate','well','label','type','path']
for reg in regions:
    for s in CANNY_SIGMAS:
        fieldnames += [f'{reg}_canny_sigma{s}_mp_{k}' for k in stats_fields]
    if not args.no_sobel:
        fieldnames += [f'{reg}_sobel_mp_{k}' for k in stats_fields]

outpath = os.path.join(OUT, args.output)
with open(outpath, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(results)

print(f"\nSaved {len(results)} rows to {outpath}")
print(f"  Plates: {sorted(set(r['plate'] for r in results))}")
print(f"  Types: {sorted(set(r['type'] for r in results))}")
print(f"  Canny sigmas: {CANNY_SIGMAS}")
print(f"  Sobel: {'yes' if not args.no_sobel else 'no'}")

# === Generate sample visualization (single-thread, small images only) ===
if not args.no_vis:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_plates = len(PLATES)
    n_cols = 2 + len(CANNY_SIGMAS)  # original + sobel + each canny sigma
    fig, axes = plt.subplots(n_plates * 2, n_cols,
                             figsize=(n_cols * 3.5, n_plates * 2 * 3.5))

    for pi, plate in enumerate(PLATES):
        plate_rows = [r for r in results if r['plate'] == plate]
        drug_rows = [r for r in plate_rows if r['type'] == 'drug']
        mut_rows = [r for r in plate_rows if r['type'] == 'mutant']

        for ti, (label, candidates) in enumerate([('drug', drug_rows),
                                                   ('mutant', mut_rows)]):
            if not candidates:
                continue
            row = random.choice(candidates)
            img = load_gray(row['path'])
            crop = center_crop(img, CROP_SIZE)
            crop_mp = model_preprocess(crop)

            ri = pi * 2 + ti
            ax_orig = axes[ri, 0]
            ax_orig.imshow(crop_mp, cmap='gray', vmin=-1, vmax=1)
            ax_orig.set_title(f'{plate} {label}\nWell {row["well"]}', fontsize=7)
            ax_orig.axis('off')

            for si, sigma in enumerate(CANNY_SIGMAS):
                ax_edge = axes[ri, 1 + si]
                edges = feature.canny(crop_mp.astype(np.float32),
                                      sigma=sigma,
                                      low_threshold=args.canny_low,
                                      high_threshold=args.canny_high)
                ax_edge.imshow(edges, cmap='gray', vmin=0, vmax=1)
                ax_edge.set_title(f'Canny σ={sigma}\ndensity={edges.mean():.3f}', fontsize=6)
                ax_edge.axis('off')

            if not args.no_sobel:
                ax_grad = axes[ri, 1 + len(CANNY_SIGMAS)]
                grad = filters.sobel(crop_mp.astype(np.float32))
                ax_grad.imshow(grad, cmap='hot')
                ax_grad.set_title(f'Sobel mag\nmean={grad.mean():.0f}', fontsize=6)
                ax_grad.axis('off')

    plt.tight_layout()
    vis_path = os.path.join(OUT, 'samples_edge.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sample visualization to {vis_path}")
