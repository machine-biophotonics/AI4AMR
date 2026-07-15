#!/usr/bin/env python3
"""Extract 7 raw image statistics from ALL images in NC/WT/control wells
   (Mutants_Data + Drugs_Data) to match Controls_Data density.
   Output: CSV appended/extracted for 13 control-mode classes."""
import numpy as np
import os, json, re, csv, argparse
from scipy.stats import entropy as sp_entropy
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--plates', default='P1,P2,P3,P4,P5,P6')
parser.add_argument('--output', default='control_extra_7stats.csv')
parser.add_argument('--workers', type=int, default=cpu_count())
args = parser.parse_args()
PLATES = args.plates.split(',')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
BASE = os.path.dirname(PROJECT_DIR)
MUTANT_BASE = os.path.join(BASE, 'Mutants_Data')
DRUG_BASE = os.path.join(BASE, 'Drugs_Data')

with open(os.path.join(PROJECT_DIR, 'plate_well_id_path.json')) as f:
    mutant_map = json.load(f)
with open(os.path.join(PROJECT_DIR, 'plate_well_ic50_mapping.json')) as f:
    drug_map = json.load(f)

# Region parameters
IMG_SIZE = 2720
CROP_SIZE = 224
STRIDE = (IMG_SIZE - CROP_SIZE) // 11
HALF_N = 2
NEIGHBORHOOD_SPAN = 4 * STRIDE + CROP_SIZE
CENTER = IMG_SIZE // 2
RAW_FIELDS = ['mean', 'std', 'snr', 'entropy', 'p1', 'p99', 'median']
REGIONS = ['full', 'center1128', 'center224']

# Target labels (the 13 control-mode classes)
NC_LABELS = ['NC_1','NC_2','NC_3','NC_4','NC_5','NC_6',
             'WT NC_1','WT NC_2','WT NC_3','WT NC_4','WT NC_5','WT NC_6']

def extract_well(fname):
    m = re.search(r'Well([A-H]\d+)', fname)
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

def process_one_image(args):
    path, plate, label = args
    fname = os.path.basename(path)
    well = extract_well(fname)
    try:
        import tifffile
        img = tifffile.imread(path)
    except Exception:
        from PIL import Image
        img = np.array(Image.open(path))
    if len(img.shape) == 3:
        img = img[:, :, 0]
    img = img.astype(np.float32)
    img_c1128 = center_crop(img, NEIGHBORHOOD_SPAN)
    img_c224  = center_crop(img, CROP_SIZE)
    row = {'plate': plate, 'well': well, 'label': label, 'image': fname, 'path': path}
    for region_name, region_img in [('full', img), ('center1128', img_c1128), ('center224', img_c224)]:
        for k, v in compute_stats(region_img).items():
            row[f'{region_name}_raw_{k}'] = v
        for k, v in compute_stats(model_preprocess(region_img), range_min=-1, range_max=1).items():
            row[f'{region_name}_mp_{k}'] = v
    return row

def get_target_wells():
    """Return list of (plate, well, label, source_path) for all target wells."""
    targets = []
    # NC_* and WT NC_* from Mutants_Data
    for plate in PLATES:
        for row_id in mutant_map.get(plate, {}):
            for col, info in mutant_map[plate][row_id].items():
                if info['id'] in NC_LABELS:
                    well_str = f"{row_id}{int(col):02d}"
                    plate_dir = os.path.join(MUTANT_BASE, plate)
                    if os.path.isdir(plate_dir):
                        for root, _, files in os.walk(plate_dir):
                            for fname in files:
                                if fname.endswith(('.tif','.tiff')) and f'Well{well_str}' in fname:
                                    targets.append((plate, well_str, info['id'],
                                                   os.path.join(root, fname)))
    # drug_control from Drugs_Data
    for plate in PLATES:
        for well_str, info in drug_map.get(plate, {}).items():
            if 'control' in info.get('antibiotic','').lower() or 'dmso' in info.get('antibiotic','').lower():
                plate_dir = os.path.join(DRUG_BASE, plate)
                if os.path.isdir(plate_dir):
                    for root, _, files in os.walk(plate_dir):
                        for fname in files:
                            if fname.endswith(('.tif','.tiff')) and f'Well{well_str}' in fname:
                                targets.append((plate, well_str, 'drug_control',
                                               os.path.join(root, fname)))
    return targets

def main():
    targets = get_target_wells()
    print(f"Found {len(targets)} images across target wells")
    by_label = defaultdict(list)
    for p, w, l, path in targets:
        by_label[l].append(path)
    for l in sorted(by_label):
        print(f"  {l:20s}: {len(by_label[l])} images")

    # Build args for parallel processing
    task_args = [(path, plate, label) for plate, well, label, path in targets]

    rows = []
    with Pool(args.workers) as pool:
        for result in tqdm(pool.imap_unordered(process_one_image, task_args),
                          total=len(task_args), desc="Extracting stats"):
            if result is not None:
                rows.append(result)

    fieldnames = ['plate', 'well', 'label', 'image', 'path']
    for reg in REGIONS:
        fieldnames += [f'{reg}_raw_{k}' for k in RAW_FIELDS]
        fieldnames += [f'{reg}_mp_{k}' for k in RAW_FIELDS]

    outpath = os.path.join(SCRIPT_DIR, args.output)
    with open(outpath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved {len(rows)} samples to {outpath}")
    print(f"  Plates: {sorted(set(r['plate'] for r in rows))}")
    print(f"  Classes: {len(set(r['label'] for r in rows))}")

if __name__ == '__main__':
    main()
