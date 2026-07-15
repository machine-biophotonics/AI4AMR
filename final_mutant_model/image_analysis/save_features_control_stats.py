#!/usr/bin/env python3
"""Extract 7 raw image statistics on 3 spatial regions from all Controls_Data images.
   Labels from plate_well_control_id_path.json.
   Output: CSV (1 row per image, 7 stats × 3 regions = 21 raw features + 3 regions × model-preprocessed)."""

import numpy as np
import os, json, re, csv, argparse
from scipy.stats import entropy as sp_entropy
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

parser = argparse.ArgumentParser()
parser.add_argument('--plates', default='P1,P2,P3,P4,P5,P6')
parser.add_argument('--output', default='control_7stats.csv')
parser.add_argument('--workers', type=int, default=cpu_count(), help='Parallel workers')
args = parser.parse_args()
PLATES = args.plates.split(',')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
BASE = os.path.dirname(PROJECT_DIR)
CONTROL_BASE = os.path.join(BASE, 'Controls_Data')

with open(os.path.join(PROJECT_DIR, 'plate_well_control_id_path.json')) as f:
    control_map = json.load(f)

# Region parameters (matching save_features_all_plates.py)
IMG_SIZE = 2720
CROP_SIZE = 224
STRIDE = (IMG_SIZE - CROP_SIZE) // 11  # = 226
HALF_N = 2
NEIGHBORHOOD_SPAN = 4 * STRIDE + CROP_SIZE  # = 1128
CENTER = IMG_SIZE // 2  # = 1360
RAW_FIELDS = ['mean', 'std', 'snr', 'entropy', 'p1', 'p99', 'median']
REGIONS = ['full', 'center1128', 'center224']


def extract_well(fname):
    m = re.search(r'Well([A-H]\d+)', fname)
    return m.group(1) if m else None


def lookup_label(plate, well):
    row = well[0]
    col = str(int(well[1:]))
    return control_map.get(plate, {}).get(row, {}).get(col, {}).get('id', 'unknown')


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
    """Match model preprocessing: uint16 → 0-1 → uint8 → [-1, 1]"""
    img_01 = img.astype(np.float32) / 65535.0
    img_255 = (img_01 * 255).astype(np.uint8)
    return (img_255.astype(np.float32) / 255.0 - 0.5) / 0.5


def process_one_image(args):
    path, plate = args
    fname = os.path.basename(path)
    well = extract_well(fname)
    if well is None:
        return None
    label = lookup_label(plate, well)

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


def get_all_image_paths():
    all_paths = []
    for plate in PLATES:
        plate_dir = os.path.join(CONTROL_BASE, plate)
        if not os.path.exists(plate_dir):
            continue
        for root, _, files in os.walk(plate_dir):
            for fname in files:
                if not fname.lower().endswith(('.tif', '.tiff')):
                    continue
                all_paths.append((os.path.join(root, fname), plate))
    return all_paths


def main():
    print(f"Finding all control images across {PLATES}...")
    all_paths = get_all_image_paths()
    print(f"  Found {len(all_paths)} images")
    print(f"  Workers: {args.workers}")
    print(f"  Regions: {REGIONS}")
    print(f"  Stats per region: {RAW_FIELDS}")
    print(f"  Total features per image: {len(REGIONS) * len(RAW_FIELDS) * 2} (raw + model-preprocessed)")

    rows = []
    with Pool(args.workers) as pool:
        for result in tqdm(pool.imap_unordered(process_one_image, all_paths), total=len(all_paths), desc="Processing"):
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
