#!/usr/bin/env python3
"""Pre-crop all images to 224×224 center crops, save as float32 memmap.

Usage:
    python3 precache_crops.py

Output:
    crops_224x224_f32.npy  — memmap file, ~4.85 GB, (24192, 224, 224) float32
    crops_paths.txt        — ordered list of image paths for alignment
"""
import os, sys, glob, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROP_SIZE = 224
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'crops_224x224_f32.npy')
PATHS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'crops_paths.txt')

all_paths = sorted(set(
    p for pi in range(1, 7) for cond in ['Drugs_Data', 'Mutants_Data']
    for ext in ['*.tif', '*.tiff']
    for p in glob.glob(os.path.join(ROOT, cond, f'P{pi}', '**', ext), recursive=True)
))
N = len(all_paths)
print(f"Total images: {N}")

if os.path.exists(OUT_FILE):
    ans = input(f"  {os.path.basename(OUT_FILE)} exists. Overwrite? [y/N] ")
    if ans.lower() != 'y':
        print("  Aborted.")
        sys.exit(0)

data = np.memmap(OUT_FILE, dtype='float32', mode='w+', shape=(N, CROP_SIZE, CROP_SIZE))
t0 = time.time()
batch_save = time.time()

for i, path in enumerate(all_paths):
    try:
        import tifffile
        img = tifffile.imread(path)
    except Exception:
        from PIL import Image
        img = np.array(Image.open(path))

    if img.ndim == 3:
        img = img[:, :, 0]

    h, w = img.shape
    cy, cx = (h - CROP_SIZE) // 2, (w - CROP_SIZE) // 2
    crop = img[cy:cy + CROP_SIZE, cx:cx + CROP_SIZE]

    if img.dtype == np.uint16:
        crop = crop.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        crop = crop.astype(np.float32) / 255.0
    else:
        crop = crop.astype(np.float32)

    data[i] = crop

    if (i + 1) % 500 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (N - i - 1) / rate
        print(f"  [{i+1}/{N}] {rate:.0f} imgs/sec, ETA {eta:.0f}s", flush=True)

data.flush()
total = time.time() - t0
print(f"\nDone. {N} images in {total:.0f}s ({N/total:.0f} imgs/sec)")
print(f"  {OUT_FILE} — {N * CROP_SIZE * CROP_SIZE * 4 / 1024**3:.2f} GB")

with open(PATHS_FILE, 'w') as f:
    for p in all_paths:
        f.write(p + '\n')
print(f"  {PATHS_FILE} — {N} paths")
