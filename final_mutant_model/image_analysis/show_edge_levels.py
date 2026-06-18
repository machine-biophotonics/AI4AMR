#!/usr/bin/env python3
"""Visualize Canny at different sigmas + Sobel on sample images,
   similar to Felix's notebook but for our data.
"""
import numpy as np; np.random.seed(42)
import os, json, re, random
from PIL import Image
from skimage import feature, filters
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
BASE = os.path.dirname(PROJECT_DIR)
MUTANT_BASE = os.path.join(BASE, 'Mutants_Data')
DRUG_BASE = os.path.join(BASE, 'Drugs_Data')
OUT = os.path.join(SCRIPT_DIR, 'output_all_plates'); os.makedirs(OUT, exist_ok=True)

def load_gray(path):
    img = np.array(Image.open(path))
    if len(img.shape) == 3: img = img[:,:,0]
    return img.astype(np.float32)

def model_preprocess(img):
    img_01 = img.astype(np.float32) / 65535.0
    img_255 = (img_01 * 255).astype(np.uint8)
    return (img_255.astype(np.float32) / 255.0 - 0.5) / 0.5

def center_crop(img, size):
    h, w = img.shape
    half = size // 2
    cy, cx = h // 2, w // 2
    return img[cy-half:cy+half, cx-half:cx+half]

def extract_well(fname):
    m = re.search(r'Well([A-Z]\d+)_', fname)
    return m.group(1) if m else None

for label, base_dir in [('drug', DRUG_BASE), ('mutant', MUTANT_BASE)]:
    plate_dir = os.path.join(base_dir, 'P1')
    files = []
    for root, _, fnames in os.walk(plate_dir):
        for f in fnames:
            if f.endswith(('.tif','.tiff')):
                files.append(os.path.join(root, f))
    path = random.choice(files)
    img = load_gray(path)
    crop = center_crop(img, 448)
    crop_mp = model_preprocess(crop)

    titles = ['Original', r'Canny $\sigma$=1', r'Canny $\sigma$=2',
              r'Canny $\sigma$=3', r'Canny $\sigma$=5', 'Sobel mag']

    figs = []
    figs.append(crop_mp)
    for s in [1, 2, 3, 5]:
        figs.append(feature.canny(crop_mp.astype(np.float32), sigma=s).astype(float))
    figs.append(filters.sobel(crop_mp.astype(np.float32)))

    n = len(figs)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'P1 {label}  —  center 448×448 crop  (model-preprocessed)', fontsize=14, fontweight='bold')

    cmaps = ['gray', 'gray', 'gray', 'gray', 'gray', 'hot']
    for i, (ax, f, t, cm) in enumerate(zip(axes.flat, figs, titles, cmaps)):
        ax.imshow(f, cmap=cm)
        if 'Canny' in t:
            ax.set_xlabel(f'density = {f.mean():.3f}', fontsize=9)
        elif 'Sobel' in t:
            ax.set_xlabel(f'mean = {f.mean():.0f}', fontsize=9)
        ax.set_title(t, fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    outpath = os.path.join(OUT, f'edge_levels_P1_{label}.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outpath}")
