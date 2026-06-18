#!/usr/bin/env python3
"""Compare basic pixel statistics between drug and mutant microscopy images."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json, glob, re
from PIL import Image
from collections import defaultdict

def load_image(path):
    img = np.array(Image.open(path))
    if len(img.shape) == 3:
        img = img[:, :, 0]
    return img.astype(np.float32)

def extract_well(fname):
    m = re.search(r'Well([A-Z]\d+)_', fname)
    return m.group(1) if m else None

def compute_stats(img):
    return {
        'mean': float(img.mean()),
        'std': float(img.std()),
        'min': float(img.min()),
        'max': float(img.max()),
        'p1': float(np.percentile(img, 1)),
        'p99': float(np.percentile(img, 99)),
        'median': float(np.median(img)),
        'skew': float(((img - img.mean())**3).mean() / (img.std()**3 + 1e-8)),
    }

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

MUTANT_BASE = os.path.join(os.path.dirname(PROJECT_DIR), 'Mutants_Data')
DRUG_BASE = os.path.join(os.path.dirname(PROJECT_DIR), 'Drugs_Data')

with open(os.path.join(PROJECT_DIR, 'plate_well_id_path.json')) as f:
    mutant_map = json.load(f)
with open(os.path.join(PROJECT_DIR, 'plate_well_ic50_mapping.json')) as f:
    drug_map = json.load(f)

results = []

# Mutant images
print("Loading mutant images...")
for plate in ['P1','P2','P3','P4','P5','P6']:
    plate_dir = os.path.join(MUTANT_BASE, plate)
    if not os.path.exists(plate_dir): continue
    for root, _, files in os.walk(plate_dir):
        for fname in files:
            if not fname.endswith(('.tif','.tiff','.png')): continue
            well = extract_well(fname)
            if not well: continue
            label = mutant_map.get(plate, {}).get(well, {}).get('id', 'unknown')
            path = os.path.join(root, fname)
            img = load_image(path)
            stats = compute_stats(img)
            stats['plate'] = plate
            stats['well'] = well
            stats['label'] = label
            stats['type'] = 'mutant'
            results.append(stats)
    print(f"  {plate} done")

# Drug images
print("Loading drug images...")
for plate in ['P1','P2','P3','P4','P5','P6']:
    plate_dir = os.path.join(DRUG_BASE, plate)
    if not os.path.exists(plate_dir): continue
    for root, _, files in os.walk(plate_dir):
        for fname in files:
            if not fname.endswith(('.tif','.tiff','.png')): continue
            well = extract_well(fname)
            if not well: continue
            info = drug_map.get(plate, {}).get(well, {})
            label = f"{info.get('antibiotic','unknown')}_{info.get('ic50_multiple','')}"
            path = os.path.join(root, fname)
            img = load_image(path)
            stats = compute_stats(img)
            stats['plate'] = plate
            stats['well'] = well
            stats['label'] = label
            stats['type'] = 'drug'
            results.append(stats)
    print(f"  {plate} done")

OUT = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUT, exist_ok=True)

# 1. Per-gene/per-drug mean brightness
gene_means = defaultdict(list)
drug_means = defaultdict(list)
for r in results:
    if r['type'] == 'mutant':
        gene = r['label'].rsplit('_',1)[0] if '_' in r['label'] else r['label']
        gene_means[gene].append(r['mean'])
    else:
        drug_means[r['label']].append(r['mean'])

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

genes = sorted(gene_means.keys())
means = [np.mean(gene_means[g]) for g in genes]
stds = [np.std(gene_means[g]) for g in genes]
axes[0].bar(range(len(genes)), means, yerr=stds, capsize=3)
axes[0].set_xticks(range(len(genes)))
axes[0].set_xticklabels(genes, rotation=90, fontsize=6)
axes[0].set_ylabel('Mean pixel intensity')
axes[0].set_title('Mutant lines (per gene)')

drugs = sorted(drug_means.keys())
means_d = [np.mean(drug_means[d]) for d in drugs]
stds_d = [np.std(drug_means[d]) for d in drugs]
axes[1].bar(range(len(drugs)), means_d, yerr=stds_d, capsize=3)
axes[1].set_xticks(range(len(drugs)))
axes[1].set_xticklabels(drugs, rotation=90, fontsize=6)
axes[1].set_ylabel('Mean pixel intensity')
axes[1].set_title('Drug treatments')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'mean_intensity_comparison.png'), dpi=150)
plt.close()

# 2. Brightness distributions: mutant vs drug
mut_means = [r['mean'] for r in results if r['type'] == 'mutant']
drug_means_all = [r['mean'] for r in results if r['type'] == 'drug']

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(mut_means, bins=80, alpha=0.6, label=f'Mutant (n={len(mut_means)})')
ax.hist(drug_means_all, bins=80, alpha=0.6, label=f'Drug (n={len(drug_means_all)})')
ax.set_xlabel('Mean pixel intensity')
ax.set_ylabel('Count')
ax.legend()
ax.set_title('Brightness distribution: Drug vs Mutant')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'brightness_histogram.png'), dpi=150)
plt.close()

# 3. Contrast (std) distribution
mut_stds = [r['std'] for r in results if r['type'] == 'mutant']
drug_stds = [r['std'] for r in results if r['type'] == 'drug']

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(mut_stds, bins=80, alpha=0.6, label=f'Mutant (n={len(mut_stds)})')
ax.hist(drug_stds, bins=80, alpha=0.6, label=f'Drug (n={len(drug_stds)})')
ax.set_xlabel('Pixel intensity std (contrast)')
ax.set_ylabel('Count')
ax.legend()
ax.set_title('Contrast distribution: Drug vs Mutant')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'contrast_histogram.png'), dpi=150)
plt.close()

# 4. Per-plate brightness
plate_mut = defaultdict(list)
plate_drug = defaultdict(list)
for r in results:
    if r['type'] == 'mutant':
        plate_mut[r['plate']].append(r['mean'])
    else:
        plate_drug[r['plate']].append(r['mean'])

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, data, title in [(axes[0], plate_mut, 'Mutant'), (axes[1], plate_drug, 'Drug')]:
    plates = sorted(data.keys())
    means_p = [np.mean(data[p]) for p in plates]
    stds_p = [np.std(data[p]) for p in plates]
    ax.bar(range(len(plates)), means_p, yerr=stds_p, capsize=5)
    ax.set_xticks(range(len(plates)))
    ax.set_xticklabels(plates)
    ax.set_ylabel('Mean brightness')
    ax.set_title(f'{title} — per plate')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'per_plate_brightness.png'), dpi=150)
plt.close()

# 5. Scatter: mean vs std (each point = one image)
fig, ax = plt.subplots(figsize=(8, 6))
mut_pts = [(r['mean'], r['std']) for r in results if r['type'] == 'mutant']
drug_pts = [(r['mean'], r['std']) for r in results if r['type'] == 'drug']
ax.scatter([p[0] for p in mut_pts], [p[1] for p in mut_pts], s=1, alpha=0.3, label='Mutant')
ax.scatter([p[0] for p in drug_pts], [p[1] for p in drug_pts], s=1, alpha=0.3, label='Drug')
ax.set_xlabel('Mean (brightness)')
ax.set_ylabel('Std (contrast)')
ax.legend()
ax.set_title('Brightness vs Contrast: Drug vs Mutant')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'brightness_vs_contrast.png'), dpi=150)
plt.close()

# 6. Summary stats
print(f"\n{'='*60}")
print("SUMMARY STATS")
print(f"{'='*60}")
print(f"{'':20} {'Mutant':>10} {'Drug':>10}")
for metric in ['mean','std','p1','p99','median']:
    mut_vals = [r[metric] for r in results if r['type'] == 'mutant']
    drug_vals = [r[metric] for r in results if r['type'] == 'drug']
    print(f"{metric:20} {np.mean(mut_vals):>10.2f} {np.mean(drug_vals):>10.2f}")
    print(f"{'':20} ±{np.std(mut_vals):>8.2f} ±{np.std(drug_vals):>8.2f}")

print(f"\nMutant images: {len(mut_means)}")
print(f"Drug images: {len(drug_means_all)}")
print(f"\nPlots saved to {OUT}/")
