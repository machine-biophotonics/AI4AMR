#!/usr/bin/env python3
"""Applies per-crop RANK NORMALIZATION before computing stats.
   Each pixel replaced by its rank within the crop → uniform distribution.
   Eliminates ALL pixel-stat confounds."""
import argparse, numpy as np, os, json, re, csv, random, warnings
from PIL import Image; from scipy.stats import entropy as sp_entropy; from tqdm import tqdm
warnings.filterwarnings('ignore'); random.seed(42); np.random.seed(42)
parser = argparse.ArgumentParser()
parser.add_argument('--output', default='all_plates_features_ranknorm.csv')
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
    h, _ = np.histogram(img, bins=256, range=(0,1)); p = h/h.sum(); p = p[p>0]
    return {'mean':float(img.mean()),'std':float(img.std()),'snr':float(img.mean()/(img.std()+1e-8)),
            'entropy':float(sp_entropy(p)),'p1':float(np.percentile(img,1)),
            'p99':float(np.percentile(img,99)),'median':float(np.median(img))}
def rank_normalize(img):
    flat = img.flatten(); ranks = np.argsort(np.argsort(flat)).astype(np.float32)
    return (ranks / (len(ranks)-1)).reshape(img.shape)
def center_crop(img, size):
    half=size//2; h,w=img.shape; return img[h//2-half:h//2+half, w//2-half:w//2+half]
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
        imgs = {'full':rank_normalize(img), 'center1128':rank_normalize(center_crop(img,NEIGHBORHOOD_SPAN)),
                'center224':rank_normalize(center_crop(img,CROP_SIZE))}
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
