#!/usr/bin/env python3
"""
Extract bag embeddings from all 6 plates using the trained multi-head MIL model.
Uses get_projected_features() → 1280-dim pooled embedding (before classifier heads).
"""

import os, sys, json, glob, re, argparse, csv
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import mil_model
from mil_model import MILEncoder, MultiCropDataset, extract_well_from_filename

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'Mutants_Data')
CHECKPOINT = os.path.join(SCRIPT_DIR, 'multi_head_mutant', 'fold_P6', 'best_model_acc.pth')
MUTANT_MAPPING = os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'multi_head_mutant')

all_plates = ['P1', 'P2', 'P3', 'P4']

# ── Load mutant mapping ──
with open(MUTANT_MAPPING, 'r') as f:
    mutant_data = json.load(f)

# Build plate_maps: {plate: {well: gene_name}}
plate_maps = {}
for plate in all_plates:
    plate_maps[plate] = {}
    if plate in mutant_data:
        for row, cols in mutant_data[plate].items():
            for col, info in cols.items():
                if 'id' in info:
                    well = f"{row}{int(col):02d}"
                    plate_maps[plate][well] = info['id']

# Collect all unique gene classes
all_classes = sorted(set(
    label for pm in plate_maps.values() for label in pm.values() if label
))
class_to_idx = {c: i for i, c in enumerate(all_classes)}
num_classes = len(all_classes)
print(f"Classes: {num_classes}")

# ── Build model ──
model = MILEncoder(
    num_classes=num_classes,
    num_heads=4,
    dropout=0.0,
    use_contrastive=True,
    num_channels=1,
    pretrained='micronet',
    backbone='efficientnet_b0',
    pooling='attention',
    multi_head=True,
    n_multi_heads=4,
)
checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model = model.to(device)
model.eval()
print(f"Model loaded from {CHECKPOINT}")


def get_image_paths_for_plate(plate_key):
    plate_dir = os.path.join(BASE_DIR, plate_key)
    paths = []
    for pattern in ['*.tif', '*.tiff', '*.png']:
        paths.extend(glob.glob(os.path.join(plate_dir, '**', pattern), recursive=True))
    return paths


def get_gene_label(plate_key, well):
    if well and plate_key in plate_maps and well in plate_maps[plate_key]:
        return plate_maps[plate_key][well]
    return None


# ── Extract embeddings per plate ──
for plate in all_plates:
    print(f"\n{'='*60}")
    print(f"Processing plate {plate}...")
    
    image_paths = get_image_paths_for_plate(plate)
    if not image_paths:
        print(f"WARNING: No images found for {plate} at {os.path.join(BASE_DIR, plate)}")
        continue
    
    # Build labels
    labels = []
    filtered_paths = []
    for path in image_paths:
        well = extract_well_from_filename(os.path.basename(path))
        gene = get_gene_label(plate, well)
        if gene and gene in class_to_idx:
            labels.append(class_to_idx[gene])
            filtered_paths.append(path)
    
    print(f"  {len(filtered_paths)} images with valid labels")
    
    # Create dataset (deterministic: augment=False, epoch=0 → center position)
    dataset = MultiCropDataset(
        filtered_paths, labels, None,
        neighborhood=3, grid_size=12, augment=False,
        seed=SEED, num_channels=1,
        extraction_mode='neighborhood',
    )
    dataset.set_epoch(0)
    
    loader = DataLoader(
        dataset, batch_size=32, shuffle=False,
        num_workers=8, pin_memory=True,
    )
    
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'  Extracting {plate}', leave=False):
            images = batch[0].to(device)
            embeddings = model.get_projected_features(images)
            all_embeddings.append(embeddings.cpu().numpy())
    
    embeddings_np = np.concatenate(all_embeddings, axis=0)
    
    # Save
    np.save(os.path.join(OUTPUT_DIR, f'embeddings_{plate}.npy'), embeddings_np)
    
    # Also save metadata
    meta_path = os.path.join(OUTPUT_DIR, f'metadata_{plate}.csv')
    with open(meta_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['plate', 'well', 'gene', 'class_idx', 'image_path'])
        for p in filtered_paths:
            well = extract_well_from_filename(os.path.basename(p))
            gene = get_gene_label(plate, well)
            w.writerow([plate, well, gene, class_to_idx.get(gene, -1), p])
    
    print(f"  Saved {len(embeddings_np)} embeddings → embeddings_{plate}.npy ({embeddings_np.shape})")

print(f"\nDone! Embeddings saved to {OUTPUT_DIR}")
