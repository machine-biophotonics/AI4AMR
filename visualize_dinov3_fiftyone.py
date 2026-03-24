#!/usr/bin/env python3
"""
Visualize DINOv3 embeddings in FiftyOne
"""

import os
import json
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMB_TRAIN_PATH = os.path.join(BASE_DIR, "dinov3_embeddings_train_c512.npz")
EMB_VAL_PATH = os.path.join(BASE_DIR, "dinov3_embeddings_val_c512.npz")
EMB_TEST_PATH = os.path.join(BASE_DIR, "dinov3_embeddings_test_c512.npz")

with open(os.path.join(BASE_DIR, 'plate_well_id_path.json'), 'r') as f:
    plate_data = json.load(f)

plate_maps = {}
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_maps[plate] = {}
    for row, wells in plate_data[plate].items():
        for col, info in wells.items():
            well = f"{row}{int(col):02d}"
            plate_maps[plate][well] = info['id']

all_labels = sorted(set(label for pm in plate_maps.values() for label in pm.values()))
label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

def load_embeddings():
    train_data = np.load(EMB_TRAIN_PATH)
    val_data = np.load(EMB_VAL_PATH)
    test_data = np.load(EMB_TEST_PATH)
    
    print(f"Train: {train_data['embeddings'].shape}")
    print(f"Val: {val_data['embeddings'].shape}")
    print(f"Test: {test_data['embeddings'].shape}")
    
    return train_data, val_data, test_data

def create_dataset(split='train'):
    import glob
    import re
    
    if split == 'train':
        paths = []
        for p in ['P1', 'P2', 'P3', 'P4']:
            paths.extend(glob.glob(os.path.join(BASE_DIR, p, '*.tif')))
    elif split == 'val':
        paths = glob.glob(os.path.join(BASE_DIR, 'P5', '*.tif'))
    else:
        paths = glob.glob(os.path.join(BASE_DIR, 'P6', '*.tif'))
    
    return paths

def build_fiftyone_dataset():
    """Build FiftyOne dataset from embeddings."""
    
    train_data, val_data, test_data = load_embeddings()
    
    dataset = fo.Dataset("dinov3_embeddings")
    
    splits = [
        ('train', train_data, create_dataset('train')),
        ('val', val_data, create_dataset('val')),
        ('test', test_data, create_dataset('test')),
    ]
    
    for split_name, data, paths in splits:
        embeddings = data['embeddings']
        labels = data['labels']
        
        print(f"Processing {split_name}: {len(paths)} paths, {len(embeddings)} embeddings")
        
        samples = []
        for i, (emb, label_idx) in enumerate(zip(embeddings, labels)):
            if i >= len(paths):
                break
            
            img_path = paths[i]
            if not os.path.exists(img_path):
                continue
            
            sample = fo.Sample(
                filepath=img_path,
                embedding=emb.tolist(),
                ground_truth=fo.Classification(label=idx_to_label[label_idx]),
                split=split_name,
            )
            samples.append(sample)
        
        dataset.add_samples(samples)
        print(f"  Added {len(samples)} samples")
    
    return dataset

if __name__ == '__main__':
    print("Building FiftyOne dataset...")
    dataset = build_fiftyone_dataset()
    
    print("\nLaunching FiftyOne...")
    session = fo.launch_app(dataset)
    
    print(f"\nDataset: {dataset.name}")
    print(f"Total samples: {len(dataset)}")
    print("\nUse session.view() to filter by split/label")