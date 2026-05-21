#!/usr/bin/env python3
"""Patch labels in both/fold_Plate_1/embeddings npz to fix drug labels.

Drug-path samples currently have mutant labels (because MUTANT_DATA
returns first match for all wells).  This script re-labels drug
samples using the correct drug name from plate_well_ic50_mapping.json.
"""

import os, json, re, numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(SCRIPT_DIR, 'both/fold_Plate_1/embeddings_Plate_1_mil_n3.npz')

with open(os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')) as f:
    IC50 = json.load(f)

data = np.load(NPZ_PATH, allow_pickle=True)
emb = data['embeddings']
# Convert to object array to avoid fixed-width truncation
labels_raw = [str(l) for l in data['labels']]
labels = np.array(labels_raw, dtype=object)
paths = data['paths']
classes = dict(data['classes'].item())  # 0..184 → label string

drug_mask = np.array(['Drugs_Data' in str(p) for p in paths])
mut_mask = np.array(['Mutants_Data' in str(p) for p in paths])

print(f"Total: {len(labels)}  Drug: {drug_mask.sum()}  Mutant: {mut_mask.sum()}")
print(f"Current unique labels: {len(set(labels))}")

# Helper to extract plate + well from a full path
_well_re = re.compile(r'Well(\w{3})')

def extract_plate_well(path):
    parts = str(path).split('/')
    plate = None
    for p in parts:
        if re.match(r'^P[1-6]$', p):
            plate = p
            break
    m = _well_re.search(str(path))
    well = m.group(1) if m else None
    return plate, well

def drug_label(plate, well):
    """Return correct drug-class label from IC50 mapping."""
    info = IC50.get(plate, {}).get(well)
    if info is None:
        return None
    ab = info.get('antibiotic', '')
    dose = info.get('ic50_multiple', '')
    if not ab or not dose:
        return None
    if dose == 'control':
        return 'control'
    dose_str = dose if 'x' in str(dose) else f"{dose}x"
    return f"{ab.replace(' ', '_')}_{dose_str}"

# Patch labels for drug-path samples
changed = 0
for i in np.where(drug_mask)[0]:
    plate, well = extract_plate_well(paths[i])
    if plate and well:
        correct = drug_label(plate, well)
        if correct and labels[i] != correct:
            labels[i] = correct
            changed += 1

print(f"Changed: {changed} drug labels")
print(f"New unique labels: {len(set(labels))}")

# Build new classes dict from all unique labels
all_new = sorted(set(str(l) for l in labels))
new_classes = {i: name for i, name in enumerate(all_new)}
# Map old class indices → new

print(f"New classes: {len(new_classes)}")

# Save patched npz
out_path = NPZ_PATH.replace('.npz', '_patched.npz')
np.savez_compressed(out_path,
                    embeddings=emb,
                    labels=np.array(labels, dtype=object),
                    paths=paths,
                    classes=new_classes)
print(f"Saved: {out_path}")
