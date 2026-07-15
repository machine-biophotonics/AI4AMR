#!/usr/bin/env python3
"""
Consolidate all DINOv3 CLS token embeddings (center crop) into a single file
with full metadata: label, plate (P1-P6), data source (control/mutant/drug).

Output:
    features_all.npz     — numpy arrays (embeddings, label_indices)
    features_metadata.csv — per-sample metadata (source, plate, well, image_name, label)
    features_label_map.json — index → label string mapping

Usage:
    python3 save_features_with_metadata.py
"""

import os
import re
import json
import csv
import glob
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")

# Mapping files
CONTROL_MAP = os.path.join(os.path.dirname(BASE_DIR), "final_mutant_model", "plate_well_control_id_path.json")
MUTANT_MAP = os.path.join(BASE_DIR, "plate_well_id_path.json")
DRUG_MAP = os.path.join(BASE_DIR, "plate_well_ic50_mapping.json")

OUTPUT_NPZ = os.path.join(BASE_DIR, "features_all.npz")
OUTPUT_CSV = os.path.join(BASE_DIR, "features_metadata.csv")
OUTPUT_LABEL_MAP = os.path.join(BASE_DIR, "features_label_map.json")


def load_well_map(path: str, map_type: str):
    """Load the appropriate well-to-label mapping."""
    with open(path) as f:
        data = json.load(f)

    well_to_label = {}
    for plate, plate_data in data.items():
        well_to_label[plate] = {}
        if map_type in ("control", "mutant"):
            # { "A": { "1": {"id": "..."} } }
            for row, cols in plate_data.items():
                for col, info in cols.items():
                    well = f"{row}{int(col):02d}"
                    well_to_label[plate][well] = info.get("id", "WT")
        elif map_type == "drug":
            # { "A01": {"antibiotic": "...", "ic50_multiple": "..."} }
            for well, info in plate_data.items():
                antibiotic = info.get("antibiotic", "unknown")
                ic50 = info.get("ic50_multiple", "unknown")
                label = f"{antibiotic}_{ic50}" if ic50 != "control" else "drug_control"
                well_to_label[plate][well] = label
    return well_to_label


def extract_well_from_filename(filename: str):
    match = re.search(r'Well([A-H]\d+)', filename)
    return match.group(1) if match else None


def extract_source_plate(dirname: str):
    """Parse 'Controls_P1' -> ('control', 'P1')"""
    match = re.match(r'(Controls|Mutants|Drugs)_(P\d)', dirname)
    if match:
        src = match.group(1).lower().rstrip('s')  # controls→control, mutants→mutant, drugs→drug
        return src, match.group(2)
    return None, None


def main():
    print("=" * 60)
    print("Consolidating DINOv3 embeddings with metadata")
    print("=" * 60)

    print("\nLoading label maps...")
    control_map = load_well_map(CONTROL_MAP, "control")
    mutant_map = load_well_map(MUTANT_MAP, "mutant")
    drug_map = load_well_map(DRUG_MAP, "drug")

    source_map = {
        "control": control_map,
        "mutant": mutant_map,
        "drug": drug_map,
    }

    all_embeddings = []
    all_metadata = []

    source_dirs = sorted([
        d for d in os.listdir(EMBEDDINGS_DIR)
        if os.path.isdir(os.path.join(EMBEDDINGS_DIR, d))
        and d != "metadata.json"
    ])

    print(f"Found {len(source_dirs)} source directories: {source_dirs}")

    for dirname in source_dirs:
        source, plate = extract_source_plate(dirname)
        if source is None:
            continue

        dir_path = os.path.join(EMBEDDINGS_DIR, dirname)
        well_dirs = sorted([
            d for d in os.listdir(dir_path)
            if os.path.isdir(os.path.join(dir_path, d))
        ])

        label_map = source_map.get(source, {}).get(plate, {})

        for well_dir_name in tqdm(well_dirs, desc=f"{dirname}"):
            well_path = os.path.join(dir_path, well_dir_name)
            npy_files = sorted(glob.glob(os.path.join(well_path, "*.npy")))

            # Extract well ID from the subdirectory name or from the npy filename
            well = extract_well_from_filename(well_dir_name)
            if well is None and npy_files:
                well = extract_well_from_filename(os.path.basename(npy_files[0]))

            label = label_map.get(well, "unknown")

            for npy_path in npy_files:
                emb = np.load(npy_path)
                img_name = os.path.splitext(os.path.basename(npy_path))[0]

                all_embeddings.append(emb)
                all_metadata.append({
                    "source": source,
                    "plate": plate,
                    "well": well or "unknown",
                    "image_name": img_name,
                    "label": label,
                })

    if len(all_embeddings) == 0:
        print("No embeddings found!")
        return

    all_embeddings = np.array(all_embeddings, dtype=np.float32)

    # Build label index
    unique_labels = sorted(set(m["label"] for m in all_metadata))
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    label_indices = np.array([label_to_idx[m["label"]] for m in all_metadata], dtype=np.int32)

    print(f"\nTotal samples: {len(all_embeddings)}")
    print(f"Embedding dim: {all_embeddings.shape[1]}")
    print(f"Unique labels: {len(unique_labels)}")
    print(f"Shape: {all_embeddings.shape}")

    # Per-source counts
    from collections import Counter
    source_counts = Counter(m["source"] for m in all_metadata)
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src}: {cnt}")
    plate_counts = Counter(m["plate"] for m in all_metadata)
    for pl, cnt in sorted(plate_counts.items()):
        print(f"  {pl}: {cnt}")

    print(f"\nSaving to {OUTPUT_NPZ} ...")
    np.savez_compressed(OUTPUT_NPZ, embeddings=all_embeddings, label_indices=label_indices)

    print(f"Saving CSV to {OUTPUT_CSV} ...")
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "plate", "well", "image_name", "label"])
        writer.writeheader()
        writer.writerows(all_metadata)

    print(f"Saving label map to {OUTPUT_LABEL_MAP} ...")
    with open(OUTPUT_LABEL_MAP, "w") as f:
        json.dump({"label_to_idx": label_to_idx, "idx_to_label": {str(i): lbl for lbl, i in label_to_idx.items()}}, f, indent=2)

    print(f"\nDone! Files saved:")
    print(f"  {OUTPUT_NPZ}")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_LABEL_MAP}")


if __name__ == "__main__":
    main()
