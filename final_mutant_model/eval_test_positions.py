#!/usr/bin/env python3
# Must be set before any imports to suppress inductor SM warning
import warnings
warnings.filterwarnings("ignore", message=".*Not enough SMs to use max_autotune_gemm.*")

import os
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "ATEN,CPP"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCH_CUDNN_DETERMINISTIC"] = "1"

"""
Per-position test evaluation for MIL models.
For each test image, evaluate the model at every valid grid position (10x10=100)
using a 3x3 neighborhood crop bag. Saves per-position true/predicted labels to CSV.

Supports metabolomics_mutant modes:
- Standard: test plate = P1-P4, all 3 timepoints
- --timepoint_split: test = one timepoint (T1/T2/T3), all 4 plates
- --include_timepoint_in_labels: 288 classes (T{g}_gene)
"""

import argparse
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import glob
import json
import re
import csv
from collections import Counter

from mil_model import MILEncoder, MultiCropDataset, extract_well_from_filename

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

parser = argparse.ArgumentParser()
parser.add_argument('--data_mode', type=str, default='metabolomics_mutant', choices=['metabolomics_mutant'])
parser.add_argument('--data_root', type=str, default=None, help='Path to Metabolomics Mutants folder')
parser.add_argument('--fold_dir', type=str, required=True, help='Folder containing best_model_acc.pth')
parser.add_argument('--test_plate', type=str, default='P1', help='P1-P4 (standard/tp_labels) or T1/T2/T3 (timepoint_split)')
parser.add_argument('--timepoint_split', action='store_true', default=False)
parser.add_argument('--include_timepoint_in_labels', action='store_true', default=False)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--backbone', type=str, default='efficientnet_b0')
parser.add_argument('--pretrained', type=str, default='micronet')
parser.add_argument('--num_channels', type=int, default=1)
parser.add_argument('--pooling', type=str, default='attention')
parser.add_argument('--neighborhood', type=int, default=3)
parser.add_argument('--grid_size', type=int, default=12)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--chunk_positions', type=int, default=50,
                    help='Max positions forwarded at once (memory vs speed)')
parser.add_argument('--checkpoint', type=str, default='best_model_acc.pth')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if args.data_root:
    BASE_DIR = args.data_root
else:
    BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Metabolomics_Data', 'Mutants')

METABOLOMICS_MAPPING_PATH = os.path.join(SCRIPT_DIR, 'plate_metabolomics_mutant_mapping.json')
with open(METABOLOMICS_MAPPING_PATH, 'r') as f:
    metabolomics_data = json.load(f)

# Build plate_maps: physical plate (P1-P4) -> well -> gene id
plate_maps = {}
for plate_key, rows in metabolomics_data.items():
    physical_plate = plate_key.split('_')[0]
    if physical_plate not in plate_maps:
        plate_maps[physical_plate] = {}
        for row_letter, cols in rows.items():
            for col_num, info in cols.items():
                well = f"{row_letter}{int(col_num):02d}"
                plate_maps[physical_plate][well] = info['id']


def get_image_paths_for_plate(plate: str) -> list:
    timepoints = [f"{plate}_T{t}" for t in [1, 2, 3]]
    valid_paths = []
    for tp in timepoints:
        plate_dir = os.path.join(BASE_DIR, tp)
        if not os.path.exists(plate_dir):
            continue
        paths = []
        for pattern in ['*.tif', '*.tiff', '*.png']:
            paths.extend(glob.glob(os.path.join(plate_dir, '**', pattern), recursive=True))
        for path in paths:
            well = extract_well_from_filename(os.path.basename(path))
            if well and well in plate_maps.get(plate, {}):
                valid_paths.append(path)
    return valid_paths


def get_timepoint_from_path(path: str) -> str:
    match = re.search(r'_T(\d)', path, re.IGNORECASE)
    return f"T{match.group(1)}" if match else 'T1'


def get_tp_num(path: str) -> int:
    match = re.search(r'_T(\d)', path, re.IGNORECASE)
    return int(match.group(1)) if match else None


# Build classes
base_classes = sorted(set(label for pm in plate_maps.values() for label in pm.values() if label))
if args.include_timepoint_in_labels:
    all_classes = sorted([f"T{t}_{gene}" for t in [1, 2, 3] for gene in base_classes])
else:
    all_classes = base_classes
class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
num_classes = len(all_classes)
print(f"Number of classes: {num_classes}")

plate_keys_for_match = ['P1', 'P2', 'P3', 'P4']


def metabolomics_label_extractor(path, pm=plate_maps, all_plates_local=plate_keys_for_match):
    path_lower = path.lower()
    for p in all_plates_local:
        p_lower = p.lower()
        if f'/{p_lower}_t' in path_lower or f'\\{p_lower}_t\\' in path_lower:
            plate_key = p
            break
    else:
        return None
    well = extract_well_from_filename(os.path.basename(path))
    if well and plate_key in pm and well in pm[plate_key]:
        gene = pm[plate_key][well]
        if args.include_timepoint_in_labels:
            tp = get_timepoint_from_path(path)
            return f"{tp}_{gene}"
        return gene
    return None


# Build test paths
test_paths, test_labels = [], []
if args.timepoint_split:
    test_tp = int(args.test_plate[-1])
    print(f"Timepoint split eval: test timepoint T{test_tp} (all 4 plates)")
    for plate in ['P1', 'P2', 'P3', 'P4']:
        for path in get_image_paths_for_plate(plate):
            label = metabolomics_label_extractor(path)
            if label in class_to_idx and get_tp_num(path) == test_tp:
                test_paths.append(path)
                test_labels.append(class_to_idx[label])
else:
    print(f"Test plate: {args.test_plate}")
    for path in get_image_paths_for_plate(args.test_plate):
        label = metabolomics_label_extractor(path)
        if label in class_to_idx:
            test_paths.append(path)
            test_labels.append(class_to_idx[label])

test_labels = np.array(test_labels)
print(f"Test: {len(test_paths)} images")

# Load model
print(f"Loading model from {os.path.join(args.fold_dir, args.checkpoint)}...")
model = MILEncoder(num_classes=num_classes, num_heads=args.num_heads, dropout=args.dropout,
                   use_contrastive=True, num_channels=args.num_channels, pretrained=args.pretrained,
                   backbone=args.backbone, pooling=args.pooling)
checkpoint = torch.load(os.path.join(args.fold_dir, args.checkpoint), map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Build dataset to reuse positions/stride/transform/_load_image (augment=False = center transform, no jitter)
dataset = MultiCropDataset(test_paths, test_labels.tolist(), None,
                           neighborhood=args.neighborhood, grid_size=args.grid_size,
                           augment=False, seed=42, num_channels=args.num_channels)
positions = dataset.positions
stride = dataset.stride
transform = dataset.transform
half_n = args.neighborhood // 2
crop_size = args.crop_size
print(f"Valid positions per image: {len(positions)}")

# Output files
fold_tag = args.test_plate
csv_path = os.path.join(args.fold_dir, f"test_positions_fold_{fold_tag}.csv")
summary_csv_path = os.path.join(args.fold_dir, f"test_positions_summary_fold_{fold_tag}.csv")
summary_json_path = os.path.join(args.fold_dir, f"test_positions_summary_fold_{fold_tag}.json")

per_pos_correct = [0] * len(positions)
per_pos_total = [0] * len(positions)
image_correct = []
image_majority_correct = []

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'timepoint', 'position_idx', 'pos_x', 'pos_y',
                     'true_label', 'true_idx', 'predicted_label', 'predicted_idx',
                     'prob_true', 'correct'])

    start_time = time.time()
    for img_idx, (path, true_idx) in enumerate(zip(test_paths, test_labels)):
        img_pil = dataset._load_image(path)
        tp = get_timepoint_from_path(path)

        # Build [n_positions, 9, C, H, W] bag for all positions
        position_preds = np.zeros(len(positions), dtype=np.int64)
        position_probs = np.zeros(len(positions), dtype=np.float32)
        position_bags = []

        for pos_idx, (left, top) in enumerate(positions):
            bag = []
            for di in range(-half_n, half_n + 1):
                for dj in range(-half_n, half_n + 1):
                    l = left + dj * stride
                    t = top + di * stride
                    crop = img_pil.crop((l, t, l + crop_size, t + crop_size))
                    crop = np.array(crop)
                    crop = transform(image=crop)['image']
                    bag.append(crop)
            position_bags.append(torch.stack(bag))  # [9, C, H, W]

        # Forward in chunks to limit peak memory
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            for chunk_start in range(0, len(positions), args.chunk_positions):
                chunk = position_bags[chunk_start:chunk_start + args.chunk_positions]
                tensor = torch.stack(chunk).to(device)  # [B, 9, C, H, W]
                outputs = model(tensor)  # [B, C]
                probs = torch.softmax(outputs.float(), dim=1)
                preds = probs.argmax(dim=1)
                chunk_len = len(chunk)
                position_preds[chunk_start:chunk_start + chunk_len] = preds.cpu().numpy()
                position_probs[chunk_start:chunk_start + chunk_len] = probs[:, true_idx].cpu().numpy()

        n_correct = 0
        for pos_idx in range(len(positions)):
            pred_idx = int(position_preds[pos_idx])
            correct = (pred_idx == int(true_idx))
            n_correct += int(correct)
            per_pos_correct[pos_idx] += int(correct)
            per_pos_total[pos_idx] += 1
            writer.writerow([path, tp, pos_idx, positions[pos_idx][0], positions[pos_idx][1],
                             all_classes[int(true_idx)], int(true_idx),
                             all_classes[pred_idx], pred_idx,
                             round(float(position_probs[pos_idx]), 6), int(correct)])

        image_correct.append(n_correct / len(positions))
        # Majority vote across positions
        votes = Counter(position_preds.tolist())
        majority = votes.most_common(1)[0][0]
        image_majority_correct.append(int(majority == int(true_idx)))

        if (img_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (img_idx + 1) / elapsed
            eta = (len(test_paths) - img_idx - 1) / rate if rate > 0 else 0
            print(f"  {img_idx+1}/{len(test_paths)} images, {rate:.1f} img/s, ETA {eta/60:.1f} min")

print(f"\nPer-position results saved to {csv_path}")

# Summary
overall_acc = sum(per_pos_correct) / sum(per_pos_total) * 100.0
image_mean_acc = float(np.mean(image_correct)) * 100.0
image_majority_acc = float(np.mean(image_majority_correct)) * 100.0

print(f"Overall position-level accuracy: {overall_acc:.2f}%")
print(f"Per-image mean accuracy: {image_mean_acc:.2f}%")
print(f"Per-image majority-vote accuracy: {image_majority_acc:.2f}%")

with open(summary_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['position_idx', 'pos_x', 'pos_y', 'correct', 'total', 'accuracy'])
    for pos_idx, (left, top) in enumerate(positions):
        acc = 100.0 * per_pos_correct[pos_idx] / max(per_pos_total[pos_idx], 1)
        writer.writerow([pos_idx, left, top, per_pos_correct[pos_idx], per_pos_total[pos_idx], round(acc, 2)])

summary = {
    'config': {'test_plate': args.test_plate, 'timepoint_split': args.timepoint_split,
               'include_timepoint_in_labels': args.include_timepoint_in_labels,
               'num_classes': num_classes, 'neighborhood': args.neighborhood,
               'grid_size': args.grid_size, 'num_test_images': len(test_paths),
               'num_positions': len(positions), 'checkpoint': args.checkpoint},
    'results': {'position_level_acc': round(overall_acc, 3),
                'image_mean_acc': round(image_mean_acc, 3),
                'image_majority_vote_acc': round(image_majority_acc, 3)},
    'per_position_accuracy': [round(100.0 * per_pos_correct[i] / max(per_pos_total[i], 1), 2)
                              for i in range(len(positions))]
}
with open(summary_json_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary saved to {summary_json_path}")
