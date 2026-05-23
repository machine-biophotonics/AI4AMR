#!/usr/bin/env python3
"""
Extract VAE latent vectors at all valid grid positions from the test set.
For each well on the test plate, extracts latents z (32-dim), mu, logvar,
bag embedding (1280-dim), attention weights, logits, and predicted/true labels.

Usage:
    python3 extract_test_latents.py --test_plate P1 --data_mode both
    python3 extract_test_latents.py --test_plate P1 --data_mode both --checkpoint path/to/checkpoint.pth
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", message=".*Not enough SMs to use max_autotune_gemm.*")
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import json
import glob
import random
import csv
import time
import re
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from mil_model import MultiCropDataset, extract_well_from_filename
from vae_model import MILVAE

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def load_image(img_path: str, num_channels: int = 1) -> Image.Image:
    try:
        import tifffile
        img_array = tifffile.imread(img_path)
    except ImportError:
        img_array = np.array(Image.open(img_path))
    except Exception:
        img_array = np.array(Image.open(img_path))

    if len(img_array.shape) == 3:
        img_array = img_array[:, :, 0]

    if img_array.dtype == np.uint16:
        img_array = img_array.astype(np.float32) / 65535.0
    elif img_array.dtype == np.uint8:
        img_array = img_array.astype(np.float32) / 255.0
    elif img_array.dtype in (np.float32, np.float64):
        img_array = img_array.astype(np.float32)

    img_uint8 = (img_array * 255).astype(np.uint8)
    if num_channels == 1:
        return Image.fromarray(img_uint8, mode='L')
    else:
        return Image.fromarray(img_uint8, mode='L').convert('RGB')


def extract_well_from_path(img_path: str) -> str | None:
    """Extract well label (e.g. A01) from image path using the filename."""
    return extract_well_from_filename(os.path.basename(img_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_plate', type=str, default='P1')
    parser.add_argument('--data_mode', type=str, default='both', choices=['drug', 'mutant', 'both'])
    parser.add_argument('--drug_no_concentration', action='store_true')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint. Default: mil_vae_{data_mode}/fold_{test_plate}/checkpoint_mil_latest.pth')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory. Default: mil_vae_{data_mode}/fold_{test_plate}')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='efficientnet_b0')
    parser.add_argument('--pretrained', type=str, default='imagenet')
    parser.add_argument('--pooling', type=str, default='attention')
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--neighborhood', type=int, default=3)
    parser.add_argument('--grid_size', type=int, default=12)
    parser.add_argument('--extraction_mode', type=str, default='neighborhood')
    parser.add_argument('--no_pixel_decoder', action='store_true')
    parser.add_argument('--no_feature_decoder', action='store_true')
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    if args.data_root:
        BASE_DIR = args.data_root
    elif args.data_mode == 'drug':
        BASE_DIR = os.path.join(PROJECT_ROOT, 'Drugs_Data')
    else:
        BASE_DIR = os.path.join(PROJECT_ROOT, 'Mutants_Data')

    IC50_MAPPING_PATH = os.path.join(PROJECT_ROOT, 'final_mutant_model', 'plate_well_ic50_mapping.json')
    MUTANT_MAPPING_PATH = os.path.join(PROJECT_ROOT, 'final_mutant_model', 'plate_well_id_path.json')

    with open(IC50_MAPPING_PATH, 'r') as f:
        ic50_data = json.load(f)
    with open(MUTANT_MAPPING_PATH, 'r') as f:
        mutant_data = json.load(f)

    plate_maps = {}
    for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
        plate_maps[plate] = {}
        if args.data_mode in ('drug', 'both') and plate in ic50_data:
            for well, info in ic50_data[plate].items():
                antibiotic = info.get('antibiotic', '')
                ic50_mult = info.get('ic50_multiple', '')
                if antibiotic and ic50_mult:
                    if args.drug_no_concentration:
                        drug_class = antibiotic.replace(' ', '_')
                    else:
                        if ic50_mult == 'control':
                            drug_class = 'control'
                        else:
                            ic50_str = ic50_mult if 'x' in ic50_mult else f"{ic50_mult}x"
                            drug_class = f"{antibiotic.replace(' ', '_')}_{ic50_str}"
                    plate_maps[plate][f"drug_{well}"] = drug_class
        if args.data_mode in ('mutant', 'both') and plate in mutant_data:
            for row, cols in mutant_data[plate].items():
                for col, info in cols.items():
                    if 'id' in info:
                        well = f"{row}{int(col):02d}"
                        plate_maps[plate][f"mutant_{well}"] = info['id']

    all_plates = ['Plate_1', 'Plate_2', 'Plate_3', 'Plate_4', 'Plate_5', 'Plate_6']

    def get_image_paths(plate):
        plate_key = f"P{plate.split('_')[-1]}"
        search_dirs = []
        if args.data_mode in ('drug', 'both'):
            drug_base = os.path.join(PROJECT_ROOT, 'Drugs_Data')
            search_dirs.append((os.path.join(drug_base, plate_key), 'drug'))
        if args.data_mode in ('mutant', 'both'):
            mutant_base = os.path.join(PROJECT_ROOT, 'Mutants_Data')
            search_dirs.append((os.path.join(mutant_base, plate_key), 'mutant'))
        valid = []
        for plate_dir, source_type in search_dirs:
            if not os.path.exists(plate_dir):
                continue
            patterns = ['*.tif', '*.tiff', '*.png']
            for pattern in patterns:
                paths = glob.glob(os.path.join(plate_dir, '**', pattern), recursive=True)
                for path in paths:
                    well = extract_well_from_filename(os.path.basename(path))
                    composite = f"{source_type}_{well}"
                    if composite and composite in plate_maps.get(plate_key, {}):
                        valid.append(path)
        return valid

    test_norm = f"Plate_{args.test_plate[-1]}" if 'P' in args.test_plate.upper() and args.test_plate[-1].isdigit() else args.test_plate

    # Build class mapping
    all_classes_set = set()
    for pm in plate_maps.values():
        for lbl in pm.values():
            if lbl:
                all_classes_set.add(lbl)
    all_classes = sorted(all_classes_set)
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    num_classes = len(all_classes)
    print(f"Total classes: {num_classes}")

    def extract_label(path):
        path_lower = path.lower()
        for pn in range(1, 7):
            if f'/p{pn}/' in path_lower:
                plate_key = f'P{pn}'
                break
        else:
            return None
        well = extract_well_from_filename(os.path.basename(path))
        if well is None:
            return None
        if '/mutants_data/' in path_lower:
            prefix = 'mutant_'
        else:
            prefix = 'drug_'
        cw = f"{prefix}{well}"
        if plate_key in plate_maps and cw in plate_maps[plate_key]:
            return plate_maps[plate_key][cw]
        return None

    def get_source_type(path):
        path_lower = path.lower()
        return 'mutant' if '/mutants_data/' in path_lower else 'drug'

    # Collect test set
    test_paths, test_labels_str = [], []
    for p in get_image_paths(test_norm):
        lbl = extract_label(p)
        if lbl in class_to_idx:
            test_paths.append(p)
            test_labels_str.append(lbl)

    print(f"Test set: {len(test_paths)} images from {test_norm}")
    test_labels = [class_to_idx[lbl] for lbl in test_labels_str]

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    use_pixel_decoder = not args.no_pixel_decoder
    use_feature_decoder = not args.no_feature_decoder

    model = MILVAE(
        num_classes=num_classes,
        latent_dim=args.latent_dim,
        beta=0.1,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_contrastive=True,
        num_channels=args.num_channels,
        pretrained=args.pretrained,
        backbone=args.backbone,
        pooling=args.pooling,
        img_size=224,
        feature_decoder=use_feature_decoder,
        pixel_decoder=use_pixel_decoder,
    ).to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    data_mode_folder = args.data_mode
    if args.data_mode == 'drug' and args.drug_no_concentration:
        data_mode_folder = 'drug_noconcentration'

    default_output_dir = os.path.join(SCRIPT_DIR, f'mil_vae_{data_mode_folder}', f'fold_{args.test_plate}')
    output_dir = args.output_dir or default_output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = os.path.join(output_dir, 'checkpoint_mil_latest.pth')
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(output_dir, 'best_mil_vae.pth')

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded from epoch {ckpt.get('epoch', '?')}, val_auc={ckpt.get('val_auc', '?'):.4f}")
    else:
        model.load_state_dict(ckpt)
        print(f"  Loaded best_mil_vae.pth")

    # ------------------------------------------------------------------
    # Create dataset to get valid positions
    # ------------------------------------------------------------------
    sample_dataset = MultiCropDataset(
        test_paths[:1], test_labels[:1], None,
        neighborhood=args.neighborhood,
        grid_size=args.grid_size,
        augment=False,
        seed=SEED,
        num_channels=args.num_channels,
        extraction_mode=args.extraction_mode,
    )
    valid_positions = sample_dataset.positions
    num_positions = len(valid_positions)
    stride = sample_dataset.stride
    crop_size = sample_dataset.crop_size
    image_size = sample_dataset.image_size
    num_neighbors = args.neighborhood * args.neighborhood

    norm_mean = [0.5] if args.num_channels == 1 else [0.485, 0.456, 0.406]
    norm_std = [0.5] if args.num_channels == 1 else [0.229, 0.224, 0.225]
    transform = A.Compose([
        A.Normalize(mean=norm_mean, std=norm_std),
        ToTensorV2(),
    ])

    print(f"Grid: {args.grid_size}x{args.grid_size}, Valid positions: {num_positions}, "
          f"3x3 neighborhood: {num_neighbors} crops/position")
    print(f"Image size: {image_size}, Crop size: {crop_size}, Stride: {stride:.1f}")

    # ------------------------------------------------------------------
    # Extract latents at all positions
    # ------------------------------------------------------------------
    half_n = args.neighborhood // 2
    records = []

    print(f"\nExtracting latents for {len(test_paths)} test images...")
    start_time = time.time()

    for img_idx, (img_path, true_label_str) in enumerate(zip(test_paths, test_labels_str)):
        well = extract_well_from_path(img_path)
        source = get_source_type(img_path)
        true_label_idx = class_to_idx[true_label_str]
        composite_key = f"{source}_{well}"

        image = load_image(img_path, args.num_channels)
        img_w, img_h = image.size

        position_latents = []
        position_mu = []
        position_logvar = []
        position_bag = []
        position_logits = []
        position_probs = []
        position_pred = []
        position_attention = []
        grid_rows = []
        grid_cols = []

        for pos_idx, (center_left, center_top) in enumerate(valid_positions):
            crops_list = []
            for di in range(-half_n, half_n + 1):
                for dj in range(-half_n, half_n + 1):
                    left = int(center_left + dj * stride)
                    top = int(center_top + di * stride)
                    left = max(0, min(left, img_w - crop_size))
                    top = max(0, min(top, img_h - crop_size))
                    crop = image.crop((left, top, left + crop_size, top + crop_size))
                    crop = np.array(crop)
                    crop = transform(image=crop)['image']
                    crops_list.append(crop)

            crops = torch.stack(crops_list).unsqueeze(0).to(device)

            with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                results = model(crops)

            z = results['z'].cpu().numpy()
            mu = results['mu'].cpu().numpy()
            logvar = results['logvar'].cpu().numpy()
            bag = results['bag'].cpu().numpy()
            logits = results['logits'].cpu().numpy()
            probs = F.softmax(results['logits'], dim=1).cpu().numpy()
            pred = np.argmax(logits, axis=1)
            attn = results['attn_weights'].cpu().numpy()

            position_latents.append(z[0])
            position_mu.append(mu[0])
            position_logvar.append(logvar[0])
            position_bag.append(bag[0])
            position_logits.append(logits[0])
            position_probs.append(probs[0])
            position_pred.append(pred[0])
            position_attention.append(attn[0])

            # Compute grid row/col from position index
            grid_col = pos_idx % (args.grid_size - 2)
            grid_row = pos_idx // (args.grid_size - 2)
            grid_rows.append(grid_row)
            grid_cols.append(grid_col)

        rec = {
            'img_idx': img_idx,
            'img_path': img_path,
            'well': well,
            'source': source,
            'composite_key': composite_key,
            'true_label': true_label_str,
            'true_label_idx': true_label_idx,
            'pred_label': idx_to_class[position_pred[0]],
            'pred_label_idx': int(position_pred[0]),
            'num_positions': num_positions,
            'latents': np.stack(position_latents),           # [P, 32]
            'mu': np.stack(position_mu),                      # [P, 32]
            'logvar': np.stack(position_logvar),              # [P, 32]
            'bag': np.stack(position_bag),                    # [P, 1280]
            'logits': np.stack(position_logits),              # [P, num_classes]
            'probs': np.stack(position_probs),                # [P, num_classes]
            'predictions': np.array(position_pred),           # [P]
            'attention': np.stack(position_attention),        # [P, num_heads, num_neighbors]
            'grid_rows': np.array(grid_rows),                 # [P]
            'grid_cols': np.array(grid_cols),                 # [P]
        }
        records.append(rec)

        if (img_idx + 1) % 20 == 0 or img_idx == len(test_paths) - 1:
            elapsed = time.time() - start_time
            print(f"  [{img_idx + 1}/{len(test_paths)}] "
                  f"well={well} true={true_label_str} pred={rec['pred_label']} "
                  f"({elapsed / (img_idx + 1):.1f}s/img)")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pt_path = os.path.join(output_dir, f'test_latents_{args.test_plate}_{timestamp}.pt')
    csv_path = os.path.join(output_dir, f'test_latents_{args.test_plate}_{timestamp}.csv')

    print(f"\nSaving to {pt_path} ...")
    torch.save({
        'args': vars(args),
        'checkpoint': ckpt_path,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'records': records,
        'num_positions': num_positions,
        'grid_shape': (args.grid_size - 2, args.grid_size - 2),
    }, pt_path)

    # CSV: one row per (well, grid_position)
    csv_cols = [
        'well', 'source', 'true_label', 'pred_label',
        'grid_row', 'grid_col',
    ] + [f'z_{i}' for i in range(args.latent_dim)] + ['correct']
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(csv_cols)
        for rec in records:
            for pos_i in range(rec['num_positions']):
                row = [
                    rec['well'],
                    rec['source'],
                    rec['true_label'],
                    rec['pred_label'],
                    int(rec['grid_rows'][pos_i]),
                    int(rec['grid_cols'][pos_i]),
                ]
                row.extend(rec['latents'][pos_i].tolist())
                row.append(1 if rec['predictions'][pos_i] == rec['true_label_idx'] else 0)
                w.writerow(row)

    print(f"Saved {len(records)} images × {num_positions} positions to:")
    print(f"  {pt_path}")
    print(f"  {csv_path}")
    print(f"\nPer-well majority vote accuracy:")
    correct_vote = 0
    for rec in records:
        vote_pred = np.bincount(rec['predictions']).argmax()
        if vote_pred == rec['true_label_idx']:
            correct_vote += 1
    print(f"  {correct_vote}/{len(records)} = {100 * correct_vote / len(records):.1f}%")


if __name__ == '__main__':
    main()
