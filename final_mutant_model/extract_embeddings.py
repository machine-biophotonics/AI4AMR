#!/usr/bin/env python3
"""
Extract 1280-d bag embeddings from trained MILEncoder for all plates.
Processes both mutant and drug plates through backbone → attention → head_proj.
Saves embeddings with metadata for downstream ComBat correction + cross-domain prediction.
"""

import os
import sys
import json
import argparse
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALL_PLATES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']

def parse_well_from_filename(img_path: str) -> Optional[str]:
    filename = os.path.basename(img_path)
    import re
    match = re.search(r'Well(\w\d+)_', filename)
    return match.group(1) if match else None

def load_image(img_path: str, num_channels: int = 1) -> Image.Image:
    """Load image with same normalization as training."""
    try:
        import tifffile
        img_array = tifffile.imread(str(img_path))
    except (ImportError, Exception):
        img_array = np.array(Image.open(str(img_path)))

    if len(img_array.shape) == 3:
        img_array = img_array[:, :, 0]

    if img_array.dtype == np.uint16:
        img_array = img_array.astype(np.float32) / 65535.0
    elif img_array.dtype == np.uint8:
        img_array = img_array.astype(np.float32) / 255.0
    elif img_array.dtype in (np.float32, np.float64):
        img_array = img_array.astype(np.float32)

    if num_channels == 1:
        return Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
    else:
        return Image.fromarray((img_array * 255).astype(np.uint8), mode='L').convert('RGB')

def extract_mil_crops(img_path: str, crop_size: int = 224, grid_size: int = 12,
                      neighborhood: int = 3, num_channels: int = 1) -> torch.Tensor:
    """Extract 3x3 neighborhood crops as a single bag tensor [1, 9, C, H, W]."""
    img = load_image(img_path, num_channels)

    if num_channels == 1:
        img_np = np.array(img)
        if len(img_np.shape) == 2:
            img_np = img_np[np.newaxis, ...]
        else:
            img_np = np.transpose(img_np, (2, 0, 1))
    else:
        img_np = np.array(img)
        img_np = np.transpose(img_np, (2, 0, 1))

    w = img_np.shape[2] if len(img_np.shape) == 3 else img_np.shape[1]
    h = img_np.shape[1]

    stride = (w - crop_size) // (grid_size - 1) if grid_size > 1 else 0
    half_n = neighborhood // 2

    # Find center position
    center_i, center_j = grid_size // 2, grid_size // 2
    center_left = center_j * stride
    center_top = center_i * stride

    crops_list = []
    for di in range(-half_n, half_n + 1):
        for dj in range(-half_n, half_n + 1):
            left = center_left + dj * stride
            top = center_top + di * stride
            left = max(0, min(left, w - crop_size))
            top = max(0, min(top, h - crop_size))

            if num_channels == 1:
                crop_np = img_np[:, top:top + crop_size, left:left + crop_size]
            else:
                crop_np = img_np[:, top:top + crop_size, left:left + crop_size]

            if num_channels == 1:
                mean = np.array([0.5], dtype=np.float32).reshape(1, 1, 1)
                std = np.array([0.5], dtype=np.float32).reshape(1, 1, 1)
            else:
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

            crop_np = crop_np.astype(np.float32) / 255.0
            crop_np = (crop_np - mean) / std
            crops_list.append(torch.from_numpy(crop_np).float())

    return torch.stack(crops_list).unsqueeze(0)

def get_images_for_plate(plate_dir: str) -> list[Path]:
    """Get all TIFF images recursively in a plate directory."""
    paths = sorted(Path(plate_dir).rglob('*.tif'))
    paths += sorted(Path(plate_dir).rglob('*.tiff'))
    return paths

def get_drug_label(well: str, plate_key: str, ic50_data: dict,
                   drug_no_concentration: bool = False) -> Optional[str]:
    """Get drug class label for a well."""
    if plate_key not in ic50_data or well not in ic50_data[plate_key]:
        return None
    info = ic50_data[plate_key][well]
    antibiotic = info.get('antibiotic', '')
    ic50_multiple = info.get('ic50_multiple', '')
    if not antibiotic or not ic50_multiple:
        return None
    if drug_no_concentration:
        return antibiotic.replace(' ', '_')
    if ic50_multiple == 'control':
        return 'control'
    ic50_str = ic50_multiple if 'x' in str(ic50_multiple) else f"{ic50_multiple}x"
    return f"{antibiotic.replace(' ', '_')}_{ic50_str}"

def get_mutant_label(well: str, plate_key: str, mutant_data: dict) -> Optional[str]:
    """Get mutant gene label for a well."""
    if plate_key not in mutant_data:
        return None
    row = well[0]
    col = str(int(well[1:]))
    if row in mutant_data[plate_key] and col in mutant_data[plate_key][row]:
        return mutant_data[plate_key][row][col].get('id', None)
    return None


def main():
    parser = argparse.ArgumentParser(description='Extract bag embeddings from trained MIL model')
    parser.add_argument('--checkpoint', type=str, default='mutant/fold_Plate_1/best_model_acc.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='embeddings',
                        help='Output directory for saved embeddings')
    parser.add_argument('--num_channels', type=int, default=1, help='Input channels (1=grayscale)')
    parser.add_argument('--num_heads', type=int, default=4, help='Attention heads')
    parser.add_argument('--pooling', type=str, default='attention', help='Pooling method')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for extraction')
    parser.add_argument('--drug_no_concentration', action='store_true', default=False,
                        help='Group drugs by antibiotic name only')
    parser.add_argument('--dry_run', action='store_true', help='Dry run - count only')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load label mappings
    with open(os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')) as f:
        ic50_data = json.load(f)
    with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')) as f:
        mutant_data = json.load(f)

    # Load model checkpoint
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(SCRIPT_DIR, checkpoint_path)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    has_contrastive = any('contrastive_head' in k for k in checkpoint['model_state_dict'].keys())
    num_classes = checkpoint['model_state_dict']['classifier.1.weight'].shape[0]
    print(f"  num_classes: {num_classes}, has_contrastive: {has_contrastive}")

    from mil_model import MILEncoder
    model = MILEncoder(
        num_classes=num_classes,
        num_heads=args.num_heads,
        attention_temp=0.5,
        dropout=0.0,
        use_contrastive=has_contrastive,
        num_channels=args.num_channels,
        pooling=args.pooling
    )
    # Load only backbone + attention + head_proj (skip classifier for embedding extraction)
    filtered_sd = {k: v for k, v in checkpoint['model_state_dict'].items()
                   if 'classifier' not in k}
    model.load_state_dict(filtered_sd, strict=False)
    model = model.to(device)
    model.eval()
    print("Model loaded, classifier skipped (using as feature extractor)")

    # Collect all plate data sources
    data_sources = {}
    for plate in ALL_PLATES:
        # Mutant data
        mut_dir = os.path.join(os.path.dirname(SCRIPT_DIR), 'Mutants_Data', plate)
        if os.path.exists(mut_dir):
            images = get_images_for_plate(mut_dir)
            if images:
                data_sources[f'mutant_{plate}'] = {
                    'domain': 'mutant',
                    'plate': plate,
                    'images': images,
                    'label_fn': lambda w, pk=plate: get_mutant_label(w, pk, mutant_data)
                }
                print(f"  mutant_{plate}: {len(images)} images")

        # Drug data
        drug_dir = os.path.join(os.path.dirname(SCRIPT_DIR), 'Drugs_Data', plate)
        if os.path.exists(drug_dir):
            images = get_images_for_plate(drug_dir)
            if images:
                data_sources[f'drug_{plate}'] = {
                    'domain': 'drug',
                    'plate': plate,
                    'images': images,
                    'label_fn': lambda w, pk=plate, ic50=ic50_data: get_drug_label(w, pk, ic50, args.drug_no_concentration)
                }
                print(f"  drug_{plate}: {len(images)} images")

    print(f"\nTotal data sources: {len(data_sources)}")

    if args.dry_run:
        total = sum(len(v['images']) for v in data_sources.values())
        print(f"Total images to process: {total}")
        sys.exit(0)

    # Extract embeddings
    os.makedirs(args.output_dir, exist_ok=True)

    for source_name, source_info in data_sources.items():
        images = source_info['images']
        domain = source_info['domain']
        plate = source_info['plate']
        label_fn = source_info['label_fn']

        output_path = os.path.join(args.output_dir, f'embeddings_{source_name}.npz')
        if os.path.exists(output_path):
            print(f"  Skipping {source_name} (already exists)")
            continue

        all_embeddings = []
        all_labels = []
        all_paths = []
        all_wells = []

        print(f"\nExtracting {source_name} ({domain}, {plate}): {len(images)} images")

        for i in tqdm(range(0, len(images), args.batch_size), desc=source_name):
            batch_paths = images[i:i + args.batch_size]
            batch_tensors = []

            for img_path in batch_paths:
                try:
                    bag = extract_mil_crops(
                        str(img_path),
                        crop_size=224,
                        grid_size=12,
                        neighborhood=3,
                        num_channels=args.num_channels
                    )
                    batch_tensors.append(bag)
                except Exception as e:
                    print(f"  Error processing {img_path}: {e}")
                    continue

            if not batch_tensors:
                continue

            batch = torch.cat(batch_tensors, dim=0).to(device)
            with torch.no_grad():
                embs = model.get_mil_embeddings(batch)

            all_embeddings.append(embs.cpu().numpy())
            for img_path in batch_paths:
                well = parse_well_from_filename(str(img_path))
                label = label_fn(well) if well else None
                all_labels.append(label if label else 'unknown')
                all_paths.append(str(img_path))
                all_wells.append(well if well else 'unknown')

        if not all_embeddings:
            print(f"  No embeddings extracted for {source_name}")
            continue

        embeddings_np = np.concatenate(all_embeddings, axis=0)
        labels_np = np.array(all_labels)
        paths_np = np.array(all_paths)
        wells_np = np.array(all_wells)

        print(f"  Shape: {embeddings_np.shape}, Labels: {len(set(labels_np))}")

        np.savez_compressed(
            output_path,
            embeddings=embeddings_np,
            labels=labels_np,
            paths=paths_np,
            wells=wells_np,
            domain=domain,
            plate=plate
        )
        print(f"  Saved: {output_path}")

    # Save metadata about classes
    # Collect all mutant classes
    all_mutant_classes = sorted(set(
        info['id'] for pk in mutant_data
        for row in mutant_data[pk].values()
        for col, info in row.items()
        if 'id' in info
    ))
    print(f"\nUnique mutant classes: {len(all_mutant_classes)}")

    # Collect all drug classes
    all_drug_classes = set()
    for pk in ic50_data:
        for well, info in ic50_data[pk].items():
            label = get_drug_label(well, pk, ic50_data, args.drug_no_concentration)
            if label:
                all_drug_classes.add(label)
    all_drug_classes = sorted(all_drug_classes)
    print(f"Unique drug classes: {len(all_drug_classes)}")

    np.savez_compressed(
        os.path.join(args.output_dir, 'class_mappings.npz'),
        mutant_classes=np.array(all_mutant_classes, dtype=object),
        drug_classes=np.array(all_drug_classes, dtype=object),
    )

    print("\nDone! All embeddings extracted.")


if __name__ == '__main__':
    main()
