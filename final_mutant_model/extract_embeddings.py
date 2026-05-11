#!/usr/bin/env python3
"""
Extract embeddings from center crop for cross-domain analysis.
Uses trained MILEncoder checkpoint as frozen feature extractor.
Extracts 3x3 neighborhood (9 crops) like training, uses attention pooling.
Supports: mutant, drug, drug_on_mutant, mutant_on_drug modes.
"""

import os
import sys
import json
import argparse
import glob
import re
from typing import Optional, List

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path


SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract embeddings from center crop')
    parser.add_argument('--fold', type=str, default='P6',
                        help='Fold to process (e.g., P6). Default: P6.')
    parser.add_argument('--data_mode', type=str, default='drug', choices=['drug', 'mutant', 'both'],
                        help='Data mode: drug (Drugs_Data), mutant (Mutants_Data), both')
    parser.add_argument('--drug_on_mutant', action='store_true', default=False,
                        help='Use drug-trained model on mutant images (cross-domain)')
    parser.add_argument('--mutant_on_drug', action='store_true', default=False,
                        help='Use mutant-trained model on drug images (cross-domain)')
    parser.add_argument('--drug_no_concentration', action='store_true', default=False,
                        help='Group drugs by antibiotic name only')
    parser.add_argument('--crop_size', type=int, default=224, help='Crop size (default: 224)')
    parser.add_argument('--grid_size', type=int, default=12, help='Grid size (default: 12)')
    parser.add_argument('--neighborhood', type=int, default=3, choices=[3, 5, 7, 9, 11],
                        help='Neighborhood size: 3=(3x3=9 crops), 5=(5x5=25 crops), etc. Default: 3')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process')
    parser.add_argument('--sample_per_class', type=int, default=None,
                        help='Sample N images per class')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for sampling')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                        help='Checkpoint filename to use')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to parent folder containing P1-P6')
    parser.add_argument('--use_sc_mil', action='store_true',
                        help='Use SC-MIL trained model')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout for inference (default: 0.0)')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (default: 4)')
    parser.add_argument('--num_channels', type=int, default=1,
                        help='Number of input channels (default: 1)')
    parser.add_argument('--pretrained', type=str, default='micronet', choices=['imagenet', 'micronet'],
                        help='Pretrained weights')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0', choices=['efficientnet_b0', 'mobilenet_v3_small', 'mobilenet_v2'],
                        help='Backbone architecture (must match training)')
    parser.add_argument('--pooling', type=str, default='attention', choices=['attention', 'simple_attention', 'mean', 'max'],
                        help='Pooling method')
    parser.add_argument('--embedding_type', type=str, default='mil', choices=['backbone', 'mil', 'projected'],
                        help='Embedding type: mil (attention pooled), backbone (1280-dim), projected (256-dim)')
    parser.add_argument('--output_name', type=str, default=None,
                        help="Output filename")
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='Dry run - do not save results')
    
    args: argparse.Namespace = parser.parse_args()
    
    # Load JSON mappings
    ic50_path = os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')
    if os.path.exists(ic50_path):
        with open(ic50_path, 'r') as f:
            IC50_DATA: dict = json.load(f)
    else:
        IC50_DATA = {}
        print(f"Warning: {ic50_path} not found")
    
    mutant_path = os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')
    if os.path.exists(mutant_path):
        with open(mutant_path, 'r') as f:
            MUTANT_DATA: dict = json.load(f)
    else:
        MUTANT_DATA = {}
        print(f"Warning: {mutant_path} not found")
    
    # Handle cross-domain modes
    if args.mutant_on_drug:
        # Mutant model on drug images
        print(f"\n*** MUTANT ON DRUG MODE ***")
        print(f"  Using mutant-trained model to extract embeddings from drug images")
        data_mode_folder = 'mutant_on_drug'
        actual_data_mode = 'drug'
        actual_model_mode = 'mutant'
    elif args.drug_on_mutant:
        # Drug model on mutant images
        print(f"\n*** DRUG ON MUTANT MODE ***")
        print(f"  Using drug-trained model to extract embeddings from mutant images")
        data_mode_folder = 'drug_on_mutant'
        actual_data_mode = 'mutant'
        actual_model_mode = 'drug'
    else:
        data_mode_folder = args.data_mode
        actual_data_mode = args.data_mode
        actual_model_mode = args.data_mode
    
    if actual_data_mode == 'drug' and args.drug_no_concentration:
        data_mode_folder = 'drug_noconcentration'
    
    # Build classes based on model mode (not data mode)
    all_classes: list[str] = []
    
    if actual_model_mode == 'drug':
        drug_classes: set = set()
        for plate, wells in IC50_DATA.items():
            for well, info in wells.items():
                antibiotic = info.get('antibiotic', '')
                ic50_multiple = info.get('ic50_multiple', '')
                if antibiotic and ic50_multiple:
                    if args.drug_no_concentration:
                        drug_classes.add(antibiotic.replace(' ', '_'))
                    else:
                        if ic50_multiple == 'control':
                            drug_classes.add('control')
                        else:
                            ic50_str = ic50_multiple if 'x' in str(ic50_multiple) else f"{ic50_multiple}x"
                            antibiotic_clean = antibiotic.replace(' ', '_')
                            drug_classes.add(f"{antibiotic_clean}_{ic50_str}")
        all_classes = sorted(drug_classes)
    elif actual_model_mode == 'mutant':
        mutant_classes: set = set()
        for plate, rows in MUTANT_DATA.items():
            for row, cols in rows.items():
                for col, info in cols.items():
                    if 'id' in info:
                        mutant_classes.add(info['id'])
        all_classes = sorted(mutant_classes)
    else:  # both
        drug_classes: set = set()
        for plate, wells in IC50_DATA.items():
            for well, info in wells.items():
                antibiotic = info.get('antibiotic', '')
                ic50_multiple = info.get('ic50_multiple', '')
                if antibiotic and ic50_multiple:
                    if args.drug_no_concentration:
                        drug_classes.add(antibiotic.replace(' ', '_'))
                    else:
                        if ic50_multiple == 'control':
                            drug_classes.add('control')
                        else:
                            ic50_str = ic50_multiple if 'x' in str(ic50_multiple) else f"{ic50_multiple}x"
                            antibiotic_clean = antibiotic.replace(' ', '_')
                            drug_classes.add(f"{antibiotic_clean}_{ic50_str}")
        mutant_classes: set = set()
        for plate, rows in MUTANT_DATA.items():
            for row, cols in rows.items():
                for col, info in cols.items():
                    if 'id' in info:
                        mutant_classes.add(info['id'])
        all_classes = sorted(drug_classes | mutant_classes)
    
    classes = {i: name for i, name in enumerate(all_classes)}
    num_classes: int = args.num_classes if args.num_classes is not None else len(classes)
    print(f"Loaded {num_classes} classes from model_mode={actual_model_mode}")

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    from mil_model import MILEncoder
    
    def _load_image(img_path: str) -> Image.Image:
        import numpy as np
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
        elif img_array.dtype == np.float32 or img_array.dtype == np.float64:
            img_array = img_array.astype(np.float32)
        
        if args.num_channels == 1:
            return Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
        else:
            return Image.fromarray((img_array * 255).astype(np.uint8), mode='L').convert('RGB')
    
    test_plate: str = args.fold
    neighborhood: int = args.neighborhood
    num_crops_per_position: int = neighborhood * neighborhood
    
    if 'Plate_' in test_plate:
        plate_num = test_plate.split('_')[-1]
        image_plate_key = f'P{plate_num}'
    else:
        image_plate_key = test_plate
    
    # Determine fold_key for checkpoint
    if 'Plate_' in test_plate:
        fold_key = test_plate
    else:
        fold_key = f'Plate_{test_plate.replace("P", "")}'
    
    # Checkpoint path - use model's folder
    if os.path.sep in args.checkpoint or os.path.exists(args.checkpoint):
        checkpoint_path = args.checkpoint
    else:
        checkpoint_folder = actual_model_mode
        fold_dir = os.path.join(SCRIPT_DIR, checkpoint_folder, f'fold_{fold_key}')
        checkpoint_path = os.path.join(fold_dir, args.checkpoint)
    
    # Data directory - use data mode
    if args.data_root:
        BASE_DIR = args.data_root
        image_dir = os.path.join(BASE_DIR, image_plate_key)
    elif actual_data_mode == 'drug':
        image_dir = os.path.join(os.path.dirname(SCRIPT_DIR), 'Drugs_Data', image_plate_key)
    elif actual_data_mode == 'both':
        image_dir = None  # Will collect from both dirs below
    else:
        image_dir = os.path.join(os.path.dirname(SCRIPT_DIR), 'Mutants_Data', image_plate_key)
    
    crop_size = args.crop_size
    grid_size = args.grid_size
    
    print(f"\n{'='*60}")
    print(f"Embedding Extraction (MATCHING TRAINING)")
    print(f"  fold: {test_plate}")
    print(f"  checkpoint: {checkpoint_path}")
    if actual_data_mode == 'both':
        print(f"  image_dir: Drugs_Data/{image_plate_key} + Mutants_Data/{image_plate_key}")
    else:
        print(f"  image_dir: {image_dir}")
    print(f"  model_mode: {actual_model_mode}")
    print(f"  data_mode: {actual_data_mode}")
    print(f"  neighborhood: {neighborhood}x{neighborhood} ({num_crops_per_position} crops)")
    print(f"  embedding_type: {args.embedding_type}")
    print(f"  pooling: {args.pooling}")
    print(f"{'='*60}")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Infer exact num_classes from checkpoint to match training architecture
    ckpt_num_classes = checkpoint['model_state_dict']['classifier.1.weight'].shape[0]
    print(f"  Checkpoint num_classes: {ckpt_num_classes}")
    
    has_contrastive = any('contrastive_head' in k for k in checkpoint['model_state_dict'].keys())
    use_sc_mil = has_contrastive or args.use_sc_mil
    
    model = MILEncoder(
        num_classes=ckpt_num_classes, 
        num_heads=args.num_heads, 
        attention_temp=0.5, 
        dropout=args.dropout,
        use_contrastive=use_sc_mil,
        num_channels=args.num_channels,
        pretrained=args.pretrained,
        backbone=args.backbone,
        pooling=args.pooling
    )
    model = model.to(device)
    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    print(f"Model loaded and frozen (SC-MIL: {use_sc_mil})")

    def extract_mil_crops(img_path: str, crop_size: int, grid_size: int, neighborhood: int):
        """Extract neighborhood crops like training: NxN around center position."""
        img = _load_image(img_path)
        
        if args.num_channels == 1:
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
        
        valid_positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * stride
                top = i * stride
                if left + crop_size <= w and top + crop_size <= h:
                    can_left = left - half_n * stride >= 0
                    can_right = left + half_n * stride + crop_size <= w
                    can_top = top - half_n * stride >= 0
                    can_bottom = top + half_n * stride + crop_size <= h
                    if can_left and can_right and can_top and can_bottom:
                        valid_positions.append((left, top))
        
        crops = []
        
        for center_x, center_y in valid_positions:
            for dy in range(-half_n, half_n + 1):
                for dx in range(-half_n, half_n + 1):
                    left = center_x + dx * stride
                    top = center_y + dy * stride
                    
                    if args.num_channels == 1:
                        crop_np = img_np[:, top:top+crop_size, left:left+crop_size]
                    else:
                        crop_np = img_np[:, top:top+crop_size, left:left+crop_size]
                    
                    if args.num_channels == 1:
                        mean = np.array([0.5], dtype=np.float32).reshape(1, 1, 1)
                        std = np.array([0.5], dtype=np.float32).reshape(1, 1, 1)
                    else:
                        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
                        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
                    
                    crop_np = crop_np.astype(np.float32) / 255.0
                    crop_np = (crop_np - mean) / std
                    crop_tensor = torch.from_numpy(crop_np).float()
                    
                    crops.append(crop_tensor)
        
        return crops

    def get_embeddings(model: MILEncoder, img_path: str, embedding_type: str = 'mil') -> np.ndarray:
        """Extract embeddings using MIL (matches training)."""
        crops = extract_mil_crops(img_path, crop_size, grid_size, neighborhood)
        
        if not crops:
            return None
        
        crops_tensor = torch.stack(crops).unsqueeze(0).to(device)
        
        with torch.no_grad():
            if embedding_type == 'mil':
                embedding = model.get_mil_embeddings(crops_tensor)
            elif embedding_type == 'backbone':
                embedding = model.get_backbone_features(crops_tensor)
            elif embedding_type == 'projected':
                embedding = model.get_projected_features(crops_tensor)
            else:
                embedding = model.get_mil_embeddings(crops_tensor)
        
        return embedding[0].cpu().numpy()

    def get_well_from_filename(img_path: str) -> Optional[str]:
        filename = os.path.basename(img_path)
        match = re.search(r'Well(\w\d+)_', filename)
        return match.group(1) if match else None

    def get_label(plate: str, well: Optional[str], mode: str, img_path: str = '') -> Optional[str]:
        if not well:
            return None
        
        plate_key = plate
        
        if mode == 'both':
            # Disambiguate by source directory (matches train_mil.py both_extractor)
            path_lower = img_path.lower()
            if '/drugs_data/' in path_lower or '\\drugs_data\\' in path_lower:
                source = 'drug'
            elif '/mutants_data/' in path_lower or '\\mutants_data\\' in path_lower:
                source = 'mutant'
            else:
                source = 'mutant'  # default fallback
            
            if source == 'drug':
                if plate_key in IC50_DATA and well in IC50_DATA[plate_key]:
                    info = IC50_DATA[plate_key][well]
                    antibiotic = info.get('antibiotic', '')
                    ic50_multiple = info.get('ic50_multiple', '')
                    if antibiotic and ic50_multiple:
                        if args.drug_no_concentration:
                            return antibiotic.replace(' ', '_')
                        else:
                            if ic50_multiple == 'control':
                                return 'control'
                            ic50_str = ic50_multiple if 'x' in str(ic50_multiple) else f"{ic50_multiple}x"
                            antibiotic_clean = antibiotic.replace(' ', '_')
                            return f"{antibiotic_clean}_{ic50_str}"
            else:  # mutant
                row = well[0]
                col = well[1:].lstrip('0') or '0'
                try:
                    if plate_key in MUTANT_DATA and row in MUTANT_DATA[plate_key]:
                        if col in MUTANT_DATA[plate_key][row]:
                            return MUTANT_DATA[plate_key][row][col].get('id', None)
                except:
                    pass
            return None
        
        if mode in ['mutant', 'both']:
            row = well[0]
            col = well[1:].lstrip('0') or '0'
            try:
                if plate_key in MUTANT_DATA and row in MUTANT_DATA[plate_key]:
                    if col in MUTANT_DATA[plate_key][row]:
                        return MUTANT_DATA[plate_key][row][col].get('id', None)
            except:
                pass
        
        if mode in ['drug', 'both']:
            if plate_key in IC50_DATA and well in IC50_DATA[plate_key]:
                info = IC50_DATA[plate_key][well]
                antibiotic = info.get('antibiotic', '')
                ic50_multiple = info.get('ic50_multiple', '')
                if antibiotic and ic50_multiple:
                    if args.drug_no_concentration:
                        return antibiotic.replace(' ', '_')
                    else:
                        if ic50_multiple == 'control':
                            return 'control'
                        ic50_str = ic50_multiple if 'x' in str(ic50_multiple) else f"{ic50_multiple}x"
                        antibiotic_clean = antibiotic.replace(' ', '_')
                        return f"{antibiotic_clean}_{ic50_str}"
        
        return None

    if actual_data_mode == 'both':
        drug_dir = os.path.join(os.path.dirname(SCRIPT_DIR), 'Drugs_Data', image_plate_key)
        mutant_dir = os.path.join(os.path.dirname(SCRIPT_DIR), 'Mutants_Data', image_plate_key)
        search_dirs = []
        if os.path.exists(drug_dir):
            search_dirs.append(drug_dir)
            print(f"  Including drug images from: {drug_dir}")
        if os.path.exists(mutant_dir):
            search_dirs.append(mutant_dir)
            print(f"  Including mutant images from: {mutant_dir}")
        if not search_dirs:
            print(f"ERROR: No data directories found for {image_plate_key}")
            return
        image_files = []
        for d in search_dirs:
            image_files.extend(glob.glob(os.path.join(d, '**', '*.tif'), recursive=True))
            image_files.extend(glob.glob(os.path.join(d, '**', '*.tiff'), recursive=True))
        image_files = list(set(image_files))
    else:
        if not os.path.exists(image_dir):
            print(f"ERROR: Image directory not found: {image_dir}")
            return
        image_files = list(set(
            glob.glob(os.path.join(image_dir, '**', '*.tif'), recursive=True) +
            glob.glob(os.path.join(image_dir, '**', '*.tiff'), recursive=True)
        ))
    
    if not image_files:
        print(f"ERROR: No image files found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} image files (recursive search)")
    
    if args.sample_per_class:
        import random
        random.seed(args.random_seed)
        sampled_files = []
        class_to_files = {}
        
        for f in image_files:
            well = get_well_from_filename(f)
            label = get_label(test_plate, well, actual_data_mode, f)
            if label not in class_to_files:
                class_to_files[label] = []
            class_to_files[label].append(f)
        
        for label, files in class_to_files.items():
            n = min(args.sample_per_class, len(files))
            sampled = random.sample(files, n)
            sampled_files.extend(sampled)
        
        image_files = sampled_files
        print(f"Sampled to {len(image_files)} images ({args.sample_per_class} per class)")
    
    if args.max_images and len(image_files) > args.max_images:
        image_files = image_files[:args.max_images]
        print(f"Limited to {args.max_images} images")

    embeddings_list = []
    labels_list = []
    paths_list = []
    
    for f in tqdm(image_files, desc="Extracting embeddings"):
        embedding = get_embeddings(model, f, args.embedding_type)
        
        if embedding is not None:
            well = get_well_from_filename(f)
            label = get_label(test_plate, well, actual_data_mode, f)
            
            embeddings_list.append(embedding)
            labels_list.append(label if label else 'unknown')
            paths_list.append(f)
    
    embeddings_array = np.array(embeddings_list)
    
    print(f"\nExtracted {len(embeddings_array)} embeddings")
    print(f"Embedding shape: {embeddings_array.shape}")
    print(f"Embedding type: {args.embedding_type}")
    print(f"  neighborhood: {neighborhood}x{neighborhood} ({num_crops_per_position} crops)")
    
    if args.dry_run:
        print("Dry run - not saving")
        return
    
    if args.output_name:
        output_path = args.output_name
    else:
        output_dir = os.path.join(SCRIPT_DIR, data_mode_folder, f'fold_{fold_key}')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"embeddings_{fold_key}_{args.embedding_type}_n{neighborhood}.npz")
    
    np.savez(
        output_path,
        embeddings=embeddings_array,
        labels=np.array(labels_list),
        paths=np.array(paths_list),
        classes=classes
    )
    
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()