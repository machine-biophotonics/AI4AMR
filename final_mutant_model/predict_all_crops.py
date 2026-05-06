#!/usr/bin/env python3
"""
Predict all crops for each test image in final_mutant_model experiment.
Uses MILEncoder model with attention pooling for prediction.
Supports both standard MIL and SC-MIL trained models.
Supports configurable crop neighborhood: 3x3, 5x5, 7x7, 9x9, 11x11
"""

import os
import sys
import json
import argparse
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path


SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    parser = argparse.ArgumentParser(description='Predict all crops for final_mutant_model')
    parser.add_argument('--fold', type=str, default=None,
                        help='Fold to predict (e.g., P6). If not specified, uses P6.')
    parser.add_argument('--data_mode', type=str, default='mutant', choices=['drug', 'mutant', 'both'],
                        help='Data mode: drug (Drugs_Data), mutant (Mutants_Data), both')
    parser.add_argument('--crop_size', type=int, default=224, help='Crop size (default: 224)')
    parser.add_argument('--grid_size', type=int, default=12, help='Grid size (default: 12)')
    parser.add_argument('--extraction_mode', type=str, default='neighborhood', choices=['neighborhood', 'raster'],
                        help='Crop extraction mode: neighborhood (N×N grids around positions) or raster (all crops in tiling grid)')
    parser.add_argument('--crop_neighborhood', type=int, default=5, choices=[3, 5, 7, 9, 11],
                        help='Neighborhood size: 3=(3x3=9 crops), 5=(5x5=25 crops), 7=(7x7=49 crops), 9=(9x9=81 crops), 11=(11x11=121 crops)')
    parser.add_argument('--no_mil_mode', action='store_true',
                        help='Disable MIL mode (use single crops instead of neighborhoods)')
    parser.add_argument('--mil_mode', dest='mil_mode', action='store_true', default=True,
                        help='Use MIL mode with configurable neighborhood (default: True)')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process')
    parser.add_argument('--sample_per_class', type=int, default=None,
                        help='Sample N images per class (e.g., 1 for 1 per class)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for sampling')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--checkpoint', type=str, default='best_model_acc.pth',
                        help='Checkpoint filename to use (best_model_acc.pth, best_model_auc.pth, best_model_loss.pth)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to parent folder containing P1-P6 (default: parent of script dir)')
    parser.add_argument('--use_sc_mil', action='store_true',
                        help='Use SC-MIL trained model (uses MILEncoder with contrastive head)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout for inference (default: 0.0, set to 0 to disable)')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (default: 4)')
    parser.add_argument('--num_channels', type=int, default=1,
                        help='Number of input channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--pretrained', type=str, default='imagenet', choices=['imagenet', 'micronet'],
                        help='Pretrained weights: imagenet (default) or micronet')
    parser.add_argument('--filter_class', type=str, default=None,
                        help='Filter predictions by class name pattern (e.g., _2x, control)')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='Dry run - do not save any results')
    
    args: argparse.Namespace = parser.parse_args()
    
    # Load JSON mappings based on data_mode
    if args.data_mode in ['drug', 'both']:
        with open(os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json'), 'r') as f:
            IC50_DATA: dict = json.load(f)
    else:
        IC50_DATA = {}
    
    if args.data_mode in ['mutant', 'both']:
        with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json'), 'r') as f:
            MUTANT_DATA: dict = json.load(f)
    else:
        MUTANT_DATA = {}
    
    # Set BASE_DIR based on data_root and data_mode
    if args.data_root:
        BASE_DIR: str = args.data_root
    elif args.data_mode == 'drug':
        BASE_DIR: str = os.path.join(os.path.dirname(SCRIPT_DIR), 'Drugs_Data')
    else:
        BASE_DIR: str = os.path.join(os.path.dirname(SCRIPT_DIR), 'Mutants_Data')
    
    crop_size: int = args.crop_size
    grid_size: int = args.grid_size
    neighborhood: int = args.crop_neighborhood
    extraction_mode: str = args.extraction_mode
    mil_mode: bool = args.mil_mode
    
    num_crops_per_position: int = neighborhood * neighborhood
    
    if extraction_mode == 'raster':
        total_crops = grid_size * grid_size
        print(f"Config: RASTER - {total_crops} crops ({grid_size}x{grid_size} grid), extraction_mode={extraction_mode}")
    else:
        print(f"Config: NEIGHBORHOOD - {num_crops_per_position} crops/position ({neighborhood}x{neighborhood}), extraction_mode={extraction_mode}")
    
    # Build classes based on data_mode (same logic as training)
    all_classes: list[str] = []
    if args.data_mode == 'drug':
        drug_classes: set = set()
        for plate, wells in IC50_DATA.items():
            for well, info in wells.items():
                antibiotic = info.get('antibiotic', '')
                ic50_multiple = info.get('ic50_multiple', '')
                if antibiotic and ic50_multiple:
                    if ic50_multiple == 'control':
                        drug_classes.add('control')
                    else:
                        ic50_str = ic50_multiple if 'x' in str(ic50_multiple) else f"{ic50_multiple}x"
                        antibiotic_clean = antibiotic.replace(' ', '_')
                        drug_classes.add(f"{antibiotic_clean}_{ic50_str}")
        all_classes = sorted(drug_classes)
    elif args.data_mode == 'mutant':
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
    idx_to_label = classes
    label_to_idx = {v: k for k, v in classes.items()}
    
    num_classes: int = args.num_classes if args.num_classes is not None else len(classes)
    print(f"Loaded {num_classes} classes from data_mode={args.data_mode}")

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    from mil_model import MILEncoder
    
    # Define _load_image function - EXACT same as trial_daniel (min-max normalization)
    def _load_image(img_path: str) -> Image.Image:
        """Load image - EXACT same normalization as trial_daniel: sample / (2^bit_depth - 1)"""
        import numpy as np
        try:
            import tifffile
            img_array = tifffile.imread(img_path)
        except ImportError:
            img_array = np.array(Image.open(img_path))
        except Exception:
            img_array = np.array(Image.open(img_path))
        
        # Handle multi-channel or single-channel
        if len(img_array.shape) == 3:
            img_array = img_array[:, :, 0]
        
        # EXACT same normalization as trial_daniel (dataset.py line 207)
        if img_array.dtype == np.uint16:
            # uint16: divide by (2^16 - 1) = 65535
            img_array = img_array.astype(np.float32) / 65535.0
        elif img_array.dtype == np.uint8:
            # uint8: divide by (2^8 - 1) = 255
            img_array = img_array.astype(np.float32) / 255.0
        elif img_array.dtype == np.float32 or img_array.dtype == np.float64:
            img_array = img_array.astype(np.float32)
        
        # Convert to PIL Image
        if args.num_channels == 1:
            return Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
        else:
            return Image.fromarray((img_array * 255).astype(np.uint8), mode='L').convert('RGB')
    
    test_plate: str = args.fold if args.fold else 'P6'
    
    # Checkpoint path: use data_mode folder if specified checkpoint doesn't contain path separator
    if os.path.sep in args.checkpoint or os.path.exists(args.checkpoint):
        # Direct path provided
        checkpoint_path = args.checkpoint
    else:
        # Use data_mode folder (e.g., drug/fold_P6/, mutant/fold_P6/)
        fold_dir = os.path.join(SCRIPT_DIR, args.data_mode, f'fold_{test_plate}')
        checkpoint_path = os.path.join(fold_dir, args.checkpoint)
    
    image_dir: str = os.path.join(BASE_DIR, test_plate)
    output_dir: str = os.path.join(SCRIPT_DIR, args.data_mode, f'fold_{test_plate}')

    print(f'\n{"="*60}')
    print(f'Processing fold: test plate={test_plate}')
    print(f'  checkpoint: {checkpoint_path}')
    print(f'  image_dir: {image_dir}')
    print(f'  mil_mode: {mil_mode}')
    print(f'  neighborhood: {neighborhood}x{neighborhood} ({num_crops_per_position} crops per position)')
    print(f'{"="*60}')
    
    if not os.path.exists(checkpoint_path):
        # Show available in data_mode folder
        fold_dir = os.path.join(SCRIPT_DIR, args.data_mode, f'fold_{test_plate}')
        print(f'ERROR: Checkpoint not found: {checkpoint_path}')
        if os.path.exists(fold_dir):
            print(f'Available checkpoints in {fold_dir}:')
            for f in os.listdir(fold_dir):
                if f.endswith('.pth'):
                    print(f'  - {f}')
        else:
            print(f'Directory does not exist: {fold_dir}')
        return
    
    checkpoint: dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    has_contrastive = any('contrastive_head' in k for k in checkpoint['model_state_dict'].keys())
    print(f'  sc_mil_checkpoint: {has_contrastive}')
    use_sc_mil = has_contrastive or args.use_sc_mil
    
    model: MILEncoder = MILEncoder(
        num_classes=num_classes, 
        num_heads=args.num_heads, 
        attention_temp=0.5, 
        dropout=args.dropout,
        use_contrastive=use_sc_mil,
        num_channels=args.num_channels,
        pretrained=args.pretrained
    )
    model = model.to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    print(f"MILEncoder model loaded successfully (SC-MIL: {use_sc_mil})")

    def extract_all_crops(img_path: str, crop_size: int, grid_size: int) -> list[tuple[torch.Tensor, int, int, int]]:
        """Extract all crops from an image in deterministic order using proper image loading."""
        # Use _load_image for proper 16-bit handling (min-max normalization like trial_daniel)
        img: Image.Image = _load_image(img_path)
        
        # Convert to numpy for processing
        if args.num_channels == 1:
            img_np: np.ndarray = np.array(img)
            # Add channel dimension if needed
            if len(img_np.shape) == 2:
                img_np = img_np[np.newaxis, ...]
            else:
                img_np = np.transpose(img_np, (2, 0, 1))
        else:
            img_np = np.array(img)
            img_np = np.transpose(img_np, (2, 0, 1))
        
        w: int = img_np.shape[2] if len(img_np.shape) == 3 else img_np.shape[1]
        h: int = img_np.shape[1]
        
        stride: int = (w - crop_size) // (grid_size - 1) if grid_size > 1 else 0
        positions: list[tuple[int, int]] = []
        for row in range(grid_size):
            for col in range(grid_size):
                left = col * stride
                top = row * stride
                if left + crop_size <= w and top + crop_size <= h:
                    positions.append((left, top))
        
        crops: list[tuple[torch.Tensor, int, int, int]] = []
        
        for idx, (left, top) in enumerate(positions):
            row = idx // grid_size
            col = idx % grid_size
            
            # Extract crop from numpy array
            if args.num_channels == 1:
                crop_np = img_np[:, top:top+crop_size, left:left+crop_size]
            else:
                crop_np = img_np[:, top:top+crop_size, left:left+crop_size]
            
            # Normalize using ImageNet stats
            if args.num_channels == 1:
                mean: np.ndarray = np.array([0.5], dtype=np.float32).reshape(1, 1, 1)
                std: np.ndarray = np.array([0.5], dtype=np.float32).reshape(1, 1, 1)
            else:
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
            
            crop_np = crop_np.astype(np.float32) / 255.0
            crop_np = (crop_np - mean) / std
            crop_tensor: torch.Tensor = torch.from_numpy(crop_np).float()
            
            crops.append((crop_tensor, row, col, idx))
        
        return crops

    def extract_mil_crops(img_path: str, crop_size: int, grid_size: int, neighborhood: int) -> list[tuple[torch.Tensor, int, int, int]]:
        """
        Extract positions with configurable neighborhood crops using proper image loading.
        
        Args:
            img_path: Path to image
            crop_size: Size of each crop
            grid_size: Grid size for positions
            neighborhood: Neighborhood size (3, 5, 7, 9, 11)
        
        Returns:
            List of (crop_tensor, position_idx, local_row, local_col)
            with num_crops = neighborhood * neighborhood crops per position
        """
        # Use _load_image for proper 16-bit handling (min-max normalization like trial_daniel)
        img: Image.Image = _load_image(img_path)
        
        # Convert to numpy for processing
        if args.num_channels == 1:
            img_np: np.ndarray = np.array(img)
            if len(img_np.shape) == 2:
                img_np = img_np[np.newaxis, ...]
            else:
                img_np = np.transpose(img_np, (2, 0, 1))
        else:
            img_np = np.array(img)
            img_np = np.transpose(img_np, (2, 0, 1))
        
        w: int = img_np.shape[2] if len(img_np.shape) == 3 else img_np.shape[1]
        h: int = img_np.shape[1]
        
        stride: int = (w - crop_size) // (grid_size - 1) if grid_size > 1 else 0
        
        half_n: int = neighborhood // 2
        
        # Find valid center positions that can accommodate the full neighborhood
        valid_positions: list[tuple[int, int]] = []
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * stride
                top = i * stride
                if left + crop_size <= w and top + crop_size <= h:
                    # Check if we can extract full neighborhood around this position
                    can_left = left - half_n * stride >= 0
                    can_right = left + half_n * stride + crop_size <= w
                    can_top = top - half_n * stride >= 0
                    can_bottom = top + half_n * stride + crop_size <= h
                    if can_left and can_right and can_top and can_bottom:
                        valid_positions.append((left, top))
        
        crops: list[tuple[torch.Tensor, int, int, int]] = []
        
        for pos_idx, (center_x, center_y) in enumerate(valid_positions):
            for dy in range(-half_n, half_n + 1):
                for dx in range(-half_n, half_n + 1):
                    left = center_x + dx * stride
                    top = center_y + dy * stride
                    
                    # Extract crop from numpy array
                    if args.num_channels == 1:
                        crop_np = img_np[:, top:top+crop_size, left:left+crop_size]
                    else:
                        crop_np = img_np[:, top:top+crop_size, left:left+crop_size]
                    
                    # Normalize using ImageNet stats
                    if args.num_channels == 1:
                        mean: np.ndarray = np.array([0.5], dtype=np.float32).reshape(1, 1, 1)
                        std: np.ndarray = np.array([0.5], dtype=np.float32).reshape(1, 1, 1)
                    else:
                        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
                        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
                    
                    crop_np = crop_np.astype(np.float32) / 255.0
                    crop_np = (crop_np - mean) / std
                    crop_tensor: torch.Tensor = torch.from_numpy(crop_np).float()
                    
                    # Normalize local position to 0-based index within neighborhood
                    local_row = dy + half_n
                    local_col = dx + half_n
                    
                    crops.append((crop_tensor, pos_idx, local_row, local_col))
        
        return crops

    def parse_well_from_filename(img_path: str) -> Optional[str]:
        """Parse well position from image filename."""
        filename: str = os.path.basename(img_path)
        parts: list[str] = filename.split('_')
        for part in parts:
            if part.startswith('Well'):
                well_str: str = part.replace('Well', '')
                if len(well_str) == 3:
                    row: str = well_str[0]
                    col: str = well_str[1:]  # Keep leading zero (e.g., '01' not '1')
                    return row + col
                return well_str
        return None

    def get_ground_truth_label(plate: str, well: Optional[str]) -> Optional[str]:
        """Get ground truth label from appropriate JSON based on data_mode."""
        if not well:
            return None
        
        # For mutant mode, use MUTANT_DATA
        if args.data_mode in ['mutant', 'both'] and plate in MUTANT_DATA:
            row: str = well[0]
            col: str = well[1:]
            if row in MUTANT_DATA[plate]:
                if col in MUTANT_DATA[plate][row]:
                    return MUTANT_DATA[plate][row][col].get('id', None)
        
        # For drug mode, use IC50_DATA
        if args.data_mode in ['drug', 'both'] and plate in IC50_DATA:
            if well in IC50_DATA[plate]:
                info = IC50_DATA[plate][well]
                antibiotic = info.get('antibiotic', '')
                ic50_multiple = info.get('ic50_multiple', '')
                if antibiotic and ic50_multiple:
                    if ic50_multiple == 'control':
                        return 'control'
                    ic50_str = ic50_multiple if 'x' in str(ic50_multiple) else f"{ic50_multiple}x"
                    antibiotic_clean = antibiotic.replace(' ', '_')
                    return f"{antibiotic_clean}_{ic50_str}"
        
        return None

    def predict_image(model: MILEncoder, img_path: str, plate: str, batch_size: int) -> list[dict]:
        """Predict all crops for a single image using MIL attention."""
        if extraction_mode == 'raster':
            all_crops = extract_all_crops(img_path, crop_size, grid_size)
            n_positions = 1
            num_crops = len(all_crops)
        elif mil_mode:
            all_crops = extract_mil_crops(img_path, crop_size, grid_size, neighborhood)
            n_positions = len(all_crops) // num_crops_per_position
            num_crops = num_crops_per_position
        else:
            all_crops = extract_all_crops(img_path, crop_size, grid_size)
            n_positions = len(all_crops)
            num_crops = 1
        
        results: list[dict] = []
        
        if extraction_mode == 'raster':
            batch_tensors = torch.stack([c[0] for c in all_crops]).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, attn_weights = model(batch_tensors, return_attention=True)
                probs = torch.softmax(logits, dim=1)
            
            pooled_pred_idx = int(probs[0].argmax(dim=0).item())
            pooled_confidence = float(probs[0].max(dim=0).values.item())
            pooled_probs_np = probs[0].cpu().numpy().tolist()
            pooled_attn_np = attn_weights[0].cpu().numpy().tolist()
            
            well = parse_well_from_filename(img_path)
            gt_label = get_ground_truth_label(plate, well) if well else None
            gt_idx = label_to_idx.get(gt_label, -1) if gt_label else -1
            
            results.append({
                'image_path': img_path,
                'image_name': os.path.basename(img_path),
                'plate': plate,
                'well': well,
                'ground_truth_label': gt_label,
                'ground_truth_idx': gt_idx,
                'position_index': 0,
                'extraction_mode': 'raster',
                'num_crops': num_crops,
                'predicted_class_idx': pooled_pred_idx,
                'predicted_class_name': idx_to_label.get(pooled_pred_idx, 'unknown'),
                'confidence': pooled_confidence,
                'probs': pooled_probs_np,
                'attention': pooled_attn_np,
            })
        elif mil_mode:
            # Calculate center index for extracting center crop info
            center_idx = num_crops_per_position // 2
            
            for pos_idx in range(n_positions):
                pos_crops = all_crops[pos_idx * num_crops_per_position:(pos_idx + 1) * num_crops_per_position]
                batch_tensors = torch.stack([c[0] for c in pos_crops]).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits, attn_weights = model(batch_tensors, return_attention=True)
                    probs = torch.softmax(logits, dim=1)
                
                pooled_pred_idx = int(probs[0].argmax(dim=0).item())
                pooled_confidence = float(probs[0].max(dim=0).values.item())
                pooled_probs_np = probs[0].cpu().numpy().tolist()
                pooled_attn_np = attn_weights[0].cpu().numpy().tolist()
                
                center_crop = pos_crops[center_idx]
                crop_tensor, pos_id, local_row, local_col = center_crop
                
                well = parse_well_from_filename(img_path)
                gt_label = get_ground_truth_label(plate, well) if well else None
                gt_idx = label_to_idx.get(gt_label, -1) if gt_label else -1
                
                results.append({
                    'image_path': img_path,
                    'image_name': os.path.basename(img_path),
                    'plate': plate,
                    'well': well,
                    'ground_truth_label': gt_label,
                    'ground_truth_idx': gt_idx,
                    'position_index': pos_idx,
                    'neighborhood_size': neighborhood,
                    'num_crops_per_position': num_crops_per_position,
                    'local_row': local_row,
                    'local_col': local_col,
                    'predicted_class_idx': pooled_pred_idx,
                    'predicted_class_name': idx_to_label.get(pooled_pred_idx, 'unknown'),
                    'confidence': pooled_confidence,
                    'probs': pooled_probs_np,
                    'attention': pooled_attn_np,
                })
        else:
            for i in range(0, len(all_crops), batch_size):
                batch_crops = all_crops[i:i+batch_size]
                batch_tensors = torch.stack([c[0] for c in batch_crops]).to(device)
                
                with torch.no_grad():
                    logits, attn_weights = model(batch_tensors, return_attention=True)
                    probs = torch.softmax(logits, dim=1)
                    preds = probs.argmax(dim=1)
                    confidences = probs.max(dim=1).values
                
                for j, (crop_tensor, row, col, crop_idx) in enumerate(batch_crops):
                    pred_idx = int(preds[j].item())
                    confidence = float(confidences[j].item())
                    probs_np = probs[j].cpu().numpy().tolist()
                    attn_np = attn_weights[j].cpu().numpy().tolist()
                    
                    well = parse_well_from_filename(img_path)
                    gt_label = get_ground_truth_label(plate, well) if well else None
                    gt_idx = label_to_idx.get(gt_label, -1) if gt_label else -1
                    
                    results.append({
                        'image_path': img_path,
                        'image_name': os.path.basename(img_path),
                        'plate': plate,
                        'well': well,
                        'ground_truth_label': gt_label,
                        'ground_truth_idx': gt_idx,
                        'crop_index': crop_idx,
                        'grid_row': row,
                        'grid_col': col,
                        'predicted_class_idx': pred_idx,
                        'predicted_class_name': idx_to_label.get(pred_idx, 'unknown'),
                        'confidence': confidence,
                        'probs': probs_np,
                        'attention': attn_np,
                    })
        
        return results

    def compute_metrics(results: list[dict], num_classes: int) -> dict:
        """Compute metrics from crop predictions."""
        if not results:
            return {}
        
        df: pd.DataFrame = pd.DataFrame(results)
        metrics: dict = {}
        
        df_with_gt: pd.DataFrame = df[df['ground_truth_label'].notna()].copy()
        
        if len(df_with_gt) > 0:
            correct: int = int((df_with_gt['predicted_class_idx'] == df_with_gt['ground_truth_idx']).sum())
            total: int = len(df_with_gt)
            metrics['accuracy'] = correct / total
            metrics['correct'] = correct
            metrics['total_gt'] = total
            
            from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
            from sklearn.preprocessing import label_binarize
            
            y_true: np.ndarray = np.array(df_with_gt['ground_truth_idx'].tolist())
            y_pred: np.ndarray = np.array(df_with_gt['predicted_class_idx'].tolist())
            y_probs: np.ndarray = np.array(df_with_gt['probs'].tolist())
            
            results_weighted = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            
            precision_mean, recall_mean, f1_mean, _ = results_weighted
            metrics['precision'] = float(precision_mean)
            metrics['recall'] = float(recall_mean)
            metrics['f1'] = float(f1_mean)
            
            # Check if we have multiple unique classes in ground truth
            unique_classes = len(np.unique(y_true))
            
            try:
                if unique_classes > 1:
                    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
                    # Only compute ROC AUC if we have multiple classes AND probabilities
                    if y_true_bin.shape[1] > 1 and y_probs.shape[1] > 1:
                        metrics['roc_auc'] = float(roc_auc_score(y_true_bin, y_probs, average='weighted', multi_class='ovr'))
                    else:
                        metrics['roc_auc'] = None
                else:
                    metrics['roc_auc'] = None
            except Exception:
                metrics['roc_auc'] = None
            
            try:
                if unique_classes > 1:
                    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
                    if y_true_bin.shape[1] > 1 and y_probs.shape[1] > 1:
                        metrics['avg_precision'] = float(average_precision_score(y_true_bin, y_probs, average='weighted'))
                    else:
                        metrics['avg_precision'] = None
                else:
                    metrics['avg_precision'] = None
            except Exception:
                metrics['avg_precision'] = None
        else:
            metrics['accuracy'] = None
            metrics['precision'] = None
            metrics['recall'] = None
            metrics['f1'] = None
            metrics['roc_auc'] = None
            metrics['avg_precision'] = None
        
        return metrics

    # Process images - search recursively for tif/tiff files
    image_paths: list[Path] = sorted(Path(image_dir).rglob('*.tif')) + sorted(Path(image_dir).rglob('*.tiff'))
    
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    
    # Filter images by class if filter_class is specified (BEFORE sampling)
    if args.filter_class:
        filtered_paths: list[Path] = []
        for img_path in image_paths:
            well = parse_well_from_filename(str(img_path))
            label = get_ground_truth_label(test_plate, well)
            if label and args.filter_class in label:
                filtered_paths.append(img_path)
        image_paths = filtered_paths
        print(f"Filtered images by '{args.filter_class}': {len(image_paths)} images")
    
    # Sample N images per class if requested
    if args.sample_per_class:
        import random
        random.seed(args.random_seed)
        
        # Group images by their ground truth label
        images_by_label: dict[str, list[Path]] = {}
        for img_path in image_paths:
            well = parse_well_from_filename(str(img_path))
            label = get_ground_truth_label(test_plate, well)
            if label:
                if label not in images_by_label:
                    images_by_label[label] = []
                images_by_label[label].append(img_path)
        
        # Sample N per class
        sampled_paths: list[Path] = []
        for label, paths in images_by_label.items():
            if len(paths) >= args.sample_per_class:
                sampled = random.sample(paths, args.sample_per_class)
            else:
                sampled = paths  # Use all available if less than N
            sampled_paths.extend(sampled)
        
        image_paths = sampled_paths
        print(f"Sampled {len(image_paths)} images ({args.sample_per_class} per class)")
    
    print(f"Processing {len(image_paths)} images...")
    
    all_results: list[dict] = []
    
    for img_path in tqdm(image_paths, desc=f"Predicting"):
        img_path_str: str = str(img_path)
        results: list[dict] = predict_image(model, img_path_str, test_plate, args.batch_size)
        all_results.extend(results)
    
    # Filter by class pattern if requested
    if args.filter_class:
        filtered_results = []
        for r in all_results:
            gt_label = r.get('ground_truth_label', '')
            if gt_label and args.filter_class in gt_label:
                filtered_results.append(r)
        print(f"\nFiltered by '{args.filter_class}': {len(filtered_results)}/{len(all_results)} results")
        all_results = filtered_results
    
    metrics: dict = compute_metrics(all_results, num_classes)
    
    print(f"\nPrediction summary:")
    print(f"  - Images processed: {len(image_paths)}")
    print(f"  - Crops per position: {num_crops_per_position} ({neighborhood}x{neighborhood})")
    print(f"  - Total crop predictions: {len(all_results)}")
    
    # Calculate accuracy by position (1 prediction per position)
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Position-level accuracy (1 prediction per grid position)
        pos_accuracy = None
        if 'ground_truth_idx' in df.columns and 'predicted_class_idx' in df.columns:
            df_with_gt = df[df['ground_truth_idx'] != -1]
            if len(df_with_gt) > 0:
                correct = (df_with_gt['predicted_class_idx'] == df_with_gt['ground_truth_idx']).sum()
                total = len(df_with_gt)
                pos_accuracy = correct / total
                print(f"\nPosition-level accuracy: {pos_accuracy:.4f} ({correct}/{total})")
        
        # Majority voting accuracy (1 vote per image)
        if 'image_path' in df.columns and 'ground_truth_label' in df.columns:
            image_votes = {}
            for _, row in df.iterrows():
                img = row['image_path']
                if img not in image_votes:
                    image_votes[img] = {}
                pred = row['predicted_class_name']
                image_votes[img][pred] = image_votes[img].get(pred, 0) + 1
            
            correct_images = 0
            total_images = 0
            print(f"\nMajority voting per image:")
            for img, votes in image_votes.items():
                majority_pred = max(votes, key=votes.get)
                well = parse_well_from_filename(img)
                gt = get_ground_truth_label(test_plate, well)
                if gt:
                    total_images += 1
                    if majority_pred == gt:
                        correct_images += 1
                    print(f"  {os.path.basename(img)[:30]}: GT={gt}, Pred={majority_pred} ({'✓' if majority_pred == gt else '✗'})")
            
            if total_images > 0:
                maj_accuracy = correct_images / total_images
                print(f"\nMajority voting accuracy: {maj_accuracy:.4f} ({correct_images}/{total_images})")
    
    if args.dry_run:
        print(f"\n[DRY RUN] Skipping save and metrics")
        print(f"\n{'='*60}")
        print(f"Done! Fold {test_plate} (dry run - no files saved)")
        print(f"{'='*60}")
        return
    
    df: pd.DataFrame = pd.DataFrame(all_results)
    
    if df.empty or 'image_name' not in df.columns:
        print('ERROR: No valid predictions. Skipping per-image aggregation.')
        print('Results were empty.')
        return
    
    output_csv: str = os.path.join(output_dir, f'predictions_all_crops_mil.csv')
    checkpoint_name = os.path.basename(args.checkpoint).replace('.pth', '')
    if mil_mode:
        suffix = f"_{checkpoint_name}_n{neighborhood}"
        if args.use_sc_mil:
            suffix += "_scmil"
        output_csv = os.path.join(output_dir, f'predictions_all_crops_mil{suffix}.csv')
    else:
        output_csv = os.path.join(output_dir, f'predictions_all_crops{suffix}.csv')
    df.to_csv(output_csv, index=False)
    print(f"\nSaved predictions to {output_csv}")
    
    print(f"\nMetrics:")
    if metrics.get('accuracy') is not None:
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total_gt']})")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        if metrics.get('roc_auc') is not None:
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        if metrics.get('avg_precision') is not None:
            print(f"  Avg Precision: {metrics['avg_precision']:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Done! Fold {test_plate}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()