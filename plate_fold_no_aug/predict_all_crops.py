#!/usr/bin/env python3
"""
Predict all 144 crops for each test image in final_crispr_model experiment.
Uses MIL model with attention pooling for prediction.
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
    parser = argparse.ArgumentParser(description='Predict all crops for final_crispr_model')
    parser.add_argument('--fold', type=str, default=None,
                        help='Fold to predict (e.g., P6). If not specified, uses P6.')
    parser.add_argument('--crop_size', type=int, default=224, help='Crop size (default: 224)')
    parser.add_argument('--grid_size', type=int, default=12, help='Grid size (default: 12)')
    parser.add_argument('--mil_mode', action='store_true',
                        help='Use MIL mode: 100 positions with 9 crops each')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--checkpoint', type=str, default='best_model_auc.pth',
                        help='Checkpoint filename to use (best_model_auc.pth, best_model_acc.pth, best_model_loss.pth)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to parent folder containing P1-P6 (default: parent of script dir)')
    
    args: argparse.Namespace = parser.parse_args()
    
    with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json'), 'r') as f:
        PLATE_WELL_ID: dict = json.load(f)
    
    # Set BASE_DIR based on data_root argument
    if args.data_root:
        BASE_DIR: str = args.data_root
    else:
        BASE_DIR: str = os.path.dirname(SCRIPT_DIR)
    
    crop_size: int = args.crop_size
    grid_size: int = args.grid_size
    mil_mode: bool = args.mil_mode
    
    print(f"Config: crop_size={crop_size}, grid_size={grid_size}, mil_mode={mil_mode}")
    
    classes: dict[int, str] = {}
    with open(os.path.join(SCRIPT_DIR, 'classes.txt'), 'r') as f:
        for line in f:
            idx, name = line.strip().split(',', 1)
            classes[int(idx)] = name

    idx_to_label: dict[int, str] = classes
    label_to_idx: dict[str, int] = {v: k for k, v in classes.items()}
    
    num_classes: int = args.num_classes if args.num_classes is not None else len(classes)
    print(f"Loaded {num_classes} classes")

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Import the MIL model
    from mil_model import AttentionMILModel
    
# Default fold is P6 if not specified
    test_plate: str = args.fold if args.fold else 'P6'
    fold_dir: str = os.path.join(SCRIPT_DIR, f'fold_{test_plate}')
    checkpoint_path: str = os.path.join(fold_dir, args.checkpoint)
    image_dir: str = os.path.join(BASE_DIR, test_plate)
    output_dir: str = fold_dir

    print(f'\n{'='*60}')
    print(f'Processing fold: test plate={test_plate}')
    print(f'  checkpoint: {checkpoint_path}')
    print(f'  image_dir: {image_dir}')
    print(f'  mil_mode: {mil_mode}')
    print(f'{'='*60}')
    
    if not os.path.exists(checkpoint_path):
        print(f'ERROR: Checkpoint not found: {checkpoint_path}')
        print(f'Available checkpoints in {fold_dir}:')
        for f in os.listdir(fold_dir):
            if f.endswith('.pth'):
                print(f'  - {f}')
        return
    
    # Load model
    model: AttentionMILModel = AttentionMILModel(num_classes=num_classes, num_heads=4, attention_temp=0.5)
    model = model.to(device)
    
    checkpoint: dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded successfully")

    def extract_all_crops(img_path: str, crop_size: int, grid_size: int) -> list[tuple[torch.Tensor, int, int, int]]:
        """Extract all crops from an image in deterministic order."""
        img: Image.Image = Image.open(img_path).convert('RGB')
        w: int
        h: int
        w, h = img.size
        
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
            
            crop: Image.Image = img.crop((left, top, left + crop_size, top + crop_size))
            crop_np: np.ndarray = np.array(crop).astype(np.float32) / 255.0
            crop_np = np.transpose(crop_np, (2, 0, 1))
            mean: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
            std: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
            crop_np = (crop_np - mean) / std
            crop_tensor: torch.Tensor = torch.from_numpy(crop_np).float()
            
            crops.append((crop_tensor, row, col, idx))
        
        return crops

    def extract_mil_crops(img_path: str, crop_size: int, grid_size: int) -> list[tuple[torch.Tensor, int, int, int]]:
        """Extract 100 positions (with 9 crops each) matching training: center + 3x3 neighborhood."""
        img: Image.Image = Image.open(img_path).convert('RGB')
        w: int
        h: int
        w, h = img.size
        
        stride_x: int = (w - crop_size) // (grid_size - 1) if grid_size > 1 else 0
        stride_y: int = (h - crop_size) // (grid_size - 1) if grid_size > 1 else 0
        
        valid_positions: list[tuple[int, int]] = []
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * stride_x
                top = i * stride_y
                if left + crop_size <= w and top + crop_size <= h:
                    can_left = left - stride_x >= 0
                    can_right = left + stride_x + crop_size <= w
                    can_top = top - stride_y >= 0
                    can_bottom = top + stride_y + crop_size <= h
                    if can_left and can_right and can_top and can_bottom:
                        valid_positions.append((left, top))
        
        crops: list[tuple[torch.Tensor, int, int, int]] = []
        
        for pos_idx, (center_x, center_y) in enumerate(valid_positions):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    left = center_x + dx * stride_x
                    top = center_y + dy * stride_y
                    
                    crop: Image.Image = img.crop((left, top, left + crop_size, top + crop_size))
                    crop_np: np.ndarray = np.array(crop).astype(np.float32) / 255.0
                    crop_np = np.transpose(crop_np, (2, 0, 1))
                    mean: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
                    std: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
                    crop_np = (crop_np - mean) / std
                    crop_tensor: torch.Tensor = torch.from_numpy(crop_np).float()
                    
                    crops.append((crop_tensor, pos_idx, dy + 1, dx + 1))
        
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
                    col: str = well_str[1:]
                    col = str(int(col))
                    return row + col
                return well_str
        return None

    def get_ground_truth_label(plate: str, well: Optional[str]) -> Optional[str]:
        """Get ground truth label from plate_well_id_path.json."""
        if plate in PLATE_WELL_ID and well:
            row: str = well[0]
            col: str = well[1:]
            if row in PLATE_WELL_ID[plate]:
                if col in PLATE_WELL_ID[plate][row]:
                    return PLATE_WELL_ID[plate][row][col].get('id', None)
        return None

    def predict_image(model: AttentionMILModel, img_path: str, plate: str, batch_size: int) -> list[dict]:
        """Predict all crops for a single image using MIL attention."""
        if 'A01' in img_path:
            print(f"DEBUG: mil_mode={mil_mode}")
        
        if mil_mode:
            all_crops = extract_mil_crops(img_path, crop_size, grid_size)
            n_positions = len(all_crops) // 9
        else:
            all_crops = extract_all_crops(img_path, crop_size, grid_size)
            n_positions = len(all_crops)
        
        results: list[dict] = []
        
        if mil_mode:
            for pos_idx in range(n_positions):
                pos_crops = all_crops[pos_idx * 9:(pos_idx + 1) * 9]
                batch_tensors = torch.stack([c[0] for c in pos_crops]).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits, attn_weights = model(batch_tensors, return_attention=True)
                    probs = torch.softmax(logits, dim=1)
                
                pooled_pred_idx = int(probs[0].argmax(dim=0).item())
                pooled_confidence = float(probs[0].max(dim=0).values.item())
                pooled_probs_np = probs[0].cpu().numpy().tolist()
                pooled_attn_np = attn_weights[0].cpu().numpy().tolist()
                
                center_crop = pos_crops[4]
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
                y_true, y_pred, average='weighted', zero_division='warn'
            )
            
            precision_mean, recall_mean, f1_mean, _ = results_weighted
            metrics['precision'] = float(precision_mean)
            metrics['recall'] = float(recall_mean)
            metrics['f1'] = float(f1_mean)
            
            try:
                y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
                metrics['roc_auc'] = float(roc_auc_score(y_true_bin, y_probs, average='weighted', multi_class='ovr'))
            except Exception:
                metrics['roc_auc'] = None
            
            try:
                metrics['avg_precision'] = float(average_precision_score(y_true_bin, y_probs, average='weighted'))
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

    # Process images
    image_paths: list[Path] = sorted(Path(image_dir).glob('*.tif'))
    
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    
    print(f"Processing {len(image_paths)} images...")
    
    all_results: list[dict] = []
    
    for img_path in tqdm(image_paths, desc=f"Predicting"):
        img_path_str: str = str(img_path)
        results: list[dict] = predict_image(model, img_path_str, test_plate, args.batch_size)
        all_results.extend(results)
    
    metrics: dict = compute_metrics(all_results, num_classes)
    
    print(f"\nPrediction summary:")
    print(f"  - Images processed: {len(image_paths)}")
    print(f"  - Crops per image: {len(all_results) // len(image_paths) if image_paths else 0}")
    print(f"  - Total crop predictions: {len(all_results)}")
    
    df: pd.DataFrame = pd.DataFrame(all_results)
    
    if df.empty or 'image_name' not in df.columns:
        print('ERROR: No valid predictions. Skipping per-image aggregation.')
        print('Results were empty.')
        return
    
    output_csv: str = os.path.join(output_dir, f'predictions_all_crops_mil.csv')
    # Include checkpoint name in filename
    checkpoint_name = args.checkpoint.replace('.pth', '')
    if mil_mode:
        output_csv = os.path.join(output_dir, f'predictions_all_crops_mil_{checkpoint_name}.csv')
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
    
    print(f"\n{'='*60}")
    print(f"Done! Fold {test_plate}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()