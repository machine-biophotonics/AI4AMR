#!/usr/bin/env python3
# Must be set before any imports to suppress inductor SM warning
import warnings
warnings.filterwarnings("ignore", message=".*Not enough SMs to use max_autotune_gem.*")

import os
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "ATEN,CPP"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCH_CUDNN_DETERMINISTIC"] = "1"

"""
Drug prediction script - rewritten to match training pipeline exactly.
Uses exact imports and setup from train_mil.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize


def get_project_root():
    """Get the project root directory."""
    # This file is in final_mutant_model/, so parent gets CRISPRi Reference Plate Imaging/
    return Path(__file__).resolve().parent


def get_data_path():
    """Get the trial_daniel/data path."""
    project_root = get_project_root()
    # trial_daniel/data is at same level as final_mutant_model/
    trial_dir = project_root.parent / "trial_daniel"
    return trial_dir / "data"


def get_script_dir():
    """Get the final_mutant_model directory."""
    return get_project_root()


def load_classes(plate, class_merge=False):
    """Load classes from plate folder - same as training."""
    data_path = get_data_path()
    plate_dir = data_path / plate
    
    if not plate_dir.exists():
        raise FileNotFoundError(f"Plate directory not found: {plate_dir}")
    
    unique_labels = set()
    for folder in os.listdir(plate_dir):
        folder_path = plate_dir / folder
        if not folder_path.is_dir():
            continue
        
        label = folder
        if class_merge:
            if folder == 'DMSO_control':
                label = 'Control'
            else:
                for dose in ['_0.25x', '_0.5x', '_1x', '_2x']:
                    if folder.endswith(dose):
                        label = folder[:-len(dose)]
                        break
        
        unique_labels.add(label)
    
    unique_labels = sorted(unique_labels)
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_label = {i: label for i, label in enumerate(unique_labels)}
    
    return idx_to_label, label_to_idx


def get_image_paths(plate):
    """Get all image paths for a plate - recursively search for .tif files."""
    data_path = get_data_path()
    plate_dir = data_path / plate
    
    if not plate_dir.exists():
        raise FileNotFoundError(f"Plate directory not found: {plate_dir}")
    
    # Recursively find all .tif/.tiff files
    tif_files = list(plate_dir.rglob("*.tif"))
    tiff_files = list(plate_dir.rglob("*.tiff"))
    
    all_files = sorted(set(tif_files + tiff_files))
    return all_files


def get_ground_truth(img_path, label_to_idx, class_merge=False):
    """Extract ground truth from parent folder - same as training."""
    parent_folder = os.path.basename(os.path.dirname(img_path))
    
    label = parent_folder
    if class_merge:
        if parent_folder == 'DMSO_control':
            label = 'Control'
        else:
            for dose in ['_0.25x', '_0.5x', '_1x', '_2x']:
                if parent_folder.endswith(dose):
                    label = parent_folder[:-len(dose)]
                    break
    
    if label in label_to_idx:
        return label_to_idx[label], label
    return None, None


def load_image_array(img_path):
    """Load image as numpy array - same as training (mil_model.py)."""
    # Import tifffile in the function to mirror how training does it
    try:
        import tifffile
        HAS_TIFFFILE = True
    except ImportError:
        HAS_TIFFFILE = False
    
    if HAS_TIFFFILE and (img_path.lower().endswith('.tif') or img_path.lower().endswith('.tiff')):
        try:
            img = tifffile.imread(img_path)
            if img is not None and len(img.shape) > 0:
                # Handle single channel - take first if multi-channel
                if len(img.shape) == 3:
                    img = img[0] if img.shape[0] < 10 else img[:, :, 0]
                elif len(img.shape) == 2:
                    pass  # Already 2D
                return img
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    # Fallback to PIL (for non-TIFF)
    from PIL import Image
    img = Image.open(img_path)
    if img.mode != 'L':
        img = img.convert('L')
    return np.array(img)


def extract_crops_with_neighborhood(img_array, center_x, center_y, crop_size, stride, neighborhood):
    """
    Extract n x n neighborhood crops around a center position.
    Same logic as training (MultiCropDataset).
    """
    half_n = neighborhood // 2
    crops = []
    
    for dy in range(-half_n, half_n + 1):
        for dx in range(-half_n, half_n + 1):
            left = center_x + dx * stride
            top = center_y + dy * stride
            
            crop = img_array[top:top+crop_size, left:left+crop_size]
            crops.append(crop)
    
    return crops


def process_plate(plate, checkpoint_path, args):
    """Main prediction pipeline - matching training exactly."""
    
    print(f"\n{'='*60}")
    print(f"Processing plate: {plate}")
    print(f"{'='*60}")
    
    # Get project root
    project_root = get_project_root()
    script_dir = project_root / "final_mutant_model"
    
    # Load classes
    idx_to_label, label_to_idx = load_classes(plate, args.class_merge)
    num_classes = len(idx_to_label)
    print(f"Loaded {num_classes} classes")
    
    # Get image paths
    image_paths = get_image_paths(plate)
    print(f"Found {len(image_paths)} images")
    
    # Shuffle images for diverse sampling
    np.random.seed(42)
    indices = np.random.permutation(len(image_paths))
    image_paths = [image_paths[i] for i in indices]
    
    if len(image_paths) == 0:
        print("ERROR: No images found!")
        return
    
    test_img = load_image_array(str(image_paths[0]))
    h, w = test_img.shape
    print(f"Image size: {w} x {h}")
    
    # Calculate grid - same as training
    grid_size = args.grid_size
    crop_size = args.crop_size
    neighborhood = args.neighborhood
    
    stride = (w - crop_size) // (grid_size - 1)
    print(f"Grid: {grid_size}x{grid_size}, stride: {stride}, crop: {crop_size}, neighborhood: {neighborhood}")
    
    # Calculate valid positions - same logic as training
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
    
    print(f"Valid positions: {len(valid_positions)}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    sys.path.insert(0, str(script_dir))
    from mil_model import MILEncoder
    
    model = MILEncoder(
        num_classes=num_classes,
        num_heads=args.num_heads,
        dropout=0.0,  # No dropout for inference
        num_channels=1,  # Single channel for drug data
        use_contrastive=True  # Match training
    )
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f"Model loaded!")
    
    # Process images
    results = []
    normalize_value = 0.5
    
    max_images = args.max_images if args.max_images else len(image_paths)
    images_to_process = image_paths[:max_images]
    
    print(f"\nProcessing {len(images_to_process)} images...")
    
    for img_path in tqdm(images_to_process, desc="Predicting"):
        img_path = str(img_path)
        
        # Get ground truth
        gt_idx, gt_label = get_ground_truth(img_path, label_to_idx, args.class_merge)
        
        if gt_idx is None:
            # Debug: show what's happening
            parent = os.path.basename(os.path.dirname(img_path))
            print(f"DEBUG: Could not find GT for parent folder: {parent}")
            print(f"DEBUG: Available labels sample: {list(label_to_idx.keys())[:10]}")
        
        # Load image
        img_array = load_image_array(img_path)
        
        # Process each valid position
        for pos_idx, (center_x, center_y) in enumerate(valid_positions):
            # Extract neighborhood crops
            crops = extract_crops_with_neighborhood(
                img_array, center_x, center_y, 
                crop_size, stride, neighborhood
            )
            
            # Normalize each crop - same as training!
            normalized_crops = []
            for crop in crops:
                crop_float = crop.astype(np.float32)
                # Normalize: 16-bit to [0,1] then to [−1,1]
                crop_float = crop_float / 65535.0
                crop_float = (crop_float - normalize_value) / normalize_value
                # Add channel dimension
                crop_float = np.expand_dims(crop_float, axis=0)
                normalized_crops.append(crop_float)
            
            # Stack into batch
            batch = torch.from_numpy(np.stack(normalized_crops)).float()
            batch = batch.unsqueeze(0).to(device)  # Add batch dimension
            
            # Forward pass
            with torch.no_grad():
                logits, attn_weights = model(batch, return_attention=True)
                probs = torch.softmax(logits, dim=1)
            
            # Get prediction
            pred_idx = int(probs[0].argmax())
            pred_conf = float(probs[0].max())
            pred_probs = probs[0].cpu().numpy().tolist()
            
            results.append({
                'image_path': img_path,
                'ground_truth_label': gt_label,
                'ground_truth_idx': gt_idx,
                'position_index': pos_idx,
                'predicted_class_idx': pred_idx,
                'predicted_class_name': idx_to_label.get(pred_idx, 'unknown'),
                'confidence': pred_conf,
                'probs': pred_probs,
            })
    
    # Compute metrics
    df = pd.DataFrame(results)
    
    print(f"\n{'='*60}")
    print(f"Prediction Results")
    print(f"{'='*60}")
    
    # Filter to rows with ground truth
    df_with_gt = df[df['ground_truth_idx'].notna()].copy()
    
    if len(df_with_gt) > 0:
        correct = (df_with_gt['predicted_class_idx'] == df_with_gt['ground_truth_idx']).sum()
        total = len(df_with_gt)
        accuracy = correct / total
        
        print(f"\nMetrics:")
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        # More metrics
        y_true = df_with_gt['ground_truth_idx'].astype(int).tolist()
        y_pred = df_with_gt['predicted_class_idx'].astype(int).tolist()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
        
        # ROC-AUC
        try:
            y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
            y_probs = np.array(df_with_gt['probs'].tolist())
            auc = roc_auc_score(y_true_bin, y_probs, average='weighted', multi_class='ovr')
            print(f"  ROC-AUC: {auc:.4f}")
        except Exception as e:
            print(f"  ROC-AUC: Error computing - {e}")
    else:
        print("WARNING: No ground truth found for any image!")
        print(f"  Sample paths:")
        for p in image_paths[:3]:
            gt_idx, gt_label = get_ground_truth(str(p), label_to_idx, args.class_merge)
            print(f"    {p}: gt_idx={gt_idx}, gt_label={gt_label}")
    
    # Save results
    output_dir = script_dir / "fold_Plate_6"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"predictions_{plate}_drug.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved predictions to: {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Drug prediction - matching training pipeline')
    
    parser.add_argument('--plate', type=str, default='Plate_6',
                        help='Test plate (e.g., Plate_6)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path (default: fold_<plate>/best_model_auc.pth)')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='Crop size (default: 224)')
    parser.add_argument('--grid_size', type=int, default=12,
                        help='Grid size (default: 12)')
    parser.add_argument('--neighborhood', type=int, default=5,
                        help='Neighborhood size (default: 5)')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Attention heads (default: 4)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Max images to process')
    parser.add_argument('--class_merge', action='store_true',
                        help='Merge classes by antibiotic')
    
    args = parser.parse_args()
    
    # Get project paths
    project_root = get_project_root()
    script_dir = get_script_dir()
    plate = args.plate
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint) if isinstance(args.checkpoint, str) else args.checkpoint
    else:
        checkpoint_path = script_dir / f"fold_{plate}" / "best_model_auc.pth"
    
    # Convert to Path object if string
    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)
    
    print(f"Project root: {project_root}")
    print(f"Script dir: {script_dir}")
    print(f"Plate: {plate}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data path: {get_data_path()}")
    
    # Check checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return
    
    # Run prediction
    process_plate(plate, checkpoint_path, args)


if __name__ == '__main__':
    main()