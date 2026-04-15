#!/usr/bin/env python3
"""
Predict all crops for each test image with configurable crop_size and grid_size.
Saves results to CSV with embeddings, logits, probabilities, well position, and metrics.

Supports predicting on all 6 folds (each fold has a model trained on 4 plates, tested on 1).
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Any, Optional

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path


SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
BASE_DIR: str = os.path.dirname(SCRIPT_DIR)

# Load label mapping
with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json'), 'r') as f:
    PLATE_WELL_ID: dict = json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description='Predict crops for test images')
    parser.add_argument('--fold', type=str, default=None, 
                        help='Fold to predict (P1-P6). If not specified, predicts on all folds.')
    parser.add_argument('--crop_size', type=int, default=224, help='Crop size (default: 224)')
    parser.add_argument('--grid_size', type=int, default=12, help='Grid size (default: 12)')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes (default: auto-detect from classes.txt)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process per fold (default: all)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    
    args: argparse.Namespace = parser.parse_args()
    
    crop_size: int = args.crop_size
    grid_size: int = args.grid_size
    crops_per_image: int = grid_size * grid_size
    
    print(f"Config: crop_size={crop_size}, grid_size={grid_size}, crops_per_image={crops_per_image}")
    
    # Load class mappings
    classes: dict[int, str] = {}
    with open(os.path.join(SCRIPT_DIR, 'classes.txt'), 'r') as f:
        for line in f:
            idx, name = line.strip().split(',', 1)
            classes[int(idx)] = name

    idx_to_label: dict[int, str] = classes
    label_to_idx: dict[str, int] = {v: k for k, v in classes.items()}
    
    # Allow override of num_classes for excluded classes
    num_classes: int
    if args.num_classes is not None:
        num_classes = args.num_classes
    else:
        num_classes = len(classes)

    print(f"Loaded {num_classes} classes")

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine which folds to run
    folds_to_run: list[str]
    if args.fold:
        folds_to_run = [args.fold]
    else:
        folds_to_run = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']

    class EfficientNetClassifier(nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            import torchvision
            self.features = torchvision.models.efficientnet_b0(weights=None).features
            in_features: int = 1280
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features, num_classes)
            )
        
        def forward(self, x: torch.Tensor, return_embedding: bool = False) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
            if return_embedding:
                x = self.features(x)
                x = self.avgpool(x)
                embedding: torch.Tensor = torch.flatten(x, 1)
                logits: torch.Tensor = self.classifier(embedding)
                return embedding, logits
            else:
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x

    def extract_all_crops(img_path: str, crop_size: int, grid_size: int) -> list[tuple[torch.Tensor, int, int, int]]:
        """Extract all crops from an image in deterministic order."""
        img: Image.Image = Image.open(img_path).convert('RGB')
        w: int
        h: int
        w, h = img.size
        
        # Handle grid_size=1 case
        if grid_size == 1:
            stride: int = 0
            left: int = (w - crop_size) // 2
            top: int = (h - crop_size) // 2
            positions: list[tuple[int, int]] = [(left, top)]
        else:
            stride = (w - crop_size) // (grid_size - 1)
            positions = []
            for row in range(grid_size):
                for col in range(grid_size):
                    left = col * stride
                    top = row * stride
                    if left + crop_size <= w and top + crop_size <= h:
                        positions.append((left, top))

        crops: list[tuple[torch.Tensor, int, int, int]] = []
        crop_idx: int = 0
        
        for row in range(grid_size):
            for col in range(grid_size):
                if crop_idx >= len(positions):
                    break
                left, top = positions[crop_idx]
                
                crop: Image.Image = img.crop((left, top, left + crop_size, top + crop_size))
                
                crop_np: np.ndarray = np.array(crop).astype(np.float32) / 255.0
                crop_np = np.transpose(crop_np, (2, 0, 1))
                mean: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
                std: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
                crop_np = (crop_np - mean) / std
                crop_tensor: torch.Tensor = torch.from_numpy(crop_np).float()
                
                crops.append((crop_tensor, row, col, crop_idx))
                crop_idx += 1
        
        return crops

    def parse_well_from_filename(img_path: str, plate: str) -> Optional[str]:
        """Parse well position (e.g., A1, A01) from image filename."""
        filename: str = os.path.basename(img_path)
        # Format: WellA01_PointA01_0000_... or similar
        # Extract WellA01 -> A01
        parts: list[str] = filename.split('_')
        for part in parts:
            if part.startswith('Well'):
                well_str: str = part.replace('Well', '')  # e.g., "A01"
                # Convert to standard format: A01 -> A1
                if len(well_str) == 3:
                    row: str = well_str[0]
                    col: str = well_str[1:]  # "01" -> "1" for single digit
                    col = str(int(col))  # "01" -> 1 -> "1"
                    return row + col
                return well_str
        return None

    def get_ground_truth_label(plate: str, well: Optional[str]) -> Optional[str]:
        """Get ground truth label from plate_well_id_path.json."""
        if plate in PLATE_WELL_ID and well:
            # well format: "A1" -> row="A", col="1"
            row: str = well[0]
            col: str = well[1:]
            if row in PLATE_WELL_ID[plate]:
                if col in PLATE_WELL_ID[plate][row]:
                    return PLATE_WELL_ID[plate][row][col].get('id', None)
        return None

    def predict_image(model: EfficientNetClassifier, img_path: str, plate: str, batch_size: int) -> list[dict]:
        """Predict all crops for a single image, returning embeddings, logits, probs."""
        crops: list[tuple[torch.Tensor, int, int, int]] = extract_all_crops(img_path, crop_size, grid_size)
        
        results: list[dict] = []
        
        for i in range(0, len(crops), batch_size):
            batch_crops: list[tuple[torch.Tensor, int, int, int]] = crops[i:i+batch_size]
            batch_tensors: torch.Tensor = torch.stack([c[0] for c in batch_crops]).to(device)
            
            with torch.no_grad():
                embeddings: torch.Tensor
                logits: torch.Tensor
                embeddings, logits = model(batch_tensors, return_embedding=True)
                probs: torch.Tensor = torch.softmax(logits, dim=1)
                preds: torch.Tensor = probs.argmax(dim=1)
                confidences: torch.Tensor = probs.max(dim=1).values
            
            for j, (crop_tensor, row, col, crop_idx) in enumerate(batch_crops):
                pred_idx = int(preds[j].item())
                confidence = float(confidences[j].item())
                embedding = embeddings[j].cpu().numpy().tolist()
                logits_np = logits[j].cpu().numpy().tolist()
                probs_np = probs[j].cpu().numpy().tolist()
                
                # Get well position and ground truth
                well: Optional[str] = parse_well_from_filename(img_path, plate)
                gt_label: Optional[str] = get_ground_truth_label(plate, well) if well else None
                gt_idx: int = label_to_idx.get(gt_label, -1) if gt_label else -1
                
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
                    'predicted_class_name': idx_to_label[pred_idx],
                    'confidence': confidence,
                    'embedding': embedding,
                    'logits': logits_np,
                    'probs': probs_np,
                })
        
        return results

    def compute_metrics(results: list[dict], num_classes: int, idx_to_label: dict[int, str]) -> tuple[dict, pd.DataFrame]:
        """Compute per-sample and overall metrics. Returns metrics dict and per-class DataFrame."""
        if not results:
            return {}, pd.DataFrame()
        
        df: pd.DataFrame = pd.DataFrame(results)
        
        metrics: dict = {}
        per_class_df: pd.DataFrame = pd.DataFrame()
        
        # Per-sample metrics (only for samples with ground truth)
        df_with_gt: pd.DataFrame = df[df['ground_truth_label'].notna()].copy()  # type: ignore[assignment]
        
        if len(df_with_gt) > 0:
            # Accuracy
            correct: int = int((df_with_gt['predicted_class_idx'] == df_with_gt['ground_truth_idx']).sum())  # type: ignore[arg-type]
            total: int = len(df_with_gt)
            metrics['accuracy'] = correct / total
            metrics['correct'] = correct
            metrics['total_gt'] = total
            
            # Per-class metrics
            from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
            from sklearn.preprocessing import label_binarize
            
            y_true: np.ndarray = np.array(df_with_gt['ground_truth_idx'].tolist())  # type: ignore[arg-type]
            y_pred: np.ndarray = np.array(df_with_gt['predicted_class_idx'].tolist())  # type: ignore[arg-type]
            y_probs: np.ndarray = np.array(df_with_gt['probs'].tolist())
            
            # Precision, Recall, F1, Support per class
            results_per_class: Any = precision_recall_fscore_support(
                y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division='warn'
            )
            precision_arr, recall_arr, f1_arr, support_arr = results_per_class
            
            # Build per-class DataFrame
            per_class_data: list[dict] = []
            for i in range(num_classes):
                class_name: str = idx_to_label.get(i, f"class_{i}")
                per_class_data.append({
                    'class_idx': i,
                    'class_name': class_name,
                    'precision': float(precision_arr[i]),
                    'recall': float(recall_arr[i]),
                    'f1_score': float(f1_arr[i]),
                    'support': int(support_arr[i])
                })
            per_class_df = pd.DataFrame(per_class_data)
            
            # Weighted averages
            results_weighted: Any = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division='warn'
            )
            precision_mean, recall_mean, f1_mean, _ = results_weighted
            metrics['precision'] = float(precision_mean)
            metrics['recall'] = float(recall_mean)
            metrics['f1'] = float(f1_mean)
            
            # ROC-AUC (one-vs-rest)
            y_true_bin: Optional[Any] = None
            try:
                y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
                metrics['roc_auc'] = float(roc_auc_score(y_true_bin, y_probs, average='weighted', multi_class='ovr'))
            except Exception:
                metrics['roc_auc'] = None
            
            # Average Precision
            try:
                if y_true_bin is not None:
                    metrics['avg_precision'] = float(average_precision_score(y_true_bin, y_probs, average='weighted'))
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
        
        return metrics, per_class_df

    # Process each fold
    all_fold_metrics: list[dict] = []
    
    for fold in folds_to_run:
        fold_dir: str = os.path.join(SCRIPT_DIR, f'fold_{fold}')
        test_plate: str = fold  # fold_P1 -> test on P1
        checkpoint_path: str = os.path.join(fold_dir, 'best_model.pth')
        image_dir: str = os.path.join(BASE_DIR, test_plate)
        output_csv: str = os.path.join(fold_dir, 'predictions.csv')
        
        print(f"\n{'='*60}")
        print(f"Processing fold {fold}: test plate={test_plate}")
        print(f"  checkpoint: {checkpoint_path}")
        print(f"  image_dir: {image_dir}")
        print(f"  output: {output_csv}")
        print(f"{'='*60}")
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"  WARNING: Checkpoint not found, skipping fold {fold}")
            continue
        
        # Load model
        model: EfficientNetClassifier = EfficientNetClassifier(num_classes=num_classes)
        model = model.to(device)
        
        checkpoint: dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"  Model loaded successfully")
        
        # Get all tif files
        image_paths: list[Path] = sorted(Path(image_dir).glob('*.tif'))
        
        if args.max_images:
            image_paths = image_paths[:args.max_images]
        
        print(f"  Processing {len(image_paths)} images...")
        
        all_results: list[dict] = []
        
        for img_path in tqdm(image_paths, desc=f"Fold {fold}"):
            img_path_str: str = str(img_path)
            results: list[dict] = predict_image(model, img_path_str, test_plate, args.batch_size)
            all_results.extend(results)
        
        # Compute metrics
        metrics: dict
        per_class_df: pd.DataFrame
        metrics, per_class_df = compute_metrics(all_results, num_classes, idx_to_label)
        metrics['fold'] = fold
        metrics['test_plate'] = test_plate
        all_fold_metrics.append(metrics)
        
        # Save per-class metrics
        per_class_csv: str = os.path.join(fold_dir, 'per_class_metrics.csv')
        per_class_df.to_csv(per_class_csv, index=False)
        print(f"  Saved per-class metrics to {per_class_csv}")
        
        print(f"  Metrics: {metrics}")
        
        # Convert to DataFrame and save
        df: pd.DataFrame = pd.DataFrame(all_results)
        
        # Save full predictions with embeddings, logits, probs
        df.to_csv(output_csv, index=False)
        print(f"  Saved {len(all_results)} crop predictions to {output_csv}")
        
        # Save embeddings and labels as numpy for t-SNE/UMAP (fast)
        embed_npz: str = os.path.join(fold_dir, 'embeddings.npz')
        
        embeddings_arr: np.ndarray = np.array(df['embedding'].tolist())
        labels_arr: np.ndarray = df['ground_truth_label'].values
        plates_arr: np.ndarray = df['plate'].values
        wells_arr: np.ndarray = df['well'].values
        
        np.savez(embed_npz, 
                 embeddings=embeddings_arr, 
                 labels=labels_arr,
                 plates=plates_arr,
                 wells=wells_arr)
        print(f"  Saved embeddings to {embed_npz} (embeddings: {embeddings_arr.shape})")
        
        print(f"  - Images processed: {len(image_paths)}")
        print(f"  - Crops per image: {crops_per_image} ({grid_size}x{grid_size} grid)")
        print(f"  - Total crops: {len(all_results)}")
        
        print(f"  Prediction summary:")
        print(f"  - Unique predicted classes: {df['predicted_class_name'].nunique()}")
        print(f"  - Average confidence: {df['confidence'].mean():.4f}")
        
        if metrics.get('accuracy') is not None:
            print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total_gt']})")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            if metrics.get('roc_auc') is not None:
                print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            if metrics.get('avg_precision') is not None:
                print(f"  Avg Precision: {metrics['avg_precision']:.4f}")
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()

    # Save overall metrics
    metrics_df: pd.DataFrame = pd.DataFrame(all_fold_metrics)
    metrics_csv: str = os.path.join(SCRIPT_DIR, 'fold_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\nSaved fold metrics to {metrics_csv}")
    
    print(f"\n{'='*60}")
    print(f"All folds processed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()