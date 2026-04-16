#!/usr/bin/env python3
"""
Predict all 144 crops for each test image in plate_fold_diversity_new experiment.
Tests how accuracy varies with number of training plates (1-4).
Fixed test plate: P6
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

with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json'), 'r') as f:
    PLATE_WELL_ID: dict = json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description='Predict all crops for diversity experiment')
    parser.add_argument('--n_plates', type=int, nargs='+', default=None, choices=[1, 2, 3, 4],
                        help='Number of training plates (1, 2, 3, or 4). Can specify multiple, e.g., --n_plates 1 2 3 4')
    parser.add_argument('--crop_size', type=int, default=224, help='Crop size (default: 224)')
    parser.add_argument('--grid_size', type=int, default=12, help='Grid size (default: 12)')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    
    args: argparse.Namespace = parser.parse_args()
    
    crop_size: int = args.crop_size
    grid_size: int = args.grid_size
    crops_per_image: int = grid_size * grid_size
    
    n_plates_list: list[int]
    if args.n_plates is not None:
        n_plates_list = args.n_plates
    else:
        n_plates_list = [1, 2, 3, 4]
    
    print(f"Config: n_plates={n_plates_list}, crop_size={crop_size}, grid_size={grid_size}, crops_per_image={crops_per_image}")
    
    classes: dict[int, str] = {}
    with open(os.path.join(SCRIPT_DIR, 'classes.txt'), 'r') as f:
        for line in f:
            idx, name = line.strip().split(',', 1)
            classes[int(idx)] = name

    idx_to_label: dict[int, str] = classes
    label_to_idx: dict[str, int] = {v: k for k, v in classes.items()}
    
    num_classes: int
    if args.num_classes is not None:
        num_classes = args.num_classes
    else:
        num_classes = len(classes)

    print(f"Loaded {num_classes} classes")

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    TEST_PLATE = 'P6'

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

    def predict_image(model: EfficientNetClassifier, img_path: str, plate: str, batch_size: int) -> list[dict]:
        """Predict all crops for a single image."""
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
                
                well: Optional[str] = parse_well_from_filename(img_path)
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
        
        return metrics

    for n_plates in n_plates_list:
        n_plates_dir: str = os.path.join(SCRIPT_DIR, f'increase_{n_plates}_plates')
        checkpoint_path: str = os.path.join(n_plates_dir, 'best_model.pth')
        image_dir: str = os.path.join(BASE_DIR, TEST_PLATE)
        output_dir: str = n_plates_dir
        
        print(f"\n{'='*60}")
        print(f"Plate diversity prediction: {n_plates} training plates")
        print(f"  checkpoint: {checkpoint_path}")
        print(f"  image_dir: {image_dir}")
        print(f"  test_plate: {TEST_PLATE}")
        print(f"{'='*60}")
        
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint not found: {checkpoint_path}")
            continue
        
        model: EfficientNetClassifier = EfficientNetClassifier(num_classes=num_classes)
        model = model.to(device)
        
        checkpoint: dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Model loaded successfully")
        
        image_paths: list[Path] = sorted(Path(image_dir).glob('*.tif'))
        
        if args.max_images:
            image_paths = image_paths[:args.max_images]
        
        print(f"Processing {len(image_paths)} images...")
        
        all_results: list[dict] = []
        
        for img_path in tqdm(image_paths, desc=f"Predicting"):
            img_path_str: str = str(img_path)
            results: list[dict] = predict_image(model, img_path_str, TEST_PLATE, args.batch_size)
            all_results.extend(results)
        
        metrics: dict = compute_metrics(all_results, num_classes)
        
        print(f"\nPrediction summary:")
        print(f"  - Images processed: {len(image_paths)}")
        print(f"  - Crops per image: {crops_per_image}")
        print(f"  - Total crop predictions: {len(all_results)}")
        
        df: pd.DataFrame = pd.DataFrame(all_results)
        
        output_csv: str = os.path.join(output_dir, f'predictions_{n_plates}_plates.csv')
        df.to_csv(output_csv, index=False)
        print(f"\nSaved predictions to {output_csv}")
        
        print(f"\nMetrics (all {crops_per_image} crops):")
        if metrics.get('accuracy') is not None:
            print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total_gt']})")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            if metrics.get('roc_auc') is not None:
                print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        unique_images = df['image_name'].nunique()
        print(f"\n--- Per-image averaged predictions (majority vote) ---")
        
        image_preds: list[dict] = []
        for img_name in df['image_name'].unique():
            img_df = df[df['image_name'] == img_name]
            pred_counts = img_df['predicted_class_idx'].value_counts()
            majority_pred = int(pred_counts.index[0])
            well = img_df['well'].iloc[0] if 'well' in img_df.columns else None
            gt_label = img_df['ground_truth_label'].iloc[0] if 'ground_truth_label' in img_df.columns else None
            gt_idx = int(img_df['ground_truth_idx'].iloc[0]) if 'ground_truth_idx' in img_df.columns else -1
            
            avg_probs = np.mean(np.array(img_df['probs'].tolist()), axis=0)
            avg_pred = int(np.argmax(avg_probs))
            
            image_preds.append({
                'image_name': img_name,
                'well': well,
                'ground_truth_label': gt_label,
                'ground_truth_idx': gt_idx,
                'majority_vote_pred': majority_pred,
                'avg_prob_pred': avg_pred,
                'correct_majority': int(majority_pred == gt_idx) if gt_idx >= 0 else None,
                'correct_avg_prob': int(avg_pred == gt_idx) if gt_idx >= 0 else None,
            })
        
        img_df_final = pd.DataFrame(image_preds)
        img_df_final = img_df_final[img_df_final['ground_truth_label'].notna()]
        
        if len(img_df_final) > 0:
            maj_correct = int((img_df_final['majority_vote_pred'] == img_df_final['ground_truth_idx']).sum())
            avg_correct = int((img_df_final['avg_prob_pred'] == img_df_final['ground_truth_idx']).sum())
            total = len(img_df_final)
            
            maj_acc = maj_correct / total
            avg_acc = avg_correct / total
            
            print(f"Images with ground truth: {total}")
            print(f"Majority vote accuracy: {maj_acc:.4f} ({maj_correct}/{total})")
            print(f"Average probability accuracy: {avg_acc:.4f} ({avg_correct}/{total})")
            
            metrics['majority_vote_accuracy'] = maj_acc
            metrics['avg_prob_accuracy'] = avg_acc
            metrics['num_test_images'] = total
        
        img_output_csv: str = os.path.join(output_dir, f'image_predictions_{n_plates}_plates.csv')
        img_df_final.to_csv(img_output_csv, index=False)
        print(f"Saved per-image predictions to {img_output_csv}")
        
        del model
        torch.cuda.empty_cache()
        
        print(f"\n{'='*60}")
        print(f"Done! Predicted on {n_plates} plate(s) model, test plate {TEST_PLATE}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()