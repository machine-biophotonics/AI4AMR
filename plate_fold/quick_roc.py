#!/usr/bin/env python3
"""Quick compute ROC-AUC, Precision, Recall from predictions.csv + Train/Val accuracy"""

import os, glob, json, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def compute_train_val_accuracy():
    """Compute train/val accuracy from training_metrics CSV"""
    all_plates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    
    train_accs = []
    val_accs = []
    
    for test_plate in all_plates:
        fold_dir = os.path.join(SCRIPT_DIR, f'fold_{test_plate}')
        csv_files = glob.glob(os.path.join(fold_dir, 'training_metrics_*.csv'))
        if not csv_files:
            continue
        
        df = pd.read_csv(csv_files[0])
        # Values are already in percentage form (not decimal), no need to multiply by 100
        max_train = df['train_acc'].max()
        max_val = df['val_acc'].max()
        train_accs.append(max_train)
        val_accs.append(max_val)
    
    return train_accs, val_accs


def main():
    # Compute train/val accuracy first
    print("Computing train/val accuracy...")
    train_accs, val_accs = compute_train_val_accuracy()
    
    train_mean = np.mean(train_accs)
    train_std = np.std(train_accs)
    val_mean = np.mean(val_accs)
    val_std = np.std(val_accs)
    
    print(f"Train Accuracy: {train_mean:.2f}% ± {train_std:.2f}%")
    print(f"Val Accuracy: {val_mean:.2f}% ± {val_std:.2f}%")
    
    # Now compute ROC/PR
    all_plates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    num_classes = 96
    
    all_fold_aucs = []
    all_fold_prec = []
    all_fold_rec = []
    
    for test_plate in all_plates:
        fold_dir = os.path.join(SCRIPT_DIR, f'fold_{test_plate}')
        pred_file = os.path.join(fold_dir, 'predictions.csv')
        
        if not os.path.exists(pred_file):
            continue
        
        print(f"Processing {test_plate}...")
        df = pd.read_csv(pred_file)
        
        # Group by image and average probs
        image_results = []
        for img_name, group in df.groupby('image_name'):
            true_idx = group['ground_truth_idx'].iloc[0]
            probs_list = [json.loads(p) for p in group['probs'].values]
            avg_probs = np.mean(probs_list, axis=0)
            pred_idx = np.argmax(avg_probs)
            image_results.append({
                'true_idx': int(true_idx),
                'pred_idx': int(pred_idx),
                'probs': avg_probs.tolist(),
            })
        
        # Compute metrics
        y_true = np.array([r['true_idx'] for r in image_results])
        y_probs = np.array([r['probs'] for r in image_results])
        y_pred = np.array([r['pred_idx'] for r in image_results])
        
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        
        # Per-class AUC
        class_aucs = []
        class_prec = []
        class_rec = []
        
        for i in range(num_classes):
            if y_bin[:, i].sum() > 0:
                try:
                    class_aucs.append(roc_auc_score(y_bin[:, i], y_probs[:, i]))
                except:
                    class_aucs.append(np.nan)
            else:
                class_aucs.append(np.nan)
            
            tp = ((y_pred == i) & (y_true == i)).sum()
            fp = ((y_pred == i) & (y_true != i)).sum()
            fn = ((y_pred != i) & (y_true == i)).sum()
            
            class_prec.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
            class_rec.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        
        all_fold_aucs.append(class_aucs)
        all_fold_prec.append(class_prec)
        all_fold_rec.append(class_rec)
    
    # Aggregate
    mean_auc = np.nanmean(all_fold_aucs, axis=0)
    std_auc = np.nanstd(all_fold_aucs, axis=0)
    mean_prec = np.nanmean(all_fold_prec, axis=0)
    std_prec = np.nanstd(all_fold_prec, axis=0)
    mean_rec = np.nanmean(all_fold_rec, axis=0)
    std_rec = np.nanstd(all_fold_rec, axis=0)
    
    macro_auc = np.nanmean(mean_auc)
    macro_auc_std = np.nanstd(mean_auc)
    macro_prec = np.nanmean(mean_prec)
    macro_rec = np.nanmean(mean_rec)
    
    # Cross-fold std
    macro_prec_std = np.nanstd([np.nanmean(p) for p in all_fold_prec])
    macro_rec_std = np.nanstd([np.nanmean(r) for r in all_fold_rec])
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Train Accuracy: {train_mean:.2f}% ± {train_std:.2f}%")
    print(f"Val Accuracy: {val_mean:.2f}% ± {val_std:.2f}%")
    print(f"Macro ROC-AUC: {macro_auc:.4f} ± {macro_auc_std:.4f}")
    print(f"Macro Precision: {macro_prec:.4f} ± {macro_prec_std:.4f}")
    print(f"Macro Recall: {macro_rec:.4f} ± {macro_rec_std:.4f}")
    
    # Save CSV
    classes_path = os.path.join(SCRIPT_DIR, 'classes.txt')
    if os.path.exists(classes_path):
        idx_to_class = {}
        with open(classes_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    idx_to_class[int(parts[0])] = parts[1]
    else:
        idx_to_class = {i: f"Class_{i}" for i in range(num_classes)}
    
    class_names = [idx_to_class.get(i, f"Class_{i}") for i in range(num_classes)]
    
    df = pd.DataFrame({
        'class_idx': range(num_classes),
        'class_name': class_names,
        'auc_mean': mean_auc,
        'auc_std': std_auc,
        'precision_mean': mean_prec,
        'precision_std': std_prec,
        'recall_mean': mean_rec,
        'recall_std': std_rec,
    })
    
    output_dir = os.path.join(SCRIPT_DIR, 'train test results')
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'roc_pr_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

if __name__ == '__main__':
    main()