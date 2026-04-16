#!/usr/bin/env python3
"""
Generate comprehensive fold comparison plots for MIL training.
Similar to plate_fold/plot_fold_comparison.py but adapted for MIL format.
"""

import os
import glob
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 10
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize


HIERARCHY = {
    'dnaA': 'DNA Replication', 'dnaB': 'DNA Replication', 'dnaE': 'DNA Replication',
    'gyrA': 'DNA Replication', 'gyrB': 'DNA Replication',
    'parC': 'DNA Replication', 'parE': 'DNA Replication',
    'folA': 'Folate', 'folP': 'Folate',
    'lpxA': 'Lipid A', 'lpxC': 'Lipid A', 'lptA': 'Lipid A', 'lptC': 'Lipid A',
    'msbA': 'Lipid A',
    'mrdA': 'Peptidoglycan', 'mrcA': 'Peptidoglycan', 'mrcB': 'Peptidoglycan', 'murA': 'Peptidoglycan', 'murC': 'Peptidoglycan',
    'ftsI': 'Peptidoglycan', 'ftsZ': 'Peptidoglycan',
    'rplA': 'Ribosome', 'rplC': 'Ribosome', 'rpsA': 'Ribosome', 'rpsL': 'Ribosome',
    'rpoA': 'Transcription', 'rpoB': 'Transcription',
    'secA': 'Secretion', 'secY': 'Secretion',
}


def get_base_gene(label):
    if not label or label == 'nan':
        return 'Unknown'
    if '_' in str(label):
        return str(label).rsplit('_', 1)[0]
    return str(label)


def get_pathway(label):
    base = get_base_gene(label)
    if str(base).upper().startswith('WT') or str(base).upper() == 'NC':
        return 'WT'
    if base in HIERARCHY:
        return HIERARCHY[base]
    return 'Unknown'


FAMILY = {
    'dnaA': 'dna', 'dnaB': 'dna', 'dnaE': 'dna',
    'gyrA': 'gyr', 'gyrB': 'gyr',
    'parC': 'par', 'parE': 'par',
    'folA': 'fol', 'folP': 'fol',
    'lpxA': 'lpx', 'lpxC': 'lpx', 'lptA': 'lpt', 'lptC': 'lpt',
    'msbA': 'lpt',
    'mrdA': 'mrd', 'mrcA': 'mrd', 'mrcB': 'mrd', 'murA': 'mur', 'murC': 'mur',
    'ftsI': 'fts', 'ftsZ': 'fts',
    'rplA': 'rpl', 'rplC': 'rpl', 'rpsA': 'rps', 'rpsL': 'rps',
    'rpoA': 'rpo', 'rpoB': 'rpo',
    'secA': 'sec', 'secY': 'sec',
}


def get_family(label):
    base = get_base_gene(label)
    if str(base).upper().startswith('WT') or str(base).upper() == 'NC':
        return 'WT'
    if base in FAMILY:
        return FAMILY[base]
    return 'Unknown'


def load_fold_data():
    """Load training metrics from trained folds."""
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    all_plates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    
    fold_data = {}
    
    for test_plate in all_plates:
        fold_dir = os.path.join(SCRIPT_DIR, f'fold_{test_plate}')
        
        csv_files = glob.glob(os.path.join(fold_dir, 'training_metrics_*.csv'))
        if csv_files:
            best_csv = csv_files[0]
            best_rows = 0
            for csv_path in csv_files:
                try:
                    temp_df = pd.read_csv(csv_path)
                    if len(temp_df) > best_rows:
                        best_rows = len(temp_df)
                        best_csv = csv_path
                except:
                    pass
            df = pd.read_csv(best_csv)
            df = df.dropna(subset=['epoch']).copy()
            df = df.reset_index(drop=True)
            df['epoch'] = df['epoch'].astype(int)
            max_epoch = int(df['epoch'].max()) + 1
            
            if 'val_auc' not in df.columns and 'val_balanced_acc' in df.columns:
                df['val_auc'] = df['val_balanced_acc']
            fold_data[test_plate] = {
                'data': df,
                'max_epoch': max_epoch,
                'dir': fold_dir
            }
            print(f"   {test_plate}: {max_epoch} epochs")
        else:
            print(f"   {test_plate}: Not trained")
    
    return fold_data


def plot_accuracy_with_std(fold_data, output_dir):
    """Plot train/val accuracy with std + LR decay - matches plate_fold style."""
    all_data = []
    
    for test_plate, data in fold_data.items():
        df = data['data']
        max_epoch = data['max_epoch']
        
        for epoch in range(max_epoch):
            all_data.append({
                'fold': test_plate,
                'epoch': epoch,
                'train_acc': df.iloc[epoch]['train_acc'],
                'val_acc': df.iloc[epoch]['val_acc'],
                'val_auc': df.iloc[epoch]['val_auc'],
                'val_loss': df.iloc[epoch]['val_loss'],
                'lr': df.iloc[epoch]['lr'],
            })
    
    df = pd.DataFrame(all_data)
    epochs = sorted(df['epoch'].unique())
    
    train_accs, train_stds = [], []
    val_accs, val_stds = [], []
    val_aucs, val_auc_stds = [], []
    lrs = []
    
    for epoch in epochs:
        epoch_data = df[df['epoch'] == epoch]
        train_accs.append(epoch_data['train_acc'].mean())
        train_stds.append(epoch_data['train_acc'].std())
        val_accs.append(epoch_data['val_acc'].mean())
        val_stds.append(epoch_data['val_acc'].std())
        val_aucs.append(epoch_data['val_auc'].mean())
        val_auc_stds.append(epoch_data['val_auc'].std())
        lrs.append(epoch_data['lr'].mean())
    
    epochs_arr = np.array(epochs)
    train_accs = np.array(train_accs)
    train_stds = np.array(train_stds)
    val_accs = np.array(val_accs)
    val_stds = np.array(val_stds)
    val_aucs = np.array(val_aucs)
    val_auc_stds = np.array(val_auc_stds)
    
    max_val_acc = max(val_accs.max(), train_accs.max())
    y_max = 80
    
    # Match plate_fold style: accuracy + LR on secondary axis
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Train accuracy with std
    ax1.fill_between(epochs_arr, train_accs - train_stds, train_accs + train_stds, 
                     alpha=0.2, color='blue')
    ax1.plot(epochs_arr, train_accs, 'b-', linewidth=2, label='Train Acc')
    
    # Val accuracy with std
    ax1.fill_between(epochs_arr, val_accs - val_stds, val_accs + val_stds,
                    alpha=0.2, color='orange')
    ax1.plot(epochs_arr, val_accs, 'orange', linewidth=2, label='Val Acc')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(0, y_max)
    ax1.grid(True, alpha=0.3)
    
    # LR on secondary axis (log scale like plate_fold)
    ax2 = ax1.twinx()
    ax2.semilogy(epochs_arr, lrs, 'g--', linewidth=1.5, alpha=0.7, label='LR')
    ax2.set_ylabel('Learning Rate', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.title(f'Train vs Validation Accuracy (n={len(fold_data)} folds)\nMean ± Std Dev', fontsize=14)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'accuracy_lr_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Also save AUC separately
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(epochs_arr, val_aucs - val_auc_stds, val_aucs + val_auc_stds,
                    alpha=0.2, color='green')
    ax.plot(epochs_arr, val_aucs, 'g-', linewidth=2, label='Val AUC')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title(f'Validation AUC (n={len(fold_data)} folds)\nMean ± Std Dev', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.8, 1.0)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'val_auc_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def load_predictions(fold_data):
    """Load predictions from trained folds."""
    all_plates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    
    all_fold_results = {}
    
    for test_plate in all_plates:
        if test_plate not in fold_data:
            continue
            
        fold_dir = fold_data[test_plate]['dir']
        
        # Use best_model_acc predictions - include all trained folds
        pred_files = [
            os.path.join(fold_dir, 'predictions_all_crops_mil_best_model_acc.csv'),
            os.path.join(fold_dir, 'predictions_all_crops_mil_best_model_auc.csv'),
            os.path.join(fold_dir, 'predictions_all_crops_mil_100pos.csv'),
            os.path.join(fold_dir, 'predictions_all_crops.csv'),
            os.path.join(fold_dir, 'image_predictions_all_crops.csv'),
        ]
        
        pred_file = None
        for f in pred_files:
            if os.path.exists(f):
                pred_file = f
                break
        
        if pred_file:
            print(f"   Loading: {pred_file}")
            df = pd.read_csv(pred_file)
            
            # Use per-crop predictions (NOT aggregated to image level)
            crop_results = []
            for _, row in df.iterrows():
                true_label = row['ground_truth_label']
                true_idx = row['ground_truth_idx']
                pred_idx = row['predicted_class_idx']
                probs = json.loads(row['probs']) if isinstance(row['probs'], str) else row['probs']
                
                if true_label is not None and pd.notna(true_label):
                    crop_results.append({
                        'image_name': row['image_name'],
                        'true_label': true_label,
                        'true_idx': int(true_idx),
                        'pred_idx': int(pred_idx),
                        'probs': probs,
                    })
            
            all_fold_results[test_plate] = crop_results
            print(f"   {test_plate}: {len(crop_results)} crops")
        else:
            print(f"   {test_plate}: No predictions found")
    
    return all_fold_results


def plot_roc_from_predictions(all_fold_results, output_dir):
    """Plot ROC curve from predictions - matches plate_fold style."""
    if not all_fold_results:
        print("No predictions found for ROC plot")
        return None
    
    all_plates = list(all_fold_results.keys())
    num_classes = 96
    
    # Compute per-class AUC
    all_fold_aucs = {}
    all_fold_prec = {}
    all_fold_rec = {}
    
    for test_plate, preds in all_fold_results.items():
        y_true = np.array([p['true_idx'] for p in preds])
        y_probs = np.array([p['probs'] for p in preds])
        y_pred = np.array([p['pred_idx'] for p in preds])
        
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        
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
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            class_prec.append(prec)
            class_rec.append(rec)
        
        all_fold_aucs[test_plate] = class_aucs
        all_fold_prec[test_plate] = class_prec
        all_fold_rec[test_plate] = class_rec
    
    # Mean across folds
    mean_auc = np.nanmean([all_fold_aucs[p] for p in all_plates], axis=0)
    std_auc = np.nanstd([all_fold_aucs[p] for p in all_plates], axis=0)
    mean_prec = np.nanmean([all_fold_prec[p] for p in all_plates], axis=0)
    std_prec = np.nanstd([all_fold_prec[p] for p in all_plates], axis=0)
    mean_rec = np.nanmean([all_fold_rec[p] for p in all_plates], axis=0)
    std_rec = np.nanstd([all_fold_rec[p] for p in all_plates], axis=0)
    
    macro_auc = np.nanmean(mean_auc)
    macro_auc_std = np.nanstd(mean_auc)
    macro_prec = np.nanmean(mean_prec)
    macro_prec_std = np.nanstd([np.nanmean(all_fold_prec[p]) for p in all_plates])
    macro_rec = np.nanmean(mean_rec)
    macro_rec_std = np.nanstd([np.nanmean(all_fold_rec[p]) for p in all_plates])
    
    print(f"   Macro ROC-AUC: {macro_auc:.4f} ± {macro_auc_std:.4f}")
    print(f"   Macro Precision: {macro_prec:.4f}")
    print(f"   Macro Recall: {macro_rec:.4f}")
    
    # Plot ROC curve
    fpr_grid = np.linspace(0, 1, 100)
    all_tprs = []
    
    for test_plate, preds in all_fold_results.items():
        y_true = np.array([p['true_idx'] for p in preds])
        y_probs = np.array([p['probs'] for p in preds])
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        
        tprs_interp = []
        for i in range(num_classes):
            if y_bin[:, i].sum() > 0:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
                tprs_interp.append(np.interp(fpr_grid, fpr, tpr))
            else:
                tprs_interp.append(np.zeros_like(fpr_grid))
        
        mean_tpr = np.mean(tprs_interp, axis=0)
        mean_tpr[-1] = 1.0
        all_tprs.append(mean_tpr)
    
    mean_tpr = np.mean(all_tprs, axis=0)
    std_tpr = np.std(all_tprs, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(fpr_grid, mean_tpr, 'b-', linewidth=2, 
            label=f'Mean ROC (AUC = {macro_auc:.3f} ± {macro_auc_std:.3f})')
    ax.fill_between(fpr_grid, 
                   np.maximum(0, mean_tpr - std_tpr), 
                   np.minimum(1, mean_tpr + std_tpr), 
                   alpha=0.2, color='blue')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'Average ROC Curve Across {len(all_plates)} Folds\n(Macro Average ± Std)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'roc_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return {
        'macro_auc': macro_auc,
        'macro_auc_std': macro_auc_std,
        'macro_prec': macro_prec,
        'macro_prec_std': macro_prec_std,
        'macro_rec': macro_rec,
        'macro_rec_std': macro_rec_std,
    }


def plot_per_class_metrics(all_fold_results, output_dir):
    """Plot per-class accuracy and precision - matches plate_fold style."""
    if not all_fold_results:
        print("No predictions found for per-class metrics")
        return
    
    all_plates = list(all_fold_results.keys())
    num_classes = 96
    
    # Get class names
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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
    
    class_accs = {c: [] for c in range(num_classes)}
    class_precs = {c: [] for c in range(num_classes)}
    
    for test_plate, preds in all_fold_results.items():
        y_true = np.array([p['true_idx'] for p in preds])
        y_pred = np.array([p['pred_idx'] for p in preds])
        
        for i in range(num_classes):
            tp = ((y_pred == i) & (y_true == i)).sum()
            fp = ((y_pred == i) & (y_true != i)).sum()
            fn = ((y_pred != i) & (y_true == i)).sum()
            
            acc = tp / (tp + fn) if (tp + fn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            class_accs[i].append(acc)
            class_precs[i].append(prec)
    
    # Compute mean and std
    classes = []
    acc_means, acc_stds = [], []
    prec_means, prec_stds = [], []
    
    for i in range(num_classes):
        classes.append(idx_to_class.get(i, f"Class_{i}"))
        acc_means.append(np.nanmean(class_accs[i]) * 100)
        acc_stds.append(np.nanstd(class_accs[i]) * 100)
        prec_means.append(np.nanmean(class_precs[i]) * 100)
        prec_stds.append(np.nanstd(class_precs[i]) * 100)
    
    # Sort by accuracy
    sorted_indices = np.argsort(acc_means)[::-1]
    classes = [classes[i] for i in sorted_indices]
    acc_means = [acc_means[i] for i in sorted_indices]
    acc_stds = [acc_stds[i] for i in sorted_indices]
    prec_means = [prec_means[i] for i in sorted_indices]
    prec_stds = [prec_stds[i] for i in sorted_indices]
    
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(max(16, len(classes) * 0.15), 6))
    
    ax.bar(x - width/2, acc_means, width, yerr=acc_stds, 
           label='Accuracy', color='steelblue', capsize=2, alpha=0.8)
    ax.bar(x + width/2, prec_means, width, yerr=prec_stds, 
           label='Precision', color='coral', capsize=2, alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=10)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title(f'Per-Class Accuracy and Precision Across {len(all_plates)} Folds\n(Mean ± Std)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'per_class_metrics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Save CSV
    df = pd.DataFrame({
        'class': classes,
        'accuracy_mean': acc_means,
        'accuracy_std': acc_stds,
        'precision_mean': prec_means,
        'precision_std': prec_stds,
    })
    csv_path = os.path.join(output_dir, 'per_class_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


def plot_hierarchical_metrics(all_fold_results, output_dir):
    """Plot per-class accuracy & precision for gene/family/pathway - match confusion matrix."""
    if not all_fold_results:
        print("No predictions found")
        return
    
    all_plates = list(all_fold_results.keys())
    num_classes = 96
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    classes_path = os.path.join(SCRIPT_DIR, 'classes.txt')
    
    idx_to_class = {}
    if os.path.exists(classes_path):
        with open(classes_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    idx_to_class[int(parts[0])] = parts[1]
    
    def get_label_from_idx(idx):
        return idx_to_class.get(idx, f"Class_{idx}")
    
# For each level, calculate metrics per class - BOTH crop-level and image-level
    for level, get_level in [('gene', get_base_gene), ('family', get_family), ('pathway', get_pathway), ('guide', lambda x: x)]:
        print(f"  {level}...")
        
        # Collect all unique classes at this level across all folds
        all_classes = set()
        fold_data_crop = []  # Crop-level
        fold_data_image = []  # Image-level
        fold_data_well = []  # Well-level

        for test_plate, preds in all_fold_results.items():
            from collections import Counter
            
            # Crop-level: use each crop prediction directly
            class_stats_crop = {}
            for p in preds:
                true_label = get_label_from_idx(p['true_idx'])
                pred_label = get_label_from_idx(p['pred_idx'])
                
                true_level_key = get_level(true_label)
                pred_level_key = get_level(pred_label)
                
                all_classes.add(true_level_key)
                all_classes.add(pred_level_key)
                
                if true_level_key not in class_stats_crop:
                    class_stats_crop[true_level_key] = {'tp': 0, 'fp': 0, 'fn': 0}
                
                if true_level_key == pred_level_key:
                    class_stats_crop[true_level_key]['tp'] += 1
                else:
                    class_stats_crop[true_level_key]['fn'] += 1
                    if pred_level_key not in class_stats_crop:
                        class_stats_crop[pred_level_key] = {'tp': 0, 'fp': 0, 'fn': 0}
                    class_stats_crop[pred_level_key]['fp'] += 1
            
            fold_data_crop.append(class_stats_crop)
            
            # Image-level: aggregate crops to image first (majority vote)
            img_preds = {}
            for p in preds:
                img_name = p.get('image_name', f"img_{p['true_idx']}")
                true_label = get_label_from_idx(p['true_idx'])
                pred_label = get_label_from_idx(p['pred_idx'])
                
                if img_name not in img_preds:
                    img_preds[img_name] = {'true': true_label, 'preds': []}
                img_preds[img_name]['preds'].append(pred_label)
            
            class_stats_image = {}
            for img_name, data in img_preds.items():
                true_label = data['true']
                pred_counts = Counter(data['preds'])
                majority_pred = pred_counts.most_common(1)[0][0]
                
                true_level_key = get_level(true_label)
                pred_level_key = get_level(majority_pred)
                
                all_classes.add(true_level_key)
                all_classes.add(pred_level_key)
                
                if true_level_key not in class_stats_image:
                    class_stats_image[true_level_key] = {'tp': 0, 'fp': 0, 'fn': 0}
                
                if true_level_key == pred_level_key:
                    class_stats_image[true_level_key]['tp'] += 1
                else:
                    class_stats_image[true_level_key]['fn'] += 1
                    if pred_level_key not in class_stats_image:
                        class_stats_image[pred_level_key] = {'tp': 0, 'fp': 0, 'fn': 0}
                    class_stats_image[pred_level_key]['fp'] += 1
            
            fold_data_image.append(class_stats_image)
        
        # Well-level: aggregate by well (multiple images per well)
            well_preds = {}
            for p in preds:
                # Get well from image_name (e.g., "WellA01_PointA01..." -> "WellA01")
                img_name = p.get('image_name', '')
                well = img_name.split('_')[0] if '_' in img_name else img_name[:6]
                true_label = get_label_from_idx(p['true_idx'])
                pred_label = get_label_from_idx(p['pred_idx'])
                
                if well not in well_preds:
                    well_preds[well] = {'true': true_label, 'preds': []}
                well_preds[well]['preds'].append(pred_label)
            
            class_stats_well = {}
            for well, data in well_preds.items():
                true_label = data['true']
                pred_counts = Counter(data['preds'])
                majority_pred = pred_counts.most_common(1)[0][0]
                
                true_level_key = get_level(true_label)
                pred_level_key = get_level(majority_pred)
                
                all_classes.add(true_level_key)
                all_classes.add(pred_level_key)
                
                if true_level_key not in class_stats_well:
                    class_stats_well[true_level_key] = {'tp': 0, 'fp': 0, 'fn': 0}
                
                if true_level_key == pred_level_key:
                    class_stats_well[true_level_key]['tp'] += 1
                else:
                    class_stats_well[true_level_key]['fn'] += 1
                    if pred_level_key not in class_stats_well:
                        class_stats_well[pred_level_key] = {'tp': 0, 'fp': 0, 'fn': 0}
                    class_stats_well[pred_level_key]['fp'] += 1
            
            fold_data_well.append(class_stats_well)
        
        # Save fold_data for all 3 levels
        all_fold_data = {'crop': fold_data_crop, 'image': fold_data_image, 'well': fold_data_well}
        
        # Function to calculate and plot
        def calc_and_plot(classes, fold_data_dict, suffix, title_suffix):
            accs_mean = []
            accs_std = []
            precs_mean = []
            precs_std = []
            
            for cls in classes:
                fold_accs = []
                fold_precs = []
                
                for fd in fold_data_dict:
                    tp = fd.get(cls, {}).get('tp', 0)
                    fp = fd.get(cls, {}).get('fp', 0)
                    fn = fd.get(cls, {}).get('fn', 0)
                    
                    acc = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
                    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
                    
                    fold_accs.append(acc)
                    fold_precs.append(prec)
                
                accs_mean.append(np.mean(fold_accs))
                accs_std.append(np.std(fold_accs))
                precs_mean.append(np.mean(fold_precs))
                precs_std.append(np.std(fold_precs))
            
            # Sort by accuracy mean
            sorted_idx = np.argsort(accs_mean)[::-1]
            classes_sorted = [classes[i] for i in sorted_idx]
            accs_mean = [accs_mean[i] for i in sorted_idx]
            accs_std = [accs_std[i] for i in sorted_idx]
            precs_mean = [precs_mean[i] for i in sorted_idx]
            precs_std = [precs_std[i] for i in sorted_idx]
            
            # Plot with error bars
            x = np.arange(len(classes_sorted))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(max(12, len(classes_sorted) * 0.12), 6))
            
            ax.bar(x - width/2, accs_mean, width, yerr=accs_std, label='Accuracy', color='steelblue', capsize=2, alpha=0.8)
            ax.bar(x + width/2, precs_mean, width, yerr=precs_std, label='Precision', color='coral', capsize=2, alpha=0.8)
            
            ax.set_xlabel('Class', fontsize=10)
            ax.set_ylabel('Score (%)', fontsize=12)
            ax.set_title(f'{level.capitalize()}-level Accuracy & Precision\n({title_suffix}, {level.title()}: {len(classes_sorted)} classes)', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(classes_sorted, rotation=90, fontsize=8)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 100)
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, f'per_class_metrics_{level}_{suffix}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_path}")
        
        # Generate plots for crop, image, and well
        calc_and_plot(sorted(all_classes), fold_data_crop, 'crop', 'Crop-level')
        calc_and_plot(sorted(all_classes), fold_data_image, 'image', 'Image-level')
        calc_and_plot(sorted(all_classes), fold_data_well, 'well', 'Well-level')


def main():
    parser = argparse.ArgumentParser(description='Generate MIL fold comparison plots')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(SCRIPT_DIR, 'train_test_results')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading fold training data...")
    fold_data = load_fold_data()
    
    if not fold_data:
        print("No trained folds found!")
        return
    
    print(f"\n1. Generating accuracy + LR plot...")
    plot_accuracy_with_std(fold_data, output_dir)
    
    print(f"\n2. Loading predictions...")
    all_fold_results = load_predictions(fold_data)
    
    if all_fold_results:
        print(f"\n3. Generating ROC curve...")
        roc_results = plot_roc_from_predictions(all_fold_results, output_dir)
        
        print(f"\n4. Generating per-class metrics...")
        plot_per_class_metrics(all_fold_results, output_dir)
        
        print(f"\n5. Generating hierarchical metrics (guide, gene, pathway, family)...")
        plot_hierarchical_metrics(all_fold_results, output_dir)
        
        if roc_results:
            print("\n" + "="*50)
            print("SUMMARY")
            print("="*50)
            print(f"Macro ROC-AUC: {roc_results['macro_auc']:.4f} ± {roc_results['macro_auc_std']:.4f}")
            print(f"Macro Precision: {roc_results['macro_prec']:.4f} ± {roc_results['macro_prec_std']:.4f}")
            print(f"Macro Recall: {roc_results['macro_rec']:.4f} ± {roc_results['macro_rec_std']:.4f}")
    else:
        print("\nNo predictions found - skipping ROC and per-class plots")
    
    # Save fold summary
    summary = []
    for test_plate, data in fold_data.items():
        df = data['data']
        summary.append({
            'fold': test_plate,
            'best_auc': df['val_auc'].max(),
            'best_auc_epoch': df['val_auc'].idxmax(),
            'best_acc': df['val_acc'].max(),
            'best_acc_epoch': df['val_acc'].idxmax(),
            'lowest_loss': df['val_loss'].min(),
            'lowest_loss_epoch': df['val_loss'].idxmax(),
            'epochs_trained': data['max_epoch']
        })
    
    df = pd.DataFrame(summary)
    csv_path = os.path.join(output_dir, 'per_fold_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()