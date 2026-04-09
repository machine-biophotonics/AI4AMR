#!/usr/bin/env python3
"""
Generate comprehensive fold comparison plots:
- Train/Val accuracy with std (x-axis: actual epochs achieved, max ~70%)
- Learning rate decay curve
- ROC curves (macro average across folds with std dev)
- Per-class accuracy and precision with std deviation
"""

import os
import glob
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_all_fold_data():
    """Load training metrics and predictions from all folds."""
    import json
    all_plates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    
    fold_metrics = {}
    fold_predictions = {}
    
    for test_plate in all_plates:
        fold_dir = os.path.join(SCRIPT_DIR, f'fold_{test_plate}')
        
        # Load training metrics
        csv_files = glob.glob(os.path.join(fold_dir, 'training_metrics_*.csv'))
        if csv_files:
            fold_metrics[test_plate] = pd.read_csv(csv_files[0])
        
        # Load per-class metrics CSV instead of processing predictions
        per_class_file = os.path.join(fold_dir, 'per_class_metrics.csv')
        if os.path.exists(per_class_file):
            fold_predictions[test_plate] = pd.read_csv(per_class_file)
    
    return fold_metrics, fold_predictions


def plot_accuracy_with_std(fold_metrics, output_dir):
    """Plot train/val accuracy with std - x-axis limited to max achieved."""
    all_data = []
    
    for test_plate, df in fold_metrics.items():
        for epoch in range(len(df)):
            all_data.append({
                'fold': test_plate,
                'epoch': epoch,
                'train_acc': df.iloc[epoch]['train_acc'],
                'val_acc': df.iloc[epoch]['val_acc'],
                'lr': df.iloc[epoch]['lr'],
            })
    
    plot_df = pd.DataFrame(all_data)
    epochs = sorted(plot_df['epoch'].unique())
    
    train_means, train_stds = [], []
    val_means, val_stds = [], []
    lr_means = []
    
    for epoch in epochs:
        epoch_data = plot_df[plot_df['epoch'] == epoch]
        train_means.append(epoch_data['train_acc'].mean())
        train_stds.append(epoch_data['train_acc'].std())
        val_means.append(epoch_data['val_acc'].mean())
        val_stds.append(epoch_data['val_acc'].std())
        lr_means.append(epoch_data['lr'].mean())
    
    train_means = np.array(train_means)
    train_stds = np.array(train_stds)
    val_means = np.array(val_means)
    val_stds = np.array(val_stds)
    lr_means = np.array(lr_means)
    
    # Find max achieved (to limit y-axis)
    max_val = max(val_means.max(), train_means.max())
    y_max = min(100, max_val + 10)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    epochs_arr = np.array(epochs)
    
    # Train accuracy
    ax1.fill_between(epochs_arr, train_means - train_stds, train_means + train_stds, 
                    alpha=0.2, color='blue')
    ax1.plot(epochs_arr, train_means, 'b-', linewidth=2, label='Train Acc')
    
    # Val accuracy
    ax1.fill_between(epochs_arr, val_means - val_stds, val_means + val_stds, 
                    alpha=0.2, color='orange')
    ax1.plot(epochs_arr, val_means, 'orange', linewidth=2, label='Val Acc')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, color='black')
    ax1.set_ylim(0, y_max)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    # Learning rate on secondary axis
    ax2 = ax1.twinx()
    ax2.semilogy(epochs_arr, lr_means, 'g--', linewidth=1.5, alpha=0.7, label='LR')
    ax2.set_ylabel('Learning Rate', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title(f'Train vs Validation Accuracy (n={len(fold_metrics)} folds)\nMean ± Std Dev', fontsize=14)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'accuracy_lr_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return plot_df


def compute_roc_per_class(predictions, num_classes):
    """Compute per-class ROC AUC from predictions."""
    y_true = np.array([p['true_label'] for p in predictions])
    y_probs = np.array([p['avg_probs'] for p in predictions])
    
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    
    auc_per_class = {}
    for i in range(num_classes):
        if y_bin[:, i].sum() > 0:
            try:
                auc_per_class[i] = roc_auc_score(y_bin[:, i], y_probs[:, i])
            except:
                auc_per_class[i] = np.nan
        else:
            auc_per_class[i] = np.nan
    
    return auc_per_class


def plot_roc_comparison(fold_predictions, output_dir):
    """Plot average ROC curve across folds with std deviation."""
    if not fold_predictions:
        print("No predictions found for ROC plot")
        return
    
    num_classes = len(fold_predictions[list(fold_predictions.keys())[0]][0]['avg_probs'])
    
    all_fold_tprs = []
    all_fold_aucs = []
    
    for test_plate, preds in fold_predictions.items():
        y_true = np.array([p['true_label'] for p in preds])
        y_probs = np.array([p['avg_probs'] for p in preds])
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        
        # Compute ROC for each class
        fpr_grid = np.linspace(0, 1, 100)
        tprs_interp = []
        aucs = []
        
        for i in range(num_classes):
            if y_bin[:, i].sum() > 0:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
                tpr_interp = np.interp(fpr_grid, fpr, tpr)
                tpr_interp[0] = 0.0
                tprs_interp.append(tpr_interp)
                aucs.append(auc(fpr, tpr))
        
        mean_tpr = np.mean(tprs_interp, axis=0)
        mean_tpr[-1] = 1.0
        all_fold_tprs.append(mean_tpr)
        all_fold_aucs.append(aucs)
    
    # Compute mean and std across folds
    all_fold_tprs = np.array(all_fold_tprs)
    mean_tpr = np.mean(all_fold_tprs, axis=0)
    std_tpr = np.std(all_fold_tprs, axis=0)
    
    all_fold_aucs = np.array(all_fold_aucs)
    mean_auc_per_class = np.nanmean(all_fold_aucs, axis=0)
    std_auc_per_class = np.nanstd(all_fold_aucs, axis=0)
    macro_auc = np.nanmean(mean_auc_per_class)
    macro_auc_std = np.nanstd(mean_auc_per_class)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(fpr_grid, mean_tpr, 'b-', linewidth=2, 
            label=f'Mean ROC (AUC = {macro_auc:.3f} ± {macro_auc_std:.3f})')
    ax.fill_between(fpr_grid, 
                   np.maximum(0, mean_tpr - std_tpr), 
                   np.minimum(1, mean_tpr + std_tpr), 
                   alpha=0.2, color='blue')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'Average ROC Curve Across {len(fold_predictions)} Folds\n(Macro Average ± Std)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'roc_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return mean_auc_per_class, std_auc_per_class


def plot_per_class_metrics(fold_predictions, output_dir):
    """Plot per-class accuracy and precision with std deviation across folds."""
    if not fold_predictions:
        print("No per-class metrics found")
        return
    
    # Each fold's data is now a DataFrame
    # Collect metrics per class across folds
    all_classes = fold_predictions[list(fold_predictions.keys())[0]]['class_name'].tolist()
    class_accs = {c: [] for c in all_classes}
    class_precs = {c: [] for c in all_classes}
    
    for test_plate, df in fold_predictions.items():
        # Recall = accuracy for this data
        for _, row in df.iterrows():
            cls_name = row['class_name']
            # recall is accuracy for each class
            recall = row['recall'] * 100 if 'recall' in row else 0
            precision = row['precision'] * 100 if 'precision' in row else 0
            class_accs[cls_name].append(recall)
            class_precs[cls_name].append(precision)
    
    # Compute mean and std
    classes = sorted(all_classes)
    acc_means = [np.nanmean(class_accs[c]) for c in classes]
    acc_stds = [np.nanstd(class_accs[c]) for c in classes]
    prec_means = [np.nanmean(class_precs[c]) for c in classes]
    prec_stds = [np.nanstd(class_precs[c]) for c in classes]
    
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
    
    bars1 = ax.bar(x - width/2, acc_means, width, yerr=acc_stds, 
                   label='Accuracy', color='steelblue', capsize=2, alpha=0.8)
    bars2 = ax.bar(x + width/2, prec_means, width, yerr=prec_stds, 
                   label='Precision', color='coral', capsize=2, alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=10)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title(f'Per-Class Accuracy and Precision Across {len(fold_predictions)} Folds\n(Mean ± Std)', fontsize=14)
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
    
    # Save as CSV
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
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate fold comparison plots')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    
    if args.output_dir is None:
        output_dir = os.path.join(SCRIPT_DIR, 'train test results')
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading fold data...")
    fold_metrics, fold_predictions = load_all_fold_data()
    
    print("\n1. Generating accuracy + LR plot...")
    plot_accuracy_with_std(fold_metrics, output_dir)
    
    print("\n2. Generating ROC comparison plot...")
    print("   (Skipping - requires test predictions per fold)")
    # plot_roc_comparison(fold_predictions, output_dir)
    
    print("\n3. Generating per-class metrics plot...")
    plot_per_class_metrics(fold_predictions, output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()