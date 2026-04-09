#!/usr/bin/env python3
"""
Generate ROC and Precision-Recall curves for each fold.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import ast


def load_classes(fold_dir):
    classes_path = os.path.join(os.path.dirname(fold_dir), 'classes.txt')
    if os.path.exists(classes_path):
        with open(classes_path) as f:
            return [line.strip() for line in f]
    return None


def get_predictions_data(df, sample_frac=0.2):
    n_samples = max(1, int(len(df) * sample_frac))
    df_sample = df.sample(n=n_samples, random_state=42)
    
    y_true = df_sample['ground_truth_idx'].values
    
    probs_list = []
    for p in df_sample['probs'].values:
        probs_list.append(np.array(ast.literal_eval(p)))
    y_score = np.array(probs_list)
    
    return y_true, y_score


def plot_roc_curves(y_true, y_score, classes, output_path, top_n=8):
    n_classes = len(classes)
    
    aps = []
    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        if y_true_binary.sum() > 0:
            ap = average_precision_score(y_true_binary, y_score[:, i])
            aps.append((i, ap, classes[i]))
        else:
            aps.append((i, 0, classes[i]))
    
    aps_sorted = sorted(aps, key=lambda x: x[1], reverse=True)
    top_indices = [x[0] for x in aps_sorted[:top_n]]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.get_cmap('tab10', top_n)(np.linspace(0, 1, top_n))
    
    for idx, (class_idx, ap, class_name) in enumerate(aps_sorted[:top_n]):
        y_true_binary = (y_true == class_idx).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_score[:, class_idx])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[idx], lw=2,
                label=f'{class_name} (AUC={roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'Top {top_n} Classes by AUC', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return aps_sorted


def plot_pr_curves(y_true, y_score, classes, output_path, top_n=8):
    n_classes = len(classes)
    
    aps = []
    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        if y_true_binary.sum() > 0:
            ap = average_precision_score(y_true_binary, y_score[:, i])
            aps.append((i, ap, classes[i]))
        else:
            aps.append((i, 0, classes[i]))
    
    aps_sorted = sorted(aps, key=lambda x: x[1], reverse=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.get_cmap('tab10', top_n)(np.linspace(0, 1, top_n))
    
    for idx, (class_idx, ap, class_name) in enumerate(aps_sorted[:top_n]):
        y_true_binary = (y_true == class_idx).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_score[:, class_idx])
        
        ax.plot(recall, precision, color=colors[idx], lw=2,
                label=f'{class_name} (AP={ap:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Top {top_n} Classes by Average Precision', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate ROC and PR curves')
    parser.add_argument('--folds', type=str, default='P2,P3,P4,P5', help='Comma-separated folds')
    args = parser.parse_args()
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    folds = args.folds.split(',')
    
    output_dir = os.path.join(SCRIPT_DIR, 'roc_pr_curves')
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for fold in folds:
        fold_dir = os.path.join(SCRIPT_DIR, f'fold_{fold}')
        csv_path = os.path.join(fold_dir, 'predictions.csv')
        
        if not os.path.exists(csv_path):
            print(f"Skipping {fold}: no predictions.csv")
            continue
        
        print(f"Processing {fold}...")
        
        df = pd.read_csv(csv_path)
        
        classes_path = os.path.join(SCRIPT_DIR, 'classes.txt')
        if os.path.exists(classes_path):
            with open(classes_path) as f:
                classes = [line.strip() for line in f]
        else:
            classes = [f'Class_{i}' for i in range(96)]
        
        y_true, y_score = get_predictions_data(df)
        
        n_classes = len(classes)
        if y_score.shape[1] != n_classes:
            n_classes = y_score.shape[1]
            classes = classes[:n_classes]
        
        roc_path = os.path.join(output_dir, f'roc_curves_{fold}.png')
        aps_sorted = plot_roc_curves(y_true, y_score, classes, roc_path)
        
        pr_path = os.path.join(output_dir, f'pr_curves_{fold}.png')
        plot_pr_curves(y_true, y_score, classes, pr_path)
        
        for i, (class_idx, ap, class_name) in enumerate(aps_sorted[:8]):
            all_results.append({
                'fold': fold,
                'class_idx': class_idx,
                'class_name': class_name,
                'average_precision': ap
            })
        
        print(f"  Saved ROC and PR curves")
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(output_dir, 'top_classes_metrics.csv'), index=False)
    
    print(f"\nSaved to {output_dir}/")


if __name__ == '__main__':
    main()