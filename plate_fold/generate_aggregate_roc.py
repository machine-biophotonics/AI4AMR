#!/usr/bin/env python3
"""
Generate aggregate ROC curves with mean ± std across folds.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score
import ast


def get_predictions_data_sample(df, sample_frac=0.2):
    n_samples = max(1, int(len(df) * sample_frac))
    df_sample = df.sample(n=n_samples, random_state=42)
    
    y_true = df_sample['ground_truth_idx'].values
    
    probs_list = []
    for p in df_sample['probs'].values:
        probs_list.append(np.array(ast.literal_eval(p)))
    y_score = np.array(probs_list)
    
    return y_true, y_score


def compute_roc_per_class(y_true, y_score, n_classes):
    results = {}
    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        if y_true_binary.sum() > 0:
            fpr, tpr, _ = roc_curve(y_true_binary, y_score[:, i])
            roc_auc = auc(fpr, tpr)
            results[i] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    return results


def main():
    parser = argparse.ArgumentParser(description='Generate aggregate ROC curves')
    parser.add_argument('--folds', type=str, default='P2,P3,P4', help='Comma-separated folds')
    args = parser.parse_args()
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    folds = args.folds.split(',')
    
    output_dir = os.path.join(SCRIPT_DIR, 'roc_pr_curves')
    os.makedirs(output_dir, exist_ok=True)
    
    classes_path = os.path.join(SCRIPT_DIR, 'classes.txt')
    if os.path.exists(classes_path):
        with open(classes_path) as f:
            classes = [line.strip() for line in f]
    else:
        classes = [f'Class_{i}' for i in range(96)]
    
    all_class_metrics = {}
    
    for fold in folds:
        fold_dir = os.path.join(SCRIPT_DIR, f'fold_{fold}')
        csv_path = os.path.join(fold_dir, 'predictions.csv')
        
        if not os.path.exists(csv_path):
            print(f"Skipping {fold}: no predictions.csv")
            continue
        
        print(f"Processing {fold}...")
        
        df = pd.read_csv(csv_path)
        y_true, y_score = get_predictions_data_sample(df)
        
        n_classes = min(len(classes), y_score.shape[1])
        
        class_results = compute_roc_per_class(y_true, y_score, n_classes)
        
        for i, res in class_results.items():
            if i not in all_class_metrics:
                all_class_metrics[i] = {'aucs': [], 'class_name': classes[i] if i < len(classes) else f'Class_{i}'}
            all_class_metrics[i]['aucs'].append(res['auc'])
    
    avg_aucs = [(i, np.mean(d['aucs']), np.std(d['aucs']), d['class_name']) 
                 for i, d in all_class_metrics.items()]
    avg_aucs_sorted = sorted(avg_aucs, key=lambda x: x[1], reverse=True)
    
    top_classes = [x[0] for x in avg_aucs_sorted[:8]]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.get_cmap('tab10', 8)(np.linspace(0, 1, 8))
    
    for idx, (class_idx, mean_auc, std_auc, class_name) in enumerate(avg_aucs_sorted[:8]):
        fold_curves = []
        
        for fold in folds:
            fold_dir = os.path.join(SCRIPT_DIR, f'fold_{fold}')
            csv_path = os.path.join(fold_dir, 'predictions.csv')
            
            if not os.path.exists(csv_path):
                continue
            
            df = pd.read_csv(csv_path)
            y_true, y_score = get_predictions_data_sample(df)
            
            y_true_binary = (y_true == class_idx).astype(int)
            if y_true_binary.sum() > 0:
                fpr, tpr, _ = roc_curve(y_true_binary, y_score[:, class_idx])
                fold_curves.append((fpr, tpr))
        
        if fold_curves:
            all_fpr = np.unique(np.concatenate([f for f, t in fold_curves]))
            interp_tprs = []
            for fpr, tpr in fold_curves:
                interp_tpr = np.interp(all_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                interp_tprs.append(interp_tpr)
            
            mean_tpr = np.mean(interp_tprs, axis=0)
            mean_tpr[-1] = 1.0
            
            ax.plot(all_fpr, mean_tpr, color=colors[idx], lw=2,
                    label=f'{class_name} (AUC={mean_auc:.3f}±{std_auc:.3f})')
            
            ax.fill_between(all_fpr, 
                            np.maximum(mean_tpr - np.std(interp_tprs, axis=0), 0),
                            np.minimum(mean_tpr + np.std(interp_tprs, axis=0), 1),
                            color=colors[idx], alpha=0.2)
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Aggregate ROC Curves (Top 8 Classes, Mean ± Std)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aggregate_roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    results = [(class_name, mean_auc, std_auc) for _, mean_auc, std_auc, class_name in avg_aucs_sorted]
    results_df = pd.DataFrame(results, columns=['class_name', 'mean_auc', 'std_auc'])
    results_df.to_csv(os.path.join(output_dir, 'aggregate_roc_metrics.csv'), index=False)
    
    print(f"\nSaved aggregate ROC to {output_dir}/aggregate_roc_curves.png")
    print(f"Top 8 classes by mean AUC:")
    for class_name, mean_auc, std_auc in results[:8]:
        print(f"  {class_name}: {mean_auc:.3f} ± {std_auc:.3f}")


if __name__ == '__main__':
    main()