#!/usr/bin/env python3
"""
Generate trial pathway confusion matrices for final_crispr_model.
9 plots: 3 levels (crop, image, well) × 3 types (raw, percent, binary)
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils import (
    TRIAL_PATHWAY,
    aggregate_crop_to_well,
    load_prediction_data,
    logger
)


def get_base_gene(label) -> str:
    """Extract base gene name from label."""
    from utils import get_base_gene as _get_base_gene
    return _get_base_gene(label)


def get_trial_pathway(label) -> str:
    """Get trial pathway for a label."""
    base = get_base_gene(label)
    if base.upper().startswith('WT') or base.upper() == 'NC':
        return 'WT'
    if base in TRIAL_PATHWAY:
        return TRIAL_PATHWAY[base]
    return base


def map_trial_pathway(labels) -> list[str]:
    """Map labels to trial pathways."""
    return [get_trial_pathway(l) for l in labels]


def plot_binary_cm(cm_sum: np.ndarray, labels: list[str], title: str, output_path: str) -> None:
    """Plot binary confusion matrix."""
    n = len(labels)
    
    cm_binary = np.zeros((n, n))
    for i in range(n):
        if cm_sum[i, i] > 0.5:
            cm_binary[i, i] = 1
        for j in range(n):
            if i != j and cm_sum[i, j] > 0.5:
                cm_binary[i, j] = 1
    
    random_baseline = 1.0 / n
    n_above_random = np.sum(np.diag(cm_sum) > random_baseline)
    n_above_50 = np.sum(np.diag(cm_sum) > 0.5)
    
    fig, ax = plt.subplots(figsize=(max(8, n*0.5), max(8, n*0.5)))
    
    sns.heatmap(cm_binary, annot=False, cmap='Blues', xticklabels=labels,
                yticklabels=labels, ax=ax, vmin=0, vmax=1,
                cbar_kws={'label': '0=Wrong, 1=Correct', 'shrink': 0.8},
                linewidths=0.5, linecolor='white', square=True)
    
    for i in range(n):
        rect = patches.Rectangle((i, i), 1, 1, linewidth=3, edgecolor='#FF4444',
                                  facecolor='none', zorder=10)
        ax.add_patch(rect)
    
    ax.set_xlabel('Predicted Pathway', fontsize=10)
    ax.set_ylabel('True Pathway', fontsize=10)
    ax.set_title(f'{title}\n{n_above_50}/{n} > 50%, {n_above_random}/{n} > Random({random_baseline*100:.1f}%)', 
                 fontsize=11, fontweight='bold')
    ax.set_xticks(np.arange(n) + 0.5, labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(n) + 0.5, labels, rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_raw_counts(cm_sum: np.ndarray, labels: list[str], title: str, output_path: str) -> None:
    """Plot raw count confusion matrix."""
    n = len(labels)
    
    fig, ax = plt.subplots(figsize=(max(8, n*0.5), max(8, n*0.5)))
    
    sns.heatmap(cm_sum, annot=True, fmt='.0f', cmap='Blues', xticklabels=labels,
                yticklabels=labels, ax=ax, cbar_kws={'label': 'Count', 'shrink': 0.8},
                linewidths=0.5, linecolor='white', annot_kws={'size': 10}, square=True)
    
    for i in range(n):
        rect = patches.Rectangle((i, i), 1, 1, linewidth=3, edgecolor='#FF4444',
                                  facecolor='none', zorder=10)
        ax.add_patch(rect)
    
    ax.set_xlabel('Predicted Pathway', fontsize=10)
    ax.set_ylabel('True Pathway', fontsize=10)
    ax.set_title(f'{title}\n(Raw Counts)', fontsize=11, fontweight='bold')
    ax.set_xticks(np.arange(n) + 0.5, labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(n) + 0.5, labels, rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_percentage_cm(cm_sum: np.ndarray, labels: list[str], title: str, output_path: str) -> None:
    """Plot percentage confusion matrix."""
    n = len(labels)
    
    random_baseline = 100.0 / n
    n_above_random = np.sum(np.diag(cm_sum) * 100 > random_baseline)
    n_above_50 = np.sum(np.diag(cm_sum) * 100 > 50)
    
    fig, ax = plt.subplots(figsize=(max(8, n*0.5), max(8, n*0.5)))
    
    sns.heatmap(cm_sum * 100, annot=True, fmt='.1f', cmap='Blues', xticklabels=labels,
                yticklabels=labels, ax=ax, vmin=0, vmax=100,
                cbar_kws={'label': 'Percentage (%)', 'shrink': 0.8},
                linewidths=0.5, linecolor='white', annot_kws={'size': 10}, square=True)
    
    for i in range(n):
        rect = patches.Rectangle((i, i), 1, 1, linewidth=3, edgecolor='#FF4444',
                                  facecolor='none', zorder=10)
        ax.add_patch(rect)
    
    ax.set_xlabel('Predicted Pathway', fontsize=10)
    ax.set_ylabel('True Pathway', fontsize=10)
    ax.set_title(f'{title}\n{n_above_50}/{n} > 50%, {n_above_random}/{n} > Random({random_baseline:.1f}%)', 
                 fontsize=11, fontweight='bold')
    ax.set_xticks(np.arange(n) + 0.5, labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(n) + 0.5, labels, rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate trial pathway confusion matrices')
    parser.add_argument('--folds', type=str, default='P4,P5,P6', help='Comma-separated folds')
    parser.add_argument('--csv_name', type=str, default=None)
    parser.add_argument('--prediction_csv', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    folds = args.folds.split(',')

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(SCRIPT_DIR, 'aggregate', 'trial_pathway')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Aggregating across folds: {folds}")
    print(f"Output directory: {output_dir}")

    all_fold_data = load_prediction_data(folds, SCRIPT_DIR, prefer_mil=True)
    
    if not all_fold_data:
        logger.error("No folds loaded!")
        return

    results: list[dict] = []

    levels = [('crop', 'Crop'), ('image', 'Image'), ('well', 'Well')]
    plot_types = [
        ('raw', 'Raw Counts', plot_raw_counts),
        ('percent', 'Percentage', plot_percentage_cm),
        ('binary', 'Binary', plot_binary_cm),
    ]

    for level_key, level_name in levels:
        fold_raw_cms: list[tuple[list[str], np.ndarray]] = []
        fold_norm_cms: list[tuple[list[str], np.ndarray]] = []
        fold_accs: list[float] = []
        all_labels_set: set[str] = set()

        for fold, df_valid in all_fold_data.items():
            image_df, well_df = aggregate_crop_to_well(df_valid)
            level_data = {'crop': df_valid, 'image': image_df, 'well': well_df}
            level_df = level_data[level_key]

            true_col = 'ground_truth_label' if level_key == 'crop' else 'true_label'
            true_mapped = map_trial_pathway(level_df[true_col].values)

            if level_key == 'crop':
                pred_mapped = map_trial_pathway(level_df['predicted_class_name'].values)
            else:
                pred_mapped = map_trial_pathway(level_df['pred_majority'].values)

            acc = np.mean(np.array(true_mapped) == np.array(pred_mapped))
            fold_accs.append(acc)

            all_labels = sorted(set(true_mapped) | set(pred_mapped))
            all_labels_set.update(all_labels)

            cm_raw = confusion_matrix(true_mapped, pred_mapped, labels=all_labels, normalize=None)
            fold_raw_cms.append((all_labels, cm_raw))

            cm_norm = confusion_matrix(true_mapped, pred_mapped, labels=all_labels, normalize='true')
            fold_norm_cms.append((all_labels, cm_norm))

        all_labels = sorted(all_labels_set)
        n_classes = len(all_labels)
        random_baseline = 100.0 / n_classes

        cm_sum_raw = np.zeros((n_classes, n_classes))
        for labels_list, cm in fold_raw_cms:
            label_to_idx = {l: j for j, l in enumerate(labels_list)}
            for j, l in enumerate(labels_list):
                if l in label_to_idx:
                    cm_sum_raw[label_to_idx[l], :] += cm[j, :]

        row_sums = cm_sum_raw.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_sum_normalized = cm_sum_raw / row_sums

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)

        n_above_random = np.sum(np.diag(cm_sum_normalized) * 100 > random_baseline)
        n_above_50 = np.sum(np.diag(cm_sum_normalized) * 100 > 50)

        title = f'Trial Pathway - {level_name} Level ({len(fold_raw_cms)} folds) Acc: {100*mean_acc:.1f}%±{100*std_acc:.1f}%'

        for type_key, type_name, plot_func in plot_types:
            output_path = os.path.join(output_dir, f'{type_key}_cm_{level_key}_{type_key}.png')
            plot_func(cm_sum_normalized if type_key != 'raw' else cm_sum_raw, 
                     all_labels, title, output_path)
            print(f"Saved: {output_path}")

        results.append({
            'level': level_name,
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'n_folds': len(fold_raw_cms),
            'n_classes': n_classes,
            'random_baseline': random_baseline,
            'classes_above_50': n_above_50,
            'classes_above_random': n_above_random
        })

        print(f"  {level_name}: {100*mean_acc:.2f}% ± {100*std_acc:.2f}% | {n_above_50}/{n_classes} > 50%, {n_above_random}/{n_classes} > Random({random_baseline:.1f}%)")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'trial_pathway_metrics.csv'), index=False)
    print(f"\nSaved to {output_dir}/")


if __name__ == '__main__':
    main()
