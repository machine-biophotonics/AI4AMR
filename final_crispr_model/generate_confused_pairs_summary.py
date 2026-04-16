#!/usr/bin/env python3
"""
Generate top confused pairs summary table and bar charts across ALL folds with std.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from utils import (
    aggregate_to_level,
    load_prediction_data,
    logger
)


def get_confused_pairs(df: pd.DataFrame, level: str, top_n: int = 30) -> pd.DataFrame:
    """Get top confused pairs for a given level."""
    true_col = 'ground_truth_label' if level == 'crop' else 'true_label'
    pred_col = 'predicted_class_name' if level == 'crop' else 'pred_label'
    
    true_labels_arr = df[true_col].values
    pred_labels_arr = df[pred_col].values
    
    labels = sorted(set(true_labels_arr) | set(pred_labels_arr))
    cm = confusion_matrix(true_labels_arr, pred_labels_arr, labels=labels)
    
    pairs: list[dict] = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and cm[i, j] > 0:
                total = cm[i].sum()
                pairs.append({
                    'true_class': true_label,
                    'predicted_class': pred_label,
                    'count': int(cm[i, j]),
                    'total_true': int(total),
                    'confusion_rate': float(cm[i, j]) / total * 100 if total > 0 else 0
                })
    
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values('count', ascending=False).head(top_n)
    
    return pairs_df


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate top confused pairs summary')
    parser.add_argument('--folds', type=str, default='P1,P2,P3,P4,P5,P6', help='Comma-separated folds')
    parser.add_argument('--top_n', type=int, default=20, help='Number of top pairs')
    parser.add_argument('--output_dir', type=str, default='analysis', help='Output directory')
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    folds = args.folds.split(',')
    levels = ['crop', 'image', 'well']
    
    all_fold_data = load_prediction_data(folds, SCRIPT_DIR, prefer_mil=True)
    
    if not all_fold_data:
        logger.error("No folds loaded!")
        return
    
    print(f"\nLoaded {len(all_fold_data)} folds: {list(all_fold_data.keys())}")
    
    all_results: list[pd.DataFrame] = []
    
    for level in levels:
        print(f"\n{'='*60}")
        print(f"Level: {level.upper()}")
        print(f"{'='*60}")
        
        fold_pairs: dict[tuple[str, str], list[dict]] = {}
        all_pairs_set: set[tuple[str, str]] = set()
        
        for fold, df_valid in all_fold_data.items():
            level_df = aggregate_to_level(df_valid, level)
            pairs_df = get_confused_pairs(level_df, level, top_n=50)
            
            for _, row in pairs_df.iterrows():
                pair_key = (str(row['true_class']), str(row['predicted_class']))
                if pair_key not in fold_pairs:
                    fold_pairs[pair_key] = []
                fold_pairs[pair_key].append({
                    'count': row['count'],
                    'confusion_rate': row['confusion_rate']
                })
                all_pairs_set.add(pair_key)
        
        pair_stats: list[dict] = []
        for pair_key in all_pairs_set:
            counts = [p['count'] for p in fold_pairs[pair_key]]
            rates = [p['confusion_rate'] for p in fold_pairs[pair_key]]
            
            pair_stats.append({
                'true_class': pair_key[0],
                'predicted_class': pair_key[1],
                'mean_count': np.mean(counts),
                'std_count': np.std(counts),
                'min_count': np.min(counts),
                'max_count': np.max(counts),
                'mean_rate': np.mean(rates),
                'std_rate': np.std(rates),
                'n_folds': len(counts)
            })
        
        pairs_df = pd.DataFrame(pair_stats)
        pairs_df = pairs_df.sort_values('mean_count', ascending=False).head(args.top_n)
        
        csv_output = os.path.join(OUTPUT_DIR, f'all_folds_top_confused_pairs_{level}.csv')
        pairs_df.to_csv(csv_output, index=False)
        print(f"  Saved: {csv_output}")
        
        fig, ax = plt.subplots(figsize=(14, max(7, len(pairs_df) * 0.4)))
        
        pairs_df = pairs_df.copy()
        pairs_df['pair'] = pairs_df['true_class'] + ' → ' + pairs_df['predicted_class']
        
        errors = np.minimum(pairs_df['std_count'].values, pairs_df['mean_count'].values)
        
        colors = plt.cm.Blues(pairs_df['mean_rate'].values / 100)
        bars = ax.barh(range(len(pairs_df)), pairs_df['mean_count'].values, 
                       xerr=errors, color=colors, capsize=3, error_kw={'elinewidth': 1})
        
        ax.set_yticks(range(len(pairs_df)))
        ax.set_yticklabels(pairs_df['pair'].values, fontsize=9)
        ax.invert_yaxis()
        
        for i, (mean, std, rate) in enumerate(zip(
            pairs_df['mean_count'].values, 
            pairs_df['std_count'].values, 
            pairs_df['mean_rate'].values
        )):
            label = f'{mean:.0f}±{std:.0f} ({rate:.1f}%)'
            ax.text(mean + std + max(pairs_df['mean_count']) * 0.02, i, label, 
                    va='center', fontsize=7)
        
        ax.set_xlabel('Count (mean ± std across folds)', fontsize=11)
        ax.set_title(f'Top Confused Pairs - {level.upper()} Level\n({len(all_fold_data)} folds)', 
                     fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        png_output = os.path.join(OUTPUT_DIR, f'all_folds_top_confused_pairs_{level}.png')
        plt.savefig(png_output, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {png_output}")
        
        print(f"\n{level.upper()} Level - Top 10 Confused Pairs:")
        for _, row in pairs_df.head(10).iterrows():
            print(f"  {row['true_class']} -> {row['predicted_class']}: "
                  f"{row['mean_count']:.0f}+/-{row['std_count']:.0f} ({row['mean_rate']:.1f}+/-{row['std_rate']:.1f}%)")
        
        all_results.append(pairs_df)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    
    for idx, (level, pairs_df) in enumerate(zip(levels, all_results)):
        ax = axes[idx]
        pairs_df = pairs_df.head(10).copy()
        pairs_df['pair'] = pairs_df['true_class'] + ' → ' + pairs_df['predicted_class']
        
        errors = np.minimum(pairs_df['std_count'].values, pairs_df['mean_count'].values)
        colors = plt.cm.Blues(pairs_df['mean_rate'].values / 100)
        
        ax.barh(range(len(pairs_df)), pairs_df['mean_count'].values,
               xerr=errors, color=colors, capsize=2, error_kw={'elinewidth': 1})
        
        ax.set_yticks(range(len(pairs_df)))
        ax.set_yticklabels(pairs_df['pair'].values, fontsize=8)
        ax.invert_yaxis()
        
        ax.set_xlabel('Count (mean ± std)', fontsize=10)
        ax.set_title(f'{level.upper()} Level', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle(f'Top 10 Confused Pairs per Level ({len(all_fold_data)} folds)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    combined_png = os.path.join(OUTPUT_DIR, 'all_folds_top_confused_pairs_combined.png')
    plt.savefig(combined_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved combined: {combined_png}")


if __name__ == '__main__':
    main()
