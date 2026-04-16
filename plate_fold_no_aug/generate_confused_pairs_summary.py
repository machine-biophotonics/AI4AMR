#!/usr/bin/env python3
"""
Generate top confused pairs summary table and bar charts at crop, image, and well levels.
"""

import pandas as pd
import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix

def get_base_gene(label):
    if not label or label == 'nan':
        return 'Unknown'
    if '_' in str(label):
        return str(label).rsplit('_', 1)[0]
    return str(label)


def aggregate_to_level(df, level):
    """Aggregate predictions to crop, image, or well level."""
    if level == 'crop':
        return df.copy()
    
    elif level == 'image':
        results = []
        for img_name, group in df.groupby('image_name'):
            true_label = group['ground_truth_label'].iloc[0]
            pred_counts = Counter(group['predicted_class_name'].values)
            majority_pred = pred_counts.most_common(1)[0][0]
            results.append({
                'image_name': img_name,
                'true_label': true_label,
                'pred_label': majority_pred
            })
        return pd.DataFrame(results)
    
    elif level == 'well':
        results = []
        for well, group in df.groupby('well'):
            if pd.isna(well):
                continue
            true_label = group['ground_truth_label'].iloc[0]
            pred_counts = Counter(group['predicted_class_name'].values)
            majority_pred = pred_counts.most_common(1)[0][0]
            results.append({
                'well': well,
                'true_label': true_label,
                'pred_label': majority_pred
            })
        return pd.DataFrame(results)
    
    return df


def get_confused_pairs(df, level, top_n=20):
    """Get top confused pairs for a given level."""
    true_col = 'ground_truth_label' if level == 'crop' else 'true_label'
    pred_col = 'predicted_class_name' if level == 'crop' else 'pred_label'
    
    true_labels = df[true_col].values
    pred_labels = df[pred_col].values
    
    labels = sorted(set(true_labels) | set(pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    
    pairs = []
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


def plot_confused_pairs(pairs_df, level, output_path):
    """Create bar chart of confused pairs."""
    fig, ax = plt.subplots(figsize=(12, max(6, len(pairs_df) * 0.35)))
    
    # Create labels
    pairs_df = pairs_df.copy()
    pairs_df['pair'] = pairs_df['true_class'] + ' → ' + pairs_df['predicted_class']
    
    # Color by confusion rate
    colors = plt.cm.Reds(pairs_df['confusion_rate'].values / 100)
    
    bars = ax.barh(range(len(pairs_df)), pairs_df['count'].values, color=colors)
    ax.set_yticks(range(len(pairs_df)))
    ax.set_yticklabels(pairs_df['pair'].values, fontsize=9)
    ax.invert_yaxis()
    
    # Add count and percentage labels
    for i, (count, rate) in enumerate(zip(pairs_df['count'].values, pairs_df['confusion_rate'].values)):
        ax.text(count + max(pairs_df['count']) * 0.01, i, f'{count} ({rate:.1f}%)', 
                va='center', fontsize=8)
    
    ax.set_xlabel('Count', fontsize=11)
    ax.set_title(f'Top Confused Pairs - {level.upper()} Level', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_combined_bars(all_results, levels, output_path):
    """Create combined bar chart for all levels."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    for idx, (level, pairs_df) in enumerate(zip(levels, all_results)):
        ax = axes[idx]
        pairs_df = pairs_df.copy()
        pairs_df['pair'] = pairs_df['true_class'] + ' → ' + pairs_df['predicted_class']
        pairs_df = pairs_df.head(10)  # Top 10 per level
        
        colors = plt.cm.Blues(pairs_df['confusion_rate'].values / 100)
        bars = ax.barh(range(len(pairs_df)), pairs_df['count'].values, color=colors)
        ax.set_yticks(range(len(pairs_df)))
        ax.set_yticklabels(pairs_df['pair'].values, fontsize=8)
        ax.invert_yaxis()
        
        for i, (count, rate) in enumerate(zip(pairs_df['count'].values, pairs_df['confusion_rate'].values)):
            ax.text(count + max(pairs_df['count']) * 0.01, i, f'{count} ({rate:.1f}%)', 
                    va='center', fontsize=7)
        
        ax.set_xlabel('Count', fontsize=10)
        ax.set_title(f'{level.upper()} Level', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Top 10 Confused Pairs per Level', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved combined: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate top confused pairs summary')
    parser.add_argument('--fold', type=str, default='P3', help='Fold to use')
    parser.add_argument('--top_n', type=int, default=20, help='Number of top pairs')
    parser.add_argument('--output_dir', type=str, default='analysis', help='Output directory')
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Try different CSV names
    csv_options = [
        f'fold_{args.fold}/predictions_all_crops_mil_best_model_acc.csv',
        f'fold_{args.fold}/predictions_all_crops.csv',
    ]
    csv_path = None
    for opt in csv_options:
        if os.path.exists(os.path.join(SCRIPT_DIR, opt)):
            csv_path = os.path.join(SCRIPT_DIR, opt)
            break
    if csv_path is None:
        csv_path = os.path.join(SCRIPT_DIR, csv_options[0])
    
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df_valid = df[df['ground_truth_label'].notna()].copy()
    
    levels = ['crop', 'image', 'well']
    
    all_results = []
    
    for level in levels:
        print(f"\n{'='*60}")
        print(f"Level: {level.upper()}")
        print(f"{'='*60}")
        
        agg_df = aggregate_to_level(df_valid, level)
        print(f"Total samples: {len(agg_df)}")
        
        pairs_df = get_confused_pairs(agg_df, level, args.top_n)
        pairs_df['level'] = level
        
        # Save to CSV
        csv_output = os.path.join(OUTPUT_DIR, f'top_confused_pairs_{level}.csv')
        pairs_df.to_csv(csv_output, index=False)
        print(f"  Saved: {csv_output}")
        
        # Generate bar chart PNG
        png_output = os.path.join(OUTPUT_DIR, f'top_confused_pairs_{level}.png')
        plot_confused_pairs(pairs_df, level, png_output)
        
        all_results.append(pairs_df)
    
    # Combined bar chart
    combined_png = os.path.join(OUTPUT_DIR, 'top_confused_pairs_combined.png')
    plot_combined_bars(all_results, levels, combined_png)
    
    # Combined CSV
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_output = os.path.join(OUTPUT_DIR, 'top_confused_pairs_combined.csv')
    combined_df.to_csv(combined_output, index=False)
    print(f"\nSaved combined CSV: {combined_output}")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("SUMMARY - Top 5 Confused Pairs per Level")
    print(f"{'='*60}")
    for level in levels:
        level_df = all_results[levels.index(level)]
        print(f"\n{level.upper()} Level:")
        for i, row in level_df.head(5).iterrows():
            print(f"  {row['true_class']} -> {row['predicted_class']}: {row['count']} ({row['confusion_rate']:.1f}%)")


if __name__ == '__main__':
    main()
