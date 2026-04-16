#!/usr/bin/env python3
"""
Generate training and validation curves for each fold.
"""

import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob


def find_latest_metrics(fold_dir):
    pattern = os.path.join(fold_dir, 'training_metrics_*.csv')
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def plot_fold_curves(metrics_path, fold_name, output_dir):
    df = pd.read_csv(metrics_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(df['epoch'], df['train_loss'], label='Train', linewidth=2)
    axes[0].plot(df['epoch'], df['val_loss'], label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{fold_name} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(df['epoch'], df['train_acc'], label='Train', linewidth=2)
    axes[1].plot(df['epoch'], df['val_acc'], label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{fold_name} - Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Balanced Accuracy
    axes[2].plot(df['epoch'], df['val_balanced_acc'] * 100, label='Val', linewidth=2, color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Balanced Acc (%)')
    axes[2].set_title(f'{fold_name} - Balanced Acc')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_curves_{fold_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return df['val_acc'].max(), df['val_balanced_acc'].max()


def main():
    parser = argparse.ArgumentParser(description='Generate training curves for folds')
    parser.add_argument('--folds', type=str, default='P2,P3,P4,P5', help='Comma-separated folds')
    args = parser.parse_args()
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    folds = args.folds.split(',')
    
    output_dir = os.path.join(SCRIPT_DIR, 'training_curves')
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for fold in folds:
        fold_dir = os.path.join(SCRIPT_DIR, f'fold_{fold}')
        metrics_path = find_latest_metrics(fold_dir)
        
        if not metrics_path:
            print(f"Skipping {fold}: no training metrics found")
            continue
        
        print(f"Plotting {fold} from {metrics_path}")
        best_acc, best_bal = plot_fold_curves(metrics_path, fold, output_dir)
        
        results.append({
            'fold': fold,
            'best_val_acc': best_acc,
            'best_val_balanced_acc': best_bal
        })
        
        print(f"  Best Val Acc: {best_acc:.2f}%, Best Balanced: {best_bal*100:.2f}%")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'fold_best_metrics.csv'), index=False)
    print(f"\nSaved to {output_dir}/")


if __name__ == '__main__':
    main()