#!/usr/bin/env python3
"""
Generate aggregate training/validation curves with mean ± std across folds.
"""

import os
import argparse
import pandas as pd
import numpy as np
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


def main():
    parser = argparse.ArgumentParser(description='Generate aggregate training curves')
    parser.add_argument('--folds', type=str, default='P2,P3,P4,P5', help='Comma-separated folds')
    args = parser.parse_args()
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    folds = args.folds.split(',')
    
    output_dir = os.path.join(SCRIPT_DIR, 'training_curves')
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = {}
    max_epochs = 0
    
    for fold in folds:
        fold_dir = os.path.join(SCRIPT_DIR, f'fold_{fold}')
        metrics_path = find_latest_metrics(fold_dir)
        
        if not metrics_path:
            print(f"Skipping {fold}: no training metrics")
            continue
        
        df = pd.read_csv(metrics_path)
        all_data[fold] = df
        max_epochs = max(max_epochs, len(df))
        print(f"Loaded {fold}: {len(df)} epochs")
    
    if not all_data:
        print("No data loaded")
        return
    
    epochs = np.arange(max_epochs)
    
    def get_series(key):
        series = []
        for fold in folds:
            if fold in all_data:
                data = all_data[fold][key].values
                if len(data) < max_epochs:
                    data = np.pad(data, (0, max_epochs - len(data)), mode='edge')
                series.append(data)
        return np.array(series)
    
    train_loss = get_series('train_loss')
    val_loss = get_series('val_loss')
    train_acc = get_series('train_acc')
    val_acc = get_series('val_acc')
    val_bal = get_series('val_balanced_acc')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Train Loss
    mean = np.nanmean(train_loss, axis=0)
    std = np.nanstd(train_loss, axis=0)
    axes[0, 0].plot(epochs, mean, label='Mean', linewidth=2)
    axes[0, 0].fill_between(epochs, mean - std, mean + std, alpha=0.3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Val Loss
    mean = np.nanmean(val_loss, axis=0)
    std = np.nanstd(val_loss, axis=0)
    axes[0, 1].plot(epochs, mean, label='Mean', linewidth=2, color='orange')
    axes[0, 1].fill_between(epochs, mean - std, mean + std, alpha=0.3, color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Train Acc
    mean = np.nanmean(train_acc, axis=0)
    std = np.nanstd(train_acc, axis=0)
    axes[1, 0].plot(epochs, mean, label='Mean', linewidth=2, color='green')
    axes[1, 0].fill_between(epochs, mean - std, mean + std, alpha=0.3, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Val Acc
    mean = np.nanmean(val_acc, axis=0)
    std = np.nanstd(val_acc, axis=0)
    axes[1, 1].plot(epochs, mean, label='Mean', linewidth=2, color='red')
    axes[1, 1].fill_between(epochs, mean - std, mean + std, alpha=0.3, color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Aggregate Training Curves ({len(all_data)} folds)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aggregate_training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save metrics summary
    summary = []
    for fold in folds:
        if fold in all_data:
            df = all_data[fold]
            summary.append({
                'fold': fold,
                'best_train_acc': df['train_acc'].max(),
                'best_val_acc': df['val_acc'].max(),
                'best_val_balanced': df['val_balanced_acc'].max(),
                'final_train_loss': df['train_loss'].iloc[-1],
                'final_val_loss': df['val_loss'].iloc[-1]
            })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, 'aggregate_summary.csv'), index=False)
    
    print(f"\nSaved aggregate curves to {output_dir}/aggregate_training_curves.png")
    print(f"Summary:\n{summary_df}")


if __name__ == '__main__':
    main()