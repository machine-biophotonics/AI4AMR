#!/usr/bin/env python3
"""
Generate fold comparison plots for MIL training.
Handles partial training (early stopping) gracefully.
"""

import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse


def load_fold_data():
    """Load training metrics from trained folds."""
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    all_plates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    
    fold_data = {}
    
    for test_plate in all_plates:
        fold_dir = os.path.join(SCRIPT_DIR, f'fold_{test_plate}')
        
        csv_files = glob.glob(os.path.join(fold_dir, 'training_metrics_*.csv'))
        if csv_files:
            df = pd.read_csv(csv_files[0])
            max_epoch = int(df['epoch'].max()) + 1
            fold_data[test_plate] = {
                'data': df,
                'max_epoch': max_epoch
            }
            print(f"   {test_plate}: {max_epoch} epochs")
        else:
            print(f"   {test_plate}: Not trained")
    
    return fold_data


def plot_accuracy_comparison(fold_data, output_dir):
    """Plot train/val accuracy with std across folds."""
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
    
    train_accs = []
    train_stds = []
    val_accs = []
    val_stds = []
    val_aucs = []
    val_auc_stds = []
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
    
    # Plot 1: Accuracy
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    epochs_arr = np.array(epochs)
    
    ax1.fill_between(epochs_arr, np.array(train_accs) - np.array(train_stds), 
                     np.array(train_accs) + np.array(train_stds), alpha=0.2, color='blue')
    ax1.plot(epochs_arr, train_accs, 'b-', linewidth=2, label='Train Acc')
    
    ax1.fill_between(epochs_arr, np.array(val_accs) - np.array(val_stds),
                     np.array(val_accs) + np.array(val_stds), alpha=0.2, color='orange')
    ax1.plot(epochs_arr, val_accs, 'orange', linewidth=2, label='Val Acc')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title(f'Train vs Validation Accuracy (n={len(fold_data)} folds)\nMean ± Std', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Add best epochs markers
    for test_plate, data in fold_data.items():
        df_fold = data['data']
        best_epoch = df_fold['val_auc'].idxmax()
        best_auc = df_fold['val_auc'].max()
        ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.3)
        ax1.annotate(f'{test_plate}: E{best_epoch}\nAUC={best_auc:.3f}', 
                    xy=(best_epoch, val_accs[best_epoch] if best_epoch < len(val_accs) else val_accs[-1]),
                    fontsize=7, alpha=0.7)
    
    # Plot 2: Val AUC
    ax2.fill_between(epochs_arr, np.array(val_aucs) - np.array(val_auc_stds),
                     np.array(val_aucs) + np.array(val_auc_stds), alpha=0.2, color='green')
    ax2.plot(epochs_arr, val_aucs, 'g-', linewidth=2, label='Val AUC')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('AUC', fontsize=12)
    ax2.set_title('Validation AUC\nMean ± Std', fontsize=14)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'accuracy_lr_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Plot 3: LR decay
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_arr, lrs, 'r-', linewidth=2, label='Learning Rate')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title(f'Learning Rate Decay (n={len(fold_data)} folds)', fontsize=14)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, 'lr_decay.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return {
        'train_acc': train_accs,
        'val_acc': val_accs,
        'val_auc': val_aucs,
        'lr': lrs
    }


def plot_per_class_metrics(fold_data, output_dir):
    """Plot per-class accuracy and precision."""
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
        idx_to_class = {i: f"Class_{i}" for i in range(96)}
    
    # Aggregate predictions - this requires prediction CSVs
    # For now, just save summary metrics
    summary = []
    for test_plate, data in fold_data.items():
        df = data['data']
        best_auc = df['val_auc'].max()
        best_auc_epoch = df['val_auc'].idxmax()
        best_acc = df['val_acc'].max()
        best_acc_epoch = df['val_acc'].idxmax()
        lowest_loss = df['val_loss'].min()
        lowest_loss_epoch = df['val_loss'].idxmin()
        
        summary.append({
            'fold': test_plate,
            'best_auc': best_auc,
            'best_auc_epoch': best_auc_epoch,
            'best_acc': best_acc,
            'best_acc_epoch': best_acc_epoch,
            'lowest_loss': lowest_loss,
            'lowest_loss_epoch': lowest_loss_epoch,
            'epochs_trained': data['max_epoch']
        })
    
    summary_df = pd.DataFrame(summary)
    csv_path = os.path.join(output_dir, 'per_fold_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Print summary
    print("\n=== Fold Summary ===")
    print(summary_df.to_string(index=False))
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description='Plot MIL fold comparison')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(SCRIPT_DIR, 'train_test_results')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading fold data...")
    fold_data = load_fold_data()
    
    if not fold_data:
        print("No trained folds found!")
        return
    
    print(f"\nGenerating plots for {len(fold_data)} folds...")
    
    metrics = plot_accuracy_comparison(fold_data, output_dir)
    summary = plot_per_class_metrics(fold_data, output_dir)
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()