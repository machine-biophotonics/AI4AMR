#!/usr/bin/env python3
"""
Generate confusion matrices for cross-domain predictions.

Usage:
  # For drug_on_mutant (drug model on mutant images):
  python3 generate_cross_domain_cm.py --mode drug_on_mutant --fold Plate_1 --checkpoint best_model_acc
  
  # For mutant_on_drug (mutant model on drug images):
  python3 generate_cross_domain_cm.py --mode mutant_on_drug --fold Plate_1 --checkpoint checkpoint_epoch
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(description='Generate cross-domain confusion matrices')
    parser.add_argument('--mode', type=str, required=True, choices=['drug_on_mutant', 'mutant_on_drug'],
                        help='Cross-domain mode')
    parser.add_argument('--fold', type=str, default='Plate_1')
    parser.add_argument('--checkpoint', type=str, default='best_model_acc')
    parser.add_argument('--neighborhood', type=int, default=3)
    parser.add_argument('--top_classes', type=int, default=20, help='Number of top classes to show')
    args = parser.parse_args()
    
    # Load predictions
    csv_path = os.path.join(SCRIPT_DIR, args.mode, f'fold_{args.fold}', 
                          f'predictions_all_crops_mil_{args.checkpoint}_n{args.neighborhood}.csv')
    
    if not os.path.exists(csv_path):
        print(f"ERROR: Predictions not found: {csv_path}")
        return
    
    print(f"Loading predictions from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total predictions: {len(df)}")
    
    # Majority voting per image
    print("Performing majority voting per image...")
    image_results = []
    
    for image_path, group in df.groupby('image_path'):
        gt_label = group['ground_truth_label'].iloc[0]
        pred_counts = group['predicted_class_name'].value_counts()
        majority_pred = pred_counts.index[0]
        
        image_results.append({
            'image_path': image_path,
            'ground_truth': gt_label,
            'predicted': majority_pred
        })
    
    results_df = pd.DataFrame(image_results)
    print(f"Unique images: {len(results_df)}")
    
    # Determine labels
    if args.mode == 'drug_on_mutant':
        # Predicted = drug (antibiotics), Ground truth = mutant (genes)
        predicted_label = 'Predicted Antibiotic'
        ground_truth_label = 'Ground Truth Gene'
        x_classes = sorted(results_df['predicted'].unique())
        y_classes = sorted(results_df['ground_truth'].unique())
    else:
        # Predicted = mutant (genes), Ground truth = drug (antibiotics)
        predicted_label = 'Predicted Gene'
        ground_truth_label = 'Ground Truth Antibiotic'
        x_classes = sorted(results_df['predicted'].unique())
        y_classes = sorted(results_df['ground_truth'].unique())
    
    # Get top classes for visualization
    top_pred = results_df['predicted'].value_counts().head(args.top_classes).index.tolist()
    top_gt = results_df['ground_truth'].value_counts().head(args.top_classes).index.tolist()
    
    # Use top classes for both axes
    all_classes = list(set(top_pred) | set(top_gt))[:args.top_classes]
    
    # Create confusion matrix
    cm = np.zeros((len(all_classes), len(all_classes)), dtype=int)
    label_to_idx = {l: i for i, l in enumerate(all_classes)}
    
    for _, row in results_df.iterrows():
        if row['predicted'] in label_to_idx and row['ground_truth'] in label_to_idx:
            i = label_to_idx[row['ground_truth']]
            j = label_to_idx[row['predicted']]
            cm[i, j] += 1
    
    # Save CSV
    output_dir = os.path.join(SCRIPT_DIR, args.mode, f'fold_{args.fold}', 'confusion_matrices')
    os.makedirs(output_dir, exist_ok=True)
    
    cm_df = pd.DataFrame(cm, index=all_classes, columns=all_classes)
    cm_df.to_csv(os.path.join(output_dir, f'confusion_matrix_{args.mode}.csv'))
    
    # Plot normalized confusion matrix
    fig, ax = plt.subplots(figsize=(16, 14))
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    cm_normalized = np.nan_to_num(cm_normalized)
    
    sns.heatmap(cm_normalized, annot=False, cmap='Blues',
                xticklabels=all_classes, yticklabels=all_classes,
                cbar_kws={'label': 'Percentage (%)'}, ax=ax)
    
    ax.set_xlabel(predicted_label, fontsize=12)
    ax.set_ylabel(ground_truth_label, fontsize=12)
    
    if args.mode == 'drug_on_mutant':
        title = f'Drug-on-Mutant: Predicted Antibiotics vs Ground Truth Genes\n(Drug model predicts drug classes on mutant images)'
    else:
        title = f'Mutant-on-Drug: Predicted Genes vs Ground Truth Antibiotics\n(Mutant model predicts gene classes on drug images)'
    
    ax.set_title(title, fontsize=14)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{args.mode}.png'), dpi=150)
    plt.close()
    
    # Plot raw counts
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=all_classes, yticklabels=all_classes,
                cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel(predicted_label, fontsize=12)
    ax.set_ylabel(ground_truth_label, fontsize=12)
    ax.set_title(f'{args.mode}: Raw Counts', fontsize=14)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{args.mode}_raw.png'), dpi=150)
    plt.close()
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"CONFUSION MATRIX SUMMARY: {args.mode}")
    print(f"{'='*60}")
    print(f"Ground truth classes: {len(y_classes)}")
    print(f"Predicted classes: {len(x_classes)}")
    print(f"Total images: {len(results_df)}")
    
    # Top predicted
    print(f"\nTop 10 most predicted:")
    for pred, count in results_df['predicted'].value_counts().head(10).items():
        print(f"  {pred}: {count}")
    
    # Top ground truth
    print(f"\nTop 10 ground truth:")
    for gt, count in results_df['ground_truth'].value_counts().head(10).items():
        print(f"  {gt}: {count}")
    
    # Accuracy
    correct = (results_df['predicted'] == results_df['ground_truth']).sum()
    total = len(results_df)
    print(f"\nMajority voting accuracy: {correct}/{total} = {correct/total:.4f}")
    
    print(f"\nSaved to: {output_dir}")
    print(f"  - confusion_matrix_{args.mode}.png")
    print(f"  - confusion_matrix_{args.mode}_raw.png")
    print(f"  - confusion_matrix_{args.mode}.csv")

if __name__ == '__main__':
    main()