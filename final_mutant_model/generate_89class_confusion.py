#!/usr/bin/env python3
"""
Generate 89-class confusion matrix for drug predictions (antibiotic + concentration).
Creates:
1. Full 89x89 confusion matrix heatmap
2. Per-class accuracy and performance metrics
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
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_antibiotic_and_concentration(label):
    """Extract antibiotic name and concentration from label."""
    if not label or pd.isna(label):
        return 'Unknown', 'Unknown'
    
    label = str(label)
    
    if 'control' in label.lower() or label == 'DMSO':
        return 'DMSO', 'control'
    
    if '_' in label:
        parts = label.rsplit('_', 1)
        if len(parts) == 2:
            antibiotic = parts[0]
            conc = parts[1]
            return antibiotic, conc
    
    return label, 'Unknown'


def load_ground_truth_from_mapping():
    """Load ground truth from IC50 mapping based on plate and well."""
    mapping_path = os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')
    with open(mapping_path, 'r') as f:
        ic50_data = json.load(f)
    return ic50_data


def majority_vote_per_image_89class(df, ic50_data):
    """Perform majority voting across positions for each image - 89 class version."""
    print("Performing majority voting per image (89 classes: antibiotic + concentration)...")
    
    results = []
    
    for image_path, group in df.groupby('image_path'):
        # Get plate and well from the group
        plate = group['plate'].iloc[0]
        well = group['well'].iloc[0]
        
        # Convert plate name (Plate_1 -> P1)
        if 'Plate_' in plate:
            plate_key = f"P{plate.split('_')[-1]}"
        else:
            plate_key = plate
        
        # Get ground truth from IC50 mapping
        gt_label = 'Unknown'
        if plate_key in ic50_data and well in ic50_data[plate_key]:
            ab_info = ic50_data[plate_key][well]
            antibiotic = ab_info.get('antibiotic', '')
            ic50 = ab_info.get('ic50_multiple', '')
            if antibiotic and ic50:
                # Normalize antibiotic name
                if antibiotic == 'Clavulanic Acid':
                    antibiotic = 'Clavulanic_Acid'
                elif antibiotic == 'Polymyxin B':
                    antibiotic = 'Polymyxin_B'
                else:
                    antibiotic = antibiotic.replace(' ', '_')
                if ic50 == 'control':
                    gt_label = 'control'
                else:
                    ic50_str = ic50 if 'x' in str(ic50) else f"{ic50}x"
                    gt_label = f"{antibiotic}_{ic50_str}"
        
        # Get majority predicted class
        predictions = group['predicted_class_name'].tolist()
        pred_counts = Counter(predictions)
        majority_pred = pred_counts.most_common(1)[0][0]
        
        results.append({
            'image_path': image_path,
            'well': well,
            'plate': plate,
            'ground_truth_label': gt_label,
            'predicted_class_name': majority_pred,
            'num_positions': len(group),
            'vote_count': pred_counts.most_common(1)[0][1]
        })
    
    result_df = pd.DataFrame(results)
    print(f"Majority voted predictions: {len(result_df)} images")
    
    # Print ground truth distribution
    gt_dist = result_df['ground_truth_label'].value_counts()
    print(f"Ground truth distribution: {len(gt_dist)} unique classes")
    
    return result_df


def create_full_confusion_matrix(df_voted, output_dir):
    """Create 89x89 confusion matrix for all drug classes with concentration."""
    
    # Group classes by antibiotic (for better visualization)
    def get_antibiotic_base(class_name):
        """Extract base antibiotic name without concentration."""
        if class_name == 'control' or 'DMSO' in class_name:
            return 'DMSO'
        if '_' in class_name:
            return class_name.rsplit('_', 1)[0]
        return class_name
    
    # Get all unique classes (ground truth)
    all_classes = sorted([c for c in df_voted['ground_truth_label'].unique() if c != 'Unknown'])
    print(f"Total classes: {len(all_classes)}")
    
    # Group classes by antibiotic and sort within groups
    # Order: concentrations in order [0.25x, 0.5x, 1x, 2x, control]
    conc_order = {'0.25x': 0, '0.5x': 1, '1x': 2, '2x': 3, 'control': 4, 'Unknown': 5}
    
    def sort_key(class_name):
        ab = get_antibiotic_base(class_name)
        if '_' in class_name:
            conc = class_name.rsplit('_', 1)[-1]
            conc_idx = conc_order.get(conc, 99)
        else:
            conc_idx = 99
        return (ab, conc_idx)
    
    # Sort classes: grouped by antibiotic, then by concentration
    sorted_classes = sorted(all_classes, key=sort_key)
    
    # Print class order for debugging
    print("\nClass order (grouped by antibiotic):")
    for ab in sorted(set(get_antibiotic_base(c) for c in sorted_classes)):
        classes = [c for c in sorted_classes if get_antibiotic_base(c) == ab]
        print(f"  {ab}: {classes}")
    
    # Label mapping
    class_to_idx = {c: i for i, c in enumerate(sorted_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    
    # Filter out Unknown and NaN
    df_filtered = df_voted[
        (df_voted['ground_truth_label'] != 'Unknown') & 
        (df_voted['ground_truth_label'].notna()) &
        (df_voted['predicted_class_name'] != 'Unknown') &
        (df_voted['predicted_class_name'].notna())
    ].copy()
    
    print(f"Valid predictions (excluding Unknown): {len(df_filtered)}")
    
    if len(df_filtered) == 0:
        print("ERROR: No valid predictions!")
        return
    
    # Create confusion matrix
    # Map to indices, dropping any that don't map (NaN handling)
    gt_mapped = df_filtered['ground_truth_label'].map(class_to_idx)
    pred_mapped = df_filtered['predicted_class_name'].map(class_to_idx)
    
    # Drop any NaN values from mapping
    valid_mask = gt_mapped.notna() & pred_mapped.notna()
    gt_labels = gt_mapped[valid_mask].astype(int).values
    pred_labels = pred_mapped[valid_mask].astype(int).values
    
    # Only include classes that appear in the data
    present_classes = sorted(set(gt_labels) | set(pred_labels))
    present_class_names = [idx_to_class[i] for i in present_classes]
    
    print(f"Present classes in data: {len(present_class_names)}")
    
    # Filter to only present classes
    present_mask_gt = np.isin(gt_labels, present_classes)
    present_mask_pred = np.isin(pred_labels, present_classes)
    valid_mask = present_mask_gt & present_mask_pred
    
    gt_filtered = gt_labels[valid_mask]
    pred_filtered = pred_labels[valid_mask]
    
    # Re-index to only present classes
    reindex = {old: new for new, old in enumerate(present_classes)}
    gt_reindexed = np.array([reindex[g] for g in gt_filtered])
    pred_reindexed = np.array([reindex[p] for p in pred_filtered])
    
    cm = confusion_matrix(gt_reindexed, pred_reindexed, labels=range(len(present_classes)))
    
    # Calculate metrics
    accuracy = accuracy_score(gt_reindexed, pred_reindexed)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    per_class_accuracy = np.nan_to_num(per_class_accuracy, 0)
    
    # Create DataFrame with per-class metrics
    metrics_data = []
    for i, class_name in enumerate(present_class_names):
        if cm.sum(axis=1)[i] > 0:
            metrics_data.append({
                'class': class_name,
                'support': int(cm.sum(axis=1)[i]),
                'correct': int(cm.diagonal()[i]),
                'accuracy': per_class_accuracy[i]
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = metrics_df.sort_values('accuracy', ascending=False)
    
    print("\nTop 10 performing classes:")
    print(metrics_df.head(10).to_string(index=False))
    
    print("\nBottom 10 performing classes:")
    print(metrics_df.tail(10).to_string(index=False))
    
    # Save metrics
    metrics_df.to_csv(os.path.join(output_dir, 'per_class_metrics.csv'), index=False)
    
    # Create full 89x89 heatmap with group boxes
    num_classes = len(present_class_names)
    print(f"\nCreating full {num_classes}x{num_classes} confusion matrix heatmap...")
    
    # Find group boundaries for drawing boxes
    def find_group_boundaries(class_names):
        """Find start and end indices for each antibiotic group."""
        groups = {}
        current_group = None
        start_idx = 0
        for i, name in enumerate(class_names):
            base = get_antibiotic_base(name)
            if base != current_group:
                if current_group is not None:
                    groups[current_group] = (start_idx, i - 1)
                current_group = base
                start_idx = i
        # Last group
        if current_group is not None:
            groups[current_group] = (start_idx, len(class_names) - 1)
        return groups
    
    group_bounds = find_group_boundaries(present_class_names)
    print(f"Group boundaries: {group_bounds}")
    
    # Find DMSO position (index 40)
    dmso_idx = present_class_names.index('control') if 'control' in present_class_names else None
    print(f"DMSO index: {dmso_idx}")
    
    # Function to add group boxes
    def add_group_lines(ax, class_names, dmso_idx=None):
        """Add vertical and horizontal lines for group boundaries and DMSO."""
        # Draw lines for each group boundary
        current_group = None
        for i, name in enumerate(class_names):
            base = get_antibiotic_base(name)
            if base != current_group:
                if current_group is not None and i > 0:
                    # Draw vertical line at boundary (after previous group)
                    ax.axvline(x=i + 0.5, color='red', linewidth=1.5, linestyle='-')
                    ax.axhline(y=i + 0.5, color='red', linewidth=1.5, linestyle='-')
                current_group = base
        
        # Draw DMSO line (straight through center)
        if dmso_idx is not None:
            # Vertical line through DMSO column
            ax.axvline(x=dmso_idx + 0.5, color='green', linewidth=2, linestyle='-')
            # Horizontal line through DMSO row
            ax.axhline(y=dmso_idx + 0.5, color='green', linewidth=2, linestyle='-')
    
    # Full heatmap with smaller font and larger figure
    fig, ax = plt.subplots(figsize=(34, 30))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized, 0)
    
    sns.heatmap(cm_normalized, 
                xticklabels=present_class_names, 
                yticklabels=present_class_names,
                cmap='Blues', 
                annot=False,
                cbar_kws={'label': 'Normalized Accuracy'},
                ax=ax)
    
    # Add group boundaries and DMSO line
    add_group_lines(ax, present_class_names, dmso_idx)
    
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('True', fontsize=8)
    ax.set_title(f'89-Class Confusion Matrix (Normalized)\nRed lines = antibiotic groups | Green line = DMSO/Control | Overall Accuracy: {accuracy:.2%}', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=4)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_89class_full.png'), dpi=150)
    plt.close()
    print(f"Saved: confusion_matrix_89class_full.png")
    
    # Raw count version with lines
    fig, ax = plt.subplots(figsize=(34, 30))
    sns.heatmap(cm, 
                xticklabels=present_class_names, 
                yticklabels=present_class_names,
                cmap='Blues', 
                annot=False,
                fmt='d',
                cbar_kws={'label': 'Count'},
                ax=ax)
    
    add_group_lines(ax, present_class_names, dmso_idx)
    
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('True', fontsize=8)
    ax.set_title(f'89-Class Confusion Matrix (Raw Counts)\nRed lines = antibiotic groups | Green line = DMSO/Control | Overall Accuracy: {accuracy:.2%}', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=4)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_89class_raw.png'), dpi=150)
    plt.close()
    print(f"Saved: confusion_matrix_89class_raw.png")
    
    # Top 20
    top_classes = metrics_df.head(20)['class'].tolist()
    top_indices = [present_class_names.index(c) for c in top_classes]
    cm_top = cm[np.ix_(top_indices, top_indices)]
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_top, xticklabels=top_classes, yticklabels=top_classes, 
                cmap='Blues', annot=True, fmt='d')
    plt.xlabel('Predicted', fontsize=10)
    plt.ylabel('True', fontsize=10)
    plt.title('Top 20 Performing Classes (Raw Counts)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_top20.png'), dpi=150)
    plt.close()
    
    # Bottom 20
    bottom_classes = metrics_df.tail(20)['class'].tolist()
    bottom_indices = [present_class_names.index(c) for c in bottom_classes]
    cm_bottom = cm[np.ix_(bottom_indices, bottom_indices)]
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_bottom, xticklabels=bottom_classes, yticklabels=bottom_classes, 
                cmap='Blues', annot=True, fmt='d')
    plt.xlabel('Predicted', fontsize=10)
    plt.ylabel('True', fontsize=10)
    plt.title('Bottom 20 Performing Classes (Raw Counts)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_bottom20.png'), dpi=150)
    plt.close()
    
    # Save raw confusion matrix
    cm_df = pd.DataFrame(cm, index=present_class_names, columns=present_class_names)
    cm_df.to_csv(os.path.join(output_dir, 'confusion_matrix_89class.csv'))
    
    # Summary
    with open(os.path.join(output_dir, 'summary_89class.txt'), 'w') as f:
        f.write("89-Class Drug Classification Summary (Majority Voting)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total classes: {num_classes}\n")
        f.write(f"Overall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Total predictions: {len(df_filtered)}\n\n")
        f.write("Top 10 performing classes:\n")
        f.write(metrics_df.head(10).to_string(index=False))
        f.write("\n\nBottom 10 performing classes:\n")
        f.write(metrics_df.tail(10).to_string(index=False))
    
    print(f"\nResults saved to: {output_dir}")
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Generate 89-class confusion matrix for drug predictions')
    parser.add_argument('--fold', type=str, default='Plate_1',
                        help='Fold to analyze (e.g., Plate_1, Plate_2)')
    parser.add_argument('--checkpoint', type=str, default='best_model_acc.pth',
                        help='Checkpoint filename used for prediction')
    parser.add_argument('--neighborhood', type=int, default=3,
                        help='Neighborhood size used for prediction')
    
    args = parser.parse_args()
    
    # Construct prediction file path
    checkpoint_name = args.checkpoint.replace('.pth', '')
    predictions_csv = os.path.join(
        SCRIPT_DIR, 'drug', f'fold_{args.fold}',
        f'predictions_all_crops_mil_{checkpoint_name}_n{args.neighborhood}.csv'
    )
    
    if not os.path.exists(predictions_csv):
        print(f"ERROR: Predictions file not found: {predictions_csv}")
        return
    
    output_dir = os.path.join(SCRIPT_DIR, 'drug', f'fold_{args.fold}', 'confusion_89class')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load predictions
    print(f"Loading predictions from: {predictions_csv}")
    df = pd.read_csv(predictions_csv)
    print(f"Total predictions: {len(df)}")
    print(f"Unique images: {df['image_path'].nunique()}")
    
    # Load ground truth mapping
    print("Loading ground truth from IC50 mapping...")
    ic50_data = load_ground_truth_from_mapping()
    
    # Perform majority voting
    df_voted = majority_vote_per_image_89class(df, ic50_data)
    
    # Create confusion matrix
    accuracy = create_full_confusion_matrix(df_voted, output_dir)
    
    print(f"\n=== DONE ===")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


if __name__ == '__main__':
    main()