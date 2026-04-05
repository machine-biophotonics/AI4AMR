#!/usr/bin/env python3
"""
Plot confusion matrices at 3 levels with:
- Red box around diagonal (correct predictions)
- Yellow box around same gene family but different number (e.g., ftsZ_1 vs ftsZ_2)
"""

import os
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from collections import Counter, defaultdict
from tqdm import tqdm
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Load classes
classes = {}
with open(os.path.join(SCRIPT_DIR, 'classes.txt'), 'r') as f:
    for line in f:
        idx, name = line.strip().split(',', 1)
        classes[int(idx)] = name

idx_to_label = classes
label_to_idx = {v: k for k, v in classes.items()}
class_names = [idx_to_label[i] for i in range(len(classes))]

# Load plate mappings for ground truth
with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json'), 'r') as f:
    plate_data = json.load(f)

def extract_well_from_filename(filename):
    """Extract well ID from filename like WellA01_xxx"""
    match = re.search(r'Well(\w\d+)_', filename)
    if match:
        return match.group(1)
    return None

def get_label_from_well(plate, well):
    """Get label from plate and well ID."""
    if plate not in plate_data:
        return None
    row = well[0]
    col = str(int(well[1:]))
    if row in plate_data[plate] and col in plate_data[plate][row]:
        return plate_data[plate][row][col]['id']
    return None

def get_ground_truth_from_path(image_path):
    """Get ground truth from full image path."""
    dirname = os.path.dirname(image_path)
    plate = os.path.basename(dirname)
    image_name = os.path.basename(image_path)
    well = extract_well_from_filename(image_name)
    
    if well and plate in plate_data:
        row = well[0]
        col = str(int(well[1:]))
        if row in plate_data[plate] and col in plate_data[plate][row]:
            return plate_data[plate][row][col]['id']
    return None

def get_gene_family(class_name):
    """Extract gene family from class name.
    
    Examples:
        'ftsZ_1', 'ftsZ_2', 'ftsZ_3' -> 'ftsZ'
        'WT NC_1' -> 'WT NC'
        'NC_1' -> 'NC'
        'secA_1' -> 'secA'
    """
    # Remove trailing _N suffix
    match = re.match(r'^(.+)_\d+$', class_name)
    if match:
        return match.group(1)
    return class_name

def find_same_gene_families(class_names):
    """Find groups of classes that share the same gene family.
    
    Returns a dict: {family_name: [class_names]}
    """
    families = defaultdict(list)
    for name in class_names:
        family = get_gene_family(name)
        families[family].append(name)
    
    # Only return families with multiple members
    return {k: v for k, v in families.items() if len(v) > 1}

def plot_confusion_matrix(cm, class_names, title, save_path, accuracy=None, figsize=(22, 20)):
    """Plot confusion matrix with red diagonal boxes and yellow same-gene-family boxes.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        save_path: Where to save
        accuracy: Accuracy percentage to display in title
        figsize: Figure size
    """
    
    # Find same-gene-family groups
    families = find_same_gene_families(class_names)
    
    n_classes = len(class_names)
    
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]))
    
    # Compute accuracy if not provided
    if accuracy is None:
        accuracy = 100 * np.trace(cm) / np.sum(cm)
    
    for idx, (ax, fmt, title_suffix) in enumerate([
        (axes[0], 'd', '(Counts)'),
        (axes[1], '.2f', '(Normalized by True Class)')
    ]):
        # Normalize if needed
        if fmt == '.2f':
            cm_plot = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        else:
            cm_plot = cm
        
        # Create heatmap
        sns.heatmap(cm_plot, ax=ax, cmap='Blues', xticklabels=class_names,
                    yticklabels=class_names, fmt=fmt, 
                    vmin=0, vmax=1 if fmt == '.2f' else None,
                    cbar_kws={'shrink': 0.8})
        
        # Draw RED boxes around diagonal (correct predictions)
        for i in range(n_classes):
            # Red rectangle around diagonal cell
            rect = patches.Rectangle((i, i), 1, 1, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        
        # Draw YELLOW boxes around same gene family, different number
        for family, members in families.items():
            # Get indices of all members in this family
            member_indices = [class_names.index(m) for m in members if m in class_names]
            
            # Draw yellow boxes for off-diagonal same-family cells
            for i in member_indices:
                for j in member_indices:
                    if i != j:  # Only off-diagonal
                        # Yellow semi-transparent overlay
                        rect = patches.Rectangle((j, i), 1, 1,
                                                 linewidth=2, edgecolor='gold', 
                                                 facecolor='yellow', alpha=0.2)
                        ax.add_patch(rect)
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title(f'{title} {title_suffix}', fontsize=14)
        ax.tick_params(axis='x', rotation=90, labelsize=7)
        ax.tick_params(axis='y', rotation=0, labelsize=7)
    
    # Add accuracy text box in center top
    accuracy_text = f'ACCURACY: {accuracy:.2f}%'
    fig.text(0.5, 1.04, accuracy_text, ha='center', va='bottom', fontsize=18, fontweight='bold',
             color='darkgreen', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', edgecolor='darkgreen', alpha=0.8))
    
    # Add legend
    red_patch = patches.Patch(facecolor='none', edgecolor='red', linewidth=2, label='Correct (diagonal)')
    yellow_patch = patches.Patch(facecolor='yellow', edgecolor='gold', linewidth=2, alpha=0.3, 
                                  label='Same gene, different # (off-diagonal)')
    fig.legend(handles=[red_patch, yellow_patch], loc='upper center', 
               ncol=2, fontsize=11, bbox_to_anchor=(0.5, 0.99))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_summary_statistics(cm, class_names, families, save_path):
    """Plot summary of within-family vs between-family confusion."""
    n_classes = len(class_names)
    
    within_family_correct = 0
    within_family_total = 0
    between_family_correct = 0
    between_family_total = 0
    
    family_set = {m for members in families.values() for m in members}
    
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            count = cm[i, j]
            
            true_family = get_gene_family(true_name)
            pred_family = get_gene_family(pred_name)
            
            if true_name in family_set:
                within_family_total += count
                if i == j:
                    within_family_correct += count
            else:
                between_family_total += count
                if i == j:
                    between_family_correct += count
    
    # Print summary
    print("\n" + "="*60)
    print("WITHIN-FAMILY vs BETWEEN-FAMILY CONFUSION")
    print("="*60)
    if within_family_total > 0:
        print(f"Within gene family: {within_family_correct}/{within_family_total} = {100*within_family_correct/within_family_total:.1f}% correct")
    if between_family_total > 0:
        print(f"Between gene family: {between_family_correct}/{between_family_total} = {100*between_family_correct/between_family_total:.1f}% correct")
    
    # Count same-family confusions
    family_confusion = defaultdict(int)
    for true_name in class_names:
        for pred_name in class_names:
            if true_name != pred_name:
                true_family = get_gene_family(true_name)
                pred_family = get_gene_family(pred_name)
                if true_family == pred_family and true_family in families:
                    i = class_names.index(true_name)
                    j = class_names.index(pred_name)
                    family_confusion[true_family] += cm[i, j]
    
    # Top confused families
    print(f"\nTop 10 most confused within gene families:")
    for family, count in sorted(family_confusion.items(), key=lambda x: -x[1])[:10]:
        members = families[family]
        print(f"  {family} ({', '.join(members)}): {count} confusions")
    
    return within_family_correct, within_family_total, between_family_correct, between_family_total

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot confusion matrices')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to crop predictions CSV (default: auto-detect)')
    parser.add_argument('--exclude_classes', nargs='*', default=[],
                        help='List of class names to exclude from analysis')
    args = parser.parse_args()
    
    # Find CSV file
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = os.path.join(SCRIPT_DIR, 'crop_predictions_544_5.csv')
        if not os.path.exists(csv_path):
            csv_path = os.path.join(SCRIPT_DIR, 'crop_predictions_full.csv')
        if not os.path.exists(csv_path):
            csv_path = os.path.join(SCRIPT_DIR, 'crop_predictions_test.csv')
    
    if not os.path.exists(csv_path):
        print(f"ERROR: No CSV file found in {SCRIPT_DIR}")
        print("Available files:")
        for f in os.listdir(SCRIPT_DIR):
            if f.endswith('.csv'):
                print(f"  - {f}")
        return
    
    print(f"Loading predictions from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} crop predictions")
    
    # Get ground truth for each row
    print("Computing ground truth labels...")
    df['true_label'] = df['image_path'].apply(get_ground_truth_from_path)
    
    valid = df['true_label'].notna()
    print(f"Crops with ground truth: {valid.sum()}/{len(df)} ({100*valid.sum()/len(df):.1f}%)")
    
    if valid.sum() == 0:
        print("ERROR: No ground truth found.")
        return
    
    df_valid = df[valid].copy()
    
    # Exclude specified classes
    if args.exclude_classes:
        print(f"Excluding {len(args.exclude_classes)} classes from analysis: {args.exclude_classes}")
        exclude_set = set(args.exclude_classes)
        df_valid = df_valid[~df_valid['true_label'].isin(exclude_set)]
        df_valid = df_valid[~df_valid['predicted_class_name'].isin(exclude_set)]
        print(f"After exclusion: {len(df_valid)} crops")
    
    # Get unique classes
    true_classes = set(df_valid['true_label'].unique())
    pred_classes = set(df_valid['predicted_class_name'].unique())
    all_classes = sorted(true_classes | pred_classes)
    all_names = [c for c in all_classes if c in label_to_idx]
    
    print(f"\nClasses in data: {len(all_classes)}")
    
    # Find gene families
    families = find_same_gene_families(all_names)
    print(f"Found {len(families)} gene families with multiple members")
    
    # =========================================================================
    # LEVEL 1: CROP-LEVEL CONFUSION MATRIX
    # =========================================================================
    print("\n" + "="*60)
    print("LEVEL 1: CROP-LEVEL CONFUSION MATRIX")
    print("="*60)
    
    true_crop = df_valid['true_label'].values
    pred_crop = df_valid['predicted_class_name'].values
    
    cm_crop = confusion_matrix(true_crop, pred_crop, labels=all_names)
    acc_crop = accuracy_score(true_crop, pred_crop)
    
    print(f"Crop-level accuracy: {100*acc_crop:.2f}%")
    
    plot_confusion_matrix(
        cm_crop, all_names,
        f'Crop-Level Confusion Matrix (n={len(df_valid)})',
        os.path.join(SCRIPT_DIR, 'confusion_matrix_crop_level.png'),
        accuracy=100*acc_crop
    )
    
    plot_summary_statistics(cm_crop, all_names, families,
                           os.path.join(SCRIPT_DIR, 'crop_summary.png'))
    
    # =========================================================================
    # LEVEL 2: IMAGE-LEVEL CONFUSION MATRIX
    # =========================================================================
    print("\n" + "="*60)
    print("LEVEL 2: IMAGE-LEVEL CONFUSION MATRIX")
    print("="*60)
    
    image_results = []
    
    for img_name, group in tqdm(df_valid.groupby('image_name'), desc="Aggregating crops"):
        true_label = group['true_label'].iloc[0]
        
        # Majority vote
        pred_counts = Counter(group['predicted_class_name'].values)
        majority_pred = pred_counts.most_common(1)[0][0]
        
        # Mean probability
        probs_list = [np.array(json.loads(p)) for p in group['probs_json'].values]
        mean_probs = np.mean(probs_list, axis=0)
        mean_pred_idx = np.argmax(mean_probs)
        mean_pred = idx_to_label.get(mean_pred_idx, 'Unknown')
        
        image_results.append({
            'image_name': img_name,
            'true_label': true_label,
            'pred_majority': majority_pred,
            'pred_mean_probs': mean_pred,
        })
    
    image_df = pd.DataFrame(image_results)
    
    # Majority vote
    true_img = image_df['true_label'].values
    pred_img = image_df['pred_majority'].values
    cm_image = confusion_matrix(true_img, pred_img, labels=all_names)
    acc_image = accuracy_score(true_img, pred_img)
    
    print(f"Image-level accuracy (majority vote): {100*acc_image:.2f}%")
    
    plot_confusion_matrix(
        cm_image, all_names,
        f'Image-Level Confusion Matrix (n={len(image_df)}, Majority Vote)',
        os.path.join(SCRIPT_DIR, 'confusion_matrix_image_level.png'),
        accuracy=100*acc_image
    )
    
    # Mean probs
    pred_img_mean = image_df['pred_mean_probs'].values
    cm_image_mean = confusion_matrix(true_img, pred_img_mean, labels=all_names)
    acc_image_mean = accuracy_score(true_img, pred_img_mean)
    
    print(f"Image-level accuracy (mean probs): {100*acc_image_mean:.2f}%")
    
    plot_confusion_matrix(
        cm_image_mean, all_names,
        f'Image-Level Confusion Matrix (n={len(image_df)}, Mean Probs)',
        os.path.join(SCRIPT_DIR, 'confusion_matrix_image_level_mean_probs.png'),
        accuracy=100*acc_image_mean
    )
    
    # =========================================================================
    # LEVEL 3: WELL-LEVEL CONFUSION MATRIX
    # =========================================================================
    print("\n" + "="*60)
    print("LEVEL 3: WELL-LEVEL CONFUSION MATRIX")
    print("="*60)
    
    image_df['well'] = image_df['image_name'].apply(extract_well_from_filename)
    
    well_results = []
    
    for well, group in tqdm(image_df.groupby('well'), desc="Aggregating wells"):
        true_label = group['true_label'].iloc[0]
        
        # Majority vote
        pred_counts = Counter(group['pred_majority'].values)
        majority_pred = pred_counts.most_common(1)[0][0]
        
        # Mean probs across all images
        probs_list = []
        for _, row in group.iterrows():
            img_crops = df_valid[df_valid['image_name'] == row['image_name']]
            img_probs = np.zeros(len(classes))
            for p in img_crops['probs_json'].values:
                try:
                    parsed = json.loads(p)
                    if len(parsed) == len(classes):
                        img_probs += np.array(parsed) / len(img_crops)
                    else:
                        # Skip if probability array size doesn't match
                        pass
                except (json.JSONDecodeError, TypeError):
                    pass
            probs_list.append(img_probs)
        
        mean_probs_all = np.mean(probs_list, axis=0)
        mean_pred_idx = np.argmax(mean_probs_all)
        mean_pred = idx_to_label.get(mean_pred_idx, 'Unknown')
        
        well_results.append({
            'well': well,
            'true_label': true_label,
            'pred_majority': majority_pred,
            'pred_mean_probs': mean_pred,
        })
    
    well_df = pd.DataFrame(well_results)
    
    # Majority vote
    true_well = well_df['true_label'].values
    pred_well = well_df['pred_majority'].values
    cm_well = confusion_matrix(true_well, pred_well, labels=all_names)
    acc_well = accuracy_score(true_well, pred_well)
    
    print(f"Well-level accuracy (majority vote): {100*acc_well:.2f}%")
    
    plot_confusion_matrix(
        cm_well, all_names,
        f'Well-Level Confusion Matrix (n={len(well_df)}, Majority Vote)',
        os.path.join(SCRIPT_DIR, 'confusion_matrix_well_level.png'),
        accuracy=100*acc_well
    )
    
    plot_summary_statistics(cm_well, all_names, families,
                           os.path.join(SCRIPT_DIR, 'well_summary.png'))
    
    # Mean probs
    pred_well_mean = well_df['pred_mean_probs'].values
    cm_well_mean = confusion_matrix(true_well, pred_well_mean, labels=all_names)
    acc_well_mean = accuracy_score(true_well, pred_well_mean)
    
    print(f"Well-level accuracy (mean probs): {100*acc_well_mean:.2f}%")
    
    plot_confusion_matrix(
        cm_well_mean, all_names,
        f'Well-Level Confusion Matrix (n={len(well_df)}, Mean Probs)',
        os.path.join(SCRIPT_DIR, 'confusion_matrix_well_level_mean_probs.png'),
        accuracy=100*acc_well_mean
    )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY: ACCURACY AT EACH LEVEL")
    print("="*60)
    print(f"{'Level':<15} {'Count':<10} {'Majority Vote':<20} {'Mean Probs':<20}")
    print("-"*65)
    print(f"{'Crop':<15} {len(df_valid):<10} {100*acc_crop:<20.2f} {'N/A':<20}")
    print(f"{'Image':<15} {len(image_df):<10} {100*acc_image:<20.2f} {100*acc_image_mean:<20.2f}")
    print(f"{'Well':<15} {len(well_df):<10} {100*acc_well:<20.2f} {100*acc_well_mean:<20.2f}")
    
    # Save results
    results = {
        'crop_level': {'accuracy': float(acc_crop), 'num_samples': len(df_valid)},
        'image_level': {'accuracy_majority': float(acc_image), 'accuracy_mean_probs': float(acc_image_mean), 'num_images': len(image_df)},
        'well_level': {'accuracy_majority': float(acc_well), 'accuracy_mean_probs': float(acc_well_mean), 'num_wells': len(well_df)},
        'gene_families': {k: v for k, v in families.items()}
    }
    
    results_path = os.path.join(SCRIPT_DIR, 'confusion_matrix_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

if __name__ == '__main__':
    main()
