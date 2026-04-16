#!/usr/bin/env python3
"""
Create confusion matrix using the most confident crop per image.
This uses: for each image, take the crop with highest confidence as the image prediction.
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
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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
    match = re.search(r'Well(\w\d+)_', filename)
    if match:
        return match.group(1)
    return None

def get_label_from_well(plate, well):
    if plate not in plate_data:
        return None
    row = well[0]
    col = str(int(well[1:]))
    if row in plate_data[plate] and col in plate_data[plate][row]:
        return plate_data[plate][row][col]['id']
    return None

def get_ground_truth_from_path(image_path):
    dirname = os.path.dirname(image_path)
    plate = os.path.basename(dirname)
    filename = os.path.basename(image_path)
    well = extract_well_from_filename(filename)
    if well:
        return get_label_from_well(plate, well)
    return None

def get_gene_family(label):
    if label is None:
        return None
    if '_' in label:
        return label.rsplit('_', 1)[0]
    return label

def plot_confusion_matrix(cm, labels, title, output_path, accuracy=None):
    """Plot confusion matrix with proper figure size."""
    n_classes = len(labels)
    
    # Dynamic figure size based on number of classes
    fig_size = max(20, n_classes / 4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
    
    # Plot
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax)
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    if accuracy is not None:
        ax.set_title(f'{title}\nAccuracy: {accuracy:.2f}%', fontsize=14)
    else:
        ax.set_title(title, fontsize=14)
    
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot confusion matrix using confident crops')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to crop predictions CSV')
    parser.add_argument('--top_n', type=int, default=None,
                        help='Number of top confident crops to use')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Confidence threshold (0-1). Only use crops above this threshold')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename')
    args = parser.parse_args()
    
    top_n = args.top_n
    threshold = args.threshold
    
    # Find CSV file
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = os.path.join(SCRIPT_DIR, 'crop_predictions_test.csv')
    
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        return
    
    print(f"Loading predictions from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} crop predictions")
    
    # Get ground truth
    print("Computing ground truth labels...")
    df['true_label'] = df['image_path'].apply(get_ground_truth_from_path)
    
    valid = df['true_label'].notna()
    df_valid = df[valid].copy()
    print(f"Crops with ground truth: {len(df_valid)}/{len(df)} ({100*len(df_valid)/len(df):.1f}%)")
    
    all_names = sorted(df_valid['true_label'].unique(), key=lambda x: label_to_idx.get(x, 999))
    
    # =========================================================================
    # IMAGE-LEVEL: Use TOP N OR confidence threshold
    # =========================================================================
    if threshold is not None:
        print("\n" + "="*60)
        print(f"IMAGE-LEVEL: CROPS WITH CONFIDENCE > {threshold*100:.1f}%")
        print("="*60)
    else:
        print("\n" + "="*60)
        print(f"IMAGE-LEVEL: TOP {top_n} MOST CONFIDENT CROPS PREDICTION")
        print("="*60)
    
    image_results = []
    
    for img_name, group in tqdm(df_valid.groupby('image_name'), desc="Processing images"):
        true_label = group['true_label'].iloc[0]
        
        # Filter crops based on threshold or top_n
        if threshold is not None:
            confident_crops = group[group['confidence'] >= threshold]
            # Fallback to all crops if none meet threshold
            if len(confident_crops) == 0:
                confident_crops = group
        else:
            confident_crops = group.nlargest(top_n, 'confidence')
        
        num_crops = len(confident_crops)
        
        # Method 1: Majority vote among confident crops
        pred_counts = Counter(confident_crops['predicted_class_name'].values)
        majority_pred = pred_counts.most_common(1)[0][0]
        
        # Method 2: Mean probabilities of confident crops
        probs_list = [np.array(json.loads(p)) for p in confident_crops['probs_json'].values]
        mean_probs = np.mean(probs_list, axis=0)
        mean_pred_idx = np.argmax(mean_probs)
        mean_pred = idx_to_label.get(mean_pred_idx, 'Unknown')
        
        # Method 3: Sum of probabilities
        sum_probs = np.sum(probs_list, axis=0)
        sum_pred_idx = np.argmax(sum_probs)
        sum_pred = idx_to_label.get(sum_pred_idx, 'Unknown')
        
        max_conf = confident_crops['confidence'].max()
        
        image_results.append({
            'image_name': img_name,
            'true_label': true_label,
            'pred_majority': majority_pred,
            'pred_mean': mean_pred,
            'pred_sum': sum_pred,
            'max_confidence': max_conf,
            'num_crops_used': num_crops
        })
    
    image_df = pd.DataFrame(image_results)
    
    # Calculate accuracies for each method
    true_img = image_df['true_label'].values
    
    # Majority vote
    pred_img_majority = image_df['pred_majority'].values
    cm_image_majority = confusion_matrix(true_img, pred_img_majority, labels=all_names)
    acc_image_majority = accuracy_score(true_img, pred_img_majority)
    
    # Mean probs
    pred_img_mean = image_df['pred_mean'].values
    cm_image_mean = confusion_matrix(true_img, pred_img_mean, labels=all_names)
    acc_image_mean = accuracy_score(true_img, pred_img_mean)
    
    # Sum probs
    pred_img_sum = image_df['pred_sum'].values
    cm_image_sum = confusion_matrix(true_img, pred_img_sum, labels=all_names)
    acc_image_sum = accuracy_score(true_img, pred_img_sum)
    
    if threshold is not None:
        avg_crops = image_df['num_crops_used'].mean()
        print(f"\nImage-level accuracy (threshold > {threshold*100:.1f}%):")
        print(f"  - Avg crops used per image: {avg_crops:.1f}")
        print(f"  - Majority vote: {100*acc_image_majority:.2f}%")
        print(f"  - Mean probs:     {100*acc_image_mean:.2f}%")
        print(f"  - Sum probs:      {100*acc_image_sum:.2f}%")
    else:
        print(f"\nImage-level accuracy (Top {top_n} crops):")
        print(f"  - Majority vote: {100*acc_image_majority:.2f}%")
        print(f"  - Mean probs:     {100*acc_image_mean:.2f}%")
        print(f"  - Sum probs:      {100*acc_image_sum:.2f}%")
    
    # Use best method for final output
    best_method = 'mean' if acc_image_mean >= max(acc_image_majority, acc_image_sum) else ('majority' if acc_image_majority >= acc_image_sum else 'sum')
    best_acc = max(acc_image_majority, acc_image_mean, acc_image_sum)
    print(f"\nBest method: {best_method} ({100*best_acc:.2f}%)")
    
    # Save per-image results
    if threshold is not None:
        output_csv = args.output if args.output else os.path.join(SCRIPT_DIR, f'image_predictions_thresh{threshold}.csv')
    else:
        output_csv = args.output if args.output else os.path.join(SCRIPT_DIR, f'image_predictions_top{top_n}.csv')
    image_df.to_csv(output_csv, index=False)
    print(f"Saved image predictions to: {output_csv}")
    
    # Plot confusion matrix for best method
    if best_method == 'majority':
        cm_best = cm_image_majority
        best_pred = pred_img_majority
    elif best_method == 'mean':
        cm_best = cm_image_mean
        best_pred = pred_img_mean
    else:
        cm_best = cm_image_sum
        best_pred = pred_img_sum
    
    if threshold is not None:
        output_png = args.output.replace('.csv', '.png') if args.output else os.path.join(SCRIPT_DIR, f'confusion_matrix_thresh{threshold}.png')
        title_suffix = f"Confidence > {threshold*100:.1f}%"
    else:
        output_png = args.output.replace('.csv', '.png') if args.output else os.path.join(SCRIPT_DIR, f'confusion_matrix_top{top_n}.png')
        title_suffix = f"Top {top_n} Crops"
    
    plot_confusion_matrix(
        cm_best, all_names,
        f'Image-Level ({title_suffix}, {best_method})\nn={len(image_df)}',
        output_png,
        accuracy=100*best_acc
    )
    
    # =========================================================================
    # WELL-LEVEL: Aggregate images per well
    # =========================================================================
    if threshold is not None:
        print("\n" + "="*60)
        print(f"WELL-LEVEL: CROPS WITH CONFIDENCE > {threshold*100:.1f}%")
    else:
        print("\n" + "="*60)
        print(f"WELL-LEVEL: TOP {top_n} MOST CONFIDENT IMAGES PER WELL")
    
    # Use the best prediction from image-level for each image
    image_df['best_pred'] = image_df[f'pred_{best_method}']
    image_df['best_conf'] = image_df['max_confidence']
    
    # Extract well from image path
    def get_well(path):
        filename = os.path.basename(path)
        match = re.search(r'Well(\w\d+)_', filename)
        return match.group(1) if match else None
    
    image_df['well'] = image_df['image_name'].apply(get_well)
    image_df['plate'] = image_df['image_name'].apply(lambda x: os.path.dirname(
        df_valid[df_valid['image_name'] == x]['image_path'].iloc[0]
    ).split('/')[-1])
    
    # Get well's true label from plate_well_id_path.json
    def get_well_true_label(row):
        plate = row['plate']
        well = row['well']
        if plate in plate_data and well:
            row_letter = well[0]
            col_num = str(int(well[1:]))
            if row_letter in plate_data[plate] and col_num in plate_data[plate][row_letter]:
                return plate_data[plate][row_letter][col_num]['id']
        return None
    
    image_df['well_true_label'] = image_df.apply(get_well_true_label, axis=1)
    
    # For each well, take the image with highest confidence from top N crops
    well_results = []
    for (plate, well), group in image_df.groupby(['plate', 'well']):
        if group['well_true_label'].iloc[0] is None:
            continue
        
        # Get the image with highest confidence in this well
        best_image = group.loc[group['best_conf'].idxmax()]
        
        well_results.append({
            'plate': plate,
            'well': well,
            'true_label': best_image['well_true_label'],
            'pred_topn': best_image['best_pred'],
            'confidence': best_image['best_conf'],
            'is_correct': best_image['best_pred'] == best_image['well_true_label'],
            'num_images': len(group)
        })
    
    well_df = pd.DataFrame(well_results)
    
    true_well = well_df['true_label'].values
    pred_well = well_df['pred_topn'].values
    
    cm_well = confusion_matrix(true_well, pred_well, labels=all_names)
    acc_well = accuracy_score(true_well, pred_well)
    
    if threshold is not None:
        print(f"Well-level accuracy (threshold > {threshold*100:.1f}%): {100*acc_well:.2f}%")
        well_csv = os.path.join(SCRIPT_DIR, f'well_predictions_thresh{threshold}.csv')
        well_png = os.path.join(SCRIPT_DIR, f'confusion_matrix_well_thresh{threshold}.png')
    else:
        print(f"Well-level accuracy (Top {top_n} crops): {100*acc_well:.2f}%")
        well_csv = os.path.join(SCRIPT_DIR, f'well_predictions_top{top_n}.csv')
        well_png = os.path.join(SCRIPT_DIR, f'confusion_matrix_well_top{top_n}.png')
    
    print(f"Correct predictions: {well_df['is_correct'].sum()}/{len(well_df)}")
    
    # Save well-level results
    well_df.to_csv(well_csv, index=False)
    print(f"Saved well predictions to: {well_csv}")
    
    # Plot well-level confusion matrix
    if threshold is not None:
        title_suffix = f"Confidence > {threshold*100:.1f}%"
    else:
        title_suffix = f"Top {top_n} Crops"
    
    plot_confusion_matrix(
        cm_well, all_names,
        f'Well-Level ({title_suffix})\nn={len(well_df)}',
        well_png,
        accuracy=100*acc_well
    )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    if threshold is not None:
        method_desc = f"Threshold > {threshold*100:.1f}%"
    else:
        method_desc = f"Top {top_n} Crops"
    
    print("\n" + "="*60)
    print(f"SUMMARY: {method_desc}")
    print("="*60)
    print(f"Best aggregation method: {best_method}")
    print(f"{'Level':<20} {'Count':<10} {'Accuracy':<15}")
    print("-" * 45)
    print(f"{'Image':<20} {len(image_df):<10} {100*best_acc:.2f}%")
    print(f"{'Well':<20} {len(well_df):<10} {100*acc_well:.2f}%")
    
    # Save summary
    summary = {
        'method': method_desc,
        'best_method': best_method,
        'image_level_accuracy': float(best_acc),
        'image_count': len(image_df),
        'well_level_accuracy': float(acc_well),
        'well_count': len(well_df),
    }
    
    if threshold is not None:
        summary_path = os.path.join(SCRIPT_DIR, f'thresh{threshold}_summary.json')
    else:
        summary_path = os.path.join(SCRIPT_DIR, f'top{top_n}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

if __name__ == '__main__':
    main()