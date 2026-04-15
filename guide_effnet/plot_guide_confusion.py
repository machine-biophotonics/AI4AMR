#!/usr/bin/env python3
"""
Confusion matrix for guide_effnet predictions - grouped by guide number
"""

import os
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter
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

# Load plate mappings
with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json'), 'r') as f:
    plate_data = json.load(f)

def extract_well_from_filename(filename):
    match = re.search(r'Well(\w\d+)_', filename)
    return match.group(1) if match else None

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

def get_guide(label):
    """Extract guide number from label (e.g., 1 from 'dnaB_1')"""
    if label is None:
        return None
    if '_' in label:
        return int(label.split('_')[1])
    return None

def get_gene_family(label):
    """Extract gene family from label"""
    if label is None:
        return None
    if '_' in label:
        return label.split('_')[0]
    return label

def main():
    parser = argparse.ArgumentParser(description='Guide-level confusion matrix')
    parser.add_argument('--csv', type=str, default=None)
    args = parser.parse_args()
    
    # Load predictions
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = os.path.join(SCRIPT_DIR, 'guide3_crop_predictions.csv')
    
    print(f"Loading predictions from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} crop predictions")
    
    # Get ground truth
    print("Computing ground truth labels...")
    df['true_label'] = df['image_path'].apply(get_ground_truth_from_path)
    
    valid = df['true_label'].notna()
    df_valid = df[valid].copy()
    print(f"Crops with ground truth: {len(df_valid)}")
    
    # Extract guide numbers
    df_valid['true_guide'] = df_valid['true_label'].apply(get_guide)
    df_valid['pred_guide'] = df_valid['predicted_class_name'].apply(get_guide)
    
    print("\n" + "="*60)
    print("GUIDE-LEVEL ANALYSIS")
    print("="*60)
    print(f"\nTrue guide distribution:")
    print(df_valid['true_guide'].value_counts().sort_index())
    print(f"\nPredicted guide distribution:")
    print(df_valid['pred_guide'].value_counts().sort_index())
    
    # Guide-level confusion matrix
    guide_names = [1, 2, 3]
    true_guide = df_valid['true_guide'].values
    pred_guide = df_valid['pred_guide'].values
    
    cm_guide = confusion_matrix(true_guide, pred_guide, labels=guide_names)
    acc_guide = accuracy_score(true_guide, pred_guide)
    
    print(f"\nGuide-level accuracy: {100*acc_guide:.2f}%")
    print("\nGuide Confusion Matrix (rows=true, cols=pred):")
    print("        Pred")
    print("True    G1    G2    G3")
    for i, g in enumerate(guide_names):
        print(f"  G{g}   {cm_guide[i][0]:5d} {cm_guide[i][1]:5d} {cm_guide[i][2]:5d}")
    
    # Plot guide confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_guide, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Guide 1', 'Guide 2', 'Guide 3'],
                yticklabels=['Guide 1', 'Guide 2', 'Guide 3'], ax=ax)
    ax.set_xlabel('Predicted Guide', fontsize=12)
    ax.set_ylabel('True Guide', fontsize=12)
    ax.set_title(f'Guide-Level Confusion Matrix (n={len(df_valid)})\nAccuracy: {100*acc_guide:.2f}%', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'guide_confusion_matrix.png'), dpi=150)
    plt.close()
    print("\nSaved: guide_confusion_matrix.png")
    
    # =========================================================================
    # IMAGE-LEVEL ANALYSIS
    # =========================================================================
    print("\n" + "="*60)
    print("IMAGE-LEVEL ANALYSIS (per gene family, ignoring guide)")
    print("="*60)
    
    image_results = []
    for img_name, group in tqdm(df_valid.groupby('image_name'), desc="Processing images"):
        true_label = group['true_label'].iloc[0]
        true_family = get_gene_family(true_label)
        true_guide = get_guide(true_label)
        
        # Get prediction - use mean probabilities
        probs_list = [np.array(json.loads(p)) for p in group['probs_json'].values]
        mean_probs = np.mean(probs_list, axis=0)
        
        # Map to class names
        class_names = [idx_to_label[i] for i in range(len(mean_probs))]
        pred_idx = np.argmax(mean_probs)
        pred_label = class_names[pred_idx]
        pred_family = get_gene_family(pred_label)
        pred_guide = get_guide(pred_label)
        
        image_results.append({
            'image_name': img_name,
            'true_label': true_label,
            'true_family': true_family,
            'true_guide': true_guide,
            'pred_label': pred_label,
            'pred_family': pred_family,
            'pred_guide': pred_guide,
            'family_correct': true_family == pred_family,
            'guide_correct': true_guide == pred_guide
        })
    
    image_df = pd.DataFrame(image_results)
    
    # Gene family accuracy (ignoring guide)
    acc_family = image_df['family_correct'].mean()
    print(f"\nGene family accuracy (ignoring guide): {100*acc_family:.2f}%")
    
    # Guide prediction accuracy
    acc_guide_img = image_df['guide_correct'].mean()
    print(f"Guide prediction accuracy: {100*acc_guide_img:.2f}%")
    
    # Cross-tabulation: true guide vs predicted guide at image level
    print("\nImage-level: True Guide vs Predicted Guide")
    guide_crosstab = pd.crosstab(image_df['true_guide'], image_df['pred_guide'], 
                                  margins=True, margins_name='Total')
    print(guide_crosstab)
    
    # =========================================================================
    # GENE FAMILY CONFUSION MATRIX (ignoring guide number)
    # =========================================================================
    print("\n" + "="*60)
    print("GENE FAMILY CONFUSION MATRIX (ignoring guide)")
    print("="*60)
    
    families = sorted([f for f in image_df['true_family'].unique() if f is not None])
    
    # Create confusion matrix at gene family level
    true_family = image_df['true_family'].values
    pred_family = image_df['pred_family'].values
    
    family_cm = confusion_matrix(true_family, pred_family, labels=families)
    acc_family = accuracy_score(true_family, pred_family)
    
    print(f"\nGene family accuracy: {100*acc_family:.2f}%")
    print(f"Correct: {(true_family == pred_family).sum()}/{len(true_family)}")
    
    # Plot gene family confusion matrix
    fig_size = max(16, len(families) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
    sns.heatmap(family_cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=families, yticklabels=families, ax=ax)
    ax.set_xlabel('Predicted Gene Family', fontsize=12)
    ax.set_ylabel('True Gene Family', fontsize=12)
    ax.set_title(f'Gene Family Confusion Matrix (n={len(image_df)})\nAccuracy: {100*acc_family:.2f}%', fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'gene_family_confusion_matrix.png'), dpi=100)
    plt.close()
    print("\nSaved: gene_family_confusion_matrix.png")
    
    # Show top confused families
    print("\nTop 10 most confused gene families (true -> predicted):")
    confusion_pairs = []
    for i, true_fam in enumerate(families):
        for j, pred_fam in enumerate(families):
            if i != j and family_cm[i, j] > 0:
                confusion_pairs.append((true_fam, pred_fam, family_cm[i, j]))
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    for true_fam, pred_fam, count in confusion_pairs[:10]:
        print(f"  {true_fam} -> {pred_fam}: {count}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Gene family accuracy (ignoring guide): {100*acc_family:.2f}%")
    print(f"Total images: {len(image_df)}")
    
    # =========================================================================
    # WELL-LEVEL ANALYSIS
    # =========================================================================
    print("\n" + "="*60)
    print("WELL-LEVEL ANALYSIS")
    print("="*60)
    
    # Extract well from image path
    def get_well(path):
        filename = os.path.basename(path)
        match = re.search(r'Well(\w\d+)_', filename)
        return match.group(1) if match else None
    
    image_df['well'] = image_df['image_name'].apply(get_well)
    image_df['plate'] = image_df['image_name'].apply(lambda x: 
        os.path.dirname(df_valid[df_valid['image_name'] == x]['image_path'].iloc[0]).split('/')[-1]
    )
    
    # Get well's true label
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
    image_df['well_true_family'] = image_df['well_true_label'].apply(get_gene_family)
    
    # Aggregate to well level - use mean probabilities across all images in well
    well_results = []
    for (plate, well), group in image_df.groupby(['plate', 'well']):
        if group['well_true_label'].iloc[0] is None:
            continue
        
        true_family = group['well_true_family'].iloc[0]
        
        # Get predictions for all images in well
        # Use the best image prediction (highest confidence from mean probs)
        best_image = group.loc[group['pred_family'].apply(lambda x: True)]  # Take the most common prediction
        
        # Majority vote at well level
        family_counts = group['pred_family'].value_counts()
        pred_family = family_counts.index[0]
        
        well_results.append({
            'plate': plate,
            'well': well,
            'true_family': true_family,
            'pred_family': pred_family,
            'num_images': len(group)
        })
    
    well_df = pd.DataFrame(well_results)
    
    # Well-level accuracy
    true_well_family = well_df['true_family'].values
    pred_well_family = well_df['pred_family'].values
    acc_well = accuracy_score(true_well_family, pred_well_family)
    
    print(f"\nWell-level gene family accuracy: {100*acc_well:.2f}%")
    print(f"Correct: {(true_well_family == pred_well_family).sum()}/{len(well_df)}")
    
    # Well-level confusion matrix
    well_families = sorted([f for f in well_df['true_family'].unique() if f is not None])
    well_cm = confusion_matrix(true_well_family, pred_well_family, labels=well_families)
    
    fig_size = max(10, len(well_families) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
    sns.heatmap(well_cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=well_families, yticklabels=well_families, ax=ax)
    ax.set_xlabel('Predicted Gene Family', fontsize=12)
    ax.set_ylabel('True Gene Family', fontsize=12)
    ax.set_title(f'Well-Level Gene Family Confusion Matrix (n={len(well_df)})\nAccuracy: {100*acc_well:.2f}%', fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'well_gene_family_confusion_matrix.png'), dpi=100)
    plt.close()
    print("\nSaved: well_gene_family_confusion_matrix.png")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Image-level gene family accuracy: {100*acc_family:.2f}%")
    print(f"Well-level gene family accuracy: {100*acc_well:.2f}%")
    
    # Save summary
    summary = {
        'crop_level_guide_accuracy': float(acc_guide),
        'image_level_family_accuracy': float(acc_family),
        'image_level_guide_accuracy': float(acc_guide_img),
        'image_level_exact_match': float(accuracy_score(image_df['true_label'], image_df['pred_label'])),
    }
    with open(os.path.join(SCRIPT_DIR, 'guide_analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nSaved: guide_analysis_summary.json")

if __name__ == '__main__':
    main()