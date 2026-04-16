#!/usr/bin/env python3
"""
Generate all confusion matrices and metrics in one script.
Levels: crop, image, well
Hierarchies: guide (dnaB_1), gene (dnaB), pathway (DNA)

Features:
- Majority voting at each level (crop->image->well)
- Confusion matrices with red diagonal boxes and yellow boxes for same gene
- Saves metrics to CSV
"""

import numpy as np
import json
import os
import argparse
import ast
from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
matplotlib.use('Agg')

# Hierarchy mapping from gene_hierarchy.txt - exact pathway names
HIERARCHY = {
    'ftsZ': 'cell division', 'ftsI': 'cell division', 'murA': 'cell division', 'murC': 'cell division',
    'rpsA': 'cytoplasmic translation', 'rpsL': 'cytoplasmic translation', 
    'rplA': 'cytoplasmic translation', 'rplC': 'cytoplasmic translation',
    'mrdA': 'cell wall organization', 'mrcA': 'cell wall organization', 'mrcB': 'cell wall repair',
    'lpxA': 'lipid A biosynthetic process', 'lpxC': 'lipid A biosynthetic process',
    'lptA': 'Gram-negative-bacterium-type cell outer membrane assembly', 
    'lptC': 'Gram-negative-bacterium-type cell outer membrane assembly',
    'gyrA': 'DNA topological change', 'gyrB': 'DNA topological change',
    'rpoA': 'bacterial-type flagellum assembly', 'rpoB': 'bacterial-type flagellum assembly',
    'secA': 'intracellular protein transmembrane transport', 'secY': 'intracellular protein transmembrane transport',
    'msbA': 'lipid translocation',
    'folA': '10-formyltetrahydrofolate biosynthetic process', 'folP': 'folic acid biosynthetic process',
    'dnaE': 'DNA-templated DNA replication', 'dnaB': 'DNA replication',
    'parC': 'chromosome segregation', 'parE': 'chromosome organization',
}

# Family mapping - groups genes by base name prefix
FAMILY = {
    'ftsZ': 'Fts', 'ftsI': 'Fts', 'murA': 'Mur', 'murC': 'Mur',
    'rpsA': 'Ribosome', 'rpsL': 'Ribosome', 'rplA': 'Ribosome', 'rplC': 'Ribosome',
    'mrdA': 'Cell wall', 'mrcA': 'Cell wall', 'mrcB': 'Cell wall',
    'lpxA': 'Lipid A', 'lpxC': 'Lipid A',
    'lptA': 'LPS', 'lptC': 'LPS',
    'gyrA': 'Gyrase', 'gyrB': 'Gyrase',
    'rpoA': 'RNAP', 'rpoB': 'RNAP',
    'secA': 'Secretion', 'secY': 'Secretion',
    'msbA': 'Lipid',
    'folA': 'Folate', 'folP': 'Folate',
    'dnaE': 'DNA', 'dnaB': 'DNA',
    'parC': 'TopoIV', 'parE': 'TopoIV',
}


def get_base_gene(label):
    if not label or label == 'nan':
        return 'Unknown'
    if '_' in str(label):
        return str(label).rsplit('_', 1)[0]
    return str(label)


def get_pathway(label):
    base = get_base_gene(label)
    if 'WT' in str(base).upper() or 'NC' in str(base).upper():
        return 'WT NC'
    if base in HIERARCHY:
        return HIERARCHY[base]
    return 'Unknown'


def get_family(label):
    base = get_base_gene(label)
    if base in FAMILY:
        return FAMILY[base]
    return 'Unknown'


def plot_cm_with_boxes(cm_raw, cm_norm, labels, title, output_path, is_guide=False):
    """Plot two confusion matrices: raw counts and normalized."""
    n = len(labels)
    
    import seaborn as sns
    
    if is_guide:
        fig, ax = plt.subplots(figsize=(max(14, n*0.2), max(14, n*0.2)))
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, 
                    yticklabels=labels, ax=ax, vmin=0, vmax=1,
                    cbar_kws={'label': 'Normalized Frequency', 'shrink': 0.8},
                    linewidths=0.5, linecolor='white',
                    annot_kws={'size': 5},
                    square=True)
        
        for i, label in enumerate(labels):
            base = get_base_gene(label)
            same_gene_indices = [j for j, l in enumerate(labels) if get_base_gene(l) == base]
            if len(same_gene_indices) > 1:
                min_j = min(same_gene_indices)
                max_j = max(same_gene_indices)
                rect = patches.Rectangle((min_j, min_j), max_j - min_j + 1, max_j - min_j + 1,
                                          linewidth=3, edgecolor='#FFD700', facecolor='none', 
                                          linestyle='-', zorder=10)
                ax.add_patch(rect)
        
        for i in range(n):
            rect = patches.Rectangle((i, i), 1, 1, linewidth=2.5, edgecolor='#FF4444', 
                                      facecolor='none', zorder=10)
            ax.add_patch(rect)
        
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_title(f'{title}', fontsize=12, fontweight='bold')
        
        ax.set_xticks(np.arange(n) + 0.5, labels, rotation=90, fontsize=5)
        ax.set_yticks(np.arange(n) + 0.5, labels, rotation=0, fontsize=5)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        
    else:
        fig, axes = plt.subplots(1, 2, figsize=(max(24, n*0.35), max(12, n*0.2)))
        
        # Left: Raw counts with count and percentage
        ax1 = axes[0]
        
        row_sums = cm_raw.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_perc = cm_raw / row_sums
        
        annot = np.empty_like(cm_raw, dtype=object)
        for i in range(n):
            for j in range(n):
                if cm_raw[i, j] > 0:
                    annot[i, j] = f"{int(cm_raw[i, j])}\n({cm_perc[i, j]*100:.1f}%)"
                else:
                    annot[i, j] = ""
        
        sns.heatmap(cm_raw, annot=annot, fmt='', cmap='Blues', xticklabels=labels, 
                    yticklabels=labels, ax=ax1, vmin=0,
                    cbar_kws={'label': 'Count', 'shrink': 0.8},
                    linewidths=0.5, linecolor='white',
                    annot_kws={'size': 5},
                    square=True)
        
        ax1.set_xlabel('Predicted Label', fontsize=10)
        ax1.set_ylabel('True Label', fontsize=10)
        ax1.set_title(f'{title}\n(Counts)', fontsize=12, fontweight='bold')
        
        ax1.set_xticks(np.arange(n) + 0.5, labels, rotation=90, fontsize=5)
        ax1.set_yticks(np.arange(n) + 0.5, labels, rotation=0, fontsize=5)
        
        for spine in ax1.spines.values():
            spine.set_visible(False)
        
        # Right: Normalized
        ax2 = axes[1]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, 
                    yticklabels=labels, ax=ax2, vmin=0, vmax=1,
                    cbar_kws={'label': 'Normalized Frequency', 'shrink': 0.8},
                    linewidths=0.5, linecolor='white',
                    annot_kws={'size': 5},
                    square=True)
        
        for i, label in enumerate(labels):
            base = get_base_gene(label)
            same_gene_indices = [j for j, l in enumerate(labels) if get_base_gene(l) == base]
            if len(same_gene_indices) > 1:
                min_j = min(same_gene_indices)
                max_j = max(same_gene_indices)
                rect = patches.Rectangle((min_j, min_j), max_j - min_j + 1, max_j - min_j + 1,
                                          linewidth=3, edgecolor='#FFD700', facecolor='none', 
                                          linestyle='-', zorder=10)
                ax2.add_patch(rect)
        
        for i in range(n):
            rect = patches.Rectangle((i, i), 1, 1, linewidth=2.5, edgecolor='#FF4444', 
                                      facecolor='none', zorder=10)
            ax2.add_patch(rect)
        
        ax2.set_xlabel('Predicted Label', fontsize=10)
        ax2.set_ylabel('True Label', fontsize=10)
        ax2.set_title(f'{title}\n(Normalized)', fontsize=12, fontweight='bold')
        
        ax2.set_xticks(np.arange(n) + 0.5, labels, rotation=90, fontsize=5)
        ax2.set_yticks(np.arange(n) + 0.5, labels, rotation=0, fontsize=5)
        
        for spine in ax2.spines.values():
            spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def aggregate_crop_to_well(df):
    """Aggregate from crop -> image -> well with majority voting."""
    # Crop to Image
    image_results = []
    for img_name, group in df.groupby('image_name'):
        true_label = group['true_label'].iloc[0]
        pred_counts = Counter(group['predicted_class_name'].values)
        majority_pred = pred_counts.most_common(1)[0][0]
        well = group['well'].iloc[0] if 'well' in group.columns else None
        image_results.append({
            'image_name': img_name,
            'well': well,
            'true_label': true_label,
            'pred_majority': majority_pred
        })
    
    image_df = pd.DataFrame(image_results)
    
    # Image to Well
    well_results = []
    for well, group in image_df.groupby('well'):
        if pd.isna(well):
            continue
        true_label = group['true_label'].iloc[0]
        pred_counts = Counter(group['pred_majority'].values)
        majority_pred = pred_counts.most_common(1)[0][0]
        well_results.append({
            'well': well,
            'true_label': true_label,
            'pred_majority': majority_pred
        })
    
    well_df = pd.DataFrame(well_results)
    
    return image_df, well_df


def map_hierarchy(labels, level):
    if level == 'guide':
        return list(labels)
    elif level == 'gene':
        return [get_base_gene(l) for l in labels]
    elif level == 'pathway':
        return [get_pathway(l) for l in labels]
    elif level == 'family':
        return [get_family(l) for l in labels]
    else:
        return list(labels)


def main():
    parser = argparse.ArgumentParser(description='Generate all confusion matrices')
    parser.add_argument('--fold', type=str, default='P2', help='Fold to process (P1-P6, or all)')
    args = parser.parse_args()
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    folds = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6'] if args.fold == 'all' else [args.fold]
    
    all_results = []
    
    for fold in folds:
        fold_dir = os.path.join(SCRIPT_DIR, f'fold_{fold}')
        csv_path = os.path.join(fold_dir, 'predictions.csv')
        
        if not os.path.exists(csv_path):
            print(f"Skipping {fold}: no predictions.csv")
            continue
        
        output_dir = os.path.join(fold_dir, 'confusion')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"Processing fold: {fold}")
        print(f"{'='*50}")
        
        print(f"Loading predictions...")
        df = pd.read_csv(csv_path)
        if 'ground_truth_label' in df.columns:
            df['true_label'] = df['ground_truth_label']
        df_valid = df[df['true_label'].notna()].copy()
        
        # Aggregate levels
        image_df, well_df = aggregate_crop_to_well(df_valid)
        
        print(f"Crops: {len(df_valid)}, Images: {len(image_df)}, Wells: {len(well_df)}")
        
        # Save image embeddings for t-SNE
        print("Saving image embeddings...")
        image_embeddings = []
        for img_name, group in df_valid.groupby('image_name'):
            emb_list = []
            for emb_str in group['embedding'].values:
                emb = np.array(ast.literal_eval(emb_str))
                emb_list.append(emb)
            img_emb = np.mean(emb_list, axis=0)
            image_embeddings.append(img_emb)
        
        np.save(os.path.join(output_dir, 'image_embeddings.npy'), np.array(image_embeddings))
        np.save(os.path.join(output_dir, 'image_labels.npy'), image_df['true_label'].values)
        
        print(f"  Saved {len(image_embeddings)} image embeddings")
        
        levels = [('crop', df_valid), ('image', image_df), ('well', well_df)]
        hierarchies = ['guide', 'gene', 'pathway', 'family']
        
        for level_name, level_df in levels:
            for hier in hierarchies:
                # Map to hierarchy
                true_mapped = map_hierarchy(level_df['true_label'].values, hier)
                
                if level_name == 'crop':
                    pred_mapped = map_hierarchy(level_df['predicted_class_name'].values, hier)
                elif level_name == 'image':
                    pred_mapped = map_hierarchy(level_df['pred_majority'].values, hier)
                else:
                    pred_mapped = map_hierarchy(level_df['pred_majority'].values, hier)
                
                # Accuracy
                acc = accuracy_score(true_mapped, pred_mapped)
                
                # Per-class metrics
                all_labels = sorted(set(true_mapped) | set(pred_mapped))
                cm_raw = confusion_matrix(true_mapped, pred_mapped, labels=all_labels, normalize=None)
                cm_norm = confusion_matrix(true_mapped, pred_mapped, labels=all_labels, normalize='true')
                
                # Precision, Recall for each class
                p, r, f1, sup = precision_recall_fscore_support(true_mapped, pred_mapped, 
                                                               labels=all_labels, average=None, zero_division=0)
                
                # Save confusion matrix plot
                filename = f'cm_{level_name}_{hier}.png'
                title = f'{fold} - {level_name.capitalize()}/{hier.capitalize()} Acc: {100*acc:.1f}%'
                is_guide = (hier == 'guide')
                plot_cm_with_boxes(cm_raw, cm_norm, all_labels, title, os.path.join(output_dir, filename), is_guide=is_guide)
                
                # Store results
                for i, lbl in enumerate(all_labels):
                    all_results.append({
                        'fold': fold,
                        'level': level_name,
                        'hierarchy': hier,
                        'class': lbl,
                        'precision': float(p[i]) if i < len(p) else 0,
                        'recall': float(r[i]) if i < len(r) else 0,
                        'f1': float(f1[i]) if i < len(f1) else 0,
                        'support': int(sup[i]) if i < len(sup) else 0
                    })
                
                print(f"  {level_name}/{hier}: {100*acc:.2f}%")
        
        # Save fold metrics summary
        fold_results = [r for r in all_results if r['fold'] == fold]
        fold_df = pd.DataFrame(fold_results)
        fold_df.to_csv(os.path.join(output_dir, 'per_class_metrics.csv'), index=False)
    
    # Save all results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(SCRIPT_DIR, 'all_folds_confusion_metrics.csv'), index=False)
        print(f"\nSaved all metrics to all_folds_confusion_metrics.csv")
    
    print(f"\n{'='*50}")
    print("DONE!")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()