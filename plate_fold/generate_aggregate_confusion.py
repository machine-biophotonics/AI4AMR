#!/usr/bin/env python3
"""
Generate aggregate confusion matrices with mean and std across folds.
Averages normalized confusion matrices from all folds.
"""

import numpy as np
import os
import argparse
from collections import Counter
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

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


def aggregate_crop_to_well(df):
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


def plot_aggregate_cm(cm_mean, cm_std, labels, title, output_path, is_guide=False):
    n = len(labels)
    import seaborn as sns
    
    if is_guide:
        fig, ax = plt.subplots(figsize=(max(14, n*0.2), max(14, n*0.2)))
        
        sns.heatmap(cm_mean, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, 
                    yticklabels=labels, ax=ax, vmin=0, vmax=1,
                    cbar_kws={'label': 'Mean Normalized Frequency', 'shrink': 0.8},
                    linewidths=0.5, linecolor='white',
                    annot_kws={'size': 6},
                    square=True)
        
        for i, label in enumerate(labels):
            base = get_base_gene(label)
            same_gene_indices = [j for j, l in enumerate(labels) if get_base_gene(l) == base]
            if len(same_gene_indices) > 1:
                min_j = min(same_gene_indices)
                max_j = max(same_gene_indices)
                rect = patches.Rectangle((min_j, min_j), max_j - min_j + 1, max_j - min_j + 1,
                                          linewidth=3, edgecolor='#FFD700', facecolor='none', zorder=10)
                ax.add_patch(rect)
        
        for i in range(n):
            rect = patches.Rectangle((i, i), 1, 1, linewidth=2.5, edgecolor='#FF4444', 
                                      facecolor='none', zorder=10)
            ax.add_patch(rect)
        
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_title(f'{title}', fontsize=13, fontweight='bold')
        ax.set_xticks(np.arange(n) + 0.5, labels, rotation=90, fontsize=6)
        ax.set_yticks(np.arange(n) + 0.5, labels, rotation=0, fontsize=6)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        
    else:
        fig, axes = plt.subplots(1, 2, figsize=(max(22, n*0.35), max(11, n*0.2)))
        
        ax1 = axes[0]
        sns.heatmap(cm_mean, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, 
                    yticklabels=labels, ax=ax1, vmin=0, vmax=1,
                    cbar_kws={'label': 'Mean', 'shrink': 0.8},
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
                                          linewidth=3, edgecolor='#FFD700', facecolor='none', zorder=10)
                ax1.add_patch(rect)
        
        for i in range(n):
            rect = patches.Rectangle((i, i), 1, 1, linewidth=2.5, edgecolor='#FF4444', facecolor='none', zorder=10)
            ax1.add_patch(rect)
        
        ax1.set_xlabel('Predicted Label', fontsize=10)
        ax1.set_ylabel('True Label', fontsize=10)
        ax1.set_title(f'{title}\n(Mean)', fontsize=12, fontweight='bold')
        ax1.set_xticks(np.arange(n) + 0.5, labels, rotation=90, fontsize=5)
        ax1.set_yticks(np.arange(n) + 0.5, labels, rotation=0, fontsize=5)
        for spine in ax1.spines.values():
            spine.set_visible(False)
        
        ax2 = axes[1]
        sns.heatmap(cm_std, annot=True, fmt='.2f', cmap='Reds', xticklabels=labels, 
                    yticklabels=labels, ax=ax2, vmin=0, vmax=0.3,
                    cbar_kws={'label': 'Std Dev', 'shrink': 0.8},
                    linewidths=0.5, linecolor='white',
                    annot_kws={'size': 5},
                    square=True)
        
        ax2.set_xlabel('Predicted Label', fontsize=10)
        ax2.set_ylabel('True Label', fontsize=10)
        ax2.set_title(f'{title}\n(Std Dev)', fontsize=12, fontweight='bold')
        ax2.set_xticks(np.arange(n) + 0.5, labels, rotation=90, fontsize=5)
        ax2.set_yticks(np.arange(n) + 0.5, labels, rotation=0, fontsize=5)
        for spine in ax2.spines.values():
            spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate aggregate confusion matrices')
    parser.add_argument('--folds', type=str, default='P2,P3,P4,P5', help='Comma-separated list of folds')
    args = parser.parse_args()
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    folds = args.folds.split(',')
    
    output_dir = os.path.join(SCRIPT_DIR, 'aggregate')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Aggregating across folds: {folds}")
    
    all_fold_data = {}
    
    for fold in folds:
        fold_dir = os.path.join(SCRIPT_DIR, f'fold_{fold}')
        csv_path = os.path.join(fold_dir, 'predictions.csv')
        
        if not os.path.exists(csv_path):
            print(f"Skipping {fold}: no predictions.csv")
            continue
        
        print(f"Loading {fold}...")
        df = pd.read_csv(csv_path)
        if 'ground_truth_label' in df.columns:
            df['true_label'] = df['ground_truth_label']
        df_valid = df[df['true_label'].notna()].copy()
        
        image_df, well_df = aggregate_crop_to_well(df_valid)
        all_fold_data[fold] = {
            'crop': df_valid,
            'image': image_df,
            'well': well_df
        }
    
    levels = [('crop', 'crop'), ('image', 'image'), ('well', 'well')]
    hierarchies = ['guide', 'gene', 'pathway', 'family']
    
    results = []
    
    for level_key, level_name in levels:
        for hier in hierarchies:
            fold_cms = []
            fold_accs = []
            
            for fold, data in all_fold_data.items():
                level_df = data[level_key]
                
                true_mapped = map_hierarchy(level_df['true_label'].values, hier)
                
                if level_key == 'crop':
                    pred_mapped = map_hierarchy(level_df['predicted_class_name'].values, hier)
                else:
                    pred_mapped = map_hierarchy(level_df['pred_majority'].values, hier)
                
                acc = np.mean(np.array(true_mapped) == np.array(pred_mapped))
                fold_accs.append(acc)
                
                all_labels = sorted(set(true_mapped) | set(pred_mapped))
                cm = confusion_matrix(true_mapped, pred_mapped, labels=all_labels, normalize='true')
                fold_cms.append((all_labels, cm))
            
            all_labels = fold_cms[0][0]
            
            n_classes = len(all_labels)
            cm_arrays = np.zeros((len(fold_cms), n_classes, n_classes))
            
            for i, (labels, cm) in enumerate(fold_cms):
                label_to_idx = {l: j for j, l in enumerate(all_labels)}
                for j, l in enumerate(labels):
                    if l in label_to_idx:
                        cm_arrays[i, j, :] = cm[j, :]
            
            cm_mean = np.mean(cm_arrays, axis=0)
            cm_std = np.std(cm_arrays, axis=0)
            
            mean_acc = np.mean(fold_accs)
            std_acc = np.std(fold_accs)
            
            filename = f'agg_cm_{level_name}_{hier}.png'
            title = f'Aggregate ({len(fold_cms)} folds) - {level_name.capitalize()}/{hier.capitalize()} Acc: {100*mean_acc:.1f}%±{100*std_acc:.1f}%'
            is_guide = (hier == 'guide')
            plot_aggregate_cm(cm_mean, cm_std, all_labels, title, os.path.join(output_dir, filename), is_guide=is_guide)
            
            results.append({
                'level': level_name,
                'hierarchy': hier,
                'mean_acc': mean_acc,
                'std_acc': std_acc
            })
            
            print(f"  {level_name}/{hier}: {100*mean_acc:.2f}% ± {100*std_acc:.2f}%")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'aggregate_metrics.csv'), index=False)
    print(f"\nSaved to {output_dir}/")


if __name__ == '__main__':
    main()