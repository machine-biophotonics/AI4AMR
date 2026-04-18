#!/usr/bin/env python3
"""
Generate aggregate confusion matrices for final_crispr_model.
Same logic as plate_fold_no_aug/generate_combined_confusion.py
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

# Trial pathway mapping
TRIAL_PATHWAY = {
    'folP': 'folic acid biosynthetic process', 'folA': 'folic acid biosynthetic process',
    'ftsZ': 'cell cycle process',
    'dnaE': 'DNA-templated DNA replication', 'dnaB': 'DNA-templated DNA replication',
    'secY': 'intracellular transport', 'secA': 'intracellular transport',
    'rpoB': 'regulation of DNA-templated transcription elongation',
    'rpoA': 'regulation of DNA-templated transcription elongation',
    'parE': 'cellular component organization', 'parC': 'cellular component organization',
    'gyrB': 'cellular component organization', 'gyrA': 'cellular component organization',
    'lptC': 'lipid transport', 'lptA': 'lipid transport', 'msbA': 'lipid transport',
    'rplC': 'positive regulation of gene expression', 'rplA': 'positive regulation of gene expression',
    'rpsA': 'positive regulation of gene expression', 'rpsL': 'positive regulation of gene expression',
    'murC': 'glycosaminoglycan biosynthetic process', 'murA': 'glycosaminoglycan biosynthetic process',
    'ftsI': 'glycosaminoglycan biosynthetic process', 'ftsZ': 'glycosaminoglycan biosynthetic process',
    'mrdA': 'glycosaminoglycan biosynthetic process', 'mrcA': 'glycosaminoglycan biosynthetic process',
    'mrcB': 'glycosaminoglycan biosynthetic process', 'lpxC': 'glycosaminoglycan biosynthetic process',
    'lpxA': 'glycosaminoglycan biosynthetic process',
}

FAMILY = {
    'ftsZ': 'fts', 'ftsI': 'fts', 'murA': 'mur', 'murC': 'mur',
    'rpsA': 'rps', 'rpsL': 'rps', 'rplA': 'rpl', 'rplC': 'rpl',
    'mrdA': 'mrd', 'mrcA': 'mrc', 'mrcB': 'mrc',
    'lpxA': 'lpx', 'lpxC': 'lpx',
    'lptA': 'lpt', 'lptC': 'lpt',
    'gyrA': 'gyr', 'gyrB': 'gyr',
    'rpoA': 'rpo', 'rpoB': 'rpo',
    'secA': 'sec', 'secY': 'sec',
    'msbA': 'msb',
    'folA': 'fol', 'folP': 'fol',
    'dnaE': 'dna', 'dnaB': 'dna',
    'parC': 'par', 'parE': 'par',
}

FAMILY_GROUP = {
    'dnaB': 'dna', 'dnaE': 'dna',
    'secA': 'sec', 'secY': 'sec',
    'lptA': 'lpt', 'lptC': 'lpt',
    'lpxA': 'lpx', 'lpxC': 'lpx',
    'mrcA': 'mrc', 'mrcB': 'mrc',
    'ftsI': 'fts', 'ftsZ': 'fts',
    'gyrA': 'gyr', 'gyrB': 'gyr',
    'parC': 'par', 'parE': 'par',
    'rplA': 'rpl', 'rplC': 'rpl',
    'rpoA': 'rpo', 'rpoB': 'rpo',
    'rpsA': 'rps', 'rpsL': 'rps',
    'murA': 'mur', 'murC': 'mur',
    'folA': 'fol', 'folP': 'fol',
}


def get_base_gene(label):
    if not label or label == 'nan':
        return 'Unknown'
    if '_' in str(label):
        return str(label).rsplit('_', 1)[0]
    return str(label)


def get_pathway(label):
    base = get_base_gene(label)
    if str(base).upper().startswith('WT'):
        return 'WT'
    if str(base).upper() == 'NC':
        return 'WT'  # Group NC with WT
    if base in HIERARCHY:
        return HIERARCHY[base]
    return 'Unknown'


def get_trial_pathway(label):
    """Trial pathway mapping - use exact same as generate_trial_pathway_confusion.py"""
    base = get_base_gene(label)
    if base.upper().startswith('WT') or base.upper() == 'NC':
        return 'WT'
    if base in TRIAL_PATHWAY:
        return TRIAL_PATHWAY[base]
    return base


def get_family(label):
    base = get_base_gene(label)
    if str(base).upper().startswith('WT') or str(base).upper() == 'NC':
        return 'WT'
    if base in FAMILY:
        return FAMILY[base]
    return 'Unknown'


def map_hierarchy(labels, level):
    if level == 'guide':
        return list(labels)
    elif level == 'gene':
        return [get_base_gene(l) for l in labels]
    elif level == 'pathway':
        return [get_trial_pathway(l) for l in labels]
    elif level == 'family':
        return [get_family(l) for l in labels]
    else:
        return list(labels)


def aggregate_crop_to_well(df):
    """Aggregate crop-level predictions to image and well level."""
    image_results = []
    for img_name, group in df.groupby('image_name'):
        true_label = group['ground_truth_label'].iloc[0]
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


def plot_binary_cm(cm_sum, labels, title, output_path, row_majority=True, threshold=0.5):
    n = len(labels)
    import seaborn as sns
    
    if row_majority:
        # For each row, use the percentage of the MAX column (not binary)
        cm_display = np.zeros((n, n))
        for i in range(n):
            row = cm_sum[i, :]
            if row.sum() > 0:
                row_norm = row / row.sum()
                max_val = row_norm.max()
                max_idx = row_norm.argmax()
                cm_display[i, max_idx] = max_val * 100  # Convert to percentage
    else:
        # Original: binary based on diagonal only
        cm_display = cm_sum * 100  # Convert to percentage
        for i in range(n):
            for j in range(n):
                if i != j and cm_sum[i, j] >= threshold:
                    cm_display[i, j] = cm_sum[i, j] * 100
    
    random_baseline = 1.0 / n
    
    # Count: highest is on diagonal vs off-diagonal
    n_max_on_diagonal = 0
    for i in range(n):
        row = cm_sum[i, :]
        if row.sum() > 0:
            row_norm = row / row.sum()
            max_idx = row_norm.argmax()
            if max_idx == i:
                n_max_on_diagonal += 1
    
    n_with_majority = n_max_on_diagonal  # Total rows with any prediction
    
    n_above_random = np.sum(np.diag(cm_sum) >= random_baseline)
    
    # Percentage and count for title
    pct_majority = 100.0 * n_with_majority / n if n > 0 else 0
    pct_on_diagonal = 100.0 * n_max_on_diagonal / n if n > 0 else 0
    n_above_threshold = n_with_majority
    
    # Use same styling as percentage confusion matrix
    fig, ax = plt.subplots(figsize=(max(14, n*0.2), max(14, n*0.2)))
    
    sns.heatmap(cm_display, annot=False, cmap='Blues', xticklabels=labels,
                yticklabels=labels, ax=ax, vmin=0, vmax=100,
                cbar_kws={'label': 'Percentage (%)', 'shrink': 0.8},
                linewidths=0.3, linecolor='white',
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
    
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_title(f'{title}\n(Binary %) | Max on Diagonal: {n_max_on_diagonal}/{n} ({pct_on_diagonal:.1f}%)', 
                 fontsize=11, fontweight='bold')
    ax.set_xticks(np.arange(n) + 0.5, labels, rotation=90, fontsize=5)
    ax.set_yticks(np.arange(n) + 0.5, labels, rotation=0, fontsize=5)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_raw_counts(cm_sum, labels, title, output_path, show_annot=True):
    n = len(labels)
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(max(14, n*0.2), max(14, n*0.2)))
    
    if show_annot and n < 50:
        sns.heatmap(cm_sum, annot=True, fmt='.0f', cmap='Blues', xticklabels=labels,
                    yticklabels=labels, ax=ax,
                    cbar_kws={'label': 'Count', 'shrink': 0.8},
                    linewidths=0.5, linecolor='white',
                    annot_kws={'size': 4},
                    square=True)
    else:
        sns.heatmap(cm_sum, annot=False, cmap='Blues', xticklabels=labels,
                    yticklabels=labels, ax=ax,
                    cbar_kws={'label': 'Count', 'shrink': 0.8},
                    linewidths=0.3, linecolor='white',
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
    
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_title(f'{title}\n(Raw Counts)', fontsize=11, fontweight='bold')
    ax.set_xticks(np.arange(n) + 0.5, labels, rotation=90, fontsize=5)
    ax.set_yticks(np.arange(n) + 0.5, labels, rotation=0, fontsize=5)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_percentage_cm(cm_sum, labels, title, output_path, show_annot=True):
    n = len(labels)
    import seaborn as sns
    
    random_baseline = 100.0 / n
    
    # Count: highest is on diagonal vs off-diagonal
    n_max_on_diagonal = 0
    for i in range(n):
        row = cm_sum[i, :]
        if row.sum() > 0:
            row_norm = row / row.sum()
            max_idx = row_norm.argmax()
            if max_idx == i:
                n_max_on_diagonal += 1
    
    n_above_random = np.sum(np.diag(cm_sum) * 100 > random_baseline)
    
    # Percentage
    pct_on_diagonal = 100.0 * n_max_on_diagonal / n if n > 0 else 0
    
    fig, ax = plt.subplots(figsize=(max(14, n*0.2), max(14, n*0.2)))
    
    if show_annot and n < 50:
        sns.heatmap(cm_sum * 100, annot=True, fmt='.1f', cmap='Blues', xticklabels=labels,
                    yticklabels=labels, ax=ax, vmin=0, vmax=100,
                    cbar_kws={'label': 'Percentage (%)', 'shrink': 0.8},
                    linewidths=0.5, linecolor='white',
                    annot_kws={'size': 5},
                    square=True)
    else:
        sns.heatmap(cm_sum * 100, annot=False, cmap='Blues', xticklabels=labels,
                    yticklabels=labels, ax=ax, vmin=0, vmax=100,
                    cbar_kws={'label': 'Percentage (%)', 'shrink': 0.8},
                    linewidths=0.3, linecolor='white',
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
    
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_title(f'{title}\n(Percentage %) | Max on Diagonal: {n_max_on_diagonal}/{n} ({pct_on_diagonal:.1f}%)', 
                 fontsize=11, fontweight='bold')
    ax.set_xticks(np.arange(n) + 0.5, labels, rotation=90, fontsize=5)
    ax.set_yticks(np.arange(n) + 0.5, labels, rotation=0, fontsize=5)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate aggregate confusion matrices for final_crispr_model')
    parser.add_argument('--folds', type=str, default='P1,P2,P3,P4,P5,P6', help='Comma-separated folds')
    parser.add_argument('--single_fold', type=str, default=None,
                        help='Generate for a single fold (e.g., P1) - creates fold-specific output directory')
    parser.add_argument('--guide', action='store_true', help='Generate only guide-level')
    parser.add_argument('--family', action='store_true', help='Generate only family-level')
    parser.add_argument('--csv_name', type=str, default=None, 
                        help='CSV filename to look for (default: predictions_all_crops.csv or predictions_all_crops_mil_100pos.csv)')
    parser.add_argument('--prediction_csv', type=str, default=None,
                        help='Specific prediction CSV file to use')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for confusion matrices')
    parser.add_argument('--mixed_checkpoints', action='store_true',
                        help='Use best_model_acc for P1-P5, best_model for P6')
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Handle single fold case
    if args.single_fold:
        folds = [args.single_fold]
        output_dir = os.path.join(SCRIPT_DIR, 'aggregate', f'fold_{args.single_fold}')
    elif args.output_dir:
        folds = args.folds.split(',')
        output_dir = args.output_dir
    else:
        folds = args.folds.split(',')
        output_dir = os.path.join(SCRIPT_DIR, 'aggregate', 'combined')
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"Aggregating across folds: {folds}")
    if args.guide:
        print("Mode: Guide only")
    elif args.family:
        print("Mode: Family only")
    print(f"Output directory: {output_dir}")

    all_fold_data = {}

    for fold in folds:
        fold_dir = os.path.join(SCRIPT_DIR, f'fold_{fold}')
        
        if args.prediction_csv:
            csv_path = os.path.join(fold_dir, args.prediction_csv)
        elif args.csv_name:
            csv_path = os.path.join(fold_dir, args.csv_name)
        else:
            # Mixed checkpoint: best_model_acc for P1-P5, best_model for P6
            if args.mixed_checkpoints and fold == 'P6':
                csv_path = os.path.join(fold_dir, 'predictions_all_crops_mil_best_model.csv')
            else:
                csv_path = os.path.join(fold_dir, 'predictions_all_crops.csv')
            if not os.path.exists(csv_path):
                csv_path = os.path.join(fold_dir, 'predictions_all_crops_mil_best_model_acc.csv')
            if not os.path.exists(csv_path):
                csv_path = os.path.join(fold_dir, 'predictions_all_crops_mil_100pos.csv')
            if not os.path.exists(csv_path):
                csv_path = os.path.join(fold_dir, 'predictions_all_crops_mil.csv')
            if not os.path.exists(csv_path):
                csv_path = os.path.join(fold_dir, 'image_predictions_mil.csv')

        if not os.path.exists(csv_path):
            print(f"Skipping {fold}: no CSV file found")
            continue

        print(f"Loading {fold}...")
        df = pd.read_csv(csv_path)
        
        if 'ground_truth_label' not in df.columns:
            print(f"Skipping {fold}: no ground_truth_label column")
            continue

        df_valid = df[df['ground_truth_label'].notna()].copy()

        image_df, well_df = aggregate_crop_to_well(df_valid)
        all_fold_data[fold] = {
            'crop': df_valid,
            'image': image_df,
            'well': well_df
        }

    levels = [('crop', 'crop'), ('image', 'image'), ('well', 'well')]
    if args.guide:
        hierarchies = ['guide']
    elif args.family:
        hierarchies = ['family']
    else:
        hierarchies = ['guide', 'gene', 'pathway', 'family']

    results = []

    for level_key, level_name in levels:
        for hier in hierarchies:
            fold_raw_cms = []
            fold_norm_cms = []
            fold_accs = []
            all_labels_set = set()

            for fold, data in all_fold_data.items():
                level_df = data[level_key]

                true_col = 'ground_truth_label' if level_key == 'crop' else 'true_label'
                true_mapped = map_hierarchy(level_df[true_col].values, hier)

                if level_key == 'crop':
                    pred_mapped = map_hierarchy(level_df['predicted_class_name'].values, hier)
                else:
                    pred_mapped = map_hierarchy(level_df['pred_majority'].values, hier)

                acc = np.mean(np.array(true_mapped) == np.array(pred_mapped))
                fold_accs.append(acc)

                all_labels = sorted(set(true_mapped) | set(pred_mapped))
                all_labels_set.update(all_labels)

                cm_raw = confusion_matrix(true_mapped, pred_mapped, labels=all_labels, normalize=None)
                fold_raw_cms.append((all_labels, cm_raw))

                cm_norm = confusion_matrix(true_mapped, pred_mapped, labels=all_labels, normalize='true')
                fold_norm_cms.append((all_labels, cm_norm))

            if args.family:
                def sort_key(label):
                    base = get_base_gene(label)
                    group_key = FAMILY_GROUP.get(base, base)
                    if '_' in str(label):
                        prefix = str(label).rsplit('_', 1)[0]
                        suffix = str(label).rsplit('_', 1)[1]
                        return (group_key, base, int(suffix) if suffix.isdigit() else suffix)
                    return (base, label)

                all_labels = sorted(all_labels_set, key=sort_key)
            else:
                all_labels = sorted(all_labels_set)

            n_classes = len(all_labels)
            random_baseline = 100.0 / n_classes

            cm_sum_raw = np.zeros((n_classes, n_classes))
            for labels, cm in fold_raw_cms:
                label_to_idx = {l: j for j, l in enumerate(all_labels)}
                for j, l in enumerate(labels):
                    if l in label_to_idx:
                        cm_sum_raw[label_to_idx[l], :] += cm[j, :]

            row_sums = cm_sum_raw.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            cm_sum_normalized = cm_sum_raw / row_sums

            mean_acc = np.mean(fold_accs)
            std_acc = np.std(fold_accs)

            n_above_random = np.sum(np.diag(cm_sum_normalized) * 100 > random_baseline)
            n_above_50 = np.sum(np.diag(cm_sum_normalized) * 100 > 50)

            title = f'Aggregate ({len(fold_raw_cms)} folds) - {level_name.capitalize()}/{hier.capitalize()} Acc: {100*mean_acc:.1f}%±{100*std_acc:.1f}%'
            
            plot_binary_cm(cm_sum_normalized, all_labels, title,
                         os.path.join(output_dir, f'binary_cm_{level_name}_{hier}.png'))
            
            plot_raw_counts(cm_sum_raw, all_labels, title,
                           os.path.join(output_dir, f'raw_cm_{level_name}_{hier}.png'))
            
            show_annot = (n_classes < 50)
            plot_percentage_cm(cm_sum_normalized, all_labels, title,
                            os.path.join(output_dir, f'percent_cm_{level_name}_{hier}.png'),
                            show_annot=show_annot)

            results.append({
                'level': level_name,
                'hierarchy': hier,
                'mean_acc': mean_acc,
                'std_acc': std_acc,
                'n_folds': len(fold_raw_cms),
                'n_classes': n_classes,
                'random_baseline': random_baseline,
                'classes_above_50': n_above_50,
                'classes_above_random': n_above_random
            })

            print(f"  {level_name}/{hier}: {100*mean_acc:.2f}% ± {100*std_acc:.2f}% | {n_above_50}/{n_classes} > 50%, {n_above_random}/{n_classes} > Random({random_baseline:.1f}%)")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'combined_metrics.csv'), index=False)
    print(f"\nSaved to {output_dir}/")


if __name__ == '__main__':
    main()