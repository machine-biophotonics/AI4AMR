#!/usr/bin/env python3
"""
Generate aggregate confusion matrices for final_mutant_model.
Same logic as final_crispr_model/generate_combined_confusion.py
"""

import numpy as np
import os
import re
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

# Trial pathway mapping - user's biological process table
TRIAL_PATHWAY = {
    # Folic acid biosynthetic process
    'folP': 'Folic acid biosynthetic process',
    'folA': 'Folic acid biosynthetic process',
    # Intracellular protein transport
    'secY': 'Intracellular protein transport',
    'secA': 'Intracellular protein transport',
    # Regulation of DNA-templated transcription elongation
    'rpoB': 'Regulation of DNA-templated transcription elongation',
    'rpoA': 'Regulation of DNA-templated transcription elongation',
    # Cell envelope organization
    'lptC': 'Cell envelope organization',
    'lptA': 'Cell envelope organization',
    'msbA': 'Cell envelope organization',
    # Division septum assembly
    'ftsZ': 'Division septum assembly',
    # Regulation of translational initiation
    'rplC': 'Regulation of translational initiation',
    'rplA': 'Regulation of translational initiation',
    'rpsA': 'Regulation of translational initiation',
    'rpsL': 'Regulation of translational initiation',
    # Aminoglycan biosynthetic process
    'murC': 'Aminoglycan biosynthetic process',
    'murA': 'Aminoglycan biosynthetic process',
    'mrcB': 'Aminoglycan biosynthetic process',
    # Regulation of cell shape
    'mrdA': 'Regulation of cell shape',
    'mrcA': 'Regulation of cell shape',
    'ftsI': 'Regulation of cell shape',
    # Lipid A biosynthetic process
    'lpxC': 'Lipid A biosynthetic process',
    'lpxA': 'Lipid A biosynthetic process',
    # Chromosome organization
    'gyrB': 'Chromosome organization',
    'gyrA': 'Chromosome organization',
    'dnaB': 'Chromosome organization',
    'parE': 'Chromosome organization',
    'parC': 'Chromosome organization',
    'dnaE': 'Chromosome organization',
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


def is_wt_label(label):
    base = get_base_gene(label)
    up = str(base).upper()
    return up.startswith('WT') or up == 'NC' or up.endswith(' NC')


def get_pathway(label):
    base = get_base_gene(label)
    if is_wt_label(label):
        return 'Wild Type'
    if base in HIERARCHY:
        return HIERARCHY[base]
    return 'Unknown'


def get_trial_pathway(label):
    """Trial pathway mapping - use exact same as generate_trial_pathway_confusion.py"""
    base = get_base_gene(label)
    if is_wt_label(label):
        return 'Wild Type'
    if base in TRIAL_PATHWAY:
        return TRIAL_PATHWAY[base]
    return base


def get_family(label):
    base = get_base_gene(label)
    if is_wt_label(label):
        return 'Wild Type'
    if base in FAMILY:
        return FAMILY[base]
    return 'Unknown'


def extract_well_from_path(path):
    match = re.search(r'Well(\w\d+)_', os.path.basename(path))
    return match.group(1) if match else None


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


def draw_cm_panel(ax, cm, labels, style, show_annot=True):
    import seaborn as sns
    n = len(labels)

    if style == 'binary':
        cm_display = np.zeros((n, n))
        for i in range(n):
            row = cm[i, :]
            if row.sum() > 0:
                row_norm = row / row.sum()
                cm_display[i, row_norm.argmax()] = row_norm.max() * 100
        vmax, fmt = 100, '.1f'
    elif style == 'raw':
        cm_display = cm
        vmax, fmt = None, '.0f'
    else:
        cm_display = cm * 100
        vmax, fmt = 100, '.1f'

    annot = show_annot and n < 50
    if style == 'raw':
        sns.heatmap(cm_display, annot=annot, fmt=fmt, cmap='Blues', xticklabels=labels,
                    yticklabels=labels, ax=ax,
                    cbar_kws={'label': 'Count', 'shrink': 0.8},
                    linewidths=0.3, linecolor='white',
                    annot_kws={'size': 4}, square=True)
    else:
        sns.heatmap(cm_display, annot=annot, fmt=fmt, cmap='Blues', xticklabels=labels,
                    yticklabels=labels, ax=ax, vmin=0, vmax=vmax,
                    cbar_kws={'label': 'Percentage (%)', 'shrink': 0.8},
                    linewidths=0.3, linecolor='white',
                    annot_kws={'size': 5}, square=True)

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
    ax.set_xticks(np.arange(n) + 0.5, labels, rotation=90, fontsize=5)
    ax.set_yticks(np.arange(n) + 0.5, labels, rotation=0, fontsize=5)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_timepoint_panels(panels, style, output_path, suptitle):
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(max(14, panels[0]['n_classes'] * 0.2) * n_panels,
                                      max(14, panels[0]['n_classes'] * 0.2)))
    if n_panels == 1:
        axes = [axes]
    for ax, panel in zip(axes, panels):
        draw_cm_panel(ax, panel['cm'], panel['labels'], style)
        ax.set_title(panel['title'], fontsize=10, fontweight='bold')
    fig.suptitle(suptitle, fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate aggregate confusion matrices for final_crispr_model')
    parser.add_argument('--folds', type=str, default='P1,P2,P3,P4,P5,P6', help='Comma-separated folds')
    parser.add_argument('--single_fold', type=str, default=None,
                        help='Generate for a single fold (e.g., P1 or Plate_1) - creates fold-specific output directory')
    parser.add_argument('--guide', type=int, default=None,
                        help='Filter to specific guide number (e.g. 1 for guide 1) and skip to gene-level')
    parser.add_argument('--family', action='store_true', help='Generate only family-level')
    parser.add_argument('--csv_name', type=str, default=None, 
                        help='CSV filename to look for (default: predictions_all_crops.csv or predictions_all_crops_mil_100pos.csv)')
    parser.add_argument('--prediction_csv', type=str, default=None,
                        help='Specific prediction CSV file to use')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for confusion matrices')
    parser.add_argument('--mixed_checkpoints', action='store_true',
                        help='Use best_model_acc for P1-P5, best_model for P6')
    parser.add_argument('--data_mode', type=str, default='mutant',
                        choices=['mutant', 'metabolomics_mutant'],
                        help='Data mode: mutant (original, predictions_all_crops_*.csv in mutant/fold_Plate_X/) '
                             'or metabolomics_mutant (Felix data, test_positions_fold_PX.csv in metabolomics_mutant_hpc/fold_PX/)')
    parser.add_argument('--input_root', type=str, default=None,
                        help='Override results root. For metabolomics_mutant, point at a folder holding flat '
                             'test_positions_fold_PX.csv files (e.g. lastckpt_results/) instead of fold_PX/ subfolders')
    parser.add_argument('--timepoint_panels', action='store_true',
                        help='For metabolomics_mutant: also produce 4-panel figures (T1, T2, T3, ALL) '
                             'and report per-timepoint metrics')
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    use_metabolomics = args.data_mode == 'metabolomics_mutant'
    if use_metabolomics and args.input_root:
        mode_root = args.input_root
    elif use_metabolomics:
        mode_root = os.path.join(SCRIPT_DIR, 'metabolomics_mutant_hpc')
    else:
        mode_root = os.path.join(SCRIPT_DIR, 'mutant')

    def fold_key_for(fold_input):
        if use_metabolomics:
            if fold_input.startswith('fold_'):
                return fold_input
            return f'fold_{fold_input}'
        if fold_input.startswith('fold_'):
            return fold_input
        if 'Plate_' in fold_input:
            return f'fold_{fold_input}'
        return f'fold_Plate_{fold_input.replace("P", "")}'

    # Handle single fold case
    if args.single_fold:
        folds = [args.single_fold]
        fold_key = fold_key_for(args.single_fold)
        # Save within the fold folder
        output_dir = os.path.join(mode_root, fold_key, 'confusion_matrices')
    elif args.output_dir:
        folds = args.folds.split(',')
        output_dir = args.output_dir
    else:
        folds = args.folds.split(',')
        if use_metabolomics:
            output_dir = os.path.join(mode_root, 'confusion_matrices')
        else:
            output_dir = os.path.join(SCRIPT_DIR, 'aggregate', 'combined')
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"Aggregating across folds: {folds}")
    if args.guide is not None:
        print(f"Mode: Gene-level (filtered to guide {args.guide})")
    elif args.family:
        print("Mode: Family only")
    print(f"Output directory: {output_dir}")

    all_fold_data = {}

    for fold in folds:
        fold_input = fold
        fold_key = fold_key_for(fold_input)

        if use_metabolomics and args.input_root:
            fold_dir = mode_root
        else:
            fold_dir = os.path.join(mode_root, fold_key)
        
        if args.prediction_csv:
            csv_path = os.path.join(fold_dir, args.prediction_csv)
        elif args.csv_name:
            csv_path = os.path.join(fold_dir, args.csv_name)
        elif use_metabolomics:
            csv_path = os.path.join(fold_dir, f'test_positions_{fold_key}.csv')
        else:
            # Try various CSV file names
            csv_path = os.path.join(fold_dir, 'predictions_all_crops_mil_best_model_acc_n3.csv')
            if not os.path.exists(csv_path):
                csv_path = os.path.join(fold_dir, 'predictions_all_crops_mil_best_model.csv')
            if not os.path.exists(csv_path):
                csv_path = os.path.join(fold_dir, 'predictions_all_crops_mil_best_model_acc.csv')
            if not os.path.exists(csv_path):
                csv_path = os.path.join(fold_dir, 'predictions_all_crops.csv')
            if not os.path.exists(csv_path):
                csv_path = os.path.join(fold_dir, 'image_predictions_mil.csv')

        if not os.path.exists(csv_path):
            print(f"  Tried: {csv_path}")
            print(f"  Available files: {os.listdir(fold_dir) if os.path.exists(fold_dir) else 'folder not found'}")
            print(f"Skipping {fold}: no CSV file found")
            continue

        print(f"Loading {fold}...")
        df = pd.read_csv(csv_path)

        if use_metabolomics:
            df = df.rename(columns={'true_label': 'ground_truth_label',
                                    'predicted_label': 'predicted_class_name'})
            df['image_name'] = df['image_path']
            df['well'] = df['image_path'].apply(extract_well_from_path)

        if 'ground_truth_label' not in df.columns:
            print(f"Skipping {fold}: no ground_truth_label column")
            continue

        df_valid = df[df['ground_truth_label'].notna()].copy()

        # Filter to specific guide if requested
        if args.guide is not None:
            guide_suffix = f"_{args.guide}"
            df_valid = df_valid[df_valid['ground_truth_label'].str.endswith(guide_suffix, na=False)].copy()
            print(f"  Guide {args.guide} filter: {len(df_valid)} remaining rows")

        image_df, well_df = aggregate_crop_to_well(df_valid)

        fold_groups = {'ALL': {'crop': df_valid, 'image': image_df, 'well': well_df}}
        if use_metabolomics and args.timepoint_panels and 'timepoint' in df_valid.columns:
            for tp in ['T1', 'T2', 'T3']:
                sub = df_valid[df_valid['timepoint'] == tp]
                if len(sub) > 0:
                    t_img, t_well = aggregate_crop_to_well(sub)
                    fold_groups[tp] = {'crop': sub, 'image': t_img, 'well': t_well}
        all_fold_data[fold] = fold_groups

    levels = [('crop', 'crop'), ('image', 'image'), ('well', 'well')]
    if args.guide is not None:
        hierarchies = ['gene', 'pathway', 'family']
    elif args.family:
        hierarchies = ['family']
    else:
        hierarchies = ['guide', 'gene', 'pathway', 'family']

    results = []
    timepoint_results = []

    group_names = ['T1', 'T2', 'T3', 'ALL'] if (use_metabolomics and args.timepoint_panels) else ['ALL']

    for level_key, level_name in levels:
        for hier in hierarchies:
            group_metrics = []
            group_cms = {}

            for group_name in group_names:
                fold_raw_cms = []
                fold_accs = []
                all_labels_set = set()

                for fold, fold_groups in all_fold_data.items():
                    if group_name not in fold_groups:
                        continue
                    level_df = fold_groups[group_name][level_key]

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

                if len(fold_raw_cms) == 0:
                    continue

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

                group_metrics.append({
                    'level': level_name,
                    'hierarchy': hier,
                    'group': group_name,
                    'mean_acc': mean_acc,
                    'std_acc': std_acc,
                    'n_folds': len(fold_raw_cms),
                    'n_classes': n_classes,
                    'random_baseline': random_baseline,
                    'classes_above_50': n_above_50,
                    'classes_above_random': n_above_random
                })
                group_cms[group_name] = {
                    'cm_raw': cm_sum_raw,
                    'cm_norm': cm_sum_normalized,
                    'labels': all_labels,
                    'n_classes': n_classes
                }

                if group_name == 'ALL':
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

            if args.timepoint_panels and len(group_cms) == len(group_names):
                for style in ['binary', 'raw', 'percent']:
                    panels = []
                    for gn in group_names:
                        g = group_cms[gn]
                        gm = next(m for m in group_metrics if m['group'] == gn)
                        cm = g['cm_norm'] if style != 'raw' else g['cm_raw']
                        panels.append({
                            'cm': cm,
                            'labels': g['labels'],
                            'n_classes': g['n_classes'],
                            'title': f"{gn} | Acc: {100*gm['mean_acc']:.1f}%"
                        })
                    suptitle = f'{level_name.capitalize()}/{hier.capitalize()} - 4-fold timepoint panels'
                    plot_timepoint_panels(panels, style,
                                          os.path.join(output_dir, f'timepoint_panels_{style}_{level_name}_{hier}.png'),
                                          suptitle)
                print(f"  TP {level_name}/{hier}: " + " | ".join(
                    f"{m['group']}: {100*m['mean_acc']:.2f}%" for m in group_metrics))
                timepoint_results.extend(group_metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'combined_metrics.csv'), index=False)

    if timepoint_results:
        tp_df = pd.DataFrame(timepoint_results)
        tp_df.to_csv(os.path.join(output_dir, 'timepoint_metrics.csv'), index=False)

    print(f"\nSaved to {output_dir}/")


if __name__ == '__main__':
    main()