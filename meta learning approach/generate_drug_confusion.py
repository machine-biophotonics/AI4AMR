#!/usr/bin/env python3
"""
Generate confusion matrices for drug predictions with majority voting.
Creates:
1. 4 antibiotic-level confusion matrices (row-normalized colors)
2. 4 mechanism-of-action (MoA) grouped confusion matrices

Each matrix includes one concentration level + DMSO/control
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# Mechanism of Action grouping (use ground truth names from mapping)
MOA_GROUPS = {
    "Cell wall (PBP 2)": ["Avibactam", "Clavulanic Acid", "Meropenem", "Mecillinam", "Relebactam"],
    "Cell wall (PBP 3)": ["Aztreonam", "Ceftriaxone", "Cefepim"],
    "Cell wall (PBP 1)": ["Sulbactam", "Penicillin", "Cefsulodin"],
    "Ribosome": ["Doxicyclin", "Chloramphenicol", "Clarithromycin", "Kanamycin"],
    "Gyrase": ["Ciprofloxacin", "Norfloxacin", "Levofloxacin"],
    "Membrane integrity": ["Polymyxin B", "Colistin"],
    "RNA polymerase": ["Rifampicin"],
    "DNA synthesis": ["Trimethoprim"],
    "Control": ["DMSO"]
}

# Normalize predicted antibiotic names to match ground truth mapping
PRED_TO_GT_MAPPING = {
    "Clavulanic_Acid": "Clavulanic Acid",
    "Polymyxin_B": "Polymyxin B",
}


def normalize_antibiotic_name(ab_name):
    """Normalize predicted antibiotic name to match ground truth mapping."""
    return PRED_TO_GT_MAPPING.get(ab_name, ab_name)

# Create reverse mapping: antibiotic -> MoA group
ANTIBIOTIC_TO_MOA = {}
for moa, antibiotics in MOA_GROUPS.items():
    for ab in antibiotics:
        ANTIBIOTIC_TO_MOA[ab] = moa


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
            # Normalize to match ground truth mapping
            antibiotic = normalize_antibiotic_name(antibiotic)
            return antibiotic, conc
    
    return label, 'Unknown'


def get_moa_group(antibiotic):
    """Get MoA group for an antibiotic."""
    return ANTIBIOTIC_TO_MOA.get(antibiotic, "Unknown")


def load_ground_truth_from_mapping(predictions_csv):
    """Load ground truth from IC50 mapping based on plate and well."""
    import json
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mapping_path = os.path.join(script_dir, 'plate_well_ic50_mapping.json')
    
    with open(mapping_path, 'r') as f:
        ic50_data = json.load(f)
    
    return ic50_data


def majority_vote_per_image(df):
    """Perform majority voting across positions for each image.
    
    Key difference: Extract antibiotic from predictions (ignoring concentration)
    before majority voting. This combines votes for same antibiotic regardless
    of predicted concentration.
    """
    # Load ground truth mapping
    print("Loading ground truth from IC50 mapping...")
    ic50_data = load_ground_truth_from_mapping(None)
    
    print("Performing majority voting per image (antibiotic-level, ignoring concentration)...")
    
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
        if plate_key in ic50_data and well in ic50_data[plate_key]:
            ab_info = ic50_data[plate_key][well]
            antibiotic = ab_info.get('antibiotic', 'Unknown')
            ic50 = ab_info.get('ic50_multiple', '1x')
            gt_antibiotic = antibiotic
            gt_concentration = ic50
            gt_label = f"{antibiotic}_{ic50}"
        else:
            gt_antibiotic = 'Unknown'
            gt_concentration = 'Unknown'
            gt_label = 'Unknown'
        
        # Extract antibiotic from each prediction (ignore concentration)
        predictions = group['predicted_class_name'].tolist()
        ab_predictions = []
        for p in predictions:
            ab, _ = get_antibiotic_and_concentration(p)
            ab_predictions.append(ab)
        
        # Majority vote on antibiotic only (concentration ignored)
        ab_counts = Counter(ab_predictions)
        majority_ab = ab_counts.most_common(1)[0][0]
        
        # Also get the full prediction (with concentration) for reference
        pred_counts = Counter(predictions)
        majority_full_pred = pred_counts.most_common(1)[0][0]
        pred_antibiotic, pred_concentration = get_antibiotic_and_concentration(majority_full_pred)
        
        # Use the majority_ab (antibiotic only) for accuracy check
        # But keep the full prediction details for reference
        
        # Add MoA groups
        gt_moa = get_moa_group(gt_antibiotic)
        pred_moa = get_moa_group(majority_ab)  # Use majority_ab for MoA check
        
        results.append({
            'image_path': image_path,
            'well': group['well'].iloc[0],
            'ground_truth_label': gt_label,
            'gt_antibiotic': gt_antibiotic,
            'gt_concentration': gt_concentration,
            'gt_moa': gt_moa,
            'predicted_class_name': majority_full_pred,
            'pred_antibiotic': majority_ab,  # This is the antibiotic-level vote result
            'pred_concentration': pred_concentration,
            'pred_moa': pred_moa,
            'num_positions': len(group),
            'vote_count': ab_counts.most_common(1)[0][1]  # Vote count at antibiotic level
        })
    
    result_df = pd.DataFrame(results)
    print(f"Majority voted predictions: {len(result_df)} images")
    
    return result_df


# ========== 89-Class Functions ==========

def get_antibiotic_base(class_name: str) -> str:
    """Extract base antibiotic name without concentration."""
    if class_name == 'control' or 'DMSO' in class_name:
        return 'DMSO'
    if '_' in class_name:
        return class_name.rsplit('_', 1)[0]
    return class_name


def find_group_boundaries(class_names: list) -> dict:
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
    if current_group is not None:
        groups[current_group] = (start_idx, len(class_names) - 1)
    return groups


def add_group_boxes(ax, class_names: list) -> None:
    """Add rectangular boxes around each antibiotic group on the edges."""
    from matplotlib.patches import Rectangle
    
    groups = {}
    current_group = None
    group_start = None
    
    for i, name in enumerate(class_names):
        base = get_antibiotic_base(name)
        if base != current_group:
            if current_group is not None and group_start is not None:
                groups[current_group] = (group_start, i - 1)
            current_group = base
            group_start = i
    
    if current_group is not None and group_start is not None:
        groups[current_group] = (group_start, len(class_names) - 1)
    
    for group_name, (start_idx, end_idx) in groups.items():
        num_in_group = end_idx - start_idx + 1
        if num_in_group > 1:
            rect = Rectangle(
                (start_idx, start_idx),
                num_in_group, num_in_group,
                linewidth=2.5, edgecolor='darkred',
                facecolor='none', linestyle='-'
            )
            ax.add_patch(rect)
    
    dmso_idx = class_names.index('control') if 'control' in class_names else None
    if dmso_idx is not None:
        ax.axvline(x=dmso_idx + 0.5, color='green', linewidth=2, linestyle='-')
        ax.axhline(y=dmso_idx + 0.5, color='green', linewidth=2, linestyle='-')


def create_89class_confusion_matrix(predictions_csv: str, output_dir: str) -> float:
    """Create 89x89 confusion matrix for all drug classes with concentration."""
    
    print("\n--- Creating 89-class confusion matrix ---")
    
    # Load predictions and perform majority voting
    df = pd.read_csv(predictions_csv)
    df_voted = majority_vote_per_image(df)
    
    # Get all unique classes (ground truth)
    all_classes = sorted([c for c in df_voted['ground_truth_label'].unique() if c != 'Unknown'])
    print(f"Total classes: {len(all_classes)}")
    
    # Order by antibiotic, then concentration
    conc_order = {'0.25x': 0, '0.5x': 1, '1x': 2, '2x': 3, 'control': 4, 'Unknown': 5}
    
    def sort_key(class_name: str):
        ab = get_antibiotic_base(class_name)
        if '_' in class_name:
            conc = class_name.rsplit('_', 1)[-1]
            conc_idx = conc_order.get(conc, 99)
        else:
            conc_idx = 99
        return (ab, conc_idx)
    
    sorted_classes = sorted(all_classes, key=sort_key)
    
    # Print class order
    print("\nClass order (grouped by antibiotic):")
    for ab in sorted(set(get_antibiotic_base(c) for c in sorted_classes)):
        classes = [c for c in sorted_classes if get_antibiotic_base(c) == ab]
        print(f"  {ab}: {classes}")
    
    class_to_idx = {c: i for i, c in enumerate(sorted_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    
    # Filter out Unknown
    df_filtered = df_voted[
        (df_voted['ground_truth_label'] != 'Unknown') & 
        (df_voted['ground_truth_label'].notna()) &
        (df_voted['predicted_class_name'] != 'Unknown') &
        (df_voted['predicted_class_name'].notna())
    ].copy()
    
    print(f"Valid predictions (excluding Unknown): {len(df_filtered)}")
    
    if len(df_filtered) == 0:
        print("ERROR: No valid predictions!")
        return 0.0
    
    gt_mapped = df_filtered['ground_truth_label'].map(class_to_idx)
    pred_mapped = df_filtered['predicted_class_name'].map(class_to_idx)
    
    valid_mask = gt_mapped.notna() & pred_mapped.notna()
    gt_labels = gt_mapped[valid_mask].astype(int).values
    pred_labels = pred_mapped[valid_mask].astype(int).values
    
    present_classes = sorted(set(gt_labels) | set(pred_labels))
    present_class_names = [idx_to_class[i] for i in present_classes]
    
    print(f"Present classes in data: {len(present_class_names)}")
    
    # Filter to present classes
    present_mask_gt = np.isin(gt_labels, present_classes)
    present_mask_pred = np.isin(pred_labels, present_classes)
    valid_mask = present_mask_gt & present_mask_pred
    
    gt_filtered = gt_labels[valid_mask]
    pred_filtered = pred_labels[valid_mask]
    
    reindex = {old: new for new, old in enumerate(present_classes)}
    gt_reindexed = np.array([reindex[g] for g in gt_filtered])
    pred_reindexed = np.array([reindex[p] for p in pred_filtered])
    
    cm = confusion_matrix(gt_reindexed, pred_reindexed, labels=range(len(present_classes)))
    
    accuracy = accuracy_score(gt_reindexed, pred_reindexed)
    print(f"89-class Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    per_class_accuracy = np.nan_to_num(per_class_accuracy, 0)
    
    metrics_data = []
    for i, class_name in enumerate(present_class_names):
        if cm.sum(axis=1)[i] > 0:
            metrics_data.append({
                'class': class_name,
                'support': int(cm.sum(axis=1)[i]),
                'correct': int(cm.diagonal()[i]),
                'accuracy': per_class_accuracy[i]
            })
    
    metrics_df = pd.DataFrame(metrics_data).sort_values('accuracy', ascending=False)
    
    print("\nTop 10 performing classes:")
    print(metrics_df.head(10).to_string(index=False))
    
    print("\nBottom 10 performing classes:")
    print(metrics_df.tail(10).to_string(index=False))
    
    metrics_df.to_csv(os.path.join(output_dir, 'per_class_metrics_89class.csv'), index=False)
    
    # Full heatmap with boxes
    num_classes = len(present_class_names)
    print(f"\nCreating {num_classes}x{num_classes} confusion matrix...")
    
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
    
    add_group_boxes(ax, present_class_names)
    
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('True', fontsize=8)
    ax.set_title(f'89-Class Confusion Matrix (Normalized)\nRed lines = antibiotic groups | Green line = DMSO/Control | Overall Accuracy: {accuracy:.2%}', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=4)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_89class_full.png'), dpi=150)
    plt.close()
    print(f"Saved: confusion_matrix_89class_full.png")
    
    # Raw counts version
    fig, ax = plt.subplots(figsize=(34, 30))
    sns.heatmap(cm, 
                xticklabels=present_class_names, 
                yticklabels=present_class_names,
                cmap='Blues', 
                annot=False,
                fmt='d',
                cbar_kws={'label': 'Count'},
                ax=ax)
    
    add_group_boxes(ax, present_class_names)
    
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
    
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_top, xticklabels=top_classes, yticklabels=top_classes, 
                cmap='Blues', annot=True, fmt='d', ax=ax)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.set_title('Top 20 Performing Classes (Raw Counts)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_89class_top20.png'), dpi=150)
    plt.close()
    print(f"Saved: confusion_matrix_89class_top20.png")
    
    # Bottom 20
    bottom_classes = metrics_df.tail(20)['class'].tolist()
    bottom_indices = [present_class_names.index(c) for c in bottom_classes]
    cm_bottom = cm[np.ix_(bottom_indices, bottom_indices)]
    
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_bottom, xticklabels=bottom_classes, yticklabels=bottom_classes, 
                cmap='Blues', annot=True, fmt='d', ax=ax)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.set_title('Bottom 20 Performing Classes (Raw Counts)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_89class_bottom20.png'), dpi=150)
    plt.close()
    print(f"Saved: confusion_matrix_89class_bottom20.png")
    
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
    
    print(f"\n89-class results saved to: {output_dir}")
    
    return accuracy


def create_heatmap(cm, labels, output_path, title, accuracy, std_acc=None, show_percentage=True):
    """Create a properly normalized confusion matrix heatmap."""
    
    n = len(labels)
    
    if show_percentage:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = np.round(cm_normalized * 100, 1)
        
        annot = np.array([[f"{cm_display[i,j]}%\n({cm[i,j]})" if cm[i,j] > 0 else "0%"
                           for j in range(n)] for i in range(n)])
        
        vmax = 100
        data_for_heatmap = cm_display
        cbar_label = 'Percentage (%)'
    else:
        annot = cm
        vmax = None
        data_for_heatmap = cm
        cbar_label = 'Count'
    
    n_max_on_diagonal = 0
    for i in range(n):
        row = cm[i, :]
        if row.sum() > 0:
            if row.argmax() == i:
                n_max_on_diagonal += 1
    
    pct_on_diagonal = 100.0 * n_max_on_diagonal / n if n > 0 else 0
    acc_str = f' | Acc: {100*accuracy:.1f}%' + (f'\u00b1{100*std_acc:.1f}%' if std_acc is not None else '')
    
    fig, ax = plt.subplots(figsize=(max(16, n * 0.3), max(14, n * 0.25)))
    
    sns.heatmap(data_for_heatmap, annot=annot, fmt='', cmap='Blues', vmax=vmax,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': cbar_label},
                annot_kws={'fontsize': 7 if show_percentage else 9}, ax=ax)
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'{title}\n(Max on Diagonal: {n_max_on_diagonal}/{n} ({pct_on_diagonal:.1f}%){acc_str})',
                 fontsize=13, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_drug_confusion_matrices(predictions_csv, output_dir):
    """Create both antibiotic-level and MoA confusion matrices."""
    
    print(f"Loading predictions from: {predictions_csv}")
    df = pd.read_csv(predictions_csv)
    
    print(f"Total predictions: {len(df)}")
    print(f"Unique images: {df['image_path'].nunique()}")
    
    # Perform majority voting
    df_voted = majority_vote_per_image(df)
    
    # Get unique antibiotics and MoA groups
    all_antibiotics = sorted([a for a in df_voted['gt_antibiotic'].unique() if a != 'DMSO'])
    if 'DMSO' not in all_antibiotics:
        all_antibiotics = ['DMSO'] + all_antibiotics
    
    # MoA order (as defined)
    moa_order = ["Cell wall (PBP 2)", "Cell wall (PBP 3)", "Cell wall (PBP 1)", 
                 "Ribosome", "Gyrase", "Membrane integrity", "RNA polymerase", 
                 "DNA synthesis", "Control"]
    moa_labels = [m for m in moa_order if m in df_voted['gt_moa'].unique()]
    
    print(f"\nAntibiotics: {len(all_antibiotics)}")
    print(f"MoA groups: {len(moa_labels)}")
    
    # Label mappings
    ab_to_idx = {label: i for i, label in enumerate(all_antibiotics)}
    moa_to_idx = {label: i for i, label in enumerate(moa_labels)}
    
    # Concentrations
    concentrations = ['0.25x', '0.5x', '1x', '2x']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    all_metrics = {'antibiotic': {}, 'moa': {}}
    
    for conc in concentrations:
        print(f"\n{'='*60}")
        print(f"Concentration: {conc}")
        print(f"{'='*60}")
        
        # Filter: this concentration + DMSO
        mask = (df_voted['gt_concentration'] == conc) | (df_voted['gt_antibiotic'] == 'DMSO')
        df_conc = df_voted[mask].copy()
        
        print(f"Total images: {len(df_conc)}")
        
        # ============ ANTIBIOTIC-LEVEL ============
        print(f"\n--- Antibiotic-level ---")
        
        # Filter out unknown labels
        df_filtered = df_conc[(df_conc['gt_antibiotic'] != 'Unknown') & 
                               (df_conc['pred_antibiotic'] != 'Unknown')].copy()
        print(f"Filtered images (excluding Unknown): {len(df_filtered)}")
        
        if len(df_filtered) == 0:
            print("No valid predictions after filtering!")
            return
            
        gt_labels = df_filtered['gt_antibiotic'].map(ab_to_idx).values
        pred_labels = df_filtered['pred_antibiotic'].map(ab_to_idx).values
        
        accuracy = accuracy_score(gt_labels, pred_labels)
        all_metrics['antibiotic'][conc] = accuracy
        print(f"Antibiotic Accuracy: {accuracy:.4f}")
        
        cm = confusion_matrix(gt_labels, pred_labels, labels=range(len(all_antibiotics)))
        
        # Create heatmap with percentage
        create_heatmap(cm, all_antibiotics, 
                      os.path.join(output_dir, f'confusion_matrix_{conc}.png'),
                      f'Drug Classification - {conc}', accuracy, show_percentage=True)
        
        # Save CSV
        cm_df = pd.DataFrame(cm, index=all_antibiotics, columns=all_antibiotics)
        cm_df.to_csv(os.path.join(output_dir, f'confusion_matrix_{conc}.csv'))
        
        # ============ MoA-LEVEL ============
        print(f"\n--- MoA-level ---")
        
        # Use filtered dataframe for MoA as well
        gt_moa_labels = df_filtered['gt_moa'].map(moa_to_idx).values
        pred_moa_labels = df_filtered['pred_moa'].map(moa_to_idx).values
        
        moa_accuracy = accuracy_score(gt_moa_labels, pred_moa_labels)
        all_metrics['moa'][conc] = moa_accuracy
        print(f"MoA Accuracy: {moa_accuracy:.4f}")
        
        cm_moa = confusion_matrix(gt_moa_labels, pred_moa_labels, labels=range(len(moa_labels)))
        
        # Create MoA heatmap
        create_heatmap(cm_moa, moa_labels,
                      os.path.join(output_dir, f'confusion_matrix_moa_{conc}.png'),
                      f'MoA Classification - {conc}', moa_accuracy, show_percentage=True)
        
        # Save MoA CSV
        cm_moa_df = pd.DataFrame(cm_moa, index=moa_labels, columns=moa_labels)
        cm_moa_df.to_csv(os.path.join(output_dir, f'confusion_matrix_moa_{conc}.csv'))
    
    # Save summary
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Drug Classification Summary (Majority Voting)\n")
        f.write("="*70 + "\n\n")
        
        f.write("ANTIBIOTIC-LEVEL RESULTS\n")
        f.write("-"*40 + "\n")
        f.write("Note: Same antibiotic (any concentration) = correct\n\n")
        for conc, acc in all_metrics['antibiotic'].items():
            f.write(f"  {conc}: {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"  Average: {np.mean(list(all_metrics['antibiotic'].values())):.4f}\n\n")
        
        f.write("\nMECHANISM OF ACTION (MoA) RESULTS\n")
        f.write("-"*40 + "\n")
        f.write("Note: Same MoA group = correct\n\n")
        for conc, acc in all_metrics['moa'].items():
            f.write(f"  {conc}: {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"  Average: {np.mean(list(all_metrics['moa'].values())):.4f}\n\n")
        
        f.write("\nMoA GROUPING:\n")
        for moa, antibiotics in MOA_GROUPS.items():
            f.write(f"  {moa}: {', '.join(antibiotics)}\n")
    
    print(f"\n{'='*60}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")
    
    # Save majority-voted predictions
    voted_csv = os.path.join(output_dir, 'majority_voted_predictions.csv')
    df_voted.to_csv(voted_csv, index=False)
    print(f"Majority voted predictions: {voted_csv}")


# ========== Aggregate Multi-Level Confusion (crop/image/well × concentration × antibiotic/moa) ==========


def aggregate_image_to_well(df_voted):
    """Majority vote across images within each well."""
    well_results = []
    for well, group in df_voted.groupby('well'):
        gt_label = group['ground_truth_label'].iloc[0]
        gt_ab = group['gt_antibiotic'].iloc[0]
        gt_conc = group['gt_concentration'].iloc[0]
        gt_moa = group['gt_moa'].iloc[0]

        preds = group['pred_antibiotic'].value_counts()
        majority_ab = preds.index[0] if len(preds) > 0 else 'Unknown'

        full_preds = group['predicted_class_name'].value_counts()
        majority_full_pred = full_preds.index[0] if len(full_preds) > 0 else 'Unknown'

        well_results.append({
            'well': well,
            'ground_truth_label': gt_label,
            'gt_antibiotic': gt_ab,
            'gt_concentration': gt_conc,
            'gt_moa': gt_moa,
            'predicted_class_name': majority_full_pred,
            'pred_antibiotic': majority_ab,
            'pred_moa': get_moa_group(majority_ab),
        })
    return pd.DataFrame(well_results)


def create_drug_aggregate_confusion(all_fold_data, output_dir):
    """Per-concentration confusion matrices at crop/image/well levels, aggregated across folds.
    Computes per-fold accuracies for proper mean±std, then sums CMs for the aggregate matrix."""
    os.makedirs(output_dir, exist_ok=True)

    concentrations = ['0.25x', '0.5x', '1x', '2x']
    n_folds = len(all_fold_data)
    fold_keys = list(all_fold_data.keys())

    moa_order = ["Cell wall (PBP 2)", "Cell wall (PBP 3)", "Cell wall (PBP 1)",
                 "Ribosome", "Gyrase", "Membrane integrity", "RNA polymerase",
                 "DNA synthesis", "Control"]

    for level_name in ['crop', 'image', 'well']:
        # Collect global label sets across all folds
        all_ab_set = set()
        all_moa_set = set(moa_order)
        for fk in fold_keys:
            df = all_fold_data[fk][level_name]
            df = df[(df['gt_antibiotic'] != 'Unknown') & (df['pred_antibiotic'] != 'Unknown')]
            all_ab_set.update(df['gt_antibiotic'].unique())
            all_ab_set.update(df['pred_antibiotic'].unique())
        all_ab_set.discard('DMSO')
        all_antibiotics = ['DMSO'] + sorted(all_ab_set)
        moa_labels = [m for m in moa_order if m in all_moa_set]

        ab_to_idx = {l: i for i, l in enumerate(all_antibiotics)}
        moa_to_idx = {l: i for i, l in enumerate(moa_labels)}

        for conc in concentrations:
            # Per-fold accuracies for std dev
            fold_ab_accs = []
            fold_moa_accs = []
            cm_sum = np.zeros((len(all_antibiotics), len(all_antibiotics)), dtype=np.float64)
            cm_moa_sum = np.zeros((len(moa_labels), len(moa_labels)), dtype=np.float64)

            for fk in fold_keys:
                df = all_fold_data[fk][level_name]
                df = df[(df['gt_antibiotic'] != 'Unknown') & (df['pred_antibiotic'] != 'Unknown')]
                mask = (df['gt_concentration'] == conc) | (df['gt_antibiotic'] == 'DMSO')
                df_conc = df[mask]

                if len(df_conc) == 0:
                    continue

                gt = df_conc['gt_antibiotic'].map(ab_to_idx).values
                pred = df_conc['pred_antibiotic'].map(ab_to_idx).values
                fold_ab_accs.append(np.mean(gt == pred))
                cm_sum += confusion_matrix(gt, pred, labels=range(len(all_antibiotics)))

                gt_m = df_conc['gt_moa'].map(moa_to_idx).values
                pred_m = df_conc['pred_moa'].map(moa_to_idx).values
                fold_moa_accs.append(np.mean(gt_m == pred_m))
                cm_moa_sum += confusion_matrix(gt_m, pred_m, labels=range(len(moa_labels)))

            mean_ab = np.mean(fold_ab_accs) if fold_ab_accs else 0
            std_ab = np.std(fold_ab_accs) if len(fold_ab_accs) > 1 else 0
            mean_moa = np.mean(fold_moa_accs) if fold_moa_accs else 0
            std_moa = np.std(fold_moa_accs) if len(fold_moa_accs) > 1 else 0

            create_heatmap(cm_sum, all_antibiotics,
                           os.path.join(output_dir, f'aggregate_confusion_matrix_{level_name}_{conc}.png'),
                           f'{level_name.capitalize()} - {conc} ({n_folds} folds)',
                           mean_ab, std_acc=std_ab, show_percentage=True)
            cm_df = pd.DataFrame(cm_sum, index=all_antibiotics, columns=all_antibiotics)
            cm_df.to_csv(os.path.join(output_dir, f'aggregate_confusion_matrix_{level_name}_{conc}.csv'))

            create_heatmap(cm_moa_sum, moa_labels,
                           os.path.join(output_dir, f'aggregate_confusion_matrix_moa_{level_name}_{conc}.png'),
                           f'{level_name.capitalize()} MoA - {conc} ({n_folds} folds)',
                           mean_moa, std_acc=std_moa, show_percentage=True)
            cm_moa_df = pd.DataFrame(cm_moa_sum, index=moa_labels, columns=moa_labels)
            cm_moa_df.to_csv(os.path.join(output_dir, f'aggregate_confusion_matrix_moa_{level_name}_{conc}.csv'))

            print(f"  {level_name}/{conc}: Ab={100*mean_ab:.2f}%\u00b1{100*std_ab:.2f}%  MoA={100*mean_moa:.2f}%\u00b1{100*std_moa:.2f}%")

        # Save summary with fold-wise stats
        summary_path = os.path.join(output_dir, f'summary_{level_name}.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Drug Classification Summary - {level_name.capitalize()} Level ({n_folds} folds)\n")
            f.write("=" * 70 + "\n\n")
            f.write("ANTIBIOTIC-LEVEL\n" + "-" * 40 + "\n")
            for conc in concentrations:
                accs = []
                for fk in fold_keys:
                    df = all_fold_data[fk][level_name]
                    df = df[(df['gt_antibiotic'] != 'Unknown') & (df['pred_antibiotic'] != 'Unknown')]
                    mask = (df['gt_concentration'] == conc) | (df['gt_antibiotic'] == 'DMSO')
                    df_conc = df[mask]
                    if len(df_conc) > 0:
                        accs.append(accuracy_score(df_conc['gt_antibiotic'], df_conc['pred_antibiotic']))
                if accs:
                    f.write(f"  {conc}: {np.mean(accs):.4f} \u00b1 {np.std(accs):.4f} ({100*np.mean(accs):.2f}% \u00b1 {100*np.std(accs):.2f}%)\n")
            f.write(f"\nMoA-LEVEL\n" + "-" * 40 + "\n")
            for conc in concentrations:
                accs = []
                for fk in fold_keys:
                    df = all_fold_data[fk][level_name]
                    df = df[(df['gt_antibiotic'] != 'Unknown') & (df['pred_antibiotic'] != 'Unknown')]
                    mask = (df['gt_concentration'] == conc) | (df['gt_antibiotic'] == 'DMSO')
                    df_conc = df[mask]
                    if len(df_conc) > 0:
                        accs.append(accuracy_score(df_conc['gt_moa'], df_conc['pred_moa']))
                if accs:
                    f.write(f"  {conc}: {np.mean(accs):.4f} \u00b1 {np.std(accs):.4f} ({100*np.mean(accs):.2f}% \u00b1 {100*np.std(accs):.2f}%)\n")

    print(f"\nSaved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Generate drug confusion matrices')
    parser.add_argument('--fold', type=str, default='P6')
    parser.add_argument('--folds', type=str, default=None,
                        help='Comma-separated folds (e.g., P1,P2,P3,P4,P5,P6) for multi-fold aggregation')
    parser.add_argument('--checkpoint', type=str, default='best_model_acc.pth')
    parser.add_argument('--data_mode', type=str, default='drug')
    parser.add_argument('--neighborhood', type=int, default=3,
                        help='Neighborhood size (3 for 3x3=9 crops)')
    parser.add_argument('--csv_name', type=str, default=None,
                        help='Specific CSV filename to load (e.g., predictions_all_crops_mil_checkpoint_epoch_n3.csv)')

    args = parser.parse_args()

    # Multi-fold aggregation mode
    if args.folds:
        folds = args.folds.split(',')
        fold_keys = []
        for f in folds:
            if f.startswith('fold_'):
                fold_keys.append(f)
            elif 'Plate_' in f:
                fold_keys.append(f'fold_{f}')
            else:
                fold_keys.append(f'fold_Plate_{f.replace("P", "")}')

        csv_name = args.csv_name if args.csv_name else 'predictions_all_crops_mil_checkpoint_epoch_n3.csv'
        all_fold_data = {}

        print(f"Loading {len(folds)} folds with CSV: {csv_name}")
        for fold, fk in zip(folds, fold_keys):
            fold_dir = os.path.join(SCRIPT_DIR, args.data_mode, fk)
            csv_path = os.path.join(fold_dir, csv_name)
            if not os.path.exists(csv_path):
                print(f"  WARNING: {csv_path} not found, skipping {fold}")
                continue
            print(f"  Loading {fold}...")
            df = pd.read_csv(csv_path)
            if 'ground_truth_label' not in df.columns:
                print(f"  WARNING: no ground_truth_label in {fold}, skipping")
                continue

            # Crop-level: raw predictions with added hierarchy columns
            df_crop = df.copy()
            ab_conc = df_crop['ground_truth_label'].apply(
                lambda x: get_antibiotic_and_concentration(x))
            df_crop['gt_antibiotic'] = [a[0] for a in ab_conc]
            df_crop['gt_concentration'] = [a[1] for a in ab_conc]
            df_crop['gt_moa'] = df_crop['gt_antibiotic'].map(get_moa_group)

            pred_ab_conc = df_crop['predicted_class_name'].apply(
                lambda x: get_antibiotic_and_concentration(x))
            df_crop['pred_antibiotic'] = [a[0] for a in pred_ab_conc]
            df_crop['pred_moa'] = df_crop['pred_antibiotic'].map(get_moa_group)

            # Image-level: majority vote per image
            df_voted = majority_vote_per_image(df)

            # Well-level: majority vote per well
            df_well = aggregate_image_to_well(df_voted)

            all_fold_data[fold] = {
                'crop': df_crop,
                'image': df_voted,
                'well': df_well,
            }

        if len(all_fold_data) == 0:
            print("ERROR: No valid folds loaded!")
            return

        output_dir = os.path.join(SCRIPT_DIR, 'aggregate', 'drug_combined')
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nAggregating across {len(all_fold_data)} folds, output: {output_dir}")
        create_drug_aggregate_confusion(all_fold_data, output_dir)
        return

    # Single-fold mode
    fold_input = args.fold
    if fold_input.startswith('fold_'):
        fold_key = fold_input
    elif 'Plate_' in fold_input:
        fold_key = f'fold_{fold_input}'
    else:
        fold_key = f'fold_Plate_{fold_input.replace("P", "")}'
    fold_dir = os.path.join(SCRIPT_DIR, args.data_mode, fold_key)
    checkpoint_name = args.checkpoint.replace('.pth', '')

    if args.csv_name:
        predictions_csv = os.path.join(fold_dir, args.csv_name)
    else:
        predictions_csv = os.path.join(fold_dir, f'predictions_all_crops_mil_{checkpoint_name}_n{args.neighborhood}.csv')
        if not os.path.exists(predictions_csv):
            predictions_csv = os.path.join(fold_dir, f'predictions_all_crops_mil_{checkpoint_name}.csv')

    if not os.path.exists(predictions_csv):
        print(f"ERROR: Predictions file not found: {predictions_csv}")
        return

    output_dir = os.path.join(fold_dir, 'drug_confusion_matrices_with_concentration')
    print(f"Processing single fold {args.fold}, output: {output_dir}")

    create_drug_confusion_matrices(predictions_csv, output_dir)
    create_89class_confusion_matrix(predictions_csv, output_dir)


if __name__ == '__main__':
    main()