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


def create_heatmap(cm, labels, output_path, title, accuracy, show_percentage=True):
    """Create a properly normalized confusion matrix heatmap."""
    
    if show_percentage:
        # Row-normalize (each row sums to 1)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = np.round(cm_normalized * 100, 1)
        
        # Create annotation with both count and percentage
        annot = np.array([[f"{cm_display[i,j]}%\n({cm[i,j]})" if cm[i,j] > 0 else "0%" 
                           for j in range(len(labels))] for i in range(len(labels))])
        
        vmax = 100  # Percentage scale
    else:
        annot = cm
        vmax = None
    
    fig, ax = plt.subplots(figsize=(16, 14))
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', vmax=vmax,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count' if not show_percentage else 'Percentage (%)'},
                annot_kws={'fontsize': 7 if show_percentage else 9}, ax=ax)
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'{title}\nAccuracy: {accuracy:.2%}', fontsize=14)
    # Add accuracy on top
    ax.text(0.5, 1.02, f'Accuracy: {accuracy:.2%}', transform=ax.transAxes, 
            ha='center', fontsize=14, fontweight='bold', color='darkblue')
    
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


def main():
    parser = argparse.ArgumentParser(description='Generate drug confusion matrices')
    parser.add_argument('--fold', type=str, default='P6')
    parser.add_argument('--checkpoint', type=str, default='best_model_acc.pth')
    parser.add_argument('--data_mode', type=str, default='drug')
    parser.add_argument('--neighborhood', type=int, default=3,
                        help='Neighborhood size (3 for 3x3=9 crops)')
    
    args = parser.parse_args()
    
    fold_dir = os.path.join(SCRIPT_DIR, args.data_mode, f'fold_{args.fold}')
    checkpoint_name = args.checkpoint.replace('.pth', '')
    
    predictions_csv = os.path.join(fold_dir, f'predictions_all_crops_mil_{checkpoint_name}_n{args.neighborhood}.csv')
    
    if not os.path.exists(predictions_csv):
        predictions_csv = os.path.join(fold_dir, f'predictions_all_crops_mil_{checkpoint_name}.csv')
    
    if not os.path.exists(predictions_csv):
        print(f"ERROR: Predictions file not found: {predictions_csv}")
        return
    
    output_dir = os.path.join(fold_dir, 'drug_confusion_matrices_with_concentration')
    
    create_drug_confusion_matrices(predictions_csv, output_dir)


if __name__ == '__main__':
    main()