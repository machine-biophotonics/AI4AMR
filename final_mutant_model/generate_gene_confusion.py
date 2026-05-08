#!/usr/bin/env python3
"""
Generate confusion matrix comparing CRISPRi gene knockdowns vs predicted antibiotic target genes.
This evaluates if drug model predictions match the expected antibiotic-gene relationship.
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
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter
from typing import Dict, Set, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Antibiotic to target genes mapping (from user's summary)
ANTIBIOTIC_GENES = {
    "Cefsulodin": ["mrcA", "mrcB"],
    "Penicillin": ["mrcA", "mrcB", "ftsI"],
    "Sulbactam": ["mrcA", "mrcB", "ftsI"],
    "Avibactam": [],  # No specific gene targets
    "Mecillinam": ["mrdA"],
    "Meropenem": ["mrdA", "ftsI", "mrcA", "mrcB"],
    "Clavulanic Acid": [],  # No specific gene targets
    "Relebactam": [],  # No specific gene targets
    "Aztreonam": ["ftsI"],
    "Cefepime": ["ftsI", "mrcA", "mrcB", "mrdA"],
    "Ceftriaxone": ["ftsI", "mrcA", "mrcB"],
    "Chloramphenicol": ["rplA", "rplC"],
    "Clarithromycin": ["rplA", "rplC"],
    "Doxycycline": ["rpsA", "rpsL"],
    "Kanamycin": ["rpsA", "rpsL"],
    "Ciprofloxacin": ["gyrA", "gyrB", "parC", "parE"],
    "Levofloxacin": ["gyrA", "gyrB", "parC", "parE"],
    "Norfloxacin": ["gyrA", "gyrB", "parC", "parE"],
    "Rifampicin": ["rpoA", "rpoB"],
    "Trimethoprim": ["folA", "folP"],
    "Colistin": ["lpxA", "lpxC", "lptA", "lptC"],
    "Polymyxin B": ["lpxA", "lpxC", "lptA", "lptC"],
    "DMSO": [],  # Control
}

# Extract base gene name from mutant ID (e.g., lptA_3 -> lptA)
def get_gene_from_mutant(mutant_id: str) -> str:
    """Extract base gene name from mutant ID like 'lptA_3' -> 'lptA'"""
    if '_' in mutant_id:
        return mutant_id.rsplit('_', 1)[0]
    return mutant_id


def load_mutant_data() -> dict:
    """Load mutant ID mapping."""
    with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json'), 'r') as f:
        return json.load(f)


def get_antibiotic_from_prediction(pred_class: str) -> str:
    """Extract antibiotic name from prediction like 'Ciprofloxacin_1x' -> 'Ciprofloxacin'"""
    if '_' in pred_class:
        return pred_class.rsplit('_', 1)[0]
    return pred_class


def create_gene_confusion_matrix(predictions_csv: str, output_dir: str):
    """Create gene-level confusion matrix."""
    
    print(f"Loading predictions from: {predictions_csv}")
    df = pd.read_csv(predictions_csv)
    
    print(f"Total predictions: {len(df)}")
    print(f"Unique images: {df['image_path'].nunique()}")
    
    # Load mutant ground truth
    mutant_data = load_mutant_data()
    
    # Create gene-level predictions for each image
    results = []
    
    for image_path, group in df.groupby('image_path'):
        # Get mutant info from well
        plate = group['plate'].iloc[0]
        well = group['well'].iloc[0]
        
        # Convert plate name
        if 'Plate_' in plate:
            plate_key = f"P{plate.split('_')[-1]}"
        else:
            plate_key = plate
        
        # Get row and column from well (e.g., A01 -> A, 01 -> 1)
        row_letter = well[0] if len(well) >= 1 else ''
        col_num = well[1:] if len(well) > 1 else ''
        
        # Get mutant ID from mapping
        mutant_id = None
        if plate_key in mutant_data and row_letter in mutant_data[plate_key]:
            if col_num in mutant_data[plate_key][row_letter]:
                mutant_id = mutant_data[plate_key][row_letter][col_num].get('id', None)
        
        if mutant_id is None:
            continue
        
        # Get CRISPRi target gene
        target_gene = get_gene_from_mutant(mutant_id)
        
        # Get predicted antibiotic
        pred_class = group['predicted_class_name'].iloc[0]
        pred_antibiotic = get_antibiotic_from_prediction(pred_class)
        
        # Get predicted target genes
        pred_genes = ANTIBIOTIC_GENES.get(pred_antibiotic, [])
        
        # Check if target gene is in predicted genes
        gene_match = target_gene in pred_genes if pred_genes else False
        
        results.append({
            'image_path': image_path,
            'well': well,
            'mutant_id': mutant_id,
            'target_gene': target_gene,
            'predicted_class': pred_class,
            'predicted_antibiotic': pred_antibiotic,
            'predicted_genes': pred_genes,
            'gene_match': gene_match
        })
    
    result_df = pd.DataFrame(results)
    print(f"Total images with mutant info: {len(result_df)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== 1. Gene Match Rate ==========
    gene_match_rate = result_df['gene_match'].mean()
    print(f"\n=== Gene Match Rate: {gene_match_rate:.2%} ===")
    
    # ========== 2. Per-gene accuracy ==========
    print("\n=== Per-Gene Match Rate ===")
    gene_stats = result_df.groupby('target_gene').agg({
        'gene_match': ['sum', 'count', 'mean']
    }).round(3)
    gene_stats.columns = ['matches', 'total', 'rate']
    gene_stats = gene_stats.sort_values('rate', ascending=False)
    print(gene_stats)
    
    # Save gene statistics
    gene_stats.to_csv(os.path.join(output_dir, 'per_gene_statistics.csv'))
    
    # ========== 3. Confusion Matrix: Gene -> Predicted Antibiotic ==========
    # For each gene, which antibiotic was predicted most?
    gene_ab_matrix = pd.crosstab(
        result_df['target_gene'], 
        result_df['predicted_antibiotic'],
        normalize='index'
    ) * 100
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(gene_ab_matrix, annot=True, fmt='.1f', cmap='Blues',
                cbar_kws={'label': 'Percentage (%)'})
    plt.xlabel('Predicted Antibiotic', fontsize=12)
    plt.ylabel('CRISPRi Target Gene', fontsize=12)
    plt.title(f'Gene to Predicted Antibiotic Mapping\n(Gene Match Rate: {gene_match_rate:.1%})', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gene_to_antibiotic_heatmap.png'), dpi=150)
    plt.close()
    
    gene_ab_matrix.to_csv(os.path.join(output_dir, 'gene_to_antibiotic_matrix.csv'))
    
    # ========== 4. Per-antibiotic analysis ==========
    print("\n=== Per-Antibiotic Gene Coverage ===")
    ab_stats = []
    for ab, genes in ANTIBIOTIC_GENES.items():
        if not genes:
            continue
        # Find mutants that target these genes
        matching_mutants = result_df[result_df['target_gene'].isin(genes)]
        if len(matching_mutants) > 0:
            match_rate = matching_mutants['gene_match'].mean()
            ab_stats.append({
                'antibiotic': ab,
                'target_genes': ', '.join(genes),
                'num_mutants': len(matching_mutants),
                'correctly_matched': matching_mutants['gene_match'].sum(),
                'match_rate': match_rate
            })
    
    ab_df = pd.DataFrame(ab_stats).sort_values('match_rate', ascending=False)
    print(ab_df.to_string(index=False))
    ab_df.to_csv(os.path.join(output_dir, 'per_antibiotic_statistics.csv'), index=False)
    
    # ========== 5. Save predictions ==========
    result_df.to_csv(os.path.join(output_dir, 'gene_level_predictions.csv'), index=False)
    
    # ========== 6. Summary ==========
    summary_path = os.path.join(output_dir, 'gene_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Gene-Level Analysis Summary (Drug Model on Mutant Plates)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Overall Gene Match Rate: {gene_match_rate:.2%}\n\n")
        f.write("Gene-level performance:\n")
        f.write(gene_stats.to_string())
        f.write("\n\nAntibiotic-level performance:\n")
        f.write(ab_df.to_string(index=False))
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - gene_to_antibiotic_heatmap.png")
    print(f"  - gene_to_antibiotic_matrix.csv")
    print(f"  - per_gene_statistics.csv")
    print(f"  - per_antibiotic_statistics.csv")
    print(f"  - gene_summary.txt")
    
    return gene_match_rate


def main():
    parser = argparse.ArgumentParser(description='Generate gene-level confusion matrix for drug on mutant predictions')
    parser.add_argument('--fold', type=str, default='P1',
                        help='Fold to analyze (e.g., P1, Plate_1)')
    parser.add_argument('--checkpoint', type=str, default='best_model_acc.pth',
                        help='Checkpoint filename used for prediction')
    parser.add_argument('--neighborhood', type=int, default=3,
                        help='Neighborhood size used for prediction')
    
    args = parser.parse_args()
    
    # Construct prediction file path
    predictions_csv = os.path.join(
        SCRIPT_DIR, 'drug_on_mutant', f'fold_{args.fold}',
        f'predictions_all_crops_mil_{args.checkpoint.replace(".pth", "")}_n{args.neighborhood}.csv'
    )
    
    if not os.path.exists(predictions_csv):
        # Try alternative path
        predictions_csv = os.path.join(
            SCRIPT_DIR, 'drug_on_mutant', f'fold_{args.fold}',
            f'predictions_all_crops_mil_{args.checkpoint.replace(".pth", "")}.csv'
        )
    
    if not os.path.exists(predictions_csv):
        print(f"ERROR: Predictions file not found: {predictions_csv}")
        print("Please run predict_all_crops.py with --drug_on_mutant first!")
        return
    
    output_dir = os.path.join(SCRIPT_DIR, 'drug_on_mutant', f'fold_{args.fold}', 'gene_confusion')
    
    create_gene_confusion_matrix(predictions_csv, output_dir)


if __name__ == '__main__':
    main()