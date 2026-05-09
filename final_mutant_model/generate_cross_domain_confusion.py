#!/usr/bin/env python3
"""
Generate cross-domain confusion matrix analysis.

Two modes:
1. drug_on_mutant: Drug model predicts antibiotics on mutant images
   - Ground truth: CRISPRi gene knockdowns (e.g., lptA_3, gyrB_2)
   - Predicted: Antibiotic classes (e.g., Chloramphenicol_1x)
   - Analysis: Does predicted antibiotic's target genes match ground truth gene?

2. mutant_on_drug: Mutant model predicts genes on drug images
   - Ground truth: Antibiotic classes (e.g., Chloramphenicol_1x)
   - Predicted: Gene labels (e.g., gyrB_2, rplC_3)
   - Analysis: Does predicted gene match antibiotic's target genes?

Usage:
  # Drug on mutant (gene → antibiotic targets)
  python3 generate_cross_domain_confusion.py --mode drug_on_mutant --fold Plate_1 --checkpoint best_model_acc

  # Mutant on drug (antibiotic → gene targets)
  python3 generate_cross_domain_confusion.py --mode mutant_on_drug --fold Plate_1 --checkpoint checkpoint_epoch
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
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Antibiotic to target genes mapping
ANTIBIOTIC_GENES = {
    "Cefsulodin": ["mrcA", "mrcB"],
    "Penicillin": ["mrcA", "mrcB", "ftsI"],
    "Sulbactam": ["mrcA", "mrcB", "ftsI"],
    "Avibactam": [],
    "Mecillinam": ["mrdA"],
    "Meropenem": ["mrdA", "ftsI", "mrcA", "mrcB"],
    "Clavulanic Acid": [],
    "Relebactam": [],
    "Aztreonam": ["ftsI"],
    "Cefepim": ["ftsI", "mrcA", "mrcB", "mrdA"],
    "Ceftriaxone": ["ftsI", "mrcA", "mrcB"],
    "Cefepime": ["ftsI", "mrcA", "mrcB", "mrdA"],
    "Chloramphenicol": ["rplA", "rplC"],
    "Clarithromycin": ["rplA", "rplC"],
    "Doxicyclin": ["rpsA", "rpsL"],
    "Doxycycline": ["rpsA", "rpsL"],
    "Kanamycin": ["rpsA", "rpsL"],
    "Ciprofloxacin": ["gyrA", "gyrB", "parC", "parE"],
    "Levofloxacin": ["gyrA", "gyrB", "parC", "parE"],
    "Norfloxacin": ["gyrA", "gyrB", "parC", "parE"],
    "Rifampicin": ["rpoA", "rpoB"],
    "Trimethoprim": ["folA", "folP"],
    "Colistin": ["lpxA", "lpxC", "lptA", "lptC"],
    "Polymyxin B": ["lpxA", "lpxC", "lptA", "lptC"],
    "Polymyxin_B": ["lpxA", "lpxC", "lptA", "lptC"],
    "DMSO": [],
    "control": [],
}


def get_gene_from_id(mutant_id: str) -> str:
    """Extract base gene name from mutant ID like 'lptA_3' -> 'lptA'"""
    if '_' in mutant_id:
        return mutant_id.rsplit('_', 1)[0]
    return mutant_id


def get_antibiotic_from_prediction(pred_class: str) -> str:
    """Extract antibiotic name from prediction like 'Ciprofloxacin_1x' -> 'Ciprofloxacin'"""
    if '_' in pred_class:
        return pred_class.rsplit('_', 1)[0]
    return pred_class


def analyze_drug_on_mutant(predictions_csv: str, output_dir: str):
    """
    Drug model on mutant images:
    - Ground truth: CRISPRi gene knockdowns (e.g., lptA_3)
    - Predicted: Antibiotic classes (e.g., Chloramphenicol_1x)
    - Goal: Does predicted antibiotic target the ground truth gene?
    """
    print(f"\n{'='*60}")
    print("MODE: Drug on Mutant (Drug model predicts antibiotics on mutant images)")
    print(f"{'='*60}")
    
    print(f"Loading predictions from: {predictions_csv}")
    df = pd.read_csv(predictions_csv)
    print(f"Total predictions: {len(df)}")
    
    # Majority voting per image
    results = []
    for image_path, group in df.groupby('image_path'):
        gt_label = group['ground_truth_label'].iloc[0]  # Mutant gene
        pred_counts = group['predicted_class_name'].value_counts()
        majority_pred = pred_counts.index[0]  # Drug class
        
        # Extract gene and antibiotic
        target_gene = get_gene_from_id(gt_label) if pd.notna(gt_label) else None
        pred_antibiotic = get_antibiotic_from_prediction(majority_pred)
        pred_genes = ANTIBIOTIC_GENES.get(pred_antibiotic, [])
        
        # Check if target gene is in predicted antibiotic's target genes
        gene_match = target_gene in pred_genes if target_gene and pred_genes else False
        
        results.append({
            'image_path': image_path,
            'ground_truth_gene': gt_label,
            'target_gene': target_gene,
            'predicted_class': majority_pred,
            'predicted_antibiotic': pred_antibiotic,
            'predicted_target_genes': pred_genes,
            'gene_match': gene_match
        })
    
    result_df = pd.DataFrame(results)
    print(f"Total images analyzed: {len(result_df)}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== 1. Overall Gene Match Rate ==========
    gene_match_rate = result_df['gene_match'].mean() if len(result_df) > 0 else 0
    print(f"\n=== Gene Match Rate: {gene_match_rate:.2%} ===")
    print("(Does predicted antibiotic's target genes include the CRISPRi target gene?)")
    
    # ========== 2. Per-gene analysis ==========
    if 'target_gene' in result_df.columns and len(result_df) > 0:
        gene_stats = result_df.groupby('target_gene').agg({
            'gene_match': ['sum', 'count', 'mean']
        }).round(3)
        gene_stats.columns = ['matches', 'total', 'rate']
        gene_stats = gene_stats.sort_values('rate', ascending=False)
        print("\n=== Per-Gene Match Rate (top 10) ===")
        print(gene_stats.head(10))
        gene_stats.to_csv(os.path.join(output_dir, 'per_gene_statistics.csv'))
    
    # ========== 3. Per-antibiotic analysis ==========
    ab_stats = []
    for ab, genes in ANTIBIOTIC_GENES.items():
        if not genes:
            continue
        matching_mutants = result_df[result_df['target_gene'].isin(genes)] if 'target_gene' in result_df.columns else pd.DataFrame()
        if len(matching_mutants) > 0:
            match_rate = matching_mutants['gene_match'].mean()
            ab_stats.append({
                'antibiotic': ab,
                'target_genes': ', '.join(genes),
                'num_mutants': len(matching_mutants),
                'correctly_matched': matching_mutants['gene_match'].sum(),
                'match_rate': match_rate
            })
    
    if ab_stats:
        ab_df = pd.DataFrame(ab_stats).sort_values('match_rate', ascending=False)
        print("\n=== Per-Antibiotic Match Rate (top 10) ===")
        print(ab_df.head(10).to_string(index=False))
        ab_df.to_csv(os.path.join(output_dir, 'per_antibiotic_statistics.csv'), index=False)
    
    # ========== 4. Confusion Matrix: Gene → Predicted Antibiotic ==========
    if 'target_gene' in result_df.columns:
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
        plt.title(f'Drug-on-Mutant: Gene to Predicted Antibiotic\n(Gene Match Rate: {gene_match_rate:.1%})', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gene_to_antibiotic_heatmap.png'), dpi=150)
        plt.close()
        gene_ab_matrix.to_csv(os.path.join(output_dir, 'gene_to_antibiotic_matrix.csv'))
    
    # ========== 5. Save predictions ==========
    result_df.to_csv(os.path.join(output_dir, 'cross_domain_predictions.csv'), index=False)
    
    # ========== 6. Summary ==========
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write("Drug-on-Mutant Analysis Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Ground truth: CRISPRi gene knockdowns\n")
        f.write(f"Predicted: Antibiotic classes\n")
        f.write(f"Analysis: Does predicted antibiotic target the CRISPRi gene?\n\n")
        f.write(f"Overall Gene Match Rate: {gene_match_rate:.2%}\n")
    
    print(f"\nSaved to: {output_dir}")
    
    return gene_match_rate


def analyze_mutant_on_drug(predictions_csv: str, output_dir: str):
    """
    Mutant model on drug images:
    - Ground truth: Antibiotic classes (e.g., Chloramphenicol_1x)
    - Predicted: Gene labels (e.g., gyrB_2, rplC_3)
    - Goal: Does predicted gene match the antibiotic's target genes?
    """
    print(f"\n{'='*60}")
    print("MODE: Mutant on Drug (Mutant model predicts genes on drug images)")
    print(f"{'='*60}")
    
    print(f"Loading predictions from: {predictions_csv}")
    df = pd.read_csv(predictions_csv)
    print(f"Total predictions: {len(df)}")
    
    # Majority voting per image
    results = []
    for image_path, group in df.groupby('image_path'):
        gt_label = group['ground_truth_label'].iloc[0]  # Antibiotic
        pred_counts = group['predicted_class_name'].value_counts()
        majority_pred = pred_counts.index[0]  # Gene
        
        # Extract antibiotic and gene
        gt_antibiotic = get_antibiotic_from_prediction(gt_label) if pd.notna(gt_label) else None
        target_genes = ANTIBIOTIC_GENES.get(gt_antibiotic, []) if gt_antibiotic else []
        
        pred_gene = get_gene_from_id(majority_pred)
        
        # Check if predicted gene matches any of the antibiotic's target genes
        gene_match = pred_gene in target_genes if target_genes else False
        
        results.append({
            'image_path': image_path,
            'ground_truth_antibiotic': gt_label,
            'target_antibiotic': gt_antibiotic,
            'target_genes': target_genes,
            'predicted_gene': majority_pred,
            'predicted_base_gene': pred_gene,
            'gene_match': gene_match
        })
    
    result_df = pd.DataFrame(results)
    print(f"Total images analyzed: {len(result_df)}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== 1. Overall Gene Match Rate ==========
    gene_match_rate = result_df['gene_match'].mean() if len(result_df) > 0 else 0
    print(f"\n=== Gene Match Rate: {gene_match_rate:.2%} ===")
    print("(Does predicted gene match any of the antibiotic's target genes?)")
    
    # ========== 2. Per-antibiotic analysis ==========
    if 'target_antibiotic' in result_df.columns and len(result_df) > 0:
        ab_stats = result_df.groupby('target_antibiotic').agg({
            'gene_match': ['sum', 'count', 'mean']
        }).round(3)
        ab_stats.columns = ['matches', 'total', 'rate']
        ab_stats = ab_stats.sort_values('rate', ascending=False)
        print("\n=== Per-Antibiotic Match Rate (top 10) ===")
        print(ab_stats.head(10))
        ab_stats.to_csv(os.path.join(output_dir, 'per_antibiotic_statistics.csv'))
    
    # ========== 3. Per-gene analysis ==========
    gene_stats = []
    for gene in result_df['predicted_base_gene'].unique() if 'predicted_base_gene' in result_df.columns else []:
        gene_subset = result_df[result_df['predicted_base_gene'] == gene]
        match_count = 0
        total = len(gene_subset)
        for _, row in gene_subset.iterrows():
            if gene in row['target_genes']:
                match_count += 1
        gene_stats.append({
            'predicted_gene': gene,
            'num_predictions': total,
            'matches_antibiotic_targets': match_count,
            'match_rate': match_count / total if total > 0 else 0
        })
    
    if gene_stats:
        gene_df = pd.DataFrame(gene_stats).sort_values('match_rate', ascending=False)
        print("\n=== Per-Gene Match Rate (top 10) ===")
        print(gene_df.head(10).to_string(index=False))
        gene_df.to_csv(os.path.join(output_dir, 'per_gene_statistics.csv'), index=False)
    
    # ========== 4. Confusion Matrix: Antibiotic → Predicted Gene ==========
    if 'target_antibiotic' in result_df.columns:
        ab_gene_matrix = pd.crosstab(
            result_df['target_antibiotic'], 
            result_df['predicted_base_gene'],
            normalize='index'
        ) * 100
        
        # Get top genes for visualization
        top_genes = result_df['predicted_base_gene'].value_counts().head(15).index.tolist()
        
        plt.figure(figsize=(18, 14))
        sns.heatmap(ab_gene_matrix[top_genes], annot=True, fmt='.1f', cmap='Blues',
                    cbar_kws={'label': 'Percentage (%)'})
        plt.xlabel('Predicted Gene', fontsize=12)
        plt.ylabel('Ground Truth Antibiotic', fontsize=12)
        plt.title(f'Mutant-on-Drug: Antibiotic to Predicted Gene\n(Gene Match Rate: {gene_match_rate:.1%})', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'antibiotic_to_gene_heatmap.png'), dpi=150)
        plt.close()
        ab_gene_matrix.to_csv(os.path.join(output_dir, 'antibiotic_to_gene_matrix.csv'))
    
    # ========== 5. Save predictions ==========
    result_df.to_csv(os.path.join(output_dir, 'cross_domain_predictions.csv'), index=False)
    
    # ========== 6. Summary ==========
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write("Mutant-on-Drug Analysis Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Ground truth: Antibiotic classes\n")
        f.write(f"Predicted: Gene labels (mutant IDs)\n")
        f.write(f"Analysis: Does predicted gene match antibiotic's target genes?\n\n")
        f.write(f"Overall Gene Match Rate: {gene_match_rate:.2%}\n")
    
    print(f"\nSaved to: {output_dir}")
    
    return gene_match_rate


def main():
    parser = argparse.ArgumentParser(description='Generate cross-domain confusion matrix analysis')
    parser.add_argument('--mode', type=str, required=True, choices=['drug_on_mutant', 'mutant_on_drug'],
                        help='Cross-domain mode: drug_on_mutant or mutant_on_drug')
    parser.add_argument('--fold', type=str, default='Plate_1',
                        help='Fold to analyze (e.g., Plate_1, Plate_2)')
    parser.add_argument('--checkpoint', type=str, default='best_model_acc',
                        help='Checkpoint filename (without .pth)')
    parser.add_argument('--neighborhood', type=int, default=3,
                        help='Neighborhood size used for prediction')
    
    args = parser.parse_args()
    
    # Construct prediction file path
    predictions_csv = os.path.join(
        SCRIPT_DIR, args.mode, f'fold_{args.fold}',
        f'predictions_all_crops_mil_{args.checkpoint}_n{args.neighborhood}.csv'
    )
    
    if not os.path.exists(predictions_csv):
        print(f"ERROR: Predictions file not found: {predictions_csv}")
        return
    
    # Output directory
    output_dir = os.path.join(SCRIPT_DIR, args.mode, f'fold_{args.fold}', 'gene_confusion')
    
    # Run appropriate analysis
    if args.mode == 'drug_on_mutant':
        analyze_drug_on_mutant(predictions_csv, output_dir)
    else:
        analyze_mutant_on_drug(predictions_csv, output_dir)


if __name__ == '__main__':
    main()