#!/usr/bin/env python3
"""
Compute Cosine Similarity between Mutant, Drug, and Antibiotic embeddings (ALL Plates P1-P6)

This script:
1. Loads saved embeddings from all 6 plates (embeddings/Mutants_P1-6/ and embeddings/Drugs_P1-6/)
2. Averages embeddings across plates per well (each well has 6 plates worth of data)
3. Maps wells to:
   - Mutant: gene IDs (plate_well_id_path.json)
   - Drug: well position (for drug vs mutant)
   - Antibiotic: antibiotic_IC50 (plate_well_ic50_mapping.json)
4. Computes three similarity matrices:
   - Mutant vs Drug (96 x 96)
   - Mutant vs Antibiotic (96 x 89)
   - Drug vs Antibiotic (96 x 89)
5. Saves all matrices as CSV
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
GENE_MAPPING_FILE = os.path.join(BASE_DIR, "plate_well_id_path.json")
DRUG_MAPPING_FILE = os.path.join(BASE_DIR, "plate_well_ic50_mapping.json")

PLATES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']


def well_to_row_col(well_id: str) -> tuple:
    """Convert WellA01 to (A, 1)"""
    match = well_id.replace("Well", "")
    row = match[0]
    col = int(match[1:])
    return row, str(col)


def well_to_wellname(well_id: str) -> str:
    """Convert WellA01 to A01"""
    return well_id.replace("Well", "")


def load_gene_mapping():
    """Load the well-to-gene mapping from JSON for all plates"""
    with open(GENE_MAPPING_FILE, 'r') as f:
        mapping = json.load(f)
    return mapping


def load_drug_mapping():
    """Load the well-to-antibiotic mapping from JSON for all plates"""
    with open(DRUG_MAPPING_FILE, 'r') as f:
        mapping = json.load(f)
    return mapping


def load_embeddings_by_well_all_plates(data_type: str) -> dict:
    """Load embeddings from all plates and average per well.
    data_type: 'mutant' or 'drug'
    """
    well_embeddings = defaultdict(list)
    
    for plate in PLATES:
        embeddings_dir = os.path.join(EMBEDDINGS_DIR, f"{data_type.capitalize()}s_{plate}")
        
        if not os.path.exists(embeddings_dir):
            print(f"  WARNING: {embeddings_dir} not found, skipping...")
            continue
        
        well_folders = glob.glob(os.path.join(embeddings_dir, "Well*"))
        
        for well_folder in well_folders:
            well_id = os.path.basename(well_folder)
            
            npy_files = glob.glob(os.path.join(well_folder, "*.npy"))
            
            for npy_file in npy_files:
                embedding = np.load(npy_file)
                well_embeddings[well_id].append(embedding)
    
    # Average embeddings across all plates per well
    well_mean_embeddings = {}
    for well_id, embeddings_list in well_embeddings.items():
        if len(embeddings_list) > 0:
            well_mean_embeddings[well_id] = np.mean(embeddings_list, axis=0)
    
    return well_mean_embeddings


def load_embeddings_by_well(embeddings_dir: str) -> dict:
    """Load all embeddings and compute mean per well (single plate version)"""
    well_embeddings = defaultdict(list)
    
    well_folders = glob.glob(os.path.join(embeddings_dir, "Well*"))
    
    for well_folder in well_folders:
        well_id = os.path.basename(well_folder)
        
        npy_files = glob.glob(os.path.join(well_folder, "*.npy"))
        
        for npy_file in npy_files:
            embedding = np.load(npy_file)
            well_embeddings[well_id].append(embedding)
    
    # Compute mean embedding per well
    well_mean_embeddings = {}
    for well_id, embeddings_list in well_embeddings.items():
        if len(embeddings_list) > 0:
            well_mean_embeddings[well_id] = np.mean(embeddings_list, axis=0)
    
    return well_mean_embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def compute_mutant_drug_similarity(mutant_embeddings: dict, drug_embeddings: dict, all_gene_mapping: dict) -> pd.DataFrame:
    """Compute similarity matrix between mutant genes and drug wells (all plates averaged)"""
    
    # Get valid wells from all plates combined
    valid_mutant_wells = []
    for well_id in mutant_embeddings.keys():
        row, col = well_to_row_col(well_id)
        # Check if this well exists in any plate
        for plate in PLATES:
            if plate in all_gene_mapping and row in all_gene_mapping[plate] and col in all_gene_mapping[plate][row]:
                valid_mutant_wells.append(well_id)
                break
    
    valid_drug_wells = list(drug_embeddings.keys())
    
    # Create gene ID list - use first available mapping for naming
    gene_counts = defaultdict(int)
    unique_mutant_ids = []
    for well_id in sorted(valid_mutant_wells):
        row, col = well_to_row_col(well_id)
        gene_id = None
        for plate in PLATES:
            if plate in all_gene_mapping and row in all_gene_mapping[plate] and col in all_gene_mapping[plate][row]:
                gene_id = all_gene_mapping[plate][row][col]['id']
                break
        if gene_id is None:
            gene_id = 'Unknown'
        if gene_counts[gene_id] > 0:
            gene_id = f"{gene_id}_W{well_id.replace('Well', '')}"
        gene_counts[gene_id] += 1
        unique_mutant_ids.append(gene_id)
    
    print(f"Computing Mutant vs Drug matrix ({len(unique_mutant_ids)} x {len(valid_drug_wells)})...")
    similarity_matrix = np.zeros((len(valid_mutant_wells), len(valid_drug_wells)))
    
    for i, mutant_well in enumerate(sorted(valid_mutant_wells)):
        for j, drug_well in enumerate(sorted(valid_drug_wells)):
            sim = cosine_similarity(mutant_embeddings[mutant_well], drug_embeddings[drug_well])
            similarity_matrix[i, j] = sim
    
    df = pd.DataFrame(similarity_matrix, index=unique_mutant_ids, columns=sorted(valid_drug_wells))
    return df


def compute_antibiotic_embeddings(embeddings_dict: dict, all_drug_mapping: dict, is_drug: bool = False) -> dict:
    """Aggregate embeddings by antibiotic_IC50 across all plates"""
    
    antibiotic_embeddings = defaultdict(list)
    
    for well_id, embedding in embeddings_dict.items():
        wellname = well_to_wellname(well_id)
        
        # Check all plates for this wellname
        for plate in PLATES:
            if plate in all_drug_mapping and wellname in all_drug_mapping[plate]:
                ab = all_drug_mapping[plate][wellname]['antibiotic']
                ic50 = all_drug_mapping[plate][wellname]['ic50_multiple']
                ab_id = f"{ab}_{ic50}"
                antibiotic_embeddings[ab_id].append(embedding)
                break  # Use first match
    
    # Compute mean embedding per antibiotic
    ab_mean_embeddings = {}
    for ab_id, embeddings_list in antibiotic_embeddings.items():
        if len(embeddings_list) > 0:
            ab_mean_embeddings[ab_id] = np.mean(embeddings_list, axis=0)
    
    return ab_mean_embeddings


def compute_similarity_with_antibiotic(embeddings_dict: dict, all_gene_mapping: dict, all_drug_mapping: dict, prefix: str) -> pd.DataFrame:
    """Compute similarity matrix between embeddings and antibiotics"""
    
    # Get valid wells
    valid_wells = []
    for well_id in embeddings_dict.keys():
        row, col = well_to_row_col(well_id)
        for plate in PLATES:
            if plate in all_gene_mapping and row in all_gene_mapping[plate] and col in all_gene_mapping[plate][row]:
                valid_wells.append(well_id)
                break
    
    # Get gene IDs
    gene_counts = defaultdict(int)
    unique_gene_ids = []
    for well_id in sorted(valid_wells):
        row, col = well_to_row_col(well_id)
        gene_id = None
        for plate in PLATES:
            if plate in all_gene_mapping and row in all_gene_mapping[plate] and col in all_gene_mapping[plate][row]:
                gene_id = all_gene_mapping[plate][row][col]['id']
                break
        if gene_id is None:
            gene_id = 'Unknown'
        if gene_counts[gene_id] > 0:
            gene_id = f"{gene_id}_W{well_id.replace('Well', '')}"
        gene_counts[gene_id] += 1
        unique_gene_ids.append(gene_id)
    
    # Compute antibiotic embeddings
    ab_embeddings = compute_antibiotic_embeddings(embeddings_dict, all_drug_mapping)
    ab_ids = sorted(ab_embeddings.keys())
    
    print(f"Computing {prefix} vs Antibiotic matrix ({len(unique_gene_ids)} x {len(ab_ids)})...")
    
    similarity_matrix = np.zeros((len(valid_wells), len(ab_ids)))
    
    for i, well_id in enumerate(sorted(valid_wells)):
        for j, ab_id in enumerate(ab_ids):
            sim = cosine_similarity(embeddings_dict[well_id], ab_embeddings[ab_id])
            similarity_matrix[i, j] = sim
    
    df = pd.DataFrame(similarity_matrix, index=unique_gene_ids, columns=ab_ids)
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compute cosine similarity between embeddings (all plates)')
    parser.add_argument('--type', type=str, default='all',
                        choices=['all', 'mutant_drug', 'mutant_ab', 'drug_ab'],
                        help='Which similarity to compute: all, mutant_drug, mutant_ab, drug_ab')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for similarity matrices')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Cosine Similarity Computation (All Plates P1-P6)")
    print("="*60)
    print(f"\nPlates: {PLATES}")
    
    output_dir = args.output_dir if args.output_dir else BASE_DIR
    
    # Load embeddings from ALL plates (averaged per well)
    print("\nLoading mutant embeddings from all plates...")
    mutant_embeddings = load_embeddings_by_well_all_plates('mutant')
    print(f"  Loaded {len(mutant_embeddings)} mutant wells (averaged across 6 plates)")
    
    print("\nLoading drug embeddings from all plates...")
    drug_embeddings = load_embeddings_by_well_all_plates('drug')
    print(f"  Loaded {len(drug_embeddings)} drug wells (averaged across 6 plates)")
    
    # Load mappings for all plates
    print("\nLoading mappings for all plates...")
    all_gene_mapping = load_gene_mapping()
    all_drug_mapping = load_drug_mapping()
    print(f"  Gene mapping: {len(all_gene_mapping)} plates")
    print(f"  Drug mapping: {len(all_drug_mapping)} plates")
    
    results = {}
    
    # Mutant vs Drug
    if args.type in ['all', 'mutant_drug']:
        print("\n" + "="*40)
        print("Computing Mutant vs Drug similarity")
        print("="*40)
        mutant_drug_df = compute_mutant_drug_similarity(mutant_embeddings, drug_embeddings, all_gene_mapping)
        mutant_drug_path = os.path.join(output_dir, "similarity_mutant_vs_drug.csv")
        mutant_drug_df.to_csv(mutant_drug_path)
        print(f"Saved to: {mutant_drug_path}")
        print(f"Shape: {mutant_drug_df.shape}")
        results['mutant_drug'] = mutant_drug_df
    
    # Mutant vs Antibiotic
    if args.type in ['all', 'mutant_ab']:
        print("\n" + "="*40)
        print("Computing Mutant vs Antibiotic similarity")
        print("="*40)
        mutant_ab_df = compute_similarity_with_antibiotic(mutant_embeddings, all_gene_mapping, all_drug_mapping, "Mutant")
        mutant_ab_path = os.path.join(output_dir, "similarity_mutant_vs_antibiotic.csv")
        mutant_ab_df.to_csv(mutant_ab_path)
        print(f"Saved to: {mutant_ab_path}")
        print(f"Shape: {mutant_ab_df.shape}")
        results['mutant_ab'] = mutant_ab_df
    
    # Drug vs Antibiotic
    if args.type in ['all', 'drug_ab']:
        print("\n" + "="*40)
        print("Computing Drug vs Antibiotic similarity")
        print("="*40)
        drug_ab_df = compute_similarity_with_antibiotic(drug_embeddings, all_gene_mapping, all_drug_mapping, "Drug")
        drug_ab_path = os.path.join(output_dir, "similarity_drug_vs_antibiotic.csv")
        drug_ab_df.to_csv(drug_ab_path)
        print(f"Saved to: {drug_ab_path}")
        print(f"Shape: {drug_ab_df.shape}")
        results['drug_ab'] = drug_ab_df
    
    print(f"\n{'='*60}")
    print("All similarity matrices computed and saved!")
    print(f"{'='*60}")
    
    # Print statistics
    print("\nStatistics:")
    for name, df in results.items():
        print(f"\n{name}:")
        print(f"  Min: {df.values.min():.4f}, Max: {df.values.max():.4f}, Mean: {df.values.mean():.4f}")


if __name__ == '__main__':
    main()