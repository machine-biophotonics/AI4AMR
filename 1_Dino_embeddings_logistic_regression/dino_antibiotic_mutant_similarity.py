#!/usr/bin/env python3
"""
Generate antibiotic-mutant similarity heatmap from DINOv3 embeddings.
Shows which antibiotics cluster closest to which mutant genes.
Uses embeddings from all 6 plates (P1-P6).
"""

import os
import argparse
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, leaves_list
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(SCRIPT_DIR, "embeddings")
PLATES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']


def well_to_wellname(well_id: str) -> str:
    """Convert WellA01 to A01"""
    return well_id.replace("Well", "")


def get_gene_from_id(gene_id: str) -> str:
    """Return the full gene ID with replicate number (e.g., 'lptA_1' stays 'lptA_1')"""
    return gene_id


def extract_replicate_number(gene_id: str) -> int:
    """Extract numeric replicate from gene ID like 'lptA_3' -> 3"""
    if '_' in gene_id:
        parts = gene_id.rsplit('_', 1)
        try:
            return int(parts[1])
        except ValueError:
            return 0
    return 0


def extract_concentration_number(ab_id: str) -> float:
    """Extract concentration value for sorting (e.g., 'Avibactam_0.25x' -> 0.25)"""
    if '_' in ab_id:
        parts = ab_id.rsplit('_', 1)
        conc_str = parts[1].replace('x', '')
        try:
            return float(conc_str)
        except ValueError:
            return 0.0
    return 0.0


def load_embeddings_by_well_all_plates(data_type: str) -> dict:
    """Load embeddings from all 6 plates with plate-specific keys."""
    well_embeddings = defaultdict(list)
    
    for plate in PLATES:
        embeddings_dir = os.path.join(EMBEDDINGS_DIR, f"{data_type.capitalize()}s_{plate}")
        if not os.path.exists(embeddings_dir):
            print(f"  WARNING: {embeddings_dir} not found, skipping {plate}...")
            continue
        
        well_folders = glob.glob(os.path.join(embeddings_dir, "Well*"))
        for well_folder in well_folders:
            well_id = os.path.basename(well_folder)  # e.g., "WellA01"
            npy_files = glob.glob(os.path.join(well_folder, "*.npy"))
            for npy_file in npy_files:
                embedding = np.load(npy_file)
                # Create plate-specific key like "P1_A01"
                plate_well = f"{plate}_{well_id.replace('Well', '')}"
                well_embeddings[plate_well].append(embedding)
    
    well_mean_embeddings = {}
    for well_id, embeddings_list in well_embeddings.items():
        if len(embeddings_list) > 0:
            well_mean_embeddings[well_id] = np.mean(embeddings_list, axis=0)
    
    return well_mean_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='dino_antibiotic_mutant_heatmap.png', help='Output PNG')
    parser.add_argument('--similarity', type=str, default='cosine', choices=['cosine', 'euclidean'],
                        help='Similarity metric')
    parser.add_argument('--cluster', action='store_true', default=False,
                        help='Use hierarchical clustering for ordering')
    parser.add_argument('--top_n', type=int, default=None,
                        help='Show only top N antibiotics by number of embeddings')
    parser.add_argument('--only_antibiotics', type=str, default=None,
                        help='Comma-separated list of antibiotics to include')
    
    args = parser.parse_args()
    
    print(f"\n=== DINOv3 Antibiotic-Mutant Similarity Heatmap ===")
    print(f"Loading embeddings from all plates: {PLATES}")
    
    # Load mappings
    ic50_path = os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')
    mutant_path = os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')
    
    with open(ic50_path, 'r') as f:
        ic50_data = json.load(f)
    
    with open(mutant_path, 'r') as f:
        mutant_data = json.load(f)
    
    # Load embeddings
    print("\nLoading mutant embeddings from all plates...")
    mutant_embeddings = load_embeddings_by_well_all_plates('mutant')
    print(f"  Loaded {len(mutant_embeddings)} mutant wells")
    
    print("\nLoading drug embeddings from all plates...")
    drug_embeddings = load_embeddings_by_well_all_plates('drug')
    print(f"  Loaded {len(drug_embeddings)} drug wells")
    
    # Create well -> antibiotic mapping
    well_to_antibiotic = {}
    for plate, wells in ic50_data.items():
        for well, info in wells.items():
            ab = info.get('antibiotic', '')
            ic50 = info.get('ic50_multiple', '')
            if ab and ic50:
                ab_id = f"{ab}_{ic50}"
                well_to_antibiotic[f"{plate}_{well}"] = ab_id
    
    # Create well -> gene mapping
    well_to_gene = {}
    for plate, rows in mutant_data.items():
        for row, cols in rows.items():
            for col, info in cols.items():
                gene_id = info.get('id', '')
                if gene_id:
                    gene = get_gene_from_id(gene_id)
                    col_padded = col.zfill(2)  # Add leading zero: 1 -> 01, 10 -> 10
                    well_to_gene[f"{plate}_{row}{col_padded}"] = gene
    
    # Group embeddings by antibiotic (drugs) and gene (mutants)
    antibiotic_embeddings = defaultdict(list)
    gene_embeddings = defaultdict(list)
    
    # Process mutant embeddings - each plate's well maps to its own gene
    for well_id, emb in mutant_embeddings.items():
        # well_id is like "P1_A01", check if it exists directly in well_to_gene
        if well_id in well_to_gene:
            gene = well_to_gene[well_id]
            gene_embeddings[gene].append(emb)
    
    # Process drug embeddings - each plate's well maps to its own antibiotic
    for well_id, emb in drug_embeddings.items():
        # well_id is like "P1_A01", check if it exists directly in well_to_antibiotic
        if well_id in well_to_antibiotic:
            ab_id = well_to_antibiotic[well_id]
            antibiotic_embeddings[ab_id].append(emb)
    
    print(f"\nAntibiotics found: {len(antibiotic_embeddings)}")
    print(f"Genes found: {len(gene_embeddings)}")
    
    # Filter by --only_antibiotics if specified
    if args.only_antibiotics:
        selected_abs = set(args.only_antibiotics.split(','))
        antibiotic_embeddings = {k: v for k, v in antibiotic_embeddings.items() if k in selected_abs}
        print(f"  Filtered to {len(antibiotic_embeddings)} antibiotics")
    
    # Filter by --top_n if specified
    if args.top_n:
        top_abs = sorted(antibiotic_embeddings.keys(), 
                         key=lambda x: len(antibiotic_embeddings[x]), 
                         reverse=True)[:args.top_n]
        antibiotic_embeddings = {k: antibiotic_embeddings[k] for k in top_abs}
        print(f"  Top {args.top_n} antibiotics by count")
    
    # Compute centroids
    antibiotic_centroids = {}
    for ab, embs in antibiotic_embeddings.items():
        antibiotic_centroids[ab] = np.mean(embs, axis=0)
    
    gene_centroids = {}
    for gene, embs in gene_embeddings.items():
        gene_centroids[gene] = np.mean(embs, axis=0)
    
    # Get sorted lists (alphabetically by name, then by concentration/replicate number)
    def gene_sort_key(gene_id):
        base = gene_id.rsplit('_', 1)[0] if '_' in gene_id else gene_id
        rep = extract_replicate_number(gene_id)
        return (base, rep)
    
    def ab_sort_key(ab_id):
        name = ab_id.rsplit('_', 1)[0] if '_' in ab_id else ab_id
        conc = extract_concentration_number(ab_id)
        return (name, conc)
    
    antibiotics = sorted(antibiotic_centroids.keys(), key=ab_sort_key)
    genes = sorted(gene_centroids.keys(), key=gene_sort_key)
    
    print(f"\nAntibiotics: {len(antibiotics)}")
    print(f"Genes: {len(genes)}")
    
    # Build similarity matrix
    print(f"\nComputing similarity matrix ({len(antibiotics)} x {len(genes)})...")
    n_ab = len(antibiotics)
    n_gene = len(genes)
    similarity_matrix = np.zeros((n_ab, n_gene))
    
    for i, ab in enumerate(tqdm(antibiotics, desc="Computing similarities")):
        for j, gene in enumerate(genes):
            ab_emb = antibiotic_centroids[ab].reshape(1, -1)
            gene_emb = gene_centroids[gene].reshape(1, -1)
            
            if args.similarity == 'cosine':
                sim = cosine_similarity(ab_emb, gene_emb)[0, 0]
            else:
                dist = np.linalg.norm(ab_emb - gene_emb)
                sim = 1.0 / (1.0 + dist)
            
            similarity_matrix[i, j] = sim
    
    print(f"Matrix computed. Range: {similarity_matrix.min():.4f} to {similarity_matrix.max():.4f}")
    
    # Clustering for ordering
    if args.cluster:
        print("\nHierarchical clustering for ordering...")
        ab_linkage = linkage(similarity_matrix, method='average')
        gene_linkage = linkage(similarity_matrix.T, method='average')
        
        ab_order = leaves_list(ab_linkage)
        gene_order = leaves_list(gene_linkage)
        
        antibiotics_ordered = [antibiotics[i] for i in ab_order]
        genes_ordered = [genes[j] for j in gene_order]
        similarity_ordered = similarity_matrix[ab_order][:, gene_order]
    else:
        antibiotics_ordered = antibiotics
        genes_ordered = genes
        similarity_ordered = similarity_matrix
    
    # Create heatmap
    print("\nCreating heatmap...")
    
    fig, ax = plt.subplots(figsize=(max(24, len(genes_ordered) * 0.3), max(10, len(antibiotics_ordered) * 0.3)))
    
    sns.heatmap(
        similarity_ordered,
        xticklabels=genes_ordered,
        yticklabels=antibiotics_ordered,
        cmap='RdYlBu_r',
        annot=False,
        fmt='.2f',
        cbar_kws={'label': 'Cosine Similarity', 'shrink': 0.5},
        ax=ax,
        vmin=similarity_ordered.min(),
        vmax=similarity_ordered.max(),
        linewidths=0.1,
        linecolor='white'
    )
    
    ax.set_xlabel('Mutant Gene', fontsize=12)
    ax.set_ylabel('Antibiotic', fontsize=12)
    ax.set_title(f'DINOv3 Embeddings: Antibiotic-Mutant Similarity\n({len(antibiotics)} antibiotics x {len(genes)} genes, All Plates P1-P6)', 
                 fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=9)
    
    plt.tight_layout()
    
    output_path = os.path.join(SCRIPT_DIR, args.output)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved to: {output_path}")
    
    # Save similarity matrix as CSV
    csv_path = output_path.replace('.png', '_matrix.csv')
    sim_df = pd.DataFrame(similarity_matrix, index=antibiotics, columns=genes)
    sim_df.to_csv(csv_path)
    print(f"Saved matrix to: {csv_path}")
    
    # Print top matches
    print("\n=== Top Antibiotic-Gene Similarities ===")
    flat_sim = []
    for i, ab in enumerate(antibiotics):
        for j, gene in enumerate(genes):
            flat_sim.append((ab, gene, similarity_matrix[i, j]))
    
    flat_sim.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop 20 most similar pairs:")
    for ab, gene, sim in flat_sim[:20]:
        print(f"  {ab:30} <-> {gene:10}: {sim:.4f}")
    
    print("\nTop 20 least similar pairs:")
    for ab, gene, sim in flat_sim[-20:]:
        print(f"  {ab:30} <-> {gene:10}: {sim:.4f}")
    
    # Group antibiotics by base name (ignore concentration)
    print("\n=== Per-Antibiotic Summary (averaged across concentrations) ===")
    ab_base_similarity = defaultdict(list)
    for ab, gene, sim in flat_sim:
        base = ab.rsplit('_', 1)[0]
        ab_base_similarity[base].append((gene, sim))
    
    base_summaries = []
    for base, pairs in sorted(ab_base_similarity.items()):
        avg_sim = np.mean([p[1] for p in pairs])
        max_gene = max(pairs, key=lambda x: x[1])
        base_summaries.append((base, avg_sim, max_gene[0], max_gene[1]))
    
    base_summaries.sort(key=lambda x: x[1], reverse=True)
    
    print("\nAntibiotics ranked by average similarity to all genes:")
    for base, avg_sim, max_gene, max_sim in base_summaries:
        print(f"  {base:25}: avg={avg_sim:.4f}, best match={max_gene} ({max_sim:.4f})")


if __name__ == '__main__':
    main()