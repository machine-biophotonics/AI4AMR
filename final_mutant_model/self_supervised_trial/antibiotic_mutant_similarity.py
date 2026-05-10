#!/usr/bin/env python3
"""
Generate antibiotic-mutant similarity heatmap from self-supervised embeddings.
Shows which antibiotics cluster closest to which mutant genes.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_antibiotic_from_class(class_name: str) -> str:
    """Extract antibiotic name from class like 'Ciprofloxacin_1x' -> 'Ciprofloxacin'"""
    if class_name == 'control' or 'DMSO' in class_name:
        return 'DMSO'
    if '_' in class_name:
        return class_name.rsplit('_', 1)[0]
    return class_name


def get_gene_from_id(mutant_id: str) -> str:
    """Extract base gene name from mutant ID like 'lptA_3' -> 'lptA'"""
    if '_' in mutant_id:
        return mutant_id.rsplit('_', 1)[0]
    return mutant_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='embeddings_P1.csv', help='Input CSV')
    parser.add_argument('--output', type=str, default='antibiotic_mutant_heatmap.png', help='Output PNG')
    parser.add_argument('--similarity', type=str, default='cosine', choices=['cosine', 'euclidean'],
                        help='Similarity metric')
    
    args = parser.parse_args()
    
    input_path = os.path.join(SCRIPT_DIR, 'self_supervised_trial', args.input)
    output_dir = os.path.join(SCRIPT_DIR, 'self_supervised_trial')
    output_path = os.path.join(output_dir, args.output)
    
    print(f"Loading embeddings from: {input_path}")
    df = pd.read_csv(input_path)
    
    # Get embedding columns
    embed_cols = [c for c in df.columns if c.startswith('emb_')]
    embeddings = df[embed_cols].values
    image_names = df['image_name'].values
    data_types = df['data_type'].values
    
    # Load ground truth mappings to get proper class names
    ic50_path = os.path.join(SCRIPT_DIR, '..', 'plate_well_ic50_mapping.json')
    mutant_path = os.path.join(SCRIPT_DIR, '..', 'plate_well_id_path.json')
    
    # Load drug mapping
    with open(ic50_path, 'r') as f:
        ic50_data = json.load(f)
    
    # Load mutant mapping
    with open(mutant_path, 'r') as f:
        mutant_data = json.load(f)
    
    # Create well -> antibiotic mapping
    well_to_antibiotic = {}
    for plate, wells in ic50_data.items():
        for well, info in wells.items():
            ab = info.get('antibiotic', '')
            if ab:
                well_to_antibiotic[f"{plate}_{well}"] = ab
    
    # Create well -> gene mapping
    well_to_gene = {}
    for plate, rows in mutant_data.items():
        for row, cols in rows.items():
            for col, info in cols.items():
                gene_id = info.get('id', '')
                if gene_id:
                    gene = get_gene_from_id(gene_id)
                    well_to_gene[f"{plate}_{row}{col}"] = gene
    
    print(f"Total images: {len(df)}")
    
    # Group embeddings by antibiotic (for drugs) and gene (for mutants)
    antibiotic_embeddings = {}
    gene_embeddings = {}
    
    for i, row in df.iterrows():
        img_name = row['image_name']
        data_type = row['data_type']
        emb = embeddings[i]
        
        # Parse well from image name
        well = None
        for part in img_name.split('_'):
            if part.startswith('Well'):
                well = part.replace('Well', '')
                break
        
        # Determine plate from path
        img_path = row['image_path']
        plate = 'P1'  # Default - we'll extract properly
        if 'Drugs_Data' in img_path:
            plate = img_path.split('Drugs_Data/')[1].split('/')[0]
        elif 'Mutants_Data' in img_path:
            plate = img_path.split('Mutants_Data/')[1].split('/')[0]
        
        key = f"{plate}_{well}" if well else None
        
        if data_type == 'drug' and key and key in well_to_antibiotic:
            ab = well_to_antibiotic[key]
            if ab not in antibiotic_embeddings:
                antibiotic_embeddings[ab] = []
            antibiotic_embeddings[ab].append(emb)
        
        elif data_type == 'mutant' and key and key in well_to_gene:
            gene = well_to_gene[key]
            if gene not in gene_embeddings:
                gene_embeddings[gene] = []
            gene_embeddings[gene].append(emb)
    
    print(f"Antibiotics found: {len(antibiotic_embeddings)}")
    print(f"Genes found: {len(gene_embeddings)}")
    
    # Compute centroids
    antibiotic_centroids = {}
    for ab, embs in antibiotic_embeddings.items():
        antibiotic_centroids[ab] = np.mean(embs, axis=0)
    
    gene_centroids = {}
    for gene, embs in gene_embeddings.items():
        gene_centroids[gene] = np.mean(embs, axis=0)
    
    # Get sorted lists
    antibiotics = sorted(antibiotic_centroids.keys())
    genes = sorted(gene_centroids.keys())
    
    print(f"\nAntibiotics: {antibiotics}")
    print(f"Genes: {len(genes)} (first 10: {genes[:10]})")
    
    # Build similarity matrix
    n_ab = len(antibiotics)
    n_gene = len(genes)
    similarity_matrix = np.zeros((n_ab, n_gene))
    
    for i, ab in enumerate(antibiotics):
        for j, gene in enumerate(genes):
            ab_emb = antibiotic_centroids[ab].reshape(1, -1)
            gene_emb = gene_centroids[gene].reshape(1, -1)
            
            if args.similarity == 'cosine':
                sim = cosine_similarity(ab_emb, gene_emb)[0, 0]
            else:
                # Euclidean distance - convert to similarity
                dist = np.linalg.norm(ab_emb - gene_emb)
                sim = 1.0 / (1.0 + dist)
            
            similarity_matrix[i, j] = sim
    
    # Create heatmap
    print(f"\nCreating heatmap ({n_ab} antibiotics x {n_gene} genes)...")
    
    # Cluster for better visualization
    ab_linkage = linkage(similarity_matrix, method='average')
    gene_linkage = linkage(similarity_matrix.T, method='average')
    
    # Reorder based on clustering
    from scipy.cluster.hierarchy import leaves_list
    ab_order = leaves_list(ab_linkage)
    gene_order = leaves_list(gene_linkage)
    
    antibiotics_ordered = [antibiotics[i] for i in ab_order]
    genes_ordered = [genes[j] for j in gene_order]
    
    similarity_ordered = similarity_matrix[ab_order][:, gene_order]
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(24, 10))
    
    sns.heatmap(
        similarity_ordered,
        xticklabels=genes_ordered,
        yticklabels=antibiotics_ordered,
        cmap='RdYlBu_r',
        annot=False,
        fmt='.2f',
        cbar_kws={'label': 'Similarity'},
        ax=ax,
        vmin=similarity_ordered.min(),
        vmax=similarity_ordered.max()
    )
    
    ax.set_xlabel('Mutant Gene', fontsize=12)
    ax.set_ylabel('Antibiotic', fontsize=12)
    ax.set_title(f'Self-Supervised Embeddings: Antibiotic-Mutant Similarity\n({args.similarity.capitalize()} Similarity, Hierarchically Clustered)', fontsize=14)
    
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
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
        print(f"  {ab:25} <-> {gene:10}: {sim:.4f}")
    
    print("\nTop 20 most dissimilar pairs:")
    for ab, gene, sim in flat_sim[-20:]:
        print(f"  {ab:25} <-> {gene:10}: {sim:.4f}")


if __name__ == '__main__':
    main()