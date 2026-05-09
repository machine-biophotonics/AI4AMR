#!/usr/bin/env python3
"""
Generate a single combined similarity matrix figure with:
- 96 mutant genes + 89 antibiotics = 185 entities on both axes
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def well_to_row_col(well_id: str):
    match = well_id.replace("Well", "")
    return match[0], int(match[1:])

def well_to_wellname(well_id: str):
    return well_id.replace("Well", "")

def load_embeddings_by_well(embeddings_dir: str):
    import glob
    from collections import defaultdict
    well_embeddings = defaultdict(list)
    
    well_folders = glob.glob(os.path.join(embeddings_dir, "Well*"))
    for well_folder in well_folders:
        well_id = os.path.basename(well_folder)
        npy_files = glob.glob(os.path.join(well_folder, "*.npy"))
        for npy_file in npy_files:
            embedding = np.load(npy_file)
            well_embeddings[well_id].append(embedding)
    
    well_mean_embeddings = {}
    for well_id, embeddings_list in well_embeddings.items():
        if len(embeddings_list) > 0:
            well_mean_embeddings[well_id] = np.mean(embeddings_list, axis=0)
    
    return well_mean_embeddings

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    print("\n=== Building Combined Similarity Matrix (185 x 185) ===")
    
    # Load embeddings
    print("Loading embeddings...")
    mutant_embeddings = load_embeddings_by_well(os.path.join(BASE_DIR, "embeddings", "Mutants_P1"))
    drug_embeddings = load_embeddings_by_well(os.path.join(BASE_DIR, "embeddings", "Drugs_P1"))
    
    # Load mappings
    with open(os.path.join(BASE_DIR, "plate_well_id_path.json")) as f:
        gene_mapping = json.load(f)['P1']
    
    with open(os.path.join(BASE_DIR, "plate_well_ic50_mapping.json")) as f:
        drug_mapping = json.load(f)['P1']
    
    # Get mutant gene IDs - sort by guide number (1,2,3) then by gene name
    mutant_ids = []
    mutant_wells_sorted = sorted(mutant_embeddings.keys())
    print(f"Processing {len(mutant_wells_sorted)} wells...")
    
    for i, well_id in enumerate(mutant_wells_sorted):
        row, col = well_to_row_col(well_id)
        if row in gene_mapping and str(col) in gene_mapping[row]:
            gene_id = gene_mapping[row][str(col)]['id']
            mutant_ids.append(gene_id)
        else:
            print(f"  Warning: {well_id} not found in mapping (row={row}, col={col})")
    
    # Sort by gene name only (alphabetically) - keep same gene together
    sorted_gene_ids = sorted(mutant_ids, key=lambda x: x.rsplit('_', 1)[0])
    
    # Handle duplicate gene IDs after sorting (add well position)
    gene_counts = {}
    unique_mutant_ids = []
    for gene_id in sorted_gene_ids:
        if gene_id in gene_counts:
            gene_counts[gene_id] += 1
            gene_id = f"{gene_id}_W{gene_counts[gene_id]}"
        else:
            gene_counts[gene_id] = 1
        unique_mutant_ids.append(gene_id)
    
    # Get antibiotic IDs
    ab_ids = []
    antibiotic_set = set()
    for wellname, info in drug_mapping.items():
        ab = info['antibiotic']
        ic50 = info['ic50_multiple']
        ab_id = f"{ab}_{ic50}"
        antibiotic_set.add(ab_id)
    
    antibiotic_list = sorted(list(antibiotic_set))
    print(f"  Mutant genes: {len(unique_mutant_ids)}")
    print(f"  Antibiotics: {len(antibiotic_list)}")
    
    # Get antibiotic embeddings from drug embeddings
    ab_embeddings = {}
    for well_id, embedding in drug_embeddings.items():
        wellname = well_to_wellname(well_id)
        if wellname in drug_mapping:
            ab = drug_mapping[wellname]['antibiotic']
            ic50 = drug_mapping[wellname]['ic50_multiple']
            ab_id = f"{ab}_{ic50}"
            if ab_id not in ab_embeddings:
                ab_embeddings[ab_id] = []
            ab_embeddings[ab_id].append(embedding)
    
    # Mean embedding per antibiotic
    for ab_id in ab_embeddings:
        ab_embeddings[ab_id] = np.mean(ab_embeddings[ab_id], axis=0)
    
    print(f"  Antibiotic embeddings: {len(ab_embeddings)}")
    
    # Build combined embedding dict for all 185 entities
    all_embeddings = {}
    
    # Add mutant embeddings
    for i, well_id in enumerate(mutant_wells_sorted):
        all_embeddings[unique_mutant_ids[i]] = mutant_embeddings[well_id]
    
    # Add antibiotic embeddings
    for ab_id in antibiotic_list:
        if ab_id in ab_embeddings:
            all_embeddings[ab_id] = ab_embeddings[ab_id]
    
    # Build 185x185 similarity matrix
    print("\nComputing 185x185 similarity matrix...")
    all_entity_ids = unique_mutant_ids + antibiotic_list
    n = len(all_entity_ids)
    
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i, j] = 1.0
            elif i < j:
                sim = cosine_similarity(all_embeddings[all_entity_ids[i]], 
                                       all_embeddings[all_entity_ids[j]])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
    
    # Create DataFrame
    df = pd.DataFrame(similarity_matrix, index=all_entity_ids, columns=all_entity_ids)
    
    print(f"  Matrix shape: {df.shape}")
    
    # Save CSV
    csv_path = os.path.join(BASE_DIR, "combined_similarity_matrix_185.csv")
    df.to_csv(csv_path)
    print(f"  Saved CSV: {csv_path}")
    
    # Plot single figure - research standard (Viridis, no centering)
    print("\nGenerating combined heatmap (research standard)...")
    fig, ax = plt.subplots(figsize=(40, 36))
    
    # Viridis sequential colormap - standard for similarity/correlation matrices
    # vmin/vmax fixed for consistent coloring across all panels
    sns.heatmap(df, 
                cmap='viridis',
                vmin=0.5, vmax=1.0,
                xticklabels=True, 
                yticklabels=True,
                cbar_kws={'label': 'Cosine Similarity', 'shrink': 0.5, 'pad': 0.02},
                ax=ax,
                linewidths=0,
                linecolor='white')
    
    ax.set_title('Combined Similarity Matrix: 96 Mutant Genes + 89 Antibiotics (185 × 185)', 
                 fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Entities', fontsize=14)
    ax.set_ylabel('Entities', fontsize=14)
    
    # Adjust tick labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=4, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=4)
    
    # Add boundary box to separate quadrants (96 genes vs 89 antibiotics)
    from matplotlib.patches import Rectangle
    
    # Main quadrant separator
    ax.axhline(y=96, color='white', linewidth=3)
    ax.axvline(x=96, color='white', linewidth=3)
    
    # Group same gene names (by position in sorted list)
    gene_counts = {}
    gene_positions = []
    for gene in unique_mutant_ids:
        base = gene.rsplit('_', 1)[0]
        if base not in gene_counts:
            gene_counts[base] = 0
            gene_positions.append([])
        gene_counts[base] += 1
        gene_positions[-1].append(gene_counts[base] - 1)
    
# Add small boxes for gene groups with >1 class
    gene_positions = {}
    for i, gene in enumerate(unique_mutant_ids):
        base = gene.rsplit('_', 1)[0]
        if base not in gene_positions:
            gene_positions[base] = []
        gene_positions[base].append(i)
    
    # Draw box around each group of same gene
    for base, positions in gene_positions.items():
        if len(positions) > 1:
            for pos in positions:
                rect = Rectangle((pos, pos), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
            # Also draw encompassing box
            min_pos = min(positions)
            max_pos = max(positions)
            if max_pos > min_pos:
                rect = Rectangle((min_pos, min_pos), max_pos - min_pos + 1, max_pos - min_pos + 1,
                                linewidth=2.5, edgecolor='red', facecolor='none', linestyle='-')
                ax.add_patch(rect)
    
    # Group same antibiotic concentrations
    # Parse antibiotic names to group by base antibiotic
    ab_base_positions = {}
    for i, ab in enumerate(antibiotic_list):
        base = ab.rsplit('_', 1)[0]  # e.g., Chloramphenicol from Chloramphenicol_1x
        if base not in ab_base_positions:
            ab_base_positions[base] = []
        ab_base_positions[base].append(i)
    
    # Add small boxes for antibiotic groups (offset by 96)
    for base, positions in ab_base_positions.items():
        if len(positions) > 1:
            start = positions[0] + 96
            end = positions[-1] + 96
            rect = Rectangle((start, start), end - start + 1, end - start + 1,
                            linewidth=2.5, edgecolor='red', facecolor='none', linestyle='-')
            ax.add_patch(rect)
    
    plt.tight_layout()
    
    png_path = os.path.join(BASE_DIR, "combined_similarity_matrix_185.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # === Create diverging version showing difference from mean ===
    print("\nGenerating diverging heatmap...")
    
    # Calculate within-group and cross-group similarities
    mutant_mutant_sim = df.iloc[:96, :96].values.mean()
    drug_drug_sim = df.iloc[96:, 96:].values.mean()
    mutant_drug_sim = df.iloc[:96, 96:].values.mean()
    
    # Calculate difference from overall mean
    overall_mean = df.values.mean()
    diff_from_mean = df.values - overall_mean
    
    diff_df = pd.DataFrame(diff_from_mean, index=df.index, columns=df.columns)
    
    # Create diverging heatmap
    fig, ax = plt.subplots(figsize=(42, 38))
    
    # Use diverging colormap centered at 0
    vmax = max(abs(diff_from_mean.min()), abs(diff_from_mean.max()))
    sns.heatmap(diff_df, cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
                xticklabels=True, yticklabels=True,
                cbar_kws={'label': 'Deviation from Mean Similarity', 'shrink': 0.5},
                ax=ax, linewidths=0, linecolor='white')
    
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=4, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=4)
    
    # Add quadrant separators
    ax.axhline(y=96, color='black', linewidth=3)
    ax.axvline(x=96, color='black', linewidth=3)
    
    # Title with all statistics
    title = (f'Combined Similarity Matrix: Deviation from Mean\n'
             f'Mean: {overall_mean*100:.1f}% | Mutant-Mutant: {mutant_mutant_sim*100:.1f}% | '
             f'Drug-Drug: {drug_drug_sim*100:.1f}% | Mutant-Drug: {mutant_drug_sim*100:.1f}%')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Entities', fontsize=12)
    ax.set_ylabel('Entities', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "combined_similarity_matrix_185_deviation.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: combined_similarity_matrix_185_deviation.png")
    
    # === Create top bar showing drug-mutant similarity ===
    print("\nGenerating top similarity bar...")
    
    # Calculate mean similarity of each antibiotic to all mutants
    mutant_to_ab_sim = df.iloc[:96, 96:].mean(axis=0)  # 89 antibiotics
    ab_percent = (mutant_to_ab_sim * 100).round(1)
    
    # Calculate mean similarity of each mutant to all antibiotics  
    ab_to_mutant_sim = df.iloc[96:, :96].mean(axis=1)  # 89 antibiotics (symmetric)
    mutant_percent = (ab_to_mutant_sim * 100).round(1)
    
    # Calculate overall mean similarity between mutants and drugs
    cross_sim = df.iloc[:96, 96:].values.mean()
    mutant_mutant_sim = df.iloc[:96, :96].values.mean()
    drug_drug_sim = df.iloc[96:, 96:].values.mean()
    mean_percentage = round(cross_sim * 100, 1)
    
    # Add stats to title
    title = (f'Combined Similarity Matrix: 96 Mutant Genes + 89 Antibiotics (185 × 185)\n'
             f'Mutant-Drug: {mean_percentage}% | Mutant-Mutant: {mutant_mutant_sim*100:.1f}% | '
             f'Drug-Drug: {drug_drug_sim*100:.1f}%')
    
    # Create figure with heatmap + annotation
    fig, ax = plt.subplots(figsize=(42, 38))
    
    sns.heatmap(df, cmap='viridis', vmin=0.5, vmax=1.0,
                xticklabels=True, yticklabels=True,
                cbar_kws={'label': 'Cosine Similarity', 'shrink': 0.5},
                ax=ax, linewidths=0, linecolor='white')
    
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=4, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=4)
    
    # Add percentage labels on top for antibiotics
    for i, pct in enumerate(ab_percent.values):
        ax.text(i + 0.5, -2, f'{pct}%', ha='center', va='bottom', fontsize=3, rotation=90, color='black')
    
    # Add percentage labels on right for mutants (rotated)
    for i, pct in enumerate(mutant_percent.values):
        ax.text(-2, i + 0.5, f'{pct}%', ha='right', va='center', fontsize=3, color='black')
    
    # Add mean percentage annotation (now in title)
    ax.text(48, -10, f'Cross-Group Similarity: {mean_percentage}%', 
            ha='center', fontsize=12, fontweight='bold')
    
    # Add quadrant separators and gene/antibiotic boxes to heatmap
    ax.axhline(y=96, color='white', linewidth=3)
    ax.axvline(x=96, color='white', linewidth=3)
    
    # Gene boxes - now grouped by guide number, draw box around each gene group
    gene_positions = {}
    for i, gene in enumerate(unique_mutant_ids):
        # Get base gene name without guide number
        base = gene.rsplit('_', 1)[0]
        if base not in gene_positions:
            gene_positions[base] = []
        gene_positions[base].append(i)
    
    for base, positions in gene_positions.items():
        if len(positions) > 1:
            min_pos = min(positions)
            max_pos = max(positions)
            # Draw a large box around entire gene group (consecutive after sorting)
            rect = Rectangle((min_pos - 0.5, min_pos - 0.5), 
                            max_pos - min_pos + 1, max_pos - min_pos + 1,
                            linewidth=3, edgecolor='red', facecolor='none', linestyle='-')
            ax.add_patch(rect)
    
    # Antibiotic boxes
    for base, positions in ab_base_positions.items():
        if len(positions) > 1:
            start = positions[0] + 96
            end = positions[-1] + 96
            rect = Rectangle((start, start), end - start + 1, end - start + 1,
                            linewidth=2.5, edgecolor='red', facecolor='none', linestyle='-')
            ax.add_patch(rect)
    
    # Add labels
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Entities', fontsize=12)
    ax.set_ylabel('Entities', fontsize=12)
    
    plt.savefig(os.path.join(BASE_DIR, "combined_similarity_matrix_185_with_bars.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: combined_similarity_matrix_185_with_bars.png")
    
    print(f"\n=== Saved: {png_path} ===")
    print(f"Matrix dimensions: {n} x {n}")
    print(f"  - Rows: {len(unique_mutant_ids)} mutant genes + {len(antibiotic_list)} antibiotics")
    print(f"  - Columns: {len(unique_mutant_ids)} mutant genes + {len(antibiotic_list)} antibiotics")

if __name__ == '__main__':
    main()