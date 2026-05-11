#!/usr/bin/env python3
"""
Generate a single combined 185x185 similarity matrix from ALL 6 plates.
Loads embeddings from P1-P6, averages per well, then computes:
- 96 mutant genes + 89 antibiotics = 185 entities on both axes
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
from collections import defaultdict
from matplotlib.patches import Rectangle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLATES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']


def well_to_row_col(well_id: str):
    match = well_id.replace("Well", "")
    return match[0], int(match[1:])


def well_to_wellname(well_id: str):
    return well_id.replace("Well", "")


def load_embeddings_by_well_all_plates(data_type: str) -> dict:
    """Load embeddings from all 6 plates and average per well."""
    well_embeddings = defaultdict(list)
    
    for plate in PLATES:
        embeddings_dir = os.path.join(BASE_DIR, "embeddings", f"{data_type.capitalize()}s_{plate}")
        if not os.path.exists(embeddings_dir):
            print(f"  WARNING: {embeddings_dir} not found, skipping {plate}...")
            continue
        
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
    import argparse
    parser = argparse.ArgumentParser(description='Generate combined 185x185 similarity matrix (all 6 plates)')
    parser.add_argument('--plates', type=str, default='P1,P2,P3,P4,P5,P6')
    parser.add_argument('--output_prefix', type=str, default='combined_similarity_matrix_185')
    args = parser.parse_args()

    global PLATES
    PLATES = [p.strip() for p in args.plates.split(',')]
    
    print(f"\n=== Building Combined 185x185 Similarity Matrix (Plates: {PLATES}) ===")
    
    print("Loading embeddings from all plates...")
    mutant_embeddings = load_embeddings_by_well_all_plates('mutant')
    drug_embeddings = load_embeddings_by_well_all_plates('drug')
    print(f"  Mutant wells: {len(mutant_embeddings)} (averaged across {len(PLATES)} plates)")
    print(f"  Drug wells: {len(drug_embeddings)} (averaged across {len(PLATES)} plates)")
    
    with open(os.path.join(BASE_DIR, "plate_well_id_path.json")) as f:
        all_gene_mapping = json.load(f)
    
    with open(os.path.join(BASE_DIR, "plate_well_ic50_mapping.json")) as f:
        all_drug_mapping = json.load(f)
    
    mutant_wells_sorted = sorted(mutant_embeddings.keys())
    
    mutant_ids = []
    for well_id in mutant_wells_sorted:
        row, col = well_to_row_col(well_id)
        gene_id = None
        for plate in PLATES:
            if plate in all_gene_mapping and row in all_gene_mapping[plate] and str(col) in all_gene_mapping[plate][row]:
                gene_id = all_gene_mapping[plate][row][str(col)]['id']
                break
        if gene_id:
            mutant_ids.append(gene_id)
        else:
            mutant_ids.append(f'Unknown_{well_id}')
    
    sorted_gene_ids = sorted(mutant_ids, key=lambda x: x.rsplit('_', 1)[0])
    
    gene_counts = {}
    unique_mutant_ids = []
    for gene_id in sorted_gene_ids:
        if gene_id in gene_counts:
            gene_counts[gene_id] += 1
            gene_id = f"{gene_id}_W{gene_counts[gene_id]}"
        else:
            gene_counts[gene_id] = 1
        unique_mutant_ids.append(gene_id)
    
    antibiotic_set = set()
    for plate in PLATES:
        if plate in all_drug_mapping:
            for wellname, info in all_drug_mapping[plate].items():
                ab = info['antibiotic']
                ic50 = info['ic50_multiple']
                antibiotic_set.add(f"{ab}_{ic50}")
    
    antibiotic_list = sorted(list(antibiotic_set))
    print(f"  Mutant genes: {len(unique_mutant_ids)}")
    print(f"  Antibiotics: {len(antibiotic_list)}")
    
    ab_embeddings = {}
    for well_id, embedding in drug_embeddings.items():
        wellname = well_to_wellname(well_id)
        for plate in PLATES:
            if plate in all_drug_mapping and wellname in all_drug_mapping[plate]:
                ab = all_drug_mapping[plate][wellname]['antibiotic']
                ic50 = all_drug_mapping[plate][wellname]['ic50_multiple']
                ab_id = f"{ab}_{ic50}"
                if ab_id not in ab_embeddings:
                    ab_embeddings[ab_id] = []
                ab_embeddings[ab_id].append(embedding)
                break
    
    for ab_id in ab_embeddings:
        ab_embeddings[ab_id] = np.mean(ab_embeddings[ab_id], axis=0)
    
    all_embeddings = {}
    for i, well_id in enumerate(mutant_wells_sorted):
        all_embeddings[unique_mutant_ids[i]] = mutant_embeddings[well_id]
    for ab_id in antibiotic_list:
        if ab_id in ab_embeddings:
            all_embeddings[ab_id] = ab_embeddings[ab_id]
    
    print(f"\nComputing 185x185 similarity matrix...")
    all_entity_ids = unique_mutant_ids + antibiotic_list
    n = len(all_entity_ids)
    
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i, j] = 1.0
            elif i < j:
                sim = cosine_similarity(all_embeddings[all_entity_ids[i]], all_embeddings[all_entity_ids[j]])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
    
    df = pd.DataFrame(similarity_matrix, index=all_entity_ids, columns=all_entity_ids)
    print(f"  Matrix shape: {df.shape}")
    
    csv_path = os.path.join(BASE_DIR, f"{args.output_prefix}.csv")
    df.to_csv(csv_path)
    print(f"  Saved CSV: {csv_path}")
    
    print("\nGenerating heatmaps...")
    
    # === Main heatmap ===
    fig, ax = plt.subplots(figsize=(40, 36))
    sns.heatmap(df, cmap='viridis', vmin=0.5, vmax=1.0,
                xticklabels=True, yticklabels=True,
                cbar_kws={'label': 'Cosine Similarity', 'shrink': 0.5, 'pad': 0.02},
                ax=ax, linewidths=0, linecolor='white')
    
    ax.set_title(f'Combined Similarity Matrix: 96 Mutant Genes + 89 Antibiotics (185 × 185)\nAveraged across plates: {PLATES}',
                 fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Entities', fontsize=14)
    ax.set_ylabel('Entities', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=4, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=4)
    
    ax.axhline(y=96, color='white', linewidth=3)
    ax.axvline(x=96, color='white', linewidth=3)
    
    gene_positions = {}
    for i, gene in enumerate(unique_mutant_ids):
        base = gene.rsplit('_', 1)[0]
        if base not in gene_positions:
            gene_positions[base] = []
        gene_positions[base].append(i)
    
    for base, positions in gene_positions.items():
        if len(positions) > 1:
            min_pos = min(positions)
            max_pos = max(positions)
            rect = Rectangle((min_pos, min_pos), max_pos - min_pos + 1, max_pos - min_pos + 1,
                            linewidth=2.5, edgecolor='red', facecolor='none', linestyle='-')
            ax.add_patch(rect)
    
    ab_base_positions = {}
    for i, ab in enumerate(antibiotic_list):
        base = ab.rsplit('_', 1)[0]
        if base not in ab_base_positions:
            ab_base_positions[base] = []
        ab_base_positions[base].append(i)
    
    for base, positions in ab_base_positions.items():
        if len(positions) > 1:
            start = positions[0] + 96
            end = positions[-1] + 96
            rect = Rectangle((start, start), end - start + 1, end - start + 1,
                            linewidth=2.5, edgecolor='red', facecolor='none', linestyle='-')
            ax.add_patch(rect)
    
    plt.tight_layout()
    png_path = os.path.join(BASE_DIR, f"{args.output_prefix}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path}")
    
    # === Deviation heatmap ===
    print("Generating deviation heatmap...")
    mutant_mutant_sim = df.iloc[:96, :96].values.mean()
    drug_drug_sim = df.iloc[96:, 96:].values.mean()
    mutant_drug_sim = df.iloc[:96, 96:].values.mean()
    overall_mean = df.values.mean()
    diff_from_mean = df.values - overall_mean
    
    diff_df = pd.DataFrame(diff_from_mean, index=df.index, columns=df.columns)
    
    fig, ax = plt.subplots(figsize=(42, 38))
    vmax = max(abs(diff_from_mean.min()), abs(diff_from_mean.max()))
    sns.heatmap(diff_df, cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
                xticklabels=True, yticklabels=True,
                cbar_kws={'label': 'Deviation from Mean Similarity', 'shrink': 0.5},
                ax=ax, linewidths=0, linecolor='white')
    
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=4, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=4)
    ax.axhline(y=96, color='black', linewidth=3)
    ax.axvline(x=96, color='black', linewidth=3)
    
    title = (f'Combined Similarity: Deviation from Mean\n'
             f'Mean: {overall_mean*100:.1f}% | Mutant-Mutant: {mutant_mutant_sim*100:.1f}% | '
             f'Drug-Drug: {drug_drug_sim*100:.1f}% | Mutant-Drug: {mutant_drug_sim*100:.1f}%')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout()
    dev_path = os.path.join(BASE_DIR, f"{args.output_prefix}_deviation.png")
    plt.savefig(dev_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {dev_path}")
    
    # === With bars heatmap ===
    print("Generating heatmap with similarity bars...")
    mutant_to_ab_sim = df.iloc[:96, 96:].mean(axis=0)
    ab_percent = (mutant_to_ab_sim * 100).round(1)
    ab_to_mutant_sim = df.iloc[96:, :96].mean(axis=1)
    mutant_percent = (ab_to_mutant_sim * 100).round(1)
    cross_sim = df.iloc[:96, 96:].values.mean()
    
    title = (f'Combined 185x185 Similarity (All Plates: {PLATES})\n'
             f'Mutant-Drug: {cross_sim*100:.1f}% | Mutant-Mutant: {mutant_mutant_sim*100:.1f}% | '
             f'Drug-Drug: {drug_drug_sim*100:.1f}%')
    
    fig, ax = plt.subplots(figsize=(42, 38))
    sns.heatmap(df, cmap='viridis', vmin=0.5, vmax=1.0,
                xticklabels=True, yticklabels=True,
                cbar_kws={'label': 'Cosine Similarity', 'shrink': 0.5},
                ax=ax, linewidths=0, linecolor='white')
    
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=4, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=4)
    
    for i, pct in enumerate(ab_percent.values):
        ax.text(i + 0.5, -2, f'{pct}%', ha='center', va='bottom', fontsize=3, rotation=90, color='black')
    for i, pct in enumerate(mutant_percent.values):
        ax.text(-2, i + 0.5, f'{pct}%', ha='right', va='center', fontsize=3, color='black')
    
    ax.axhline(y=96, color='white', linewidth=3)
    ax.axvline(x=96, color='white', linewidth=3)
    
    for base, positions in gene_positions.items():
        if len(positions) > 1:
            min_pos = min(positions)
            max_pos = max(positions)
            rect = Rectangle((min_pos - 0.5, min_pos - 0.5), max_pos - min_pos + 1, max_pos - min_pos + 1,
                            linewidth=3, edgecolor='red', facecolor='none', linestyle='-')
            ax.add_patch(rect)
    
    for base, positions in ab_base_positions.items():
        if len(positions) > 1:
            start = positions[0] + 96
            end = positions[-1] + 96
            rect = Rectangle((start, start), end - start + 1, end - start + 1,
                            linewidth=2.5, edgecolor='red', facecolor='none', linestyle='-')
            ax.add_patch(rect)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Entities', fontsize=12)
    ax.set_ylabel('Entities', fontsize=12)
    
    bars_path = os.path.join(BASE_DIR, f"{args.output_prefix}_with_bars.png")
    plt.savefig(bars_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {bars_path}")
    
    print(f"\n=== DONE ===")
    print(f"Matrix: {n}x{n} ({len(unique_mutant_ids)} mutants + {len(antibiotic_list)} antibiotics)")
    print(f"Plates: {PLATES}")
    print(f"Files: {csv_path}, {png_path}, {dev_path}, {bars_path}")


if __name__ == '__main__':
    main()