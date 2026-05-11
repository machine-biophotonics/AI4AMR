#!/usr/bin/env python3
"""
Cosine similarity heatmap between drug and mutant class centroids.
Input: embeddings .npz from extract_embeddings.py (--data_mode both)
"""

import os
import json
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def fix_label(img_path: str, IC50: dict, MUT: dict) -> str:
    path_lower = img_path.lower()
    if '/drugs_data/' in path_lower:
        src = 'drug'
    elif '/mutants_data/' in path_lower:
        src = 'mutant'
    else:
        return 'unknown'
    match = re.search(r'Well(\w\d+)_', os.path.basename(img_path))
    well = match.group(1) if match else None
    if not well:
        return 'unknown'
    pk = None
    for pn in range(1, 7):
        if f'/p{pn}/' in path_lower:
            pk = f'P{pn}'
            break
    if not pk:
        return 'unknown'
    if src == 'drug':
        if pk in IC50 and well in IC50[pk]:
            info = IC50[pk][well]
            ab = info.get('antibiotic', '')
            ic = info.get('ic50_multiple', '')
            if ab and ic:
                if ic == 'control':
                    return 'control'
                return f"{ab.replace(' ', '_')}_{ic if 'x' in str(ic) else f'{ic}x'}"
    else:
        row, col_raw = well[0], well[1:].lstrip('0') or '0'
        try:
            if pk in MUT and row in MUT[pk] and col_raw in MUT[pk][row]:
                return MUT[pk][row][col_raw].get('id', None)
        except:
            pass
    return 'unknown'


def extract_gene_base(label: str) -> str:
    if '_' in label:
        parts = label.rsplit('_', 1)
        if parts[1].replace('.', '').isdigit():
            return parts[0]
    return label


def extract_antibiotic_name(label: str) -> str:
    if label == 'control':
        return 'control'
    if '_' in label:
        parts = label.rsplit('_', 1)
        suffix = parts[1]
        if suffix.endswith('x') or suffix.endswith('X'):
            return parts[0]
    return label


def extract_concentration(label: str) -> float:
    if '_' in label:
        parts = label.rsplit('_', 1)
        conc_str = parts[1].replace('x', '').replace('X', '')
        try:
            return float(conc_str)
        except ValueError:
            return 0.0
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=str, default='P1')
    parser.add_argument('--embedding_type', type=str, default='mil')
    parser.add_argument('--neighborhood', type=int, default=3)
    parser.add_argument('--cluster', action='store_true', default=True,
                        help='Hierarchical clustering (default: True)')
    parser.add_argument('--similarity', type=str, default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--output', type=str, default='similarity_heatmap.png')

    args = parser.parse_args()

    fold_key = f'Plate_{args.fold.replace("P", "")}'
    emb_path = os.path.join(SCRIPT_DIR, 'both', f'fold_{fold_key}',
                            f'embeddings_{fold_key}_{args.embedding_type}_n{args.neighborhood}.npz')
    output_dir = os.path.join(SCRIPT_DIR, 'both', f'fold_{fold_key}')
    output_png = os.path.join(output_dir, args.output)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading: {emb_path}")
    data = np.load(emb_path)
    embeddings = data['embeddings']
    paths = data['paths']

    # Load label mappings
    ic50_path = os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')
    mut_path = os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')
    IC50 = json.load(open(ic50_path)) if os.path.exists(ic50_path) else {}
    MUT = json.load(open(mut_path)) if os.path.exists(mut_path) else {}

    # Fix labels
    correct_labels = [fix_label(p, IC50, MUT) for p in paths]

    is_ctrl = lambda l: l and any(x in l.lower() for x in ['control', 'wt ', 'wild', ' nc', 'nc_'])
    is_drug = lambda l: l and '_' in l and l.rsplit('_', 1)[1].endswith('x')

    # Separate into groups
    drug_embs, drug_labels = {}, {}
    mut_embs, mut_labels = {}, {}

    for emb, label in zip(embeddings, correct_labels):
        if not label or label == 'unknown' or is_ctrl(label):
            continue
        if is_drug(label):
            ab_name = extract_antibiotic_name(label)
            if ab_name not in drug_embs:
                drug_embs[ab_name] = []
                drug_labels[ab_name] = set()
            drug_embs[ab_name].append(emb)
            drug_labels[ab_name].add(label)
        else:
            gene_base = extract_gene_base(label)
            if gene_base not in mut_embs:
                mut_embs[gene_base] = []
                mut_labels[gene_base] = set()
            mut_embs[gene_base].append(emb)
            mut_labels[gene_base].add(label)

    # Compute centroids
    drug_centroids = {k: np.mean(v, axis=0) for k, v in drug_embs.items()}
    mut_centroids = {k: np.mean(v, axis=0) for k, v in mut_embs.items()}

    antibiotics = sorted(drug_centroids.keys())
    genes = sorted(mut_centroids.keys())

    print(f"Antibiotics: {len(antibiotics)}, Genes: {len(genes)}")
    print(f"  Antibiotics: {antibiotics}")
    print(f"  Genes: {genes}")

    # Build similarity matrix
    sim = np.zeros((len(antibiotics), len(genes)))
    for i, ab in enumerate(antibiotics):
        for j, gene in enumerate(genes):
            ab_emb = drug_centroids[ab].reshape(1, -1)
            g_emb = mut_centroids[gene].reshape(1, -1)
            if args.similarity == 'cosine':
                sim[i, j] = cosine_similarity(ab_emb, g_emb)[0, 0]
            else:
                d = np.linalg.norm(ab_emb - g_emb)
                sim[i, j] = 1.0 / (1.0 + d)

    # Ordering
    if args.cluster:
        from scipy.cluster.hierarchy import linkage, leaves_list
        ab_linkage = linkage(sim, method='average')
        gene_linkage = linkage(sim.T, method='average')
        ab_order = leaves_list(ab_linkage)
        gene_order = leaves_list(gene_linkage)
        antibiotics_ordered = [antibiotics[i] for i in ab_order]
        genes_ordered = [genes[j] for j in gene_order]
        sim_ordered = sim[ab_order][:, gene_order]
    else:
        antibiotics_ordered = antibiotics
        genes_ordered = genes
        sim_ordered = sim

    # Plot
    fig, ax = plt.subplots(figsize=(max(24, len(genes) * 0.3), max(10, len(antibiotics) * 0.3)))
    sns.heatmap(sim_ordered, xticklabels=genes_ordered, yticklabels=antibiotics_ordered,
                cmap='RdYlBu_r', annot=False, fmt='.2f',
                cbar_kws={'label': 'Cosine Similarity', 'shrink': 0.5},
                ax=ax, vmin=sim.min(), vmax=sim.max(),
                linewidths=0.1, linecolor='white')
    ax.set_xlabel('Mutant Gene', fontsize=12)
    ax.set_ylabel('Antibiotic', fontsize=12)
    ax.set_title(f'Cosine Similarity: Antibiotic × Mutant Gene Centroids\n({len(antibiotics)} antibiotics × {len(genes)} genes)',
                 fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PNG: {output_png}")

    # Save CSV
    csv_path = output_png.replace('.png', '_matrix.csv')
    sim_df = pd.DataFrame(sim, index=antibiotics, columns=genes)
    sim_df.to_csv(csv_path)
    print(f"Saved CSV: {csv_path}")

    # Top matches
    print("\n=== Top 20 Most Similar Pairs ===")
    flat = [(ab, gene, sim[i, j]) for i, ab in enumerate(antibiotics) for j, gene in enumerate(genes)]
    flat.sort(key=lambda x: x[2], reverse=True)
    for ab, gene, val in flat[:20]:
        print(f"  {ab:30} <-> {gene:12}: {val:.4f}")
    print("\n=== Top 20 Least Similar Pairs ===")
    for ab, gene, val in flat[-20:]:
        print(f"  {ab:30} <-> {gene:12}: {val:.4f}")


if __name__ == '__main__':
    main()
