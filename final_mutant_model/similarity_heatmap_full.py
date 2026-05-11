#!/usr/bin/env python3
"""
Full cosine similarity heatmap with all individual antibiotic/mutant labels
and expected-match boxes overlaid.
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
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, leaves_list

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Expected antibiotic ↔ gene matches from the user's table
# Maps antibiotic display name → set of expected gene base names
EXPECTED_MATCHES = {
    'Cefsulodin': {'mrcA', 'mrcB'},
    'Penicillin': {'mrcA', 'mrcB', 'ftsI'},
    'Sulbactam': {'mrcA', 'mrcB', 'ftsI'},
    'Avibactam': set(),
    'Mecillinam': {'mrdA'},
    'Meropenem': {'mrdA', 'ftsI', 'mrcA', 'mrcB'},
    'Clavulanic_Acid': set(),
    'Relebactam': set(),
    'Aztreonam': {'ftsI'},
    'Cefepim': {'ftsI', 'mrcA', 'mrcB', 'mrdA'},
    'Ceftriaxone': {'ftsI', 'mrcA', 'mrcB'},
    'Chloramphenicol': {'rplA', 'rplC'},
    'Clarithromycin': {'rplA', 'rplC'},
    'Doxicyclin': {'rpsA', 'rpsL'},
    'Kanamycin': {'rpsA', 'rpsL'},
    'Ciprofloxacin': {'gyrA', 'gyrB', 'parC', 'parE'},
    'Levofloxacin': {'gyrA', 'gyrB', 'parC', 'parE'},
    'Norfloxacin': {'gyrA', 'gyrB', 'parC', 'parE'},
    'Rifampicin': {'rpoA', 'rpoB'},
    'Trimethoprim': {'folA', 'folP'},
    'Colistin': {'lpxA', 'lpxC', 'lptA', 'lptC'},
    'Polymyxin_B': {'lpxA', 'lpxC', 'lptA', 'lptC'},
}

# All possible gene base names (for box color coding)
GENE_SET = {'mrcA', 'mrcB', 'ftsI', 'mrdA', 'rplA', 'rplC', 'rpsA', 'rpsL',
            'gyrA', 'gyrB', 'parC', 'parE', 'rpoA', 'rpoB', 'folA', 'folP',
            'lpxA', 'lpxC', 'lptA', 'lptC', 'dnaB', 'dnaE', 'murA', 'murC',
            'secA', 'secY', 'msbA', 'ftsZ'}


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


def extract_antibiotic_base(label: str) -> str:
    """E.g., 'Ciprofloxacin_2x' -> 'Ciprofloxacin'"""
    if '_' in label:
        parts = label.rsplit('_', 1)
        if parts[1].endswith('x'):
            return parts[0]
    return label


def extract_gene_base(label: str) -> str:
    """E.g., 'mrcA_1' -> 'mrcA'"""
    if '_' in label:
        parts = label.rsplit('_', 1)
        if parts[1].replace('.', '').isdigit():
            return parts[0]
    return label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=str, default='P1')
    parser.add_argument('--embedding_type', type=str, default='mil')
    parser.add_argument('--neighborhood', type=int, default=3)
    parser.add_argument('--similarity', type=str, default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--output', type=str, default='similarity_heatmap_full.png')

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

    # Group embeddings by FULL label (with concentration/guide number)
    drug_groups = {}
    mut_groups = {}
    for emb, label in zip(embeddings, correct_labels):
        if not label or label == 'unknown' or is_ctrl(label):
            continue
        if is_drug(label):
            drug_groups.setdefault(label, []).append(emb)
        else:
            mut_groups.setdefault(label, []).append(emb)

    print(f"Unique drug labels (with concentration): {len(drug_groups)}")
    print(f"Unique mutant labels (with guide): {len(mut_groups)}")

    # Build centroid lookup
    drug_cent = {k: np.mean(v, axis=0) for k, v in drug_groups.items()}
    mut_cent = {k: np.mean(v, axis=0) for k, v in mut_groups.items()}
    drug_full = sorted(drug_cent.keys())
    mut_full = sorted(mut_cent.keys())

    # Map full labels → base names
    drug_base_of = {lab: extract_antibiotic_base(lab) for lab in drug_full}
    mut_base_of = {lab: extract_gene_base(lab) for lab in mut_full}

    n_drug, n_mut = len(drug_full), len(mut_full)
    print(f"Matrix: {n_drug} drugs × {n_mut} mutants")

    # Build similarity matrix
    sim = np.zeros((n_drug, n_mut))
    for i, dlab in enumerate(drug_full):
        for j, mlab in enumerate(mut_full):
            a = drug_cent[dlab].reshape(1, -1)
            b = mut_cent[mlab].reshape(1, -1)
            if args.similarity == 'cosine':
                sim[i, j] = cosine_similarity(a, b)[0, 0]
            else:
                sim[i, j] = 1.0 / (1.0 + np.linalg.norm(a - b))

    # Clustering
    ab_linkage = linkage(sim, method='average')
    gene_linkage = linkage(sim.T, method='average')
    ab_order = leaves_list(ab_linkage)
    gene_order = leaves_list(gene_linkage)

    drug_ordered = [drug_full[i] for i in ab_order]
    mut_ordered = [mut_full[j] for j in gene_order]
    sim_ordered = sim[ab_order][:, gene_order]

    # Build expected-match mask for ordered matrix
    match_mask = np.zeros((n_drug, n_mut), dtype=bool)
    for i_orig, dlab in enumerate(drug_full):
        db = drug_base_of[dlab]
        expected_genes = EXPECTED_MATCHES.get(db, set())
        for j_orig, mlab in enumerate(mut_full):
            gb = mut_base_of[mlab]
            if gb in expected_genes:
                # Map to ordered coords
                i_new = ab_order.tolist().index(i_orig)
                j_new = gene_order.tolist().index(j_orig)
                match_mask[i_new, j_new] = True

    # Plot
    fig_w = max(12, n_mut * 0.18)
    fig_h = max(8, n_drug * 0.25)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(sim_ordered, xticklabels=mut_ordered, yticklabels=drug_ordered,
                cmap='RdYlBu_r', annot=False, cbar_kws={'label': 'Cosine Similarity', 'shrink': 0.3},
                ax=ax, vmin=sim.min(), vmax=sim.max(),
                linewidths=0.05, linecolor='white')

    # Overlay green boxes on expected matches
    for i in range(n_drug):
        for j in range(n_mut):
            if match_mask[i, j]:
                rect = mpatches.Rectangle((j, i), 1, 1, fill=False,
                                          edgecolor='lime', linewidth=1.5)
                ax.add_patch(rect)

    ax.set_xlabel('Mutant Gene', fontsize=11)
    ax.set_ylabel('Antibiotic', fontsize=11)
    ax.set_title(f'Cosine Similarity: Antibiotic × Mutant (full labels)\n'
                 f'{n_drug} drugs × {n_mut} mutants — green boxes = expected matches',
                 fontsize=12, fontweight='bold')
    plt.xticks(rotation=90, ha='center', fontsize=5)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved PNG: {output_png}")

    # CSV
    csv_path = output_png.replace('.png', '_matrix.csv')
    sim_df = pd.DataFrame(sim, index=drug_full, columns=mut_full)
    sim_df.to_csv(csv_path)
    print(f"Saved CSV: {csv_path}")

    # Print stats
    n_expected = match_mask.sum()
    print(f"\nExpected-match cells: {n_expected} out of {n_drug * n_mut}")

    # Top matches within expected pairs
    print("\n=== Top 20 Similarities Among Expected Matches ===")
    flat = []
    for i, dlab in enumerate(drug_full):
        db = drug_base_of[dlab]
        exp = EXPECTED_MATCHES.get(db, set())
        for j, mlab in enumerate(mut_full):
            gb = mut_base_of[mlab]
            if gb in exp:
                flat.append((dlab, mlab, sim[i, j]))
    flat.sort(key=lambda x: x[2], reverse=True)
    for dlab, mlab, val in flat[:20]:
        print(f"  {dlab:35} <-> {mlab:12}: {val:.4f}")

    # Top non-expected high similarities (potential novel discoveries)
    print("\n=== Top 20 Highest Non-Expected Similarities ===")
    flat2 = []
    for i, dlab in enumerate(drug_full):
        db = drug_base_of[dlab]
        exp = EXPECTED_MATCHES.get(db, set())
        for j, mlab in enumerate(mut_full):
            gb = mut_base_of[mlab]
            if gb not in exp:
                flat2.append((dlab, mlab, sim[i, j]))
    flat2.sort(key=lambda x: x[2], reverse=True)
    for dlab, mlab, val in flat2[:20]:
        print(f"  {dlab:35} <-> {mlab:12}: {val:.4f}")


if __name__ == '__main__':
    main()
