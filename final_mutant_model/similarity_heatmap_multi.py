#!/usr/bin/env python3
"""
Multi-level cosine similarity heatmaps:
  Level 1: Full labels (antibiotic_concentration × gene_guide)
  Level 2: Drug name × Gene base name
  Level 3: MOA group × Pathway group
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

# ─── Level 3 groupings ───

MOA_GROUPS = {
    "Cell wall (PBP 2)": ["Avibactam", "Clavulanic_Acid", "Meropenem", "Mecillinam", "Relebactam"],
    "Cell wall (PBP 3)": ["Aztreonam", "Ceftriaxone", "Cefepim"],
    "Cell wall (PBP 1)": ["Sulbactam", "Penicillin", "Cefsulodin"],
    "Ribosome": ["Doxicyclin", "Chloramphenicol", "Clarithromycin", "Kanamycin"],
    "Gyrase": ["Ciprofloxacin", "Norfloxacin", "Levofloxacin"],
    "Membrane integrity": ["Polymyxin_B", "Colistin"],
    "RNA polymerase": ["Rifampicin"],
    "DNA synthesis": ["Trimethoprim"],
}
ANTIBIOTIC_TO_MOA = {ab: moa for moa, abx in MOA_GROUPS.items() for ab in abx}
ANTIBIOTIC_TO_MOA['control'] = 'Control'

TRIAL_PATHWAY = {
    'folP': 'Folic acid biosynthesis',
    'folA': 'Folic acid biosynthesis',
    'secY': 'Protein transport',
    'secA': 'Protein transport',
    'rpoB': 'Transcription elongation',
    'rpoA': 'Transcription elongation',
    'lptC': 'Cell envelope organization',
    'lptA': 'Cell envelope organization',
    'msbA': 'Cell envelope organization',
    'ftsZ': 'Division septum assembly',
    'rplC': 'Translation initiation',
    'rplA': 'Translation initiation',
    'rpsA': 'Translation initiation',
    'rpsL': 'Translation initiation',
    'murC': 'Aminoglycan biosynthesis',
    'murA': 'Aminoglycan biosynthesis',
    'mrcB': 'Aminoglycan biosynthesis',
    'mrdA': 'Cell shape regulation',
    'mrcA': 'Cell shape regulation',
    'ftsI': 'Cell shape regulation',
    'lpxC': 'Lipid A biosynthesis',
    'lpxA': 'Lipid A biosynthesis',
    'gyrB': 'Chromosome organization',
    'gyrA': 'Chromosome organization',
    'dnaB': 'Chromosome organization',
    'parE': 'Chromosome organization',
    'parC': 'Chromosome organization',
    'dnaE': 'Chromosome organization',
}
GENE_TO_PATHWAY = TRIAL_PATHWAY
GENE_TO_PATHWAY.update({'WT NC': 'WT/NC', 'NC': 'WT/NC'})

# Expected antibiotic ↔ gene matches for Level 2 boxes
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


def extract_antibiotic_name(label: str) -> str:
    if '_' in label:
        parts = label.rsplit('_', 1)
        if parts[1].endswith('x'):
            return parts[0]
    return label


def extract_gene_base(label: str) -> str:
    if '_' in label:
        parts = label.rsplit('_', 1)
        if parts[1].replace('.', '').isdigit():
            return parts[0]
    return label


def compute_and_plot(embeddings, drug_labels, mut_labels, drug_groups, mut_groups,
                     drug_base_of, mut_base_of, level_name, output_path,
                     expected=None, annot=False):
    """Generic centroid similarity + heatmap."""
    # Build centroid dicts
    dcent = {k: np.mean(v, axis=0) for k, v in drug_groups.items()}
    mcent = {k: np.mean(v, axis=0) for k, v in mut_groups.items()}

    drugs_sorted = sorted(dcent.keys())
    muts_sorted = sorted(mcent.keys())
    n_d, n_m = len(drugs_sorted), len(muts_sorted)

    print(f"\n[{level_name}] {n_d} × {n_m}")

    sim = np.zeros((n_d, n_m))
    for i, dk in enumerate(drugs_sorted):
        for j, mk in enumerate(muts_sorted):
            a = dcent[dk].reshape(1, -1)
            b = mcent[mk].reshape(1, -1)
            sim[i, j] = cosine_similarity(a, b)[0, 0]

    # Clustering
    if n_d > 2 and n_m > 2:
        try:
            ab_link = linkage(sim, method='average')
            gene_link = linkage(sim.T, method='average')
            ab_ord = leaves_list(ab_link)
            gene_ord = leaves_list(gene_link)
            drugs_ord = [drugs_sorted[i] for i in ab_ord]
            muts_ord = [muts_sorted[j] for j in gene_ord]
            sim_ord = sim[ab_ord][:, gene_ord]
        except:
            drugs_ord, muts_ord, sim_ord = drugs_sorted, muts_sorted, sim
    else:
        drugs_ord, muts_ord, sim_ord = drugs_sorted, muts_sorted, sim

    # Build match mask for expected matches if provided
    match_mask = None
    if expected is not None:
        match_mask = np.zeros((n_d, n_m), dtype=bool)
        db_to_drugs = {}
        for dk in drugs_sorted:
            db = drug_base_of.get(dk, dk)
            db_to_drugs.setdefault(db, []).append(dk)

        for i_orig, dk in enumerate(drugs_sorted):
            db = drug_base_of.get(dk, dk)
            exp_genes = expected.get(db, set())
            for j_orig, mk in enumerate(muts_sorted):
                gb = mut_base_of.get(mk, mk)
                if gb in exp_genes:
                    i_new = drugs_ord.index(dk) if drugs_ord else i_orig
                    j_new = muts_ord.index(mk) if muts_ord else j_orig
                    match_mask[i_new, j_new] = True

    fig_w = max(8, n_m * 0.22)
    fig_h = max(6, n_d * 0.28)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(sim_ord, xticklabels=muts_ord, yticklabels=drugs_ord,
                cmap='RdYlBu_r', annot=annot, fmt='.2f',
                cbar_kws={'label': 'Cosine Similarity', 'shrink': 0.3},
                ax=ax, vmin=sim.min(), vmax=sim.max(),
                linewidths=0.05, linecolor='white')

    if match_mask is not None:
        for i in range(n_d):
            for j in range(n_m):
                if match_mask[i, j]:
                    ax.add_patch(mpatches.Rectangle((j, i), 1, 1, fill=False,
                                                    edgecolor='lime', linewidth=1.5))

    ax.set_xlabel('Mutant', fontsize=11)
    ax.set_ylabel('Drug', fontsize=11)
    ax.set_title(f'{level_name}  ({n_d} × {n_m})', fontsize=13, fontweight='bold')
    plt.xticks(rotation=90, ha='center', fontsize=6)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

    csv_path = output_path.replace('.png', '_matrix.csv')
    pd.DataFrame(sim, index=drugs_sorted, columns=muts_sorted).to_csv(csv_path)
    print(f"  Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=str, default='P1')
    parser.add_argument('--embedding_type', type=str, default='mil')
    parser.add_argument('--neighborhood', type=int, default=3)
    args = parser.parse_args()

    fold_key = f'Plate_{args.fold.replace("P", "")}'
    emb_path = os.path.join(SCRIPT_DIR, 'both', f'fold_{fold_key}',
                            f'embeddings_{fold_key}_{args.embedding_type}_n{args.neighborhood}.npz')
    out_dir = os.path.join(SCRIPT_DIR, 'both', f'fold_{fold_key}')
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading: {emb_path}")
    data = np.load(emb_path)
    embeddings = data['embeddings']
    paths = data['paths']

    ic50_path = os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')
    mut_path = os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')
    IC50 = json.load(open(ic50_path)) if os.path.exists(ic50_path) else {}
    MUT = json.load(open(mut_path)) if os.path.exists(mut_path) else {}

    correct_labels = [fix_label(p, IC50, MUT) for p in paths]

    # Separate into drug vs mutant groups (include controls at all levels)
    drug_emb, drug_lab = [], []
    mut_emb, mut_lab = [], []
    for emb, lab in zip(embeddings, correct_labels):
        if not lab or lab == 'unknown':
            continue
        # 'control' is a drug; everything with '_Nx' suffix is drug; rest is mutant
        if lab == 'control' or ('_' in lab and lab.rsplit('_', 1)[1].endswith('x')):
            drug_emb.append(emb)
            drug_lab.append(lab)
        else:
            mut_emb.append(emb)
            mut_lab.append(lab)

    # ──── Level 1: Full labels ────
    print("\n═══ Level 1: Full labels ═══")
    drug_groups_l1 = {}
    for emb, lab in zip(drug_emb, drug_lab):
        drug_groups_l1.setdefault(lab, []).append(emb)
    mut_groups_l1 = {}
    for emb, lab in zip(mut_emb, mut_lab):
        mut_groups_l1.setdefault(lab, []).append(emb)
    drug_base_l1 = {lab: extract_antibiotic_name(lab) for lab in drug_groups_l1}
    mut_base_l1 = {lab: extract_gene_base(lab) for lab in mut_groups_l1}

    compute_and_plot(
        embeddings, drug_lab, mut_lab,
        drug_groups_l1, mut_groups_l1,
        drug_base_l1, mut_base_l1,
        'Level 1: Full labels',
        os.path.join(out_dir, 'heatmap_L1_full.png'),
        expected=EXPECTED_MATCHES)

    # ──── Level 2: Drug name × Gene base ────
    print("\n═══ Level 2: Drug × Gene ═══")
    drug_groups_l2 = {}
    for emb, lab in zip(drug_emb, drug_lab):
        ab = extract_antibiotic_name(lab)
        drug_groups_l2.setdefault(ab, []).append(emb)
    mut_groups_l2 = {}
    for emb, lab in zip(mut_emb, mut_lab):
        gb = extract_gene_base(lab)
        mut_groups_l2.setdefault(gb, []).append(emb)
    drug_base_l2 = {k: k for k in drug_groups_l2}
    mut_base_l2 = {k: k for k in mut_groups_l2}

    compute_and_plot(
        embeddings, list(drug_groups_l2.keys()), list(mut_groups_l2.keys()),
        drug_groups_l2, mut_groups_l2,
        drug_base_l2, mut_base_l2,
        'Level 2: Drug × Gene',
        os.path.join(out_dir, 'heatmap_L2_drug_gene.png'),
        expected=EXPECTED_MATCHES)

    # ──── Level 3: MOA × Pathway ────
    print("\n═══ Level 3: MOA × Pathway ═══")
    drug_groups_l3 = {}
    for emb, lab in zip(drug_emb, drug_lab):
        ab = extract_antibiotic_name(lab)
        moa = ANTIBIOTIC_TO_MOA.get(ab, 'Other')
        drug_groups_l3.setdefault(moa, []).append(emb)
    mut_groups_l3 = {}
    for emb, lab in zip(mut_emb, mut_lab):
        gb = extract_gene_base(lab)
        pathway = GENE_TO_PATHWAY.get(gb, 'Other')
        mut_groups_l3.setdefault(pathway, []).append(emb)

    compute_and_plot(
        embeddings, list(drug_groups_l3.keys()), list(mut_groups_l3.keys()),
        drug_groups_l3, mut_groups_l3,
        {}, {},
        'Level 3: MOA × Pathway',
        os.path.join(out_dir, 'heatmap_L3_moa_pathway.png'),
        annot=True)


if __name__ == '__main__':
    main()
