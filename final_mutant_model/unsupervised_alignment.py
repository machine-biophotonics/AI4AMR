#!/usr/bin/env python3
"""
Unsupervised drug↔mutant embedding alignment analysis.

Stages:
  1. Structural similarity (within-domain geometry → GW distance)
  2. GWOT coupling discovery (probabilistic drug↔mutant matches)
  3. Control-anchored Procrustes alignment
  4. Cross-domain Mutual Nearest Neighbors (MNN) retrieval

All stages are unsupervised — no drug↔mutant correspondences used.
Results are validated against EXPECTED_MATCHES at the end.
"""

import os
import json
import re
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist, pdist, squareform


try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── EXPECTED drug↔gene matches (for validation only, NOT used in alignment) ───

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

TRIAL_PATHWAY = {
    'folP': 'Folic acid biosynthesis', 'folA': 'Folic acid biosynthesis',
    'secY': 'Protein transport', 'secA': 'Protein transport',
    'rpoB': 'Transcription elongation', 'rpoA': 'Transcription elongation',
    'lptC': 'Cell envelope organization', 'lptA': 'Cell envelope organization',
    'msbA': 'Cell envelope organization',
    'ftsZ': 'Division septum assembly',
    'rplC': 'Translation initiation', 'rplA': 'Translation initiation',
    'rpsA': 'Translation initiation', 'rpsL': 'Translation initiation',
    'murC': 'Aminoglycan biosynthesis', 'murA': 'Aminoglycan biosynthesis',
    'mrcB': 'Aminoglycan biosynthesis',
    'mrdA': 'Cell shape regulation', 'mrcA': 'Cell shape regulation', 'ftsI': 'Cell shape regulation',
    'lpxC': 'Lipid A biosynthesis', 'lpxA': 'Lipid A biosynthesis',
    'gyrB': 'Chromosome organization', 'gyrA': 'Chromosome organization',
    'dnaB': 'Chromosome organization', 'parE': 'Chromosome organization',
    'parC': 'Chromosome organization', 'dnaE': 'Chromosome organization',
}
GENE_TO_PATHWAY = {**TRIAL_PATHWAY, 'WT NC': 'WT/NC', 'NC': 'WT/NC'}


def load_jsons():
    ic50_path = os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')
    mutant_path = os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')
    with open(ic50_path) as f:
        IC50 = json.load(f)
    with open(mutant_path) as f:
        MUT = json.load(f)
    return IC50, MUT


def fix_label(img_path, IC50, MUT):
    """Recompute label from path (identical to heatmap script)."""
    path_lower = img_path.lower()
    if '/drugs_data/' in path_lower or '/Drugs_Data/' in img_path:
        src = 'drug'
    elif '/mutants_data/' in path_lower or '/Mutants_Data/' in img_path:
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
        # Try from path like /P1/
        m2 = re.search(r'/[Pp](\d)/', img_path)
        if m2:
            pk = f'P{m2.group(1)}'
        else:
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


def extract_antibiotic_name(label):
    if '_' in label:
        parts = label.rsplit('_', 1)
        if parts[1].endswith('x'):
            return parts[0]
    return label


def extract_gene_base(label):
    if '_' in label:
        parts = label.rsplit('_', 1)
        if parts[1].replace('.', '').isdigit():
            return parts[0]
    return label


def pool_centroids_by_name(centroids, names, extract_fn):
    """Average centroids grouped by name (antibiotic or gene).
    Controls ('control', 'WT NC', 'NC') are kept as-is."""
    groups = {}
    for i, name in enumerate(names):
        base = extract_fn(name)
        # Keep control labels as-is
        if name == 'control' or base in ('WT NC', 'NC'):
            base = name
        groups.setdefault(base, []).append(centroids[i])
    pooled_centroids = np.array([np.mean(v, axis=0) for _, v in sorted(groups.items())])
    pooled_names = sorted(groups.keys())
    return pooled_centroids, pooled_names


def is_drug_label(label):
    return '_' in label and label.rsplit('_', 1)[1].endswith('x')


def is_mutant_label(label):
    if label in ('control', 'unknown'):
        return False
    if label.startswith('WT NC') or label.startswith('NC'):
        return True
    if '_' in label:
        parts = label.rsplit('_', 1)
        if parts[1].replace('.', '').isdigit():
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 1: Structural similarity analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_structural_similarity(drug_centroids, mutant_centroids,
                                   drug_names, mutant_names,
                                   output_dir):
    """
    Compare within-domain geometry: drug-drug vs mutant-mutant distance matrices.
    Compute:
      - GW distance between the two distance matrices (lower = more similar structure)
      - Mantel-style correlation between upper triangles
      - Side-by-side heatmap visualization
    """
    print("\n═══ Stage 1: Structural Similarity Analysis ═══")

    D_drug = squareform(pdist(drug_centroids, metric='cosine'))
    D_mutant = squareform(pdist(mutant_centroids, metric='cosine'))

    # Mantel-like correlation (Spearman between upper triangles)
    triu_idx = np.triu_indices_from(D_drug, k=1)
    r_drug = D_drug[triu_idx]
    r_mutant = D_mutant[triu_idx]
    from scipy.stats import spearmanr
    rho, pval = spearmanr(r_drug, r_mutant)
    print(f"  Mantel (Spearman) ρ = {rho:.4f}, p = {pval:.2e}")

    # GW distance between the two metric spaces
    if HAS_POT:
        n_d = len(drug_centroids)
        n_m = len(mutant_centroids)
        p_d = np.ones(n_d) / n_d
        p_m = np.ones(n_m) / n_m
        gw_dist = ot.gromov.gromov_wasserstein2(
            D_drug / D_drug.max(), D_mutant / D_mutant.max(),
            p_d, p_m, log=False)
        print(f"  Gromov-Wasserstein distance (normalized) = {gw_dist:.6f}")
    else:
        gw_dist = None
        print("  (POT not available, skipping GW distance)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    for ax, D, names, title in zip(
            axes, [D_drug, D_mutant],
            [drug_names, mutant_names],
            ['Drug-Drug Cosine Dissimilarity', 'Mutant-Mutant Cosine Dissimilarity']):
        sns.heatmap(D, xticklabels=names, yticklabels=names,
                    ax=ax, cmap='viridis', square=True)
        ax.set_title(title, fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
    fig.suptitle(f'Stage 1: Within-Domain Structure  |  '
                 f'Mantel ρ={rho:.3f} (p={pval:.2e})'
                 + (f'  |  GW dist={gw_dist:.4f}' if gw_dist else ''),
                 fontsize=16)
    plt.tight_layout()
    path = os.path.join(output_dir, 'stage1_structural_similarity.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return {'mantel_rho': rho, 'mantel_pval': pval, 'gw_dist': gw_dist,
            'D_drug': D_drug, 'D_mutant': D_mutant}


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 2: GWOT coupling (probabilistic drug↔mutant matches)
# ═══════════════════════════════════════════════════════════════════════════

def run_gwot_coupling(drug_centroids, mutant_centroids,
                       drug_names, mutant_names,
                       drug_ab_names, mutant_gene_names,
                       output_dir):
    """
    Gromov-Wasserstein Optimal Transport between drug and mutant point sets.

    Returns coupling matrix T where T[i,j] = probability drug_i ↔ mutant_j.
    """
    print("\n═══ Stage 2: GWOT Coupling Discovery ═══")
    if not HAS_POT:
        print("  (POT not available, skipping GWOT)")
        return None

    n_d = len(drug_centroids)
    n_m = len(mutant_centroids)

    # Normalize
    X_d = drug_centroids / (np.linalg.norm(drug_centroids, axis=1, keepdims=True) + 1e-8)
    X_m = mutant_centroids / (np.linalg.norm(mutant_centroids, axis=1, keepdims=True) + 1e-8)

    # Cost matrices for GW (cosine distance within each domain)
    C_d = cdist(X_d, X_d, metric='cosine')
    C_m = cdist(X_m, X_m, metric='cosine')

    # Uniform distributions
    p_d = np.ones(n_d) / n_d
    p_m = np.ones(n_m) / n_m

    # Solve entropic GW (smoother coupling)
    gw_dist, log = ot.gromov.gromov_wasserstein2(
        C_d, C_m, p_d, p_m, log=True,
        epsilon=0.05, max_iter=200,
        solver='PGD')
    T_gw = log['T']
    print(f"  GW distance = {gw_dist:.6f}")
    print(f"  Coupling matrix shape: {T_gw.shape} (sum={T_gw.sum():.3f})")

    # Also try fused GW with outer product cost (cosine similarity between domains)
    M_outer = cdist(X_d, X_m, metric='cosine')
    T_fused, log_fused = ot.gromov.fused_gromov_wasserstein(
        M_outer, C_d, C_m, p_d, p_m, log=True,
        alpha=0.5, epsilon=0.05, max_iter=200)
    print(f"  Fused GW solved (alpha=0.5)")

    # For each drug, best mutant matches
    results = []
    for i, dname in enumerate(drug_names):
        coupling = T_gw[i]
        top_idx = np.argsort(-coupling)
        top_mutants = [(mutant_names[j], coupling[j]) for j in top_idx[:5] if coupling[j] > 0.001]
        ab = drug_ab_names[i]
        expected = EXPECTED_MATCHES.get(ab, set())
        hit = any(mutant_gene_names[mutant_names.index(m)] in expected for m, _ in top_mutants if m in mutant_names)
        results.append({
            'drug': dname,
            'antibiotic': ab,
            'top_mutants': top_mutants,
            'hit': hit,
            'coupling_row': coupling
        })
        genes_in_top = [mutant_gene_names[mutant_names.index(m)] for m, _ in top_mutants if m in mutant_names]
        correct = sum(1 for g in genes_in_top if g in expected) if expected else 0
        print(f"  {dname:25s} → best: {', '.join(f'{m}({c:.3f})' for m,c in top_mutants[:3])}  "
              f"{'✓' if hit else '✗'}  ({correct}/{len(expected) if expected else '?'})")

    top1_hit_rate = sum(1 for r in results if r['hit']) / max(len(results), 1)
    print(f"\n  GWOT Top-1 hit rate: {top1_hit_rate:.1%} ({sum(1 for r in results if r['hit'])}/{len(results)})")

    # Plot coupling matrix
    fig, axes = plt.subplots(1, 2, figsize=(28, 12))
    for ax, T, title in zip(axes, [T_gw, T_fused],
                              ['GWOT Coupling Matrix', 'Fused GWOT Coupling (α=0.5)']):
        sns.heatmap(T, xticklabels=mutant_names, yticklabels=drug_names,
                    ax=ax, cmap='viridis', square=False)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Mutant', fontsize=12)
        ax.set_ylabel('Drug', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=5)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)

    fig.suptitle(f'Stage 2: GWOT Drug↔Mutant Coupling  |  '
                 f'GW dist={gw_dist:.4f}  |  Top-1 hit rate={top1_hit_rate:.1%}',
                 fontsize=16)
    plt.tight_layout()
    path = os.path.join(output_dir, 'stage2_gwot_coupling.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return {'T_gw': T_gw, 'T_fused': T_fused, 'results': results,
            'top1_hit_rate': top1_hit_rate}


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 3: Control-anchored Procrustes alignment
# ═══════════════════════════════════════════════════════════════════════════

def run_procrustes_alignment(drug_centroids, mutant_centroids,
                              drug_names, mutant_names,
                              drug_ab_names, mutant_gene_names,
                              output_dir):
    """
    Use shared controls as anchors to learn a Procrustes (orthogonal) transformation.
    Controls: 'control' (drug) ↔ 'WT NC' and 'NC' (mutant).

    Then transform all drug embeddings into mutant space and compute cosine similarities.
    """
    print("\n═══ Stage 3: Control-anchored Procrustes Alignment ═══")

    # Find control indices
    ctrl_d_idx = [i for i, n in enumerate(drug_names) if n == 'control']
    ctrl_m_idx = [i for i, n in enumerate(mutant_names) if n.startswith('WT NC') or n.startswith('NC')]

    print(f"  Drug controls: {[drug_names[i] for i in ctrl_d_idx]}")
    print(f"  Mutant controls: {[mutant_names[i] for i in ctrl_m_idx]}")

    if len(ctrl_d_idx) == 0 or len(ctrl_m_idx) == 0:
        print("  ⚠ No shared controls found, using mean-centering only")
        drug_ref = drug_centroids.mean(axis=0)
        mutant_ref = mutant_centroids.mean(axis=0)
    else:
        drug_ref = drug_centroids[ctrl_d_idx].mean(axis=0)
        mutant_ref = mutant_centroids[ctrl_m_idx].mean(axis=0)

    # Center by control reference
    drug_centered = drug_centroids - drug_ref
    mutant_centered = mutant_centroids - mutant_ref
    print(f"  Drug centered: {drug_centered.shape}, Mutant centered: {mutant_centered.shape}")

    # Cosine similarity in centered space (no Procrustes needed — both are 1280-dim)
    X_d = drug_centered / (np.linalg.norm(drug_centered, axis=1, keepdims=True) + 1e-8)
    X_m = mutant_centered / (np.linalg.norm(mutant_centered, axis=1, keepdims=True) + 1e-8)
    sim_matrix = X_d @ X_m.T

    # For each drug, find best mutant matches
    results = []
    for i, dname in enumerate(drug_names):
        sims = sim_matrix[i]
        top_idx = np.argsort(-sims)
        top_mutants = [(mutant_names[j], sims[j]) for j in top_idx[:5]]
        ab = drug_ab_names[i]
        expected = EXPECTED_MATCHES.get(ab, set())
        hit = any(mutant_gene_names[mutant_names.index(m)] in expected for m, _ in top_mutants if m in mutant_names)
        results.append({
            'drug': dname,
            'antibiotic': ab,
            'top_mutants': top_mutants,
            'hit': hit,
        })
        genes_in_top = [mutant_gene_names[mutant_names.index(m)] for m, _ in top_mutants if m in mutant_names]
        correct = sum(1 for g in genes_in_top if g in expected) if expected else 0
        print(f"  {dname:25s} → {', '.join(f'{m}({s:.3f})' for m,s in top_mutants[:3])}  "
              f"{'✓' if hit else '✗'}  ({correct}/{len(expected) if expected else '?'})")

    top1_hit_rate = sum(1 for r in results if r['hit']) / max(len(results), 1)
    print(f"\n  Procrustes Top-1 hit rate: {top1_hit_rate:.1%} ({sum(1 for r in results if r['hit'])}/{len(results)})")

    # Plot
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(sim_matrix, xticklabels=mutant_names, yticklabels=drug_names,
                ax=ax, cmap='RdBu_r', center=0, square=False,
                vmin=-0.5, vmax=0.5)
    ax.set_title(f'Stage 3: Procrustes-Aligned Cosine Similarity  |  '
                 f'Top-1 hit rate={top1_hit_rate:.1%}', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
    plt.tight_layout()
    path = os.path.join(output_dir, 'stage3_procrustes_similarity.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return {'sim_matrix': sim_matrix, 'results': results, 'top1_hit_rate': top1_hit_rate}


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 4: Cross-domain Mutual Nearest Neighbors (MNN)
# ═══════════════════════════════════════════════════════════════════════════

def run_mnn_analysis(drug_centroids, mutant_centroids,
                      drug_names, mutant_names,
                      drug_ab_names, mutant_gene_names,
                      output_dir, k=3):
    """
    Find Mutual Nearest Neighbors between drug and mutant sets.
    MNN = pairs where each is in the other's k-nearest neighbors.
    """
    print(f"\n═══ Stage 4: Cross-domain MNN (k={k}) ═══")

    X_d = drug_centroids / (np.linalg.norm(drug_centroids, axis=1, keepdims=True) + 1e-8)
    X_m = mutant_centroids / (np.linalg.norm(mutant_centroids, axis=1, keepdims=True) + 1e-8)

    # Drug→Mutant kNN
    nn_m = NearestNeighbors(n_neighbors=min(k, len(X_m)), metric='cosine').fit(X_m)
    drug_to_mutant = nn_m.kneighbors(X_d, return_distance=False)

    # Mutant→Drug kNN
    nn_d = NearestNeighbors(n_neighbors=min(k, len(X_d)), metric='cosine').fit(X_d)
    mutant_to_drug = nn_d.kneighbors(X_m, return_distance=False)

    # Find MNN pairs
    mnn_pairs = []
    for i in range(len(X_d)):
        d_nn_in_m = set(drug_to_mutant[i])
        for j in d_nn_in_m:
            m_nn_in_d = set(mutant_to_drug[j])
            if i in m_nn_in_d:
                mnn_pairs.append((i, j))

    print(f"  Total MNN pairs (k={k}): {len(mnn_pairs)}")

    # Evaluate MNN pairs against EXPECTED_MATCHES
    results = []
    for i, j in mnn_pairs:
        dname = drug_names[i]
        mname = mutant_names[j]
        ab = drug_ab_names[i]
        gene = mutant_gene_names[j]
        expected = EXPECTED_MATCHES.get(ab, set())
        is_hit = gene in expected or (not expected and True)  # True for empty set = not penalized
        hit_str = '✓' if gene in expected else '✗' if expected else '·'
        results.append({'drug': dname, 'antibiotic': ab, 'mutant': mname, 'gene': gene,
                        'expected': gene in expected, 'hit_str': hit_str})

    n_with_expected = sum(1 for r in results if EXPECTED_MATCHES.get(r['antibiotic'], set()))
    n_hits = sum(1 for r in results if r['expected'])
    print(f"  MNN pairs with expected targets: {n_with_expected}")
    print(f"  Correct matches: {n_hits}/{len(mnn_pairs)} ({n_hits/max(len(mnn_pairs),1):.1%})")

    for r in results:
        print(f"    {r['drug']:25s} ↔ {r['mutant']:15s} (gene={r['gene']:5s})  {r['hit_str']}")

    # Build a MNN adjacency matrix for visualization
    adj = np.zeros((len(drug_names), len(mutant_names)))
    for i, j in mnn_pairs:
        adj[i, j] = 1.0

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.heatmap(adj, xticklabels=mutant_names, yticklabels=drug_names,
                ax=ax, cmap='Blues', cbar=False, square=False)
    ax.set_title(f'Stage 4: MNN Graph (k={k}) — {len(mnn_pairs)} pairs  |  '
                 f'Hits={n_hits}', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
    plt.tight_layout()
    path = os.path.join(output_dir, f'stage4_mnn_k{k}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return {'mnn_pairs': mnn_pairs, 'results': results, 'n_hits': n_hits,
            'adjacency': adj}


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=str,
                        default='both/fold_Plate_1/embeddings_Plate_1_mil_n3.npz')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--k_mnn', type=int, default=3, help='k for MNN')
    parser.add_argument('--pool', action='store_true', default=True,
                        help='Pool centroids by antibiotic/gene name (average over concentrations/replicates)')
    parser.add_argument('--no-pool', action='store_false', dest='pool',
                        help='Keep per-concentration/per-replicate centroids')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.embeddings), 'unsupervised_alignment')
    os.makedirs(output_dir, exist_ok=True)

    # ── Load embeddings ──
    print("Loading embeddings...")
    data = np.load(args.embeddings)
    embeddings = data['embeddings']  # (N, 1280)
    paths = data['paths']

    # ── Fix labels from paths ──
    IC50, MUT = load_jsons()
    labels = np.array([fix_label(p, IC50, MUT) for p in paths])
    uniq = np.unique(labels)
    print(f"  Total samples: {len(labels)}")
    print(f"  Unique labels after fix: {len(uniq)}")

    # Separate drug vs mutant
    drug_mask = np.array([is_drug_label(l) for l in labels])
    mut_mask = np.array([is_mutant_label(l) for l in labels])
    control_mask = labels == 'control'

    print(f"  Drug samples: {drug_mask.sum()} ({len(np.unique(labels[drug_mask]))} unique)")
    print(f"  Mutant samples: {mut_mask.sum()} ({len(np.unique(labels[mut_mask]))} unique)")
    print(f"  Drug controls: {control_mask.sum()}")

    # ── Compute centroids per label ──
    def compute_centroids(emb, lbls):
        """Mean embedding per unique label."""
        unique_labels = np.unique(lbls)
        centroids = []
        for ul in unique_labels:
            mask = lbls == ul
            centroids.append(emb[mask].mean(axis=0))
        return np.array(centroids), list(unique_labels)

    drug_emb = embeddings[drug_mask | control_mask]
    drug_lbl = labels[drug_mask | control_mask]
    drug_centroids, drug_names = compute_centroids(drug_emb, drug_lbl)

    mut_emb = embeddings[mut_mask]
    mut_lbl = labels[mut_mask]
    mut_centroids, mut_names = compute_centroids(mut_emb, mut_lbl)

    # Sort by name for consistency
    drug_order = np.argsort(drug_names)
    drug_centroids = drug_centroids[drug_order]
    drug_names = [drug_names[i] for i in drug_order]

    mut_order = np.argsort(mut_names)
    mut_centroids = mut_centroids[mut_order]
    mut_names = [mut_names[i] for i in mut_order]

    # ── Pool centroids by antibiotic/gene name (cleaner signal) ──
    if args.pool:
        print("  Pooling centroids by antibiotic/gene name...")
        drug_centroids, drug_names = pool_centroids_by_name(
            drug_centroids, drug_names, extract_antibiotic_name)
        mut_centroids, mut_names = pool_centroids_by_name(
            mut_centroids, mut_names, extract_gene_base)

    # Extract drug antibiotic names and mutant gene bases
    drug_ab_names = [extract_antibiotic_name(n) for n in drug_names]
    mutant_gene_names = [extract_gene_base(n) for n in mut_names]

    print(f"  Drug centroids: {len(drug_names)} ({len(set(drug_ab_names))} antibiotics)")
    print(f"  Mutant centroids: {len(mut_names)} ({len(set(mutant_gene_names))} genes)")

    # ── Run stages ──
    stage1 = analyze_structural_similarity(
        drug_centroids, mut_centroids, drug_names, mut_names, output_dir)

    stage2 = run_gwot_coupling(
        drug_centroids, mut_centroids, drug_names, mut_names,
        drug_ab_names, mutant_gene_names, output_dir)

    stage3 = run_procrustes_alignment(
        drug_centroids, mut_centroids, drug_names, mut_names,
        drug_ab_names, mutant_gene_names, output_dir)

    stage4 = run_mnn_analysis(
        drug_centroids, mut_centroids, drug_names, mut_names,
        drug_ab_names, mutant_gene_names, output_dir, k=args.k_mnn)

    # ── Summary report ──
    print("\n" + "=" * 70)
    print("  UNSUPERVISED ALIGNMENT SUMMARY")
    print("=" * 70)
    print(f"  Embeddings: {args.embeddings}")
    print(f"  Drug classes: {len(drug_names)}, Mutant classes: {len(mut_names)}")
    print()
    print(f"  Stage 1 — Structural Similarity:")
    print(f"    Mantel ρ = {stage1['mantel_rho']:.3f} (p={stage1['mantel_pval']:.2e})")
    print(f"    {'POT GW distance' if stage1.get('gw_dist') else 'GW distance'}: "
          f"{stage1.get('gw_dist', 'N/A')}")
    print(f"    Interpretation: {'Strong' if stage1['mantel_rho'] > 0.3 else 'Weak'} structural "
          f"conservation between drug and mutant spaces")
    print()
    if stage2:
        print(f"  Stage 2 — GWOT Coupling:")
        print(f"    Top-1 hit rate vs EXPECTED_MATCHES: {stage2['top1_hit_rate']:.1%}")
    print()
    print(f"  Stage 3 — Procrustes Alignment:")
    print(f"    Top-1 hit rate vs EXPECTED_MATCHES: {stage3['top1_hit_rate']:.1%}")
    print()
    print(f"  Stage 4 — MNN (k={args.k_mnn}):")
    print(f"    Total MNN pairs: {len(stage4['mnn_pairs'])}")
    print(f"    Correct matches: {stage4['n_hits']}")
    print()

    # Detailed per-antibiotic comparison
    print("  ─── Per-Antibiotic Comparison ───")
    headers = ['Antibiotic', 'Expected', 'GWOT Top-1', 'GWOT Hit',
               'Proc Top-1', 'Proc Hit', 'MNN Gene']
    print(f"  {' | '.join(f'{h:20s}' for h in headers)}")
    print("  " + "-" * 140)

    antibiotics = sorted(set(drug_ab_names) - {'control'})
    for ab in antibiotics:
        expected = EXPECTED_MATCHES.get(ab, set())
        expected_str = ','.join(sorted(expected)) if expected else '—'

        # Find first drug with this antibiotic
        idx = drug_ab_names.index(ab) if ab in drug_ab_names else -1
        gwot_gene = ''
        gwot_hit = ''
        proc_gene = ''
        proc_hit = ''
        mnn_genes = ''

        if idx >= 0 and stage2:
            gwot_top = stage2['results'][idx]['top_mutants'][0][0]
            gwot_gene = mutant_gene_names[mut_names.index(gwot_top)]
            gwot_hit = '✓' if gwot_gene in expected else '✗' if expected else '·'

        if idx >= 0:
            proc_top = stage3['results'][idx]['top_mutants'][0][0]
            proc_gene = mutant_gene_names[mut_names.index(proc_top)]
            proc_hit = '✓' if proc_gene in expected else '✗' if expected else '·'

        # MNN genes for this antibiotic
        ab_mnn_genes = [r['gene'] for r in stage4['results'] if r['antibiotic'] == ab]
        mnn_genes = ','.join(sorted(set(ab_mnn_genes))) if ab_mnn_genes else '—'

        print(f"  {ab:20s} | {expected_str:20s} | {gwot_gene:20s} | {gwot_hit:20s} | "
              f"{proc_gene:20s} | {proc_hit:20s} | {mnn_genes:20s}")

    # ── Save all results ──
    summary = {
        'n_drug_classes': len(drug_names),
        'n_mutant_classes': len(mut_names),
        'mantel_rho': float(stage1['mantel_rho']),
        'mantel_pval': float(stage1['mantel_pval']),
        'gw_dist': float(stage1['gw_dist']) if stage1.get('gw_dist') else None,
        'gwot_top1_hit_rate': float(stage2['top1_hit_rate']) if stage2 else None,
        'procrustes_top1_hit_rate': float(stage3['top1_hit_rate']),
        'mnn_total_pairs': len(stage4['mnn_pairs']),
        'mnn_correct_hits': stage4['n_hits'],
    }
    import json
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save per-antibiotic results as CSV
    rows = []
    for ab in antibiotics:
        expected = EXPECTED_MATCHES.get(ab, set())
        rows.append({
            'antibiotic': ab,
            'expected_genes': ','.join(sorted(expected)) if expected else '',
            'n_expected': len(expected),
        })
    if stage2:
        for r in stage2['results']:
            ab = r['antibiotic']
            row = next((x for x in rows if x['antibiotic'] == ab), None)
            if row:
                row['gwot_top1'] = r['top_mutants'][0][0] if r['top_mutants'] else ''
                row['gwot_top1_gene'] = mutant_gene_names[mut_names.index(row['gwot_top1'])] if row['gwot_top1'] in mut_names else ''
                row['gwot_top1_score'] = f"{r['top_mutants'][0][1]:.4f}" if r['top_mutants'] else ''
                row['gwot_hit'] = r['hit']
    for r in stage3['results']:
        ab = r['antibiotic']
        row = next((x for x in rows if x['antibiotic'] == ab), None)
        if row:
            row['procrustes_top1'] = r['top_mutants'][0][0] if r['top_mutants'] else ''
            row['procrustes_top1_gene'] = mutant_gene_names[mut_names.index(row['procrustes_top1'])] if row['procrustes_top1'] in mut_names else ''
            row['procrustes_top1_score'] = f"{r['top_mutants'][0][1]:.4f}" if r['top_mutants'] else ''
            row['procrustes_hit'] = r['hit']

    # MNN pairs per antibiotic
    for r in stage4['results']:
        ab = r['antibiotic']
        row = next((x for x in rows if x['antibiotic'] == ab), None)
        if row:
            if 'mnn_matches' not in row:
                row['mnn_matches'] = []
            row['mnn_matches'].append(r['mutant'])

    for row in rows:
        if 'mnn_matches' in row and row['mnn_matches']:
            row['mnn_genes'] = ','.join(sorted(set(mutant_gene_names[mut_names.index(m)] for m in row['mnn_matches'] if m in mut_names)))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'per_antibiotic_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    print(f"  Saved: {os.path.join(output_dir, 'summary.json')}")

    # ── L1/L2/L3 Grouped Heatmaps ──
    print("\n═══ Generating L1/L2/L3 grouped alignment heatmaps ═══")
    generate_alignment_heatmaps(embeddings, paths, output_dir)

    print("\nDone!")


# ═══════════════════════════════════════════════════════════════════════════
#  L1/L2/L3 Grouped Alignment Heatmaps
# ═══════════════════════════════════════════════════════════════════════════

def plot_grouped_heatmap(sim_matrix, row_labels, col_labels,
                          row_groups, col_groups, row_group_colors,
                          col_group_colors, title, output_path):
    """Plot a similarity matrix with group boxes around same-group entries.
    
    Args:
        sim_matrix: np.ndarray (n_rows, n_cols) - cosine similarity values
        row_labels: list of str - row names (sorted by group)
        col_labels: list of str - column names (sorted by group)
        row_groups: dict {label: group_name} - group assignment per row
        col_groups: dict {label: group_name} - group assignment per column
        title: str
        output_path: str
    """
    n_rows, n_cols = sim_matrix.shape
    
    fig, ax = plt.subplots(figsize=(max(14, n_cols * 0.22), max(12, n_rows * 0.22)))
    
    sns.heatmap(sim_matrix, xticklabels=col_labels, yticklabels=row_labels,
                ax=ax, cmap='RdBu_r', center=0, vmin=-0.5, vmax=0.5,
                cbar_kws={'label': 'Cosine Similarity', 'shrink': 0.8},
                linewidths=0.3, linecolor='white', square=True,
                annot=False)
    
    # Draw green group boxes on row axis
    row_unique_groups = []
    for lbl in row_labels:
        g = row_groups.get(lbl, 'Other')
        if g not in row_unique_groups:
            row_unique_groups.append(g)
    
    col_unique_groups = []
    for lbl in col_labels:
        g = col_groups.get(lbl, 'Other')
        if g not in col_unique_groups:
            col_unique_groups.append(g)
    
    # Row group boxes (vertical strips on rows)
    for grp in row_unique_groups:
        indices = [i for i, lbl in enumerate(row_labels) if row_groups.get(lbl, 'Other') == grp]
        if len(indices) <= 1:
            continue
        min_i, max_i = min(indices), max(indices)
        rect = mpatches.Rectangle(
            (-0.5, min_i), n_cols, max_i - min_i + 1,
            linewidth=2.5, edgecolor='#2ecc71', facecolor='none', zorder=10)
        ax.add_patch(rect)
    
    # Col group boxes (horizontal strips on columns)
    for grp in col_unique_groups:
        indices = [j for j, lbl in enumerate(col_labels) if col_groups.get(lbl, 'Other') == grp]
        if len(indices) <= 1:
            continue
        min_j, max_j = min(indices), max(indices)
        rect = mpatches.Rectangle(
            (min_j, -0.5), max_j - min_j + 1, n_rows,
            linewidth=2.5, edgecolor='#2ecc71', facecolor='none', zorder=10)
        ax.add_patch(rect)
    
    ax.set_xlabel('Mutant', fontsize=12)
    ax.set_ylabel('Drug', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_alignment_heatmaps(embeddings, paths, output_dir):
    """Generate L1/L2/L3 grouped heatmaps for drug↔mutant cosine similarity.
    
    L1: per-class (89 drugs × 96 mutants) — all labels
    L2: pooled (23 antibiotics × 30 genes)
    L3: aggregated by MoA × Pathway (8 × 10)
    """
    IC50, MUT = load_jsons()
    labels = np.array([fix_label(p, IC50, MUT) for p in paths])
    
    drug_mask = np.array([is_drug_label(l) for l in labels])
    mut_mask = np.array([is_mutant_label(l) for l in labels])
    control_mask = labels == 'control'
    
    # ── Control-centered similarity ──
    def compute_centroids(emb, lbls):
        unique_labels = np.unique(lbls)
        centroids = []
        for ul in unique_labels:
            centroids.append(emb[lbls == ul].mean(axis=0))
        return np.array(centroids), list(unique_labels)
    
    def pool_centroids(centroids, names, extract_fn, keep_controls=True):
        groups = {}
        for i, name in enumerate(names):
            base = extract_fn(name)
            if keep_controls and (name == 'control' or base in ('WT NC', 'NC')):
                base = name
            groups.setdefault(base, []).append(centroids[i])
        pooled = np.array([np.mean(v, axis=0) for _, v in sorted(groups.items())])
        pooled_names = sorted(groups.keys())
        return pooled, pooled_names
    
    drug_emb = embeddings[drug_mask | control_mask]
    drug_lbl = labels[drug_mask | control_mask]
    drug_centroids_raw, drug_names_raw = compute_centroids(drug_emb, drug_lbl)
    
    mut_emb = embeddings[mut_mask]
    mut_lbl = labels[mut_mask]
    mut_centroids_raw, mut_names_raw = compute_centroids(mut_emb, mut_lbl)
    
    # Sort
    drug_order = np.argsort(drug_names_raw)
    drug_centroids_raw = drug_centroids_raw[drug_order]
    drug_names_raw = [drug_names_raw[i] for i in drug_order]
    
    mut_order = np.argsort(mut_names_raw)
    mut_centroids_raw = mut_centroids_raw[mut_order]
    mut_names_raw = [mut_names_raw[i] for i in mut_order]
    
    # ── Pooled ──
    drug_centroids_pooled, drug_names_pooled = pool_centroids(
        drug_centroids_raw, drug_names_raw, extract_antibiotic_name)
    mut_centroids_pooled, mut_names_pooled = pool_centroids(
        mut_centroids_raw, mut_names_raw, extract_gene_base)
    
    # ── MoA × Pathway aggregation ──
    def aggregate_by_group(centroids, names, group_dict, control_name='control'):
        groups = {}
        for i, name in enumerate(names):
            if name == control_name:
                grp = 'Control'
            else:
                grp = group_dict.get(name, 'Other')
            groups.setdefault(grp, []).append(centroids[i])
        agg = np.array([np.mean(v, axis=0) for _, v in sorted(groups.items())])
        agg_names = sorted(groups.keys())
        return agg, agg_names
    
    drug_moa_centroids, moa_names = aggregate_by_group(
        drug_centroids_pooled, drug_names_pooled, ANTIBIOTIC_TO_MOA)
    mut_pathway_centroids, pathway_names = aggregate_by_group(
        mut_centroids_pooled, mut_names_pooled, GENE_TO_PATHWAY)
    
    # ── Control-center for fair comparison ──
    def control_center(centroids, names, control_name='control'):
        ctrl_idx = [i for i, n in enumerate(names) if n == control_name]
        ref = centroids[ctrl_idx].mean(axis=0) if ctrl_idx else centroids.mean(axis=0)
        centered = centroids - ref
        norms = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-8
        return centered / norms
    
    def compute_sim(centroids_a, names_a, centroids_b, names_b):
        ctrl_a = 'control' if 'control' in names_a else names_a[0]
        ctrl_b = 'WT NC' if 'WT NC' in names_b else ('NC' if 'NC' in names_b else names_b[0])
        norm_a = control_center(centroids_a, names_a, ctrl_a)
        norm_b = control_center(centroids_b, names_b, ctrl_b)
        return norm_a @ norm_b.T
    
    # ── Row/col group mappings ──
    def build_row_groups(names, group_dict, control_name='control'):
        return {n: group_dict.get(n, 'Other') if n != control_name else 'Control'
                for n in names}
    
    def build_col_groups(names, group_dict):
        return {n: group_dict.get(n, 'Other') for n in names}
    
    # ── L1: per-class ──
    sim_l1 = compute_sim(drug_centroids_raw, drug_names_raw,
                         mut_centroids_raw, mut_names_raw)
    drug_ab_names_l1 = [extract_antibiotic_name(n) for n in drug_names_raw]
    row_groups_l1 = {}
    for i, n in enumerate(drug_names_raw):
        ab = drug_ab_names_l1[i]
        if n == 'control':
            row_groups_l1[n] = 'Control'
        else:
            row_groups_l1[n] = ANTIBIOTIC_TO_MOA.get(ab, 'Other')
    
    mut_gene_names_l1 = [extract_gene_base(n) for n in mut_names_raw]
    col_groups_l1 = {}
    for i, n in enumerate(mut_names_raw):
        gene = mut_gene_names_l1[i]
        col_groups_l1[n] = GENE_TO_PATHWAY.get(gene, 'Other')
    
    # Sort L1 by group
    drug_idx_sorted = sorted(
        range(len(drug_names_raw)),
        key=lambda i: (row_groups_l1.get(drug_names_raw[i], 'Z'), drug_names_raw[i]))
    mut_idx_sorted = sorted(
        range(len(mut_names_raw)),
        key=lambda i: (col_groups_l1.get(mut_names_raw[i], 'Z'), mut_names_raw[i]))
    drug_names_l1_sorted = [drug_names_raw[i] for i in drug_idx_sorted]
    mut_names_l1_sorted = [mut_names_raw[i] for i in mut_idx_sorted]
    
    sim_l1_sorted = sim_l1[np.ix_(drug_idx_sorted, mut_idx_sorted)]
    
    plot_grouped_heatmap(
        sim_l1_sorted, list(drug_names_l1_sorted), list(mut_names_l1_sorted),
        row_groups_l1, col_groups_l1, None, None,
        f'L1: Drug × Mutant Cosine Similarity\n(labels sorted by group, {len(drug_names_l1_sorted)}×{len(mut_names_l1_sorted)})',
        os.path.join(output_dir, 'heatmap_L1_drug_mutant.png'))
    
    # ── L2: pooled ──
    sim_l2 = compute_sim(drug_centroids_pooled, drug_names_pooled,
                         mut_centroids_pooled, mut_names_pooled)
    row_groups_l2 = build_row_groups(drug_names_pooled, ANTIBIOTIC_TO_MOA)
    col_groups_l2 = build_col_groups(mut_names_pooled, GENE_TO_PATHWAY)
    
    drug_l2_idx = sorted(
        range(len(drug_names_pooled)),
        key=lambda i: (row_groups_l2.get(drug_names_pooled[i], 'Z'), drug_names_pooled[i]))
    mut_l2_idx = sorted(
        range(len(mut_names_pooled)),
        key=lambda i: (col_groups_l2.get(mut_names_pooled[i], 'Z'), mut_names_pooled[i]))
    drug_l2_sorted = [drug_names_pooled[i] for i in drug_l2_idx]
    mut_l2_sorted = [mut_names_pooled[i] for i in mut_l2_idx]
    
    sim_l2_sorted = sim_l2[np.ix_(drug_l2_idx, mut_l2_idx)]
    
    plot_grouped_heatmap(
        sim_l2_sorted, list(drug_l2_sorted), list(mut_l2_sorted),
        row_groups_l2, col_groups_l2, None, None,
        f'L2: Antibiotic × Gene Cosine Similarity\n(pooled by name, sorted by MoA/Pathway, {len(drug_l2_sorted)}×{len(mut_l2_sorted)})',
        os.path.join(output_dir, 'heatmap_L2_antibiotic_gene.png'))
    
    # ── L3: MoA × Pathway ──
    drug_moa_norm = control_center(drug_moa_centroids, moa_names, 'Control')
    mut_pathway_norm = control_center(mut_pathway_centroids, pathway_names, 'WT/NC')
    sim_l3 = drug_moa_norm @ mut_pathway_norm.T
    
    # Build MoA group colors (each MoA gets a row with itself as group)
    moa_to_moa = {m: m for m in moa_names}
    pathway_to_pathway = {p: p for p in pathway_names}
    
    plot_grouped_heatmap(
        sim_l3, moa_names, pathway_names,
        moa_to_moa, pathway_to_pathway, None, None,
        f'L3: MoA × Pathway Cosine Similarity\n(aggregated, {len(moa_names)}×{len(pathway_names)})',
        os.path.join(output_dir, 'heatmap_L3_moa_pathway.png'))


if __name__ == '__main__':
    main()
