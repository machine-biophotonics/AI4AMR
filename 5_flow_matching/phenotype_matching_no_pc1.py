#!/usr/bin/env python3
"""Phenotype matching on PC1-removed features (drug-mutant domain effect removed)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.preprocessing import StandardScaler
import warnings, os
warnings.filterwarnings('ignore')

DATA = "/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/5_flow_matching/latent_analysis_pacmap"
PC1_DIR = "/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/5_flow_matching/interpretable_directions/pc1_removed"
OUT = os.path.join(DATA, "phenotype_matching")
os.makedirs(OUT, exist_ok=True)

print("Loading PC1-removed features...")
feats = np.load(os.path.join(PC1_DIR, "feats_no_pc1.npy")).astype(np.float64)
labels = np.load(os.path.join(DATA, "labels.npy"))
class_names = np.load(os.path.join(DATA, "class_names.npy"), allow_pickle=True)
class_names = [str(n) for n in class_names]

scaler = StandardScaler()
feats_scaled = scaler.fit_transform(feats)

# ── Mutant guide 1 ──
mutant_guide1_names = sorted(set(
    n for n in class_names
    if n.endswith('_1') and not n.startswith(('NC_', 'WT NC_', 'control'))
))
mutant_guide1_indices = [class_names.index(n) for n in mutant_guide1_names]
mutant_guide1_genes = [n.replace('_1', '') for n in mutant_guide1_names]

expected = {
    'Cefsulodin': ['mrcA', 'mrcB'],
    'Penicillin': ['mrcA', 'mrcB', 'ftsI'],
    'Sulbactam': ['mrcA', 'mrcB', 'ftsI'],
    'Avibactam': [],
    'Mecillinam': ['mrdA'],
    'Meropenem': ['mrdA', 'ftsI', 'mrcA', 'mrcB'],
    'Clavulanic_Acid': [],
    'Relebactam': [],
    'Aztreonam': ['ftsI'],
    'Cefepim': ['ftsI', 'mrcA', 'mrcB', 'mrdA'],
    'Ceftriaxone': ['ftsI', 'mrcA', 'mrcB'],
    'Chloramphenicol': ['rplA', 'rplC'],
    'Clarithromycin': ['rplA', 'rplC'],
    'Doxicyclin': ['rpsA', 'rpsL'],
    'Kanamycin': ['rpsA', 'rpsL'],
    'Ciprofloxacin': ['gyrA', 'gyrB', 'parC', 'parE'],
    'Levofloxacin': ['gyrA', 'gyrB', 'parC', 'parE'],
    'Norfloxacin': ['gyrA', 'gyrB', 'parC', 'parE'],
    'Rifampicin': ['rpoA', 'rpoB'],
    'Trimethoprim': ['folA', 'folP'],
    'Colistin': ['lpxA', 'lpxC', 'lptA', 'lptC'],
    'Polymyxin_B': ['lpxA', 'lpxC', 'lptA', 'lptC'],
}

def class_mean_feats(feats_scaled, labels, class_indices):
    return np.array([feats_scaled[labels == ci].mean(axis=0) for ci in class_indices])

def cosine_sim(A, B):
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A_norm @ B_norm.T

def sliced_wasserstein_matrix(feats, labels, drug_idx, mut_idx, num_proj=200):
    n_drugs, n_muts = len(drug_idx), len(mut_idx)
    d = feats.shape[1]
    rng = np.random.RandomState(42)
    proj = rng.randn(num_proj, d)
    proj = proj / np.linalg.norm(proj, axis=1, keepdims=True)
    cache = {}
    for ci in set(list(drug_idx) + list(mut_idx)):
        X = feats[labels == ci]
        cache[ci] = np.sort(X @ proj.T, axis=0)
    D = np.zeros((n_drugs, n_muts))
    for i, di in enumerate(drug_idx):
        for j, mj in enumerate(mut_idx):
            D[i, j] = np.sqrt(np.mean((cache[di] - cache[mj]) ** 2))
    return D

concentrations = ['0.25x', '0.5x', '1x', '2x']
conc_labels = ['0.25×', '0.5×', '1×', '2×']

def plot_and_stats(feats_scaled, suffix, title_suffix):
    mutant_means = class_mean_feats(feats_scaled, labels, mutant_guide1_indices)
    
    # Cosine grid
    fig, axes = plt.subplots(2, 2, figsize=(30, 24))
    fig.suptitle(f'Cosine Similarity (PC1-removed) — Drug vs Mutant Guide 1{title_suffix}\nRed box = expected CRISPRi gene match', fontsize=18, y=0.98)
    
    cos_stats = {}
    for idx, (conc, clabel) in enumerate(zip(concentrations, conc_labels)):
        ax = axes[idx // 2][idx % 2]
        drug_names = sorted(set(n for n in class_names if n.endswith(f'_{conc}')))
        drug_indices = [class_names.index(n) for n in drug_names]
        drug_labels = [n.replace(f'_{conc}', '') for n in drug_names]
        drug_means = class_mean_feats(feats_scaled, labels, drug_indices)
        sim = cosine_sim(drug_means, mutant_means)
        
        red_boxes = []
        for di, dname in enumerate(drug_labels):
            if dname in expected:
                for gene in expected[dname]:
                    for mj, mgene in enumerate(mutant_guide1_genes):
                        if mgene == gene:
                            red_boxes.append((di, mj))
        
        im = ax.imshow(sim, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        for di, mj in red_boxes:
            ax.add_patch(Rectangle((mj - 0.5, di - 0.5), 1, 1, linewidth=2.5, edgecolor='red', facecolor='none', linestyle='-'))
        ax.set_xticks(range(len(mutant_guide1_genes)))
        ax.set_yticks(range(len(drug_labels)))
        ax.set_xticklabels(mutant_guide1_genes, rotation=90, fontsize=6)
        ax.set_yticklabels(drug_labels, fontsize=8)
        ax.set_title(f'{clabel}', fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)
        
        # Stats
        total_expected = 0
        matched_top1 = 0
        matched_top3 = 0
        matched_top5 = 0
        for di, dname in enumerate(drug_labels):
            if dname not in expected or not expected[dname]:
                continue
            genes = expected[dname]
            order = np.argsort(-sim[di])
            top1_genes = [mutant_guide1_genes[order[0]]]
            top3_genes = [mutant_guide1_genes[o] for o in order[:3]]
            top5_genes = [mutant_guide1_genes[o] for o in order[:5]]
            for g in genes:
                total_expected += 1
                if g in top1_genes: matched_top1 += 1
                if g in top3_genes: matched_top3 += 1
                if g in top5_genes: matched_top5 += 1
        cos_stats[conc] = (total_expected, matched_top1, matched_top3, matched_top5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUT, f'cosine_similarity_grid_{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Wasserstein grid
    print(f"  Wasserstein for {suffix}...")
    fig, axes = plt.subplots(2, 2, figsize=(30, 24))
    fig.suptitle(f'Wasserstein Distance (PC1-removed) — Drug vs Mutant Guide 1{title_suffix}\nRed box = expected CRISPRi match (lower = better)', fontsize=18, y=0.98)
    
    wass_stats = {}
    for idx, (conc, clabel) in enumerate(zip(concentrations, conc_labels)):
        ax = axes[idx // 2][idx % 2]
        drug_names = sorted(set(n for n in class_names if n.endswith(f'_{conc}')))
        drug_indices = [class_names.index(n) for n in drug_names]
        drug_labels = [n.replace(f'_{conc}', '') for n in drug_names]
        wass = sliced_wasserstein_matrix(feats_scaled, labels, drug_indices, mutant_guide1_indices)
        wass_norm = wass / wass.max()
        
        red_boxes = []
        for di, dname in enumerate(drug_labels):
            if dname in expected:
                for gene in expected[dname]:
                    for mj, mgene in enumerate(mutant_guide1_genes):
                        if mgene == gene:
                            red_boxes.append((di, mj))
        
        im = ax.imshow(wass_norm, cmap='viridis_r', aspect='auto', vmin=0, vmax=1)
        for di, mj in red_boxes:
            ax.add_patch(Rectangle((mj - 0.5, di - 0.5), 1, 1, linewidth=2.5, edgecolor='red', facecolor='none', linestyle='-'))
        ax.set_xticks(range(len(mutant_guide1_genes)))
        ax.set_yticks(range(len(drug_labels)))
        ax.set_xticklabels(mutant_guide1_genes, rotation=90, fontsize=6)
        ax.set_yticklabels(drug_labels, fontsize=8)
        ax.set_title(f'{clabel}', fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)
        
        total_expected = 0
        matched_top1 = 0
        matched_top3 = 0
        matched_top5 = 0
        for di, dname in enumerate(drug_labels):
            if dname not in expected or not expected[dname]:
                continue
            genes = expected[dname]
            order = np.argsort(wass[di])
            top1_genes = [mutant_guide1_genes[order[0]]]
            top3_genes = [mutant_guide1_genes[o] for o in order[:3]]
            top5_genes = [mutant_guide1_genes[o] for o in order[:5]]
            for g in genes:
                total_expected += 1
                if g in top1_genes: matched_top1 += 1
                if g in top3_genes: matched_top3 += 1
                if g in top5_genes: matched_top5 += 1
        wass_stats[conc] = (total_expected, matched_top1, matched_top3, matched_top5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUT, f'wasserstein_grid_{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return cos_stats, wass_stats

print("\n=== PC1-REMOVED FEATURES ===")
cos_stats, wass_stats = plot_and_stats(feats_scaled, 'no_pc1', '')

print("\n--- COSINE SIMILARITY (PC1 removed) ---")
for conc in concentrations:
    tot, t1, t3, t5 = cos_stats[conc]
    print(f"  {conc}: {t1}/{tot} top-1, {t3}/{tot} top-3, {t5}/{tot} top-5")

print("\n--- WASSERSTEIN (PC1 removed) ---")
for conc in concentrations:
    tot, t1, t3, t5 = wass_stats[conc]
    print(f"  {conc}: {t1}/{tot} top-1, {t3}/{tot} top-3, {t5}/{tot} top-5")

print(f"\nDone -> {OUT}/")
