#!/usr/bin/env python3
"""
Quantitative Domain Shift Analysis: Mutants vs Drugs

Computes multiple metrics to quantify how different the embedding 
distributions are between mutant and drug domains.

Metrics:
1. Mean Cosine Similarity (cross-domain)
2. Maximum Mean Discrepancy (MMD)
3. Wasserstein Distance (Earth Mover's)
4. CORAL (second-order statistics alignment)
5. KL Divergence (approximated)
6. Jensen-Shannon Divergence
7. Hellinger Distance
8. Energy Distance
9. Within-group vs between-group ratio
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, entropy
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")


def well_to_row_col(well_id: str):
    match = well_id.replace("Well", "")
    return match[0], int(match[1:])


def well_to_wellname(well_id: str):
    return well_id.replace("Well", "")


def load_embeddings_by_well(embeddings_dir: str):
    """Load embeddings and compute mean per well"""
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


def compute_mmd(x, y, kernel='rbf'):
    """Maximum Mean Discrepancy using RBF kernel"""
    # Using linear MMD (faster)
    mean_x = np.mean(x, axis=0)
    mean_y = np.mean(y, axis=0)
    return np.linalg.norm(mean_x - mean_y)


def compute_energy_distance(x, y):
    """Energy distance between two distributions"""
    n, m = len(x), len(y)
    
    # Within-domain distances
    within_x = cdist(x, x, 'cosine').sum() / (n * n)
    within_y = cdist(y, y, 'cosine').sum() / (m * m)
    
    # Between-domain distances
    between = cdist(x, y, 'cosine').sum() / (n * m)
    
    # Energy distance formula
    energy = 2 * between - within_x - within_y
    return energy


def compute_coral_loss(x, y):
    """CORAL: difference in second-order statistics (covariance)"""
    # Center the data
    x_centered = x - np.mean(x, axis=0)
    y_centered = y - np.mean(y, axis=0)
    
    # Compute covariance
    cov_x = np.cov(x_centered, rowvar=False)
    cov_y = np.cov(y_centered, rowvar=False)
    
    # CORAL loss = squared Frobenius norm of difference
    coral = np.linalg.norm(cov_x - cov_y, ord='fro') ** 2
    return coral


def compute_kl_divergence_approx(x, y, n_bins=20):
    """Approximate KL divergence using histogram binning"""
    # Flatten and compute histograms
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # Common bins
    min_val = min(x_flat.min(), y_flat.min())
    max_val = max(x_flat.max(), y_flat.max())
    bins = np.linspace(min_val, max_val, n_bins + 1)
    
    hist_x, _ = np.histogram(x_flat, bins=bins, density=True)
    hist_y, _ = np.histogram(y_flat, bins=bins, density=True)
    
    # Normalize
    hist_x = hist_x + 1e-10
    hist_y = hist_y + 1e-10
    hist_x = hist_x / hist_x.sum()
    hist_y = hist_y / hist_y.sum()
    
    # KL divergence
    kl = entropy(hist_x, hist_y)
    return kl


def compute_jensen_shannon(x, y, n_bins=20):
    """Jensen-Shannon divergence (symmetric)"""
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    min_val = min(x_flat.min(), y_flat.min())
    max_val = max(x_flat.max(), y_flat.max())
    bins = np.linspace(min_val, max_val, n_bins + 1)
    
    hist_x, _ = np.histogram(x_flat, bins=bins, density=True)
    hist_y, _ = np.histogram(y_flat, bins=bins, density=True)
    
    hist_x = hist_x + 1e-10
    hist_y = hist_y + 1e-10
    hist_x = hist_x / hist_x.sum()
    hist_y = hist_y / hist_y.sum()
    
    m = 0.5 * (hist_x + hist_y)
    js = 0.5 * (entropy(hist_x, m) + entropy(hist_y, m))
    return js


def compute_hellinger(x, y, n_bins=20):
    """Hellinger distance between distributions"""
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    min_val = min(x_flat.min(), y_flat.min())
    max_val = max(x_flat.max(), y_flat.max())
    bins = np.linspace(min_val, max_val, n_bins + 1)
    
    hist_x, _ = np.histogram(x_flat, bins=bins, density=True)
    hist_y, _ = np.histogram(y_flat, bins=bins, density=True)
    
    hist_x = np.sqrt(hist_x + 1e-10)
    hist_y = np.sqrt(hist_y + 1e-10)
    
    hellinger = np.linalg.norm(hist_x - hist_y) / np.sqrt(2)
    return hellinger


def compute_wasserstein_1d(embeddings1, embeddings2):
    """1D Wasserstein distance (average over dimensions)"""
    wass_dims = []
    for d in range(embeddings1.shape[1]):
        w = wasserstein_distance(embeddings1[:, d], embeddings2[:, d])
        wass_dims.append(w)
    return np.mean(wass_dims)


def main():
    print("="*70)
    print("QUANTITATIVE DOMAIN SHIFT ANALYSIS: MUTANTS vs DRUGS")
    print("="*70)
    
    # Load embeddings
    print("\n[1] Loading embeddings...")
    mutant_embeddings = load_embeddings_by_well(os.path.join(EMBEDDINGS_DIR, "Mutants_P1"))
    drug_embeddings = load_embeddings_by_well(os.path.join(EMBEDDINGS_DIR, "Drugs_P1"))
    
    # Get embeddings as arrays (all image embeddings, not just well means)
    mutant_all = []
    drug_all = []
    
    for well_folder in glob.glob(os.path.join(EMBEDDINGS_DIR, "Mutants_P1", "Well*")):
        for npy_file in glob.glob(os.path.join(well_folder, "*.npy")):
            mutant_all.append(np.load(npy_file))
    
    for well_folder in glob.glob(os.path.join(EMBEDDINGS_DIR, "Drugs_P1", "Well*")):
        for npy_file in glob.glob(os.path.join(well_folder, "*.npy")):
            drug_all.append(np.load(npy_file))
    
    X = np.array(mutant_all)  # (2016, 1024)
    Y = np.array(drug_all)    # (2016, 1024)
    
    # Get well-mean embeddings
    X_mean = np.array(list(mutant_embeddings.values()))
    Y_mean = np.array(list(drug_embeddings.values()))
    
    print(f"   Mutant embeddings: {X.shape}")
    print(f"   Drug embeddings: {Y.shape}")
    print(f"   Well-mean embeddings: {X_mean.shape}, {Y_mean.shape}")
    
    # === Compute metrics ===
    results = {}
    
    print("\n[2] Computing similarity metrics...")
    
    # 2.1 Mean cosine similarity (within and between groups)
    within_mutant = cdist(X_mean, X_mean, 'cosine').mean()
    within_drug = cdist(Y_mean, Y_mean, 'cosine').mean()
    between = cdist(X_mean, Y_mean, 'cosine').mean()
    
    results['Mean Cosine Sim (Mutant-Mutant)'] = 1 - within_mutant
    results['Mean Cosine Sim (Drug-Drug)'] = 1 - within_drug
    results['Mean Cosine Sim (Mutant-Drug)'] = 1 - between
    
    print(f"   Cosine Sim - Within Mutant: {results['Mean Cosine Sim (Mutant-Mutant)']:.4f}")
    print(f"   Cosine Sim - Within Drug: {results['Mean Cosine Sim (Drug-Drug)']:.4f}")
    print(f"   Cosine Sim - Cross (Mutant-Drug): {results['Mean Cosine Sim (Mutant-Drug)']:.4f}")
    
    # 2.2 MMD (Maximum Mean Discrepancy)
    mmd = compute_mmd(X, Y)
    results['MMD (embedding space)'] = mmd
    print(f"   MMD: {mmd:.4f}")
    
    # 2.3 Energy Distance
    energy = compute_energy_distance(X, Y)
    results['Energy Distance'] = energy
    print(f"   Energy Distance: {energy:.4f}")
    
    # 2.4 CORAL (covariance alignment)
    coral = compute_coral_loss(X, Y)
    results['CORAL Loss'] = coral
    print(f"   CORAL Loss: {coral:.4f}")
    
    # 2.5 Wasserstein Distance
    wass = compute_wasserstein_1d(X, Y)
    results['Wasserstein Distance (mean)'] = wass
    print(f"   Wasserstein Distance: {wass:.4f}")
    
    # 2.6 KL Divergence
    kl = compute_kl_divergence_approx(X, Y)
    results['KL Divergence (approx)'] = kl
    print(f"   KL Divergence: {kl:.4f}")
    
    # 2.7 Jensen-Shannon Divergence
    js = compute_jensen_shannon(X, Y)
    results['Jensen-Shannon Divergence'] = js
    print(f"   Jensen-Shannon: {js:.4f}")
    
    # 2.8 Hellinger Distance
    hellinger = compute_hellinger(X, Y)
    results['Hellinger Distance'] = hellinger
    print(f"   Hellinger Distance: {hellinger:.4f}")
    
    # 2.9 Within-group vs Between-group ratio
    within_ratio = (within_mutant + within_drug) / 2
    between_ratio = between
    ratio = within_ratio / between_ratio
    results['Within/Between Ratio'] = ratio
    print(f"   Within/Between Ratio: {ratio:.4f}")
    
    # 2.10 Euclidean distance of means
    mean_dist = np.linalg.norm(X_mean.mean(axis=0) - Y_mean.mean(axis=0))
    results['Euclidean Dist (means)'] = mean_dist
    print(f"   Euclidean Dist (means): {mean_dist:.4f}")
    
    # 2.11 Distribution spread (std)
    results['Mutant Embedding Std'] = X.std()
    results['Drug Embedding Std'] = Y.std()
    print(f"   Mutant Std: {results['Mutant Embedding Std']:.4f}")
    print(f"   Drug Std: {results['Drug Embedding Std']:.4f}")
    
    # === Summary ===
    print("\n" + "="*70)
    print("SUMMARY: Domain Shift Quantification")
    print("="*70)
    
    print("\n{:<35} {:>15}".format("Metric", "Value"))
    print("-" * 50)
    for key, val in results.items():
        print("{:<35} {:>15.4f}".format(key, val))
    
    # Save results (convert numpy types to Python)
    output_path = os.path.join(BASE_DIR, "domain_shift_metrics.json")
    results_clean = {k: float(v) for k, v in results.items()}
    with open(output_path, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    print(f"\n[Saved] {output_path}")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    # Key insights
    cross_sim = results['Mean Cosine Sim (Mutant-Drug)']
    within_sim = (results['Mean Cosine Sim (Mutant-Mutant)'] + results['Mean Cosine Sim (Drug-Drug)']) / 2
    
    print(f"\n• Cross-group similarity ({cross_sim:.2%}) vs Within-group ({within_sim:.2%})")
    
    if cross_sim > within_sim * 0.95:
        print("  → Domains are HIGHLY OVERLAPPING (similar distributions)")
    elif cross_sim > within_sim * 0.85:
        print("  → Domains are MODERATELY OVERLAPPING")
    else:
        print("  → Domains are DISTINCT (different distributions)")
    
    print(f"\n• MMD = {mmd:.4f} (lower = more similar)")
    print(f"• Energy Distance = {energy:.4f} (lower = more similar)")
    print(f"• CORAL = {coral:.2f} (lower = more aligned)")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()