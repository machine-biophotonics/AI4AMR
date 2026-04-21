#!/usr/bin/env python3
"""
Analyze batch effects: quantify plate vs gene variance in predictions.
Measures biological signal (gene) vs batch effect (plate) dominance.
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, Tuple, List

import plotly.express as px
import plotly.io as pio
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
try:
    from skbio.stats.distance import permanova
    from skbio import DistanceMatrix
    HAS_SKBIO = True
except ImportError:
    HAS_SKBIO = False
    print("skbio not available, skipping PERMANOVA")


GENE_COLORS = {
    # Peptidoglycan synthesis
    'mrcB': '#E53935',
    'mrcA': '#EF5350',
    'mrdA': '#F0625D',
    'ftsI': '#E57373',
    'murA': '#FF8A80',
    'murC': '#FFAB91',
    'lpxA': '#FFCCBC',
    'lpxC': '#FF7043',

    # Ribosome proteins
    'rpsL': '#FDD835',
    'rpsA': '#FBC02D',
    'rplA': '#F9A825',
    'rplC': '#F57F17',

    # LPS transport
    'msbA': '#00ACC1',
    'lptA': '#26C6DA',
    'lptC': '#4DD0E1',

    # DNA topology
    'gyrA': '#3949AB',
    'gyrB': '#5C6BC0',
    'parC': '#7986CB',
    'parE': '#9FA8DA',

    # Protein translocation
    'secA': '#00897B',
    'secY': '#26A69A',

    # DNA replication
    'dnaB': '#7E57C2',
    'dnaE': '#9575CD',

    # RNA polymerase
    'rpoA': '#43A047',
    'rpoB': '#66BB6A',

    # Cell division
    'ftsZ': '#D81B60',

    # Folate biosynthesis
    'folA': '#7CB342',
    'folP': '#9CCC65',

    # Control
    'WT': '#424242', 'wt': '#424242'
}

GENE_COLORS_LOWER = {k.lower(): v for k, v in GENE_COLORS.items()}
GENE_COLORS_LOWER['nc'] = '#424242'
GENE_COLORS_LOWER['wt nc'] = '#424242'

PLATE_COLORS = {
    'P1': '#1f77b4',
    'P2': '#ff7f0e',
    'P3': '#2ca02c',
    'P4': '#d62728',
    'P5': '#9467bd',
    'P6': '#8c564b',
}

GUIDE_SHAPES = {
    1: 'circle',
    2: 'square',
    3: 'triangle-up',
    4: 'pentagon',
    5: 'star',
    6: 'x'
}


def get_gene_and_guide(label: str) -> Tuple[str, int]:
    """Extract gene name and guide number from label."""
    if not label or label == 'nan':
        return 'wt', 0
    label = str(label)
    if '_' in label:
        parts = label.rsplit('_', 1)
        gene = parts[0]
        try:
            guide = int(parts[1])
        except (ValueError, IndexError):
            guide = 0
    else:
        gene = label
        guide = 0
    return gene.lower(), guide


def load_predictions(fold_dir: str) -> pd.DataFrame:
    """Load prediction CSV for a fold."""
    csv_files = [
        'predictions_all_crops_mil_best_model.csv',
        'predictions_all_crops_mil_best_model_acc.csv',
        'predictions_all_crops_mil_100pos.csv',
        'predictions_all_crops_best_model.csv',
        'predictions_all_crops.csv',
    ]
    
    csv_path = None
    for f in csv_files:
        path = os.path.join(fold_dir, f)
        if os.path.exists(path):
            csv_path = path
            break
    
    if csv_path is None:
        raise FileNotFoundError(f"No prediction CSV found in {fold_dir}")
    
    df = pd.read_csv(csv_path)
    return df


def aggregate_to_image(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate crop predictions to image level."""
    image_results = []
    
    for img_name, group in df.groupby('image_name'):
        true_label = group['ground_truth_label'].iloc[0]
        pred_counts = group['predicted_class_name'].value_counts()
        majority_pred = pred_counts.index[0]
        
        probs_list = []
        for p in group['probs']:
            if isinstance(p, str):
                try:
                    probs_list.append(json.loads(p))
                except json.JSONDecodeError:
                    probs_list.append([0.0] * 96)
            else:
                probs_list.append([0.0] * 96)
        
        mean_probs = np.mean(probs_list, axis=0)
        
        image_results.append({
            'image_name': img_name,
            'true_label': true_label,
            'pred_label': majority_pred,
            'probs': mean_probs
        })
    
    return pd.DataFrame(image_results)


def compute_silhouette_gene(X: np.ndarray, labels: np.ndarray, valid_genes: set) -> float:
    """Compute silhouette score for gene labels."""
    gene_labels = []
    for l in labels:
        g = l.lower() if isinstance(l, str) else str(l)
        gene_labels.append(g if g in valid_genes else 'wt')
    
    unique_genes = list(set(gene_labels))
    if len(unique_genes) < 2:
        return 0.0
    
    gene_to_idx = {g: i for i, g in enumerate(unique_genes)}
    numeric_labels = np.array([gene_to_idx[g] for g in gene_labels])
    
    try:
        score = silhouette_score(X, numeric_labels)
        return score
    except Exception:
        return 0.0


def compute_silhouette_plate(X: np.ndarray, plates: np.ndarray) -> float:
    """Compute silhouette score for plate labels (inverted for mixing)."""
    unique_plates = list(set(plates))
    if len(unique_plates) < 2:
        return 0.0
    
    plate_to_idx = {p: i for i, p in enumerate(unique_plates)}
    numeric_labels = np.array([plate_to_idx[p] for p in plates])
    
    try:
        score = silhouette_score(X, numeric_labels)
        return score
    except Exception:
        return 0.0


def compute_kbet(X: np.ndarray, plates: np.ndarray, k: int = 50) -> float:
    """
    Compute k-nearest neighbor batch effect test.
    Returns fraction of neighbors from SAME plate (lower = better mixing).
    """
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    
    same_plate_fraction = []
    for i, plate in enumerate(plates):
        neighbors = indices[i, 1:]  # Exclude self
        same_count = sum(plates[n] == plate for n in neighbors)
        same_plate_fraction.append(same_count / k)
    
    return np.mean(same_plate_fraction)


def compute_procrustes_pair(X1: np.ndarray, X2: np.ndarray) -> float:
    """
    Compute Procrustes residual between two embeddings.
    Lower = more similar structure.
    """
    from sklearn.preprocessing import StandardScaler
    from scipy.linalg import orthogonal_procrustes
    
    n1, n2 = min(len(X1), len(X2)), min(len(X1), len(X2))
    n = min(n1, n2)
    
    X1_s = StandardScaler().fit_transform(X1[:n])
    X2_s = StandardScaler().fit_transform(X2[:n])
    
    try:
        R, scale = orthogonal_procrustes(X1_s, X2_s)
        X2_rot = X2_s @ R
        
        mse = np.mean((X1_s - X2_rot) ** 2)
        residual = np.sqrt(mse)
        return residual
    except Exception:
        return 1.0


def compute_within_gene_variance(X: np.ndarray, gene_labels: List[str]) -> float:
    """Compute average within-gene variance."""
    gene_groups = {}
    for i, g in enumerate(gene_labels):
        if g not in gene_groups:
            gene_groups[g] = []
        gene_groups[g].append(i)
    
    variances = []
    for g, indices in gene_groups.items():
        if len(indices) > 1:
            sub_X = X[indices]
            variances.append(np.mean(np.var(sub_X, axis=0)))
    
    return np.mean(variances) if variances else 0.0


def compute_between_gene_variance(X: np.ndarray, gene_labels: List[str]) -> float:
    """Compute between-gene variance (centroid spread)."""
    gene_groups = {}
    for i, g in enumerate(gene_labels):
        if g not in gene_groups:
            gene_groups[g] = []
        gene_groups[g].append(i)
    
    centroids = []
    for g, indices in gene_groups.items():
        if len(indices) > 1:
            centroids.append(np.mean(X[indices], axis=0))
    
    if len(centroids) > 1:
        return np.mean(np.var(np.array(centroids), axis=0))
    return 0.0


def compute_permanova(X: np.ndarray, labels: np.ndarray, n_perms: int = 999) -> Dict[str, float]:
    """Compute PERMANOVA for gene vs plate labels."""
    if not HAS_SKBIO:
        return {'gene_r2': 0.0, 'plate_r2': 0.0, 'gene_p': 1.0, 'plate_p': 1.0}
    
    unique_labels = list(set(labels))
    if len(unique_labels) < 2:
        return {'gene_r2': 0.0, 'plate_r2': 0.0, 'gene_p': 1.0, 'plate_p': 1.0}
    
    dist_matrix = DistanceMatrix(cdist(X, X))
    
    try:
        result = permanova(dist_matrix, labels, permutations=n_perms)
        return {
            'r2': result['test statistic'] if 'test statistic' in result else 0.0,
            'p': result['p-value'] if 'p-value' in result else 1.0
        }
    except Exception:
        return {'r2': 0.0, 'p': 1.0}


def analyze_all_folds() -> pd.DataFrame:
    """Load all folds and compute batch effect metrics."""
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    all_dfs = []
    all_X = []
    
    print("Loading all folds...")
    for fold in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
        fold_dir = os.path.join(SCRIPT_DIR, f'fold_{fold}')
        if not os.path.exists(fold_dir):
            continue
        
        try:
            df = load_predictions(fold_dir)
            df_valid = df[df['ground_truth_label'].notna()].copy()
            image_df = aggregate_to_image(df_valid)
            
            gene_guide = image_df['true_label'].apply(lambda x: get_gene_and_guide(x))
            image_df['gene'] = gene_guide.apply(lambda x: x[0])
            image_df['guide'] = gene_guide.apply(lambda x: x[1])
            
            valid_genes = set(GENE_COLORS_LOWER.keys())
            image_df.loc[~image_df['gene'].isin(valid_genes), 'gene'] = 'wt'
            image_df['guide'] = image_df['guide'].fillna(0).astype(int)
            image_df['plate'] = fold
            
            all_dfs.append(image_df)
            all_X.append(np.array(image_df['probs'].tolist()))
            print(f"Loaded fold {fold}: {len(image_df)} images")
        except Exception as e:
            print(f"Skipping fold {fold}: {e}")
    
    if not all_dfs:
        print("No folds loaded!")
        return pd.DataFrame()
    
    X_all = np.vstack(all_X)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\nTotal: {len(combined_df)} images from {len(all_dfs)} plates")
    print(f"Computing metrics...")
    
    gene_labels = combined_df['gene'].tolist()
    plate_labels = combined_df['plate'].tolist()
    
    metrics = {}
    
    print("  1. Gene Silhouette (ASW-gene)...")
    gene_sil = compute_silhouette_gene(X_all, np.array(gene_labels), GENE_COLORS_LOWER.keys())
    metrics['gene_silhouette'] = round(gene_sil, 4)
    print(f"     Gene Silhouette: {gene_sil:.4f}")
    
    print("  2. Plate Silhouette (ASW-batch, lower = better mixing)...")
    plate_sil = compute_silhouette_plate(X_all, np.array(plate_labels))
    metrics['plate_silhouette'] = round(plate_sil, 4)
    print(f"     Plate Silhouette: {plate_sil:.4f}")
    
    print("  3. kBET score (lower = better mixing)...")
    kbet = compute_kbet(X_all, np.array(plate_labels), k=30)
    metrics['kbetch_score'] = round(kbet, 4)
    print(f"     kBET (k=30): {kbet:.4f}")
    
    print("  4. Within-gene variance...")
    within_var = compute_within_gene_variance(X_all, gene_labels)
    metrics['within_gene_variance'] = round(within_var, 4)
    print(f"     Within-gene var: {within_var:.4f}")
    
    print("  5. Between-gene variance...")
    between_var = compute_between_gene_variance(X_all, gene_labels)
    metrics['between_gene_variance'] = round(between_var, 4)
    print(f"     Between-gene var: {between_var:.4f}")
    
    print("  6. Variance ratio (between/within)...")
    var_ratio = between_var / within_var if within_var > 0 else 0
    metrics['variance_ratio'] = round(var_ratio, 4)
    print(f"     Variance ratio: {var_ratio:.4f}")
    
    print("  7. Computing per-plate kBET...")
    per_plate_kbet = {}
    for fold in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
        mask = combined_df['plate'] == fold
        if mask.sum() > 30:
            X_plate = X_all[mask]
            plates_plate = np.array(plate_labels)[mask]
            kbet_p = compute_kbet(X_plate, plates_plate, k=min(30, len(X_plate)-1))
            per_plate_kbet[fold] = round(kbet_p, 4)
    
    metrics['per_plate_kbet'] = per_plate_kbet
    print(f"     Per-plate kBET: {per_plate_kbet}")
    
    print("  8. Computing Procrustes residuals (pairwise)...")
    procrustes_residuals = {}
    for i, fold1 in enumerate(['P1', 'P2', 'P3', 'P4', 'P5', 'P6']):
        for fold2 in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
            if fold1 >= fold2:
                continue
            mask1 = combined_df['plate'] == fold1
            mask2 = combined_df['plate'] == fold2
            if mask1.sum() > 0 and mask2.sum() > 0:
                X1 = X_all[mask1]
                X2 = X_all[mask2]
                res = compute_procrustes_pair(X1, X2)
                procrustes_residuals[f"{fold1}_{fold2}"] = round(res, 4)
    
    metrics['procrustes_residuals'] = procrustes_residuals
    print(f"     Mean Procrustes residual: {np.mean(list(procrustes_residuals.values())):.4f}")
    
    if HAS_SKBIO:
        print("  9. Computing PERMANOVA...")
        try:
            gene_perm = compute_permanova(X_all, np.array(gene_labels))
            metrics['permanova_gene_r2'] = round(gene_perm.get('r2', 0), 4)
            metrics['permanova_gene_p'] = round(gene_perm.get('p', 1), 4)
            print(f"     PERMANOVA gene R²: {gene_perm.get('r2', 0):.4f}, p={gene_perm.get('p', 1):.4f}")
        except Exception as e:
            print(f"     PERMANOVA failed: {e}")
    else:
        print("  9. Skipping PERMANOVA (skbio not available)")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if metrics['gene_silhouette'] > metrics['plate_silhouette']:
        print("✓ Biological signal dominates (gene silhouette > plate silhouette)")
        print(f"  Gene silhouette: {metrics['gene_silhouette']:.4f}")
        print(f"  Plate silhouette: {metrics['plate_silhouette']:.4f}")
    else:
        print("⚠ Plate effect may dominate (gene silhouette <= plate silhouette)")
        print(f"  Gene silhouette: {metrics['gene_silhouette']:.4f}")
        print(f"  Plate silhouette: {metrics['plate_silhouette']:.4f}")
    
    if metrics['kbetch_score'] < 0.5:
        print("✓ Good plate mixing (kBET < 0.5)")
    else:
        print("⚠ Poor plate mixing (kBET >= 0.5)")
    
    if metrics['variance_ratio'] > 1.0:
        print("✓ Gene variance exceeds within-gene variance (ratio > 1)")
    else:
        print("⚠ Gene variance similar to within-gene variance")
    
    return pd.DataFrame([metrics])


def main():
    parser = argparse.ArgumentParser(description='Analyze batch effects')
    parser.add_argument('--output', type=str, default='batch_effect_metrics.csv')
    args = parser.parse_args()
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    metrics_df = analyze_all_folds()
    
    output_dir = os.path.join(SCRIPT_DIR, 'train_test_results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output)
    
    metrics_df.to_csv(output_path, index=False)
    print(f"\nSaved metrics to {output_path}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()