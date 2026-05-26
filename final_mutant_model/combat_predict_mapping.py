#!/usr/bin/env python3
"""
Apply ComBat batch correction to MIL bag embeddings, then train a classifier
on mutant embeddings and predict on drug embeddings to generate drug→mutant mappings.

Pipeline:
  1. Load mutant & drug bag embeddings (1280-d)
  2. ComBat correction (domain: mutant vs drug)
  3. Train classifier on corrected mutant embeddings → predict corrected drug embeddings
  4. Generate drug→mutant mapping table + heatmap + tSNE diagnostics
"""

import os
import sys
import json
import argparse
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.spatial.distance import cdist

from combat.pycombat import pycombat


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_embeddings(emb_dir: str, domain: str = None, plate: str = None) -> dict:
    """Load embeddings .npz files matching criteria."""
    emb_dir = Path(emb_dir)
    all_data = {'embeddings': [], 'labels': [], 'wells': [], 'paths': [],
                'domains': [], 'plates': []}

    npz_files = sorted(emb_dir.glob('embeddings_*.npz'))
    for fpath in npz_files:
        fname = fpath.name
        # Parse domain and plate from filename: embeddings_{domain}_{plate}.npz
        parts = fname.replace('embeddings_', '').replace('.npz', '').split('_')
        f_domain = parts[0]  # 'mutant' or 'drug'
        f_plate = parts[1]   # e.g., 'P1'

        if domain and f_domain != domain:
            continue
        if plate and f_plate != plate:
            continue
        if fname == 'class_mappings.npz':
            continue

        data = np.load(fpath, allow_pickle=True)
        n = len(data['embeddings'])
        all_data['embeddings'].append(data['embeddings'])
        all_data['labels'].extend(str(l) for l in data['labels'])
        all_data['wells'].extend(str(w) for w in data['wells'])
        all_data['paths'].extend(str(p) for p in data['paths'])
        all_data['domains'].extend([f_domain] * n)
        all_data['plates'].extend([f_plate] * n)

    if not all_data['embeddings']:
        return None
    all_data['embeddings'] = np.concatenate(all_data['embeddings'], axis=0)
    return all_data


def compute_well_centroids(data: dict) -> dict:
    """Average embeddings per well for cleaner per-well predictions."""
    well_dict = defaultdict(lambda: {'embs': [], 'domains': [], 'plates': [], 'labels': set()})
    for i in range(len(data['embeddings'])):
        key = (data['domains'][i], data['plates'][i], data['wells'][i])
        well_dict[key]['embs'].append(data['embeddings'][i])
        well_dict[key]['domains'].append(data['domains'][i])
        well_dict[key]['plates'].append(data['plates'][i])
        well_dict[key]['labels'].add(data['labels'][i])

    result = {'embeddings': [], 'labels': [], 'wells': [],
              'domains': [], 'plates': []}
    for (dom, plate, well), v in well_dict.items():
        result['embeddings'].append(np.mean(v['embs'], axis=0))
        result['labels'].append(next(iter(v['labels'])))
        result['wells'].append(well)
        result['domains'].append(dom)
        result['plates'].append(plate)

    result['embeddings'] = np.array(result['embeddings'])
    return result


def run_combat(embeddings: np.ndarray, domain_labels: list,
               par_prior: bool = True, mean_only: bool = False) -> np.ndarray:
    """Apply ComBat batch correction.
    
    ComBat expects data as (features × samples) matrix.
    We have (samples × features), so we transpose.
    """
    # Transpose to (features × samples)
    data_t = embeddings.T  # (1280, N)

    corrected_t = pycombat(
        data=pd.DataFrame(data_t),
        batch=domain_labels,
        par_prior=par_prior,
        mean_only=mean_only
    )
    # Transpose back to (samples × features)
    return np.array(corrected_t).T


def main():
    parser = argparse.ArgumentParser(description='ComBat-based drug→mutant mapping')
    parser.add_argument('--emb_dir', type=str, default='embeddings_test',
                        help='Directory with .npz embedding files')
    parser.add_argument('--output_dir', type=str, default='combat_mapping',
                        help='Output directory for results')
    parser.add_argument('--no_combat', action='store_true',
                        help='Skip ComBat correction (baseline comparison)')
    parser.add_argument('--mean_only', action='store_true',
                        help='ComBat: mean-only correction (no scale)')
    parser.add_argument('--non_parametric', action='store_true',
                        help='ComBat: non-parametric mode')
    parser.add_argument('--n_pca', type=int, default=50,
                        help='PCA dims for plot diagnostics')
    parser.add_argument('--classifier', type=str, default='logistic',
                        choices=['logistic', 'mlp'],
                        help='Classifier type for mapping')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Top-K mutant predictions per drug')
    parser.add_argument('--drug_no_concentration', action='store_true',
                        help='Group drugs by antibiotic name')
    parser.add_argument('--ensemble_run', action='store_true',
                        help='Run all ComBat configs for comparison')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Load embeddings ──────────────────────────────────────────────────────
    print("\nLoading embeddings...")
    mutant_data = load_embeddings(args.emb_dir, domain='mutant')
    drug_data = load_embeddings(args.emb_dir, domain='drug')

    if mutant_data is None:
        print("ERROR: No mutant embeddings found!")
        sys.exit(1)
    if drug_data is None:
        print("ERROR: No drug embeddings found!")
        sys.exit(1)

    print(f"  Mutant: {mutant_data['embeddings'].shape}, "
          f"{len(set(mutant_data['labels']))} classes")
    print(f"  Drug:   {drug_data['embeddings'].shape}, "
          f"{len(set(drug_data['labels']))} classes")

    # Remove 'unknown' labeled samples
    mut_known = np.array([l != 'unknown' for l in mutant_data['labels']])
    drug_known = np.array([l != 'unknown' for l in drug_data['labels']])
    print(f"  Mutant known: {mut_known.sum()}/{len(mut_known)}")
    print(f"  Drug known:   {drug_known.sum()}/{len(drug_known)}")

    # Build class mappings
    all_mutant_classes = sorted(set(
        l for l in mutant_data['labels'] if l != 'unknown'))
    all_drug_classes = sorted(set(
        l for l in drug_data['labels'] if l != 'unknown'))

    if args.drug_no_concentration:
        # Group drug classes by antibiotic name
        drug_to_group = {}
        for dc in all_drug_classes:
            if dc == 'control':
                drug_to_group[dc] = 'control'
            else:
                drug_to_group[dc] = dc.rsplit('_', 1)[0]
        all_drug_classes = sorted(set(drug_to_group.values()))

    mut_to_idx = {c: i for i, c in enumerate(all_mutant_classes)}
    drug_to_idx = {c: i for i, c in enumerate(all_drug_classes)}

    print(f"  Mutant classes: {len(all_mutant_classes)}")
    print(f"  Drug classes:   {len(all_drug_classes)}")

    # ── Combine data for ComBat ──────────────────────────────────────────────
    # Use well-level centroids for cleaner signal (average ~21 images per well)
    mut_well = compute_well_centroids({
        k: [v for i, v in enumerate(mutant_data[k]) if mut_known[i]]
        for k in ['embeddings', 'labels', 'wells', 'domains', 'plates']
    })
    drug_well = compute_well_centroids({
        k: [v for i, v in enumerate(drug_data[k]) if drug_known[i]]
        for k in ['embeddings', 'labels', 'wells', 'domains', 'plates']
    })

    # Stack
    all_embs = np.vstack([mut_well['embeddings'], drug_well['embeddings']])
    all_domains = mut_well['domains'] + drug_well['domains']
    all_plates = mut_well['plates'] + drug_well['plates']
    all_labels = mut_well['labels'] + drug_well['labels']
    all_wells = mut_well['wells'] + drug_well['wells']

    N_mut = len(mut_well['embeddings'])
    N_drug = len(drug_well['embeddings'])
    print(f"\nWell-level data: {N_mut} mutant wells, {N_drug} drug wells")

    # ── Choose analysis mode ────────────────────────────────────────────────
    configs = []
    if args.ensemble_run:
        configs = [
            {'name': 'no_correction', 'no_combat': True, 'mean_only': False, 'non_parametric': False},
            {'name': 'combat_mean', 'no_combat': False, 'mean_only': True, 'non_parametric': False},
            {'name': 'combat_full', 'no_combat': False, 'mean_only': False, 'non_parametric': False},
            {'name': 'combat_nonparam', 'no_combat': False, 'mean_only': False, 'non_parametric': True},
        ]
    else:
        configs = [{
            'name': 'no_correction' if args.no_combat else
                    ('combat_mean' if args.mean_only else
                     ('combat_nonparam' if args.non_parametric else 'combat_full')),
            'no_combat': args.no_combat,
            'mean_only': args.mean_only,
            'non_parametric': args.non_parametric,
        }]

    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"Config: {cfg['name']}")
        print(f"{'='*70}")

        out_subdir = os.path.join(args.output_dir, cfg['name'])
        os.makedirs(out_subdir, exist_ok=True)

        # ── Apply ComBat ─────────────────────────────────────────────────────
        if cfg['no_combat']:
            corrected_embs = all_embs.copy()
        else:
            print("Applying ComBat...")
            corrected_embs = run_combat(
                all_embs,
                all_domains,
                par_prior=not cfg['non_parametric'],
                mean_only=cfg['mean_only']
            )
            print(f"  Corrected shape: {corrected_embs.shape}")

        mut_emb_corrected = corrected_embs[:N_mut]
        drug_emb_corrected = corrected_embs[N_mut:]

        # ── tSNE diagnostics ────────────────────────────────────────────────
        print("Running tSNE diagnostics...")
        pca_50 = PCA(n_components=min(50, corrected_embs.shape[0], corrected_embs.shape[1]))
        pca_embs = pca_50.fit_transform(corrected_embs)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        tsne_embs = tsne.fit_transform(pca_embs)

        fig, axes = plt.subplots(1, 3, figsize=(24, 7))

        # Domain view
        for ax, title, color_map in [
            (axes[0], f'Domain ({cfg["name"]})',
             {'mutant': '#1f77b4', 'drug': '#ff7f0e'})
        ]:
            for dom, c in color_map.items():
                mask = [d == dom for d in all_domains]
                ax.scatter(tsne_embs[mask, 0], tsne_embs[mask, 1],
                          c=c, label=dom, s=20, alpha=0.6)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend()

        # Mutant classes view
        mut_colors = plt.cm.tab20(np.linspace(0, 1, min(len(all_mutant_classes), 20)))
        mut_label_set = set(mut_well['labels'])
        mut_label_to_c = {l: mut_colors[i % 20] for i, l in enumerate(sorted(mut_label_set))}
        for i in range(N_mut):
            c = mut_label_to_c.get(mut_well['labels'][i], 'gray')
            axes[1].scatter(tsne_embs[i, 0], tsne_embs[i, 1], c=[c], s=20, alpha=0.6)
        axes[1].set_title(f'Mutant classes ({len(mut_label_set)})', fontsize=14, fontweight='bold')

        # Drug classes view
        drug_label_set = set(drug_well['labels'])
        drug_colors = plt.cm.tab20(np.linspace(0, 1, min(len(drug_label_set), 20)))
        drug_label_to_c = {l: drug_colors[i % 20] for i, l in enumerate(sorted(drug_label_set))}
        for i in range(N_drug):
            c = drug_label_to_c.get(drug_well['labels'][i], 'gray')
            axes[2].scatter(tsne_embs[N_mut + i, 0], tsne_embs[N_mut + i, 1], c=[c], s=20, alpha=0.6)
        axes[2].set_title(f'Drug classes ({len(drug_label_set)})', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(out_subdir, 'tsne_domains.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # ── Train classifier on mutant embeddings → predict drug ────────────
        print("Training classifier on mutant embeddings...")
        mut_labels_idx = np.array([mut_to_idx[l] for l in mut_well['labels']])

        if args.classifier == 'logistic':
            # Standardize
            scaler = StandardScaler()
            mut_train = scaler.fit_transform(mut_emb_corrected)
            drug_test = scaler.transform(drug_emb_corrected)

            clf = LogisticRegression(
                solver='lbfgs', max_iter=5000, C=1.0,
                random_state=42, n_jobs=-1
            )
            clf.fit(mut_train, mut_labels_idx)
        else:
            # MLP via PyTorch
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            scaler = StandardScaler()
            mut_train = scaler.fit_transform(mut_emb_corrected)
            drug_test = scaler.transform(drug_emb_corrected)

            X_t = torch.from_numpy(mut_train).float().to(device)
            y_t = torch.from_numpy(mut_labels_idx).long().to(device)

            d_in = mut_train.shape[1]
            n_cls = len(all_mutant_classes)
            model = nn.Sequential(
                nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, n_cls)
            ).to(device)

            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

            best_loss = float('inf')
            for epoch in range(200):
                model.train()
                opt.zero_grad()
                logits = model(X_t)
                loss = F.cross_entropy(logits, y_t)
                loss.backward()
                opt.step()
                sch.step()
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(model.state_dict(), os.path.join(out_subdir, 'mlp_classifier.pt'))

            model.load_state_dict(torch.load(os.path.join(out_subdir, 'mlp_classifier.pt'), weights_only=True))
            model.eval()
            clf = model

            def predict_proba(x):
                x_t = torch.from_numpy(x).float().to(device)
                with torch.no_grad():
                    logits = model(x_t)
                    return F.softmax(logits, dim=1).cpu().numpy()

            def predict(x):
                return np.argmax(predict_proba(x), axis=1)

            clf.predict = predict
            clf.predict_proba = predict_proba

        # Evaluate on mutant (self-consistency via cross-val would be better)
        mut_preds = clf.predict(mut_train)
        mut_acc = accuracy_score(mut_labels_idx, mut_preds)
        print(f"  Mutant self-accuracy: {mut_acc:.4f}")

        # ── Predict on drug embeddings ──────────────────────────────────────
        print("Predicting drug→mutant mapping...")
        drug_probs = clf.predict_proba(drug_test)
        drug_preds = np.argmax(drug_probs, axis=1)

        # ── Build drug→mutant mapping table ──────────────────────────────────
        results = []
        for i in range(N_drug):
            drug_label = drug_well['labels'][i]
            if args.drug_no_concentration:
                if drug_label == 'control':
                    drug_label_grouped = 'control'
                else:
                    drug_label_grouped = drug_label.rsplit('_', 1)[0]
            else:
                drug_label_grouped = drug_label

            top_k_idx = np.argsort(drug_probs[i])[::-1][:args.top_k]
            top_mutants = [(all_mutant_classes[j], float(drug_probs[i][j]))
                           for j in top_k_idx]

            results.append({
                'drug_label': drug_label_grouped,
                'well': drug_well['wells'][i],
                'plate': drug_well['plates'][i],
                'predicted_mutant': all_mutant_classes[drug_preds[i]],
                'confidence': float(drug_probs[i][drug_preds[i]]),
                'top_k_mutants': top_mutants,
            })

        df_results = pd.DataFrame(results)
        df_results.to_csv(os.path.join(out_subdir, 'drug_to_mutant_mapping.csv'), index=False)

        # ── Aggregate by drug class ──────────────────────────────────────────
        drug_aggregated = defaultdict(lambda: {
            'wells': [], 'predictions': Counter(), 'top3_overlap': []
        })
        for r in results:
            drug = r['drug_label']
            drug_aggregated[drug]['wells'].append(r['well'])
            drug_aggregated[drug]['predictions'][r['predicted_mutant']] += 1
            drug_aggregated[drug]['top3_overlap'].append(
                [m for m, _ in r['top_k_mutants'][:3]])

        agg_rows = []
        for drug, info in sorted(drug_aggregated.items()):
            total = sum(info['predictions'].values())
            top_pred = info['predictions'].most_common(5)
            agg_rows.append({
                'drug': drug,
                'n_wells': len(info['wells']),
                'top1_mutant': top_pred[0][0] if top_pred else '',
                'top1_pct': f"{top_pred[0][1] / total * 100:.0f}%" if top_pred else '',
                'top5_mutants': ' | '.join(f"{m} ({c}/{total})" for m, c in top_pred),
            })

        df_agg = pd.DataFrame(agg_rows)
        df_agg.to_csv(os.path.join(out_subdir, 'drug_to_mutant_aggregated.csv'), index=False)
        print(f"\nAggregated drug→mutant mapping saved.")
        print(df_agg.to_string(index=False))

        # ── Cross-domain confusion heatmap ───────────────────────────────────
        # Drug classes vs predicted mutant classes
        drug_labels_grouped = []
        for l in drug_well['labels']:
            if args.drug_no_concentration:
                drug_labels_grouped.append(
                    'control' if l == 'control' else l.rsplit('_', 1)[0])
            else:
                drug_labels_grouped.append(l)

        unique_drugs_in_data = sorted(set(drug_labels_grouped))
        unique_muts = all_mutant_classes

        conf_matrix = np.zeros((len(unique_drugs_in_data), len(unique_muts)))
        for i in range(N_drug):
            d_idx = unique_drugs_in_data.index(drug_labels_grouped[i])
            m_idx = drug_preds[i]
            conf_matrix[d_idx, m_idx] += 1

        # Normalize per drug class
        conf_matrix_norm = conf_matrix / (conf_matrix.sum(axis=1, keepdims=True) + 1e-8)

        fig, ax = plt.subplots(figsize=(max(20, len(unique_muts) * 0.4),
                                         max(12, len(unique_drugs_in_data) * 0.5)))
        sns.heatmap(conf_matrix_norm, ax=ax, cmap='YlOrRd',
                    xticklabels=unique_muts, yticklabels=unique_drugs_in_data,
                    vmin=0, vmax=1)
        ax.set_xlabel('Predicted Mutant Class', fontsize=12)
        ax.set_ylabel('Drug Class', fontsize=12)
        ax.set_title(f'Drug → Mutant Mapping ({cfg["name"]})', fontsize=14, fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
        plt.setp(ax.get_yticklabels(), fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_subdir, 'drug_to_mutant_heatmap.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved.")

        # ── Similarity analysis: known drug-target pairs ─────────────────────
        # Load IC50 mapping for drug-target info
        with open(os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')) as f:
            ic50_data = json.load(f)

        # For each drug, find the top mutant prediction and check against known MOA
        print("\n--- Drug → Top Mutant Predictions ---")
        for _, row in df_agg.iterrows():
            drug_name = row['drug']
            if drug_name == 'DMSO' or drug_name == 'control':
                continue
            print(f"  {drug_name:25s} → {row['top1_mutant']:15s} "
                  f"({row['top1_pct']}), top5: {row['top5_mutants'][:60]}")

        print(f"\n{'='*70}")
        print(f"Results saved to {out_subdir}/")
        print(f"Files:")
        for f in sorted(os.listdir(out_subdir)):
            print(f"  {f}")

    if args.ensemble_run:
        print(f"\n{'='*70}")
        print("ENSEMBLE COMPLETE")
        print("Compare results across configs:", args.output_dir)


if __name__ == '__main__':
    main()
