#!/usr/bin/env python3
"""
Evaluate shared prototype space.

Loads trained model, extracts prototype vectors, evaluates:
1. Drug-target retrieval (known relationships)
2. t-SNE visualization of drugs + mutants
3. Silhouette scores by MOA groups

Usage:
  python3 evaluate_latent.py --fold P6
"""

import os, sys, json, argparse
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cpa_model import CPAModel
from train_cpa import build_label_mappings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

KNOWN_PAIRS = {
    'Ciprofloxacin': ['gyrA_1', 'gyrA_2', 'gyrA_3',
                      'gyrB_1', 'gyrB_2', 'gyrB_3',
                      'parC_1', 'parC_2', 'parC_3',
                      'parE_1', 'parE_2', 'parE_3'],
    'Levofloxacin': ['gyrA_1', 'gyrA_2', 'gyrA_3',
                     'gyrB_1', 'gyrB_2', 'gyrB_3',
                     'parC_1', 'parC_2', 'parC_3',
                     'parE_1', 'parE_2', 'parE_3'],
    'Norfloxacin': ['gyrA_1', 'gyrA_2', 'gyrA_3',
                    'gyrB_1', 'gyrB_2', 'gyrB_3',
                    'parC_1', 'parC_2', 'parC_3',
                    'parE_1', 'parE_2', 'parE_3'],
    'Rifampicin': ['rpoA_1', 'rpoA_2', 'rpoA_3',
                   'rpoB_1', 'rpoB_2', 'rpoB_3'],
    'Mecillinam': ['mrdA_1', 'mrdA_2', 'mrdA_3', 'mrdB_1'],
    'Aztreonam': ['ftsI_1', 'ftsI_2', 'ftsI_3'],
    'Trimethoprim': ['folA_1', 'folA_2', 'folA_3'],
    'Doxicyclin': ['rpsA_1', 'rpsA_2', 'rpsL_1', 'rpsL_2'],
    'Kanamycin': ['rpsA_1', 'rpsA_2', 'rpsL_1', 'rpsL_2'],
}

MOA_GROUPS = {
    'DNA_gyrase': ['Ciprofloxacin', 'Levofloxacin', 'Norfloxacin',
                   'gyrA_1', 'gyrA_2', 'gyrA_3',
                   'gyrB_1', 'gyrB_2', 'gyrB_3',
                   'parC_1', 'parC_2', 'parC_3',
                   'parE_1', 'parE_2', 'parE_3'],
    'RNA_polymerase': ['Rifampicin',
                       'rpoA_1', 'rpoA_2', 'rpoA_3',
                       'rpoB_1', 'rpoB_2', 'rpoB_3'],
    'Cell_wall_PBP': ['Mecillinam', 'Aztreonam',
                      'mrdA_1', 'mrdA_2', 'mrdA_3', 'mrdB_1',
                      'ftsI_1', 'ftsI_2', 'ftsI_3'],
    'Folate_synthesis': ['Trimethoprim',
                         'folA_1', 'folA_2', 'folA_3'],
    'Ribosome': ['Doxicyclin', 'Kanamycin', 'Chloramphenicol',
                 'rpsA_1', 'rpsA_2',
                 'rpsL_1', 'rpsL_2'],
    'Cell_envelope': ['Colistin', 'Polymyxin_B',
                      'lptA_1', 'lptA_2', 'lptA_3',
                      'lpxA_1', 'lpxA_2', 'lpxA_3',
                      'lpxC_1', 'lpxC_2', 'lpxC_3'],
    'Cell_division': ['ftsZ_1', 'ftsZ_2', 'ftsZ_3'],
}


def load_model(ckpt, n_pert, n_cls, device):
    m = CPAModel(num_perturbations=n_pert, num_classes=n_cls).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    return m


def compute_retrieval(sim, pnames, known_pairs):
    """For each drug, check if any concentration has target mutants in top-K."""
    results = {}
    for drug_prefix, targets in known_pairs.items():
        drug_idxs = [i for i, n in enumerate(pnames)
                     if n.startswith(f'drug_{drug_prefix}_')]
        target_idxs = [i for i, n in enumerate(pnames)
                       if n.startswith('mutant_') and n[7:] in targets]
        if not drug_idxs or not target_idxs:
            continue

        best_ranks = []
        for d in drug_idxs:
            mut_scores = [(sim[d, i], i) for i, n in enumerate(pnames)
                          if n.startswith('mutant_')]
            mut_scores.sort(reverse=True)
            ranks = [r+1 for r, (_, i) in enumerate(mut_scores) if i in target_idxs]
            if ranks:
                best_ranks.append(min(ranks))

        if best_ranks:
            results[drug_prefix] = {
                'n_conc': len(drug_idxs),
                'best_rank': min(best_ranks),
                'recall_5': np.mean([r <= 5 for r in best_ranks]) * 100,
            }
    return results


def compute_silhouette(emb, pnames, moa_groups):
    idx_to_g = {}
    for gname, members in moa_groups.items():
        for prefix in members:
            if prefix.startswith('drug_') or prefix.startswith('mutant_'):
                matched = [i for i, n in enumerate(pnames) if n == prefix]
            else:
                matched = [i for i, n in enumerate(pnames)
                          if n.startswith(f'drug_{prefix}_') or n == f'mutant_{prefix}']
            for i in matched:
                idx_to_g[i] = gname
    idxs = sorted(idx_to_g.keys())
    if len(set(idx_to_g.values())) < 2: return float('nan')
    y = np.array([list(set(idx_to_g.values())).index(idx_to_g[i]) for i in idxs])
    return silhouette_score(emb[idxs], y)


def plot_tsne(prototypes, pnames, path):
    tsne = TSNE(n_components=2, random_state=42,
                perplexity=min(30, len(pnames)-1))
    xy = tsne.fit_transform(prototypes)

    fig, ax = plt.subplots(figsize=(14, 12))
    colors = {'drug': '#1f77b4', 'mutant': '#ff7f0e', 'control': '#2ca02c'}
    for i, name in enumerate(pnames):
        if name == 'control':
            c, lbl = colors['control'], 'ctrl'
        elif name.startswith('drug_'):
            c, lbl = colors['drug'], name[5:]
        elif name.startswith('mutant_'):
            c, lbl = colors['mutant'], name[7:]
        else:
            c, lbl = '#7f7f7f', name
        ax.scatter(xy[i,0], xy[i,1], c=c, s=60, alpha=0.6, edgecolors='w', linewidth=0.3)
        ax.annotate(lbl, (xy[i,0], xy[i,1]), fontsize=5, alpha=0.7)

    for cat, c in colors.items():
        ax.scatter([],[], c=c, label=cat, s=80)
    ax.legend(fontsize=12)
    ax.set_title('Prototype Embeddings (t-SNE)', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"t-SNE: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', default='P6')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()

    out = args.output_dir or os.path.join(SCRIPT_DIR, 'cpa', f'fold_{args.fold}')
    ckpt = args.checkpoint or os.path.join(out, 'best_model.pth')
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading: {ckpt}")
    (ptoi, ctoi, itoc, *_ ) = build_label_mappings()

    pnames = [None] * len(ptoi)
    for n, i in ptoi.items(): pnames[i] = n

    model = load_model(ckpt, len(ptoi), len(ctoi), device)
    prototypes = model.get_prototypes().cpu().numpy()
    print(f"Prototypes: {prototypes.shape}")
    sim = cosine_similarity(prototypes)

    # 1. Retrieval
    results = compute_retrieval(sim, pnames, KNOWN_PAIRS)
    print(f"\n{'='*55}")
    print(f"Drug → Target Gene Retrieval")
    print(f"{'='*55}")
    print(f"{'Drug':<22} {'#Conc':<7} {'Best Rank':<11} {'Recall@5':<10}")
    print(f"{'-'*55}")
    all_r = []
    for drug, info in sorted(results.items()):
        star = ' ★' if info['best_rank'] <= 5 else ''
        print(f"{drug:<22} {info['n_conc']:<7} {info['best_rank']:<11} {info['recall_5']:<8.0f}%{star}")
        all_r.append(info['best_rank'])
    if all_r:
        print(f"{'-'*55}")
        print(f"Mean Best Rank: {np.mean(all_r):.1f}  "
              f"R@1: {np.mean([r<=1 for r in all_r])*100:.0f}%  "
              f"R@5: {np.mean([r<=5 for r in all_r])*100:.0f}%")

    # 2. Silhouette
    sil = compute_silhouette(prototypes, pnames, MOA_GROUPS)
    if not np.isnan(sil): print(f"\nSilhouette (MOA): {sil:.4f}")

    # 3. t-SNE
    plot_tsne(prototypes, pnames, os.path.join(out, 'tsne_prototypes.png'))

    # 4. Per-drug nearest mutants
    print(f"\n{'='*55}")
    print("Top-5 Nearest Mutants Per Drug")
    print(f"{'='*55}")
    for drug_prefix in KNOWN_PAIRS:
        didxs = [i for i,n in enumerate(pnames) if n.startswith(f'drug_{drug_prefix}_')]
        if not didxs: continue
        print(f"\n{drug_prefix} ({len(didxs)} concentrations):")
        for d in didxs[:2]:
            name = pnames[d][5:]
            mids = [(sim[d,i], i) for i,n in enumerate(pnames) if n.startswith('mutant_')]
            mids.sort(reverse=True)
            top5 = [(pnames[i][7:], s) for s,i in mids[:5]]
            print(f"  [{name}] ", end="")
            for gn, sc in top5:
                is_t = '★' if gn in KNOWN_PAIRS.get(drug_prefix,[]) else ''
                print(f"{gn}{is_t}({sc:.3f}) ", end="")
            print()

    print(f"\nDone → {out}")


if __name__ == '__main__':
    main()
