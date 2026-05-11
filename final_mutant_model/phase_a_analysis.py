#!/usr/bin/env python3
"""
Phase A: Better analysis of existing embeddings.

A1. L3 group-level analysis (MOA × Pathway) with permutation testing
A2. Concentration-stratified analysis (higher dose → stronger match?)
A3. Cross-domain linear probes (predict MOA from mutant, pathway from drug)

All use the existing embeddings — no retraining needed.
"""

import os, json, re, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Group mappings (identical to similarity_heatmap_multi.py) ───

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

EXPECTED_MATCHES = {
    'Cefsulodin': {'mrcA', 'mrcB'},
    'Penicillin': {'mrcA', 'mrcB', 'ftsI'},
    'Sulbactam': {'mrcA', 'mrcB', 'ftsI'},
    'Mecillinam': {'mrdA'},
    'Meropenem': {'mrdA', 'ftsI', 'mrcA', 'mrcB'},
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

EXPECTED_L3 = {
    'Cell wall (PBP 1)': {'Aminoglycan biosynthesis', 'Cell shape regulation'},
    'Cell wall (PBP 2)': {'Cell shape regulation'},
    'Cell wall (PBP 3)': {'Cell shape regulation'},
    'Ribosome': {'Translation initiation'},
    'Gyrase': {'Chromosome organization'},
    'Membrane integrity': {'Lipid A biosynthesis', 'Cell envelope organization'},
    'RNA polymerase': {'Transcription elongation'},
    'DNA synthesis': {'Folic acid biosynthesis'},
    'Control': {'WT/NC'},
}

# ─── Helpers ───

def load_jsons():
    ic50 = json.load(open(os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')))
    mut = json.load(open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')))
    return ic50, mut

def fix_label(img_path, IC50, MUT):
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
        m2 = re.search(r'/[Pp](\d)/', img_path)
        if m2:
            pk = f'P{m2.group(1)}'
    if not pk:
        return 'unknown'
    if src == 'drug':
        if pk in IC50 and well in IC50[pk]:
            info = IC50[pk][well]
            ab, ic = info.get('antibiotic', ''), info.get('ic50_multiple', '')
            if ab and ic:
                return 'control' if ic == 'control' else f"{ab.replace(' ', '_')}_{ic if 'x' in str(ic) else f'{ic}x'}"
    else:
        row, col_raw = well[0], well[1:].lstrip('0') or '0'
        if pk in MUT and row in MUT[pk] and col_raw in MUT[pk][row]:
            return MUT[pk][row][col_raw].get('id', 'unknown')
    return 'unknown'

def extract_antibiotic(label):
    if '_' in label:
        parts = label.rsplit('_', 1)
        if parts[1].endswith('x'):
            return parts[0]
    return label

def extract_concentration(label):
    if '_' in label:
        parts = label.rsplit('_', 1)
        if parts[1].endswith('x'):
            return parts[1]
    return None

def extract_gene(label):
    if label.startswith('WT NC') or label.startswith('NC'):
        return label
    if '_' in label:
        parts = label.rsplit('_', 1)
        if parts[1].replace('.', '').isdigit():
            return parts[0]
    return label

def is_drug(label):
    return '_' in label and label.rsplit('_', 1)[1].endswith('x')

def get_moa(label):
    ab = extract_antibiotic(label)
    return ANTIBIOTIC_TO_MOA.get(ab, 'Unknown')

def get_pathway(label):
    gene = extract_gene(label)
    return GENE_TO_PATHWAY.get(gene, 'Unknown')


# ══════════════════════════════════════════════════════════════════════
#  A1: L3 Permutation Test
# ══════════════════════════════════════════════════════════════════════

def run_l3_permutation_test(drug_emb, drug_lbl, mut_emb, mut_lbl, output_dir, n_perm=2000):
    """
    For each MOA group, compute cosine similarity to each Pathway group.
    Permutation test: shuffle MOA labels, recompute, get p-value.
    """
    print("\n═══ A1: L3 Permutation Test ═══")

    # Get MOA per drug sample
    drug_moas = np.array([get_moa(l) for l in drug_lbl])
    mut_pathways = np.array([get_pathway(l) for l in mut_lbl])

    moas = sorted(set(drug_moas))
    pathways = sorted(set(mut_pathways))
    n_moa, n_path = len(moas), len(pathways)

    # Compute centroids
    drug_centroids = np.array([drug_emb[drug_moas == m].mean(axis=0) for m in moas])
    mut_centroids = np.array([mut_emb[mut_pathways == p].mean(axis=0) for p in pathways])

    # Normalize
    dc = drug_centroids / (np.linalg.norm(drug_centroids, axis=1, keepdims=True) + 1e-8)
    mc = mut_centroids / (np.linalg.norm(mut_centroids, axis=1, keepdims=True) + 1e-8)
    obs_sim = dc @ mc.T

    # Permutation test: shuffle MOA labels, recompute centroids
    rng = np.random.RandomState(42)
    p_values = np.ones((n_moa, n_path))
    n_sig = np.zeros((n_moa, n_path))

    for perm in range(n_perm):
        if perm % 500 == 0 and perm > 0:
            print(f"  Permutation {perm}/{n_perm}")
        shuffled = drug_moas.copy()
        rng.shuffle(shuffled)
        perm_centroids = np.array([drug_emb[shuffled == m].mean(axis=0) for m in moas])
        perm_centroids /= (np.linalg.norm(perm_centroids, axis=1, keepdims=True) + 1e-8)
        perm_sim = perm_centroids @ mc.T
        p_values[perm_sim >= obs_sim] += 1

    p_values /= (n_perm + 1)
    sig_mask = p_values < 0.05

    # Print results
    print(f"\n  L3 Similarity Matrix (Observed):")
    print(f"  {'MOA':25s} ", end="")
    for p in pathways:
        print(f"{p[:18]:18s}", end="")
    print()
    print("  " + "-" * (25 + 20 * n_path))
    for i, m in enumerate(moas):
        print(f"  {m:25s} ", end="")
        for j in range(n_path):
            marker = "●" if sig_mask[i, j] else "○"
            print(f"{obs_sim[i,j]:+.3f}{marker:4s}", end="     ")
        print()

    # Count
    n_expected_hits = sum(1 for moa in moas for p in EXPECTED_L3.get(moa, set()) if p in pathways)
    n_obs_hits = sum(1 for i, m in enumerate(moas) for p in EXPECTED_L3.get(m, set())
                     if p in pathways and obs_sim[i, pathways.index(p)] > 0)
    n_sig_expected = sum(1 for i, m in enumerate(moas) for p in EXPECTED_L3.get(m, set())
                         if p in pathways and sig_mask[i, pathways.index(p)])
    print(f"\n  Expected L3 pairs: {n_expected_hits}")
    print(f"  Positive similarity (observed): {n_obs_hits}/{n_expected_hits}")
    print(f"  Significant (p<0.05): {n_sig_expected}/{n_expected_hits}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    for ax, mat, title in zip(axes,
                               [obs_sim, -np.log10(p_values + 1e-10)],
                               ['Cosine Similarity (MOA × Pathway)',
                                '-log10(p-value) — 2000 permutations']):
        sns.heatmap(mat, xticklabels=pathways, yticklabels=moas,
                    ax=ax, cmap='RdBu_r' if 'Cosine' in title else 'viridis',
                    center=0 if 'Cosine' in title else None,
                    annot=True, fmt='.2f', annot_kws={'fontsize': 7})
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    fig.suptitle(f'A1: L3 Permutation Test | Significant: {n_sig_expected}/{n_expected_hits} expected pairs',
                 fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, 'phaseA1_l3_permutation.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return {'obs_sim': obs_sim, 'p_values': p_values, 'sig_mask': sig_mask,
            'moas': moas, 'pathways': pathways, 'n_sig_expected': n_sig_expected}


# ══════════════════════════════════════════════════════════════════════
#  A2: Concentration Dependence
# ══════════════════════════════════════════════════════════════════════

def run_concentration_analysis(drug_emb, drug_lbl, mut_emb, mut_lbl, output_dir):
    """
    For each antibiotic, check if higher concentration → stronger match to target mutant.
    """
    print("\n═══ A2: Concentration Dependence ═══")

    # Get antibiotic and concentration from each drug sample
    ab_list = np.array([extract_antibiotic(l) for l in drug_lbl])
    conc_list = np.array([extract_concentration(l) for l in drug_lbl])
    conc_order = {'0.25x': 1, '0.5x': 2, '1x': 3, '2x': 4}
    conc_vals = np.array([conc_order.get(c, 0) for c in conc_list])

    # Get gene for each mutant sample
    gene_list = np.array([extract_gene(l) for l in mut_lbl])

    antibiotics = sorted(set(ab_list))
    concentrations = ['0.25x', '0.5x', '1x', '2x']

    results = []
    for ab in antibiotics:
        expected = EXPECTED_MATCHES.get(ab, set())
        if not expected:
            continue

        # Concentrations centroids
        conc_centroids = {}
        for c in concentrations:
            mask = (ab_list == ab) & (conc_list == c)
            if mask.sum() > 0:
                cent = drug_emb[mask].mean(axis=0)
                cent /= (np.linalg.norm(cent) + 1e-8)
                conc_centroids[c] = cent

        if len(conc_centroids) < 2:
            continue

        # Target mutant centroids  (average over all guides)
        target_mask = np.zeros(len(mut_emb), dtype=bool)
        for gene in expected:
            target_mask |= (gene_list == gene)
        if target_mask.sum() == 0:
            continue
        target_cent = mut_emb[target_mask].mean(axis=0)
        target_cent /= (np.linalg.norm(target_cent) + 1e-8)

        # Non-target mutant centroids
        nontarget_mask = ~target_mask
        nontarget_cent = mut_emb[nontarget_mask].mean(axis=0)
        nontarget_cent /= (np.linalg.norm(nontarget_cent) + 1e-8)

        # Compute similarity per concentration
        sims = []
        for c in concentrations:
            if c in conc_centroids:
                s = float(conc_centroids[c] @ target_cent)
                sims.append(s)

        # Spearman correlation: higher conc → higher similarity?
        x = [conc_order[c] for c in concentrations if c in conc_centroids]
        y = [float(conc_centroids[c] @ target_cent) for c in concentrations if c in conc_centroids]
        if len(x) >= 3:
            rho, p = spearmanr(x, y)
        else:
            rho, p = 0, 1.0

        results.append({
            'antibiotic': ab, 'expected_genes': ','.join(expected),
            'n_conc': len(conc_centroids),
            'sim_0.25x': f"{conc_centroids.get('0.25x', '@target_cent'):+.3f}" if '0.25x' in conc_centroids else '—',
            'sim_0.5x': f"{sims[1]:+.3f}" if len(sims) > 1 else '—',
            'sim_1x': f"{sims[2]:+.3f}" if len(sims) > 2 else '—',
            'sim_2x': f"{sims[3]:+.3f}" if len(sims) > 3 else '—',
            'trend': f"{rho:+.2f} (p={p:.2f})",
            'increasing': '✓' if rho > 0.5 else '✗' if rho < -0.5 else '·',
        })

        trend = '↑' if rho > 0.3 else '↓' if rho < -0.3 else '→'
        print(f"  {ab:20s} | {','.join(expected):15s} | "
              f"{' '.join(f'{c}={conc_centroids.get(c, \"\")[:6]:>6s}' if c in conc_centroids else '' for c in concentrations)} | "
              f"ρ={rho:+.2f} {trend}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'phaseA2_concentration_dependence.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    n_inc = sum(1 for r in results if r.get('increasing') == '✓')
    n_dec = sum(1 for r in results if r.get('increasing') == '✗')
    print(f"  Increasing trend: {n_inc}/{len(results)}, Decreasing: {n_dec}/{len(results)}")

    return results


# ══════════════════════════════════════════════════════════════════════
#  A3: Cross-domain Linear Probes
# ══════════════════════════════════════════════════════════════════════

def run_linear_probes(drug_emb, drug_lbl, mut_emb, mut_lbl, output_dir):
    """
    Train logistic regression to predict group labels across domains:
      - Drug→Pathway: train on drug embeddings to predict pathway labels
      - Mutant→MOA: train on mutant embeddings to predict MOA labels
    """
    print("\n═══ A3: Cross-domain Linear Probes ═══")

    # Prepare labels
    drug_moas = np.array([get_moa(l) for l in drug_lbl])
    drug_pathways = np.array([get_pathway(l) for l in drug_lbl])
    mut_moas = np.array([get_moa(l) if is_drug(l) else 'Unknown' for l in mut_lbl])
    mut_pathways = np.array([get_pathway(l) for l in mut_lbl])

    # Scale embeddings
    scaler = StandardScaler()
    drug_emb_s = scaler.fit_transform(drug_emb)
    mut_emb_s = scaler.transform(mut_emb)

    probes = [
        ('Drug MOA → self (oracle)', drug_emb_s, drug_moas, None, None),
        ('Mutant Pathway → self (oracle)', mut_emb_s, mut_pathways, None, None),
        ('Drug → Predict Mutant Pathway', drug_emb_s, drug_pathways, mut_emb_s, mut_pathways),
        ('Mutant → Predict Drug MOA', mut_emb_s, mut_moas, drug_emb_s, drug_moas),
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_results = []
    for name, X_train, y_train, X_test, y_test in probes:
        # Filter unknown
        if y_test is not None:
            keep_test = y_test != 'Unknown'
            X_test = X_test[keep_test]
            y_test = y_test[keep_test]

        keep_train = y_train != 'Unknown'
        X_train = X_train[keep_train]
        y_train = y_train[keep_train]

        n_classes = len(set(y_train))
        baseline = 1.0 / n_classes

        # Cross-validated accuracy
        lr = LogisticRegression(max_iter=5000, multi_class='multinomial', solver='lbfgs')
        cv_scores = cross_val_score(lr, X_train, y_train, cv=cv, scoring='balanced_accuracy')
        lr.fit(X_train, y_train)

        result = {
            'probe': name,
            'n_train': len(X_train),
            'n_test': len(X_test) if X_test is not None else 0,
            'n_classes': n_classes,
            'random_baseline': f"{baseline:.1%}",
            'cv_balanced_acc_mean': f"{cv_scores.mean():.1%}",
            'cv_balanced_acc_std': f"{cv_scores.std():.1%}",
            'above_random': '✓' if cv_scores.mean() > baseline * 1.2 else '✗' if cv_scores.mean() < baseline else '·',
        }

        if X_test is not None and len(X_test) > 0:
            # Map test labels to train label set
            train_labels = set(y_train)
            test_mask = np.array([l in train_labels for l in y_test])
            if test_mask.sum() > 0:
                y_test_f = y_test[test_mask]
                X_test_f = X_test[test_mask]
                preds = lr.predict(X_test_f)
                bacc = balanced_accuracy_score(y_test_f, preds)
                acc = accuracy_score(y_test_f, preds)
                result['test_balanced_acc'] = f"{bacc:.1%}"
                result['test_acc'] = f"{acc:.1%}"
            else:
                result['test_balanced_acc'] = 'N/A'
                result['test_acc'] = 'N/A'
        else:
            result['test_balanced_acc'] = 'N/A'
            result['test_acc'] = 'N/A'

        all_results.append(result)

        print(f"  {name:45s} | train={result['n_train']} | "
              f"CV bal.acc={result['cv_balanced_acc_mean']}±{result['cv_balanced_acc_std']} "
              f"(baseline={result['random_baseline']}) {result['above_random']}")

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, 'phaseA3_linear_probes.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    return all_results


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', default='both/fold_Plate_1/embeddings_Plate_1_mil_n3.npz')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--n_perm', type=int, default=2000, help='Permutations for A1')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.embeddings), 'phase_a')
    os.makedirs(output_dir, exist_ok=True)

    # Load embeddings
    print("Loading embeddings...")
    data = np.load(args.embeddings)
    embeddings = data['embeddings']
    paths = data['paths']

    # Fix labels
    IC50, MUT = load_jsons()
    labels = np.array([fix_label(p, IC50, MUT) for p in paths])
    print(f"  Total: {len(labels)}, Unique: {len(np.unique(labels))}")

    # Separate drug vs mutant
    d_mask = np.array([is_drug(l) or l == 'control' for l in labels])
    m_mask = np.array([not is_drug(l) and l != 'control' and l != 'unknown' for l in labels])

    drug_emb = embeddings[d_mask]
    drug_lbl = labels[d_mask]
    mut_emb = embeddings[m_mask]
    mut_lbl = labels[m_mask]
    print(f"  Drug: {len(drug_emb)}, Mutant: {len(mut_emb)}")

    # ── Run A1 ──
    a1 = run_l3_permutation_test(drug_emb, drug_lbl, mut_emb, mut_lbl, output_dir, n_perm=args.n_perm)

    # ── Run A2 ──
    a2 = run_concentration_analysis(drug_emb, drug_lbl, mut_emb, mut_lbl, output_dir)

    # ── Run A3 ──
    a3 = run_linear_probes(drug_emb, drug_lbl, mut_emb, mut_lbl, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("  PHASE A SUMMARY")
    print("=" * 60)
    print(f"\n  A1 — L3 Permutation Test:")
    print(f"    Significant expected MOA↔Pathway pairs: {a1['n_sig_expected']}")
    print(f"\n  A2 — Concentration Dependence:")
    n_inc = sum(1 for r in a2 if r.get('increasing') == '✓')
    print(f"    Drugs with increasing trend: {n_inc}/{len(a2)}")
    print(f"\n  A3 — Cross-domain Linear Probes:")
    for r in a3:
        print(f"    {r['probe']:45s} CV={r['cv_balanced_acc_mean']} (baseline={r['random_baseline']})")

    print(f"\n  All outputs saved to {output_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()
