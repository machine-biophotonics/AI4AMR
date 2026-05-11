#!/usr/bin/env python3
"""
Phase A: GPU-accelerated analysis of existing embeddings.

A1. L3 group-level analysis (MOA × Pathway) with permutation testing (GPU)
A2. Concentration-stratified analysis (higher dose → stronger match?)
A3. Cross-domain linear probes with PyTorch GPU training
"""

import os, json, re, warnings, argparse, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    'Cefsulodin': {'mrcA', 'mrcB'}, 'Penicillin': {'mrcA', 'mrcB', 'ftsI'},
    'Sulbactam': {'mrcA', 'mrcB', 'ftsI'}, 'Mecillinam': {'mrdA'},
    'Meropenem': {'mrdA', 'ftsI', 'mrcA', 'mrcB'}, 'Aztreonam': {'ftsI'},
    'Cefepim': {'ftsI', 'mrcA', 'mrcB', 'mrdA'}, 'Ceftriaxone': {'ftsI', 'mrcA', 'mrcB'},
    'Chloramphenicol': {'rplA', 'rplC'}, 'Clarithromycin': {'rplA', 'rplC'},
    'Doxicyclin': {'rpsA', 'rpsL'}, 'Kanamycin': {'rpsA', 'rpsL'},
    'Ciprofloxacin': {'gyrA', 'gyrB', 'parC', 'parE'},
    'Levofloxacin': {'gyrA', 'gyrB', 'parC', 'parE'},
    'Norfloxacin': {'gyrA', 'gyrB', 'parC', 'parE'},
    'Rifampicin': {'rpoA', 'rpoB'}, 'Trimethoprim': {'folA', 'folP'},
    'Colistin': {'lpxA', 'lpxC', 'lptA', 'lptC'},
    'Polymyxin_B': {'lpxA', 'lpxC', 'lptA', 'lptC'},
}
EXPECTED_L3 = {
    'Cell wall (PBP 1)': {'Aminoglycan biosynthesis', 'Cell shape regulation'},
    'Cell wall (PBP 2)': {'Cell shape regulation'},
    'Cell wall (PBP 3)': {'Cell shape regulation'},
    'Ribosome': {'Translation initiation'}, 'Gyrase': {'Chromosome organization'},
    'Membrane integrity': {'Lipid A biosynthesis', 'Cell envelope organization'},
    'RNA polymerase': {'Transcription elongation'},
    'DNA synthesis': {'Folic acid biosynthesis'}, 'Control': {'WT/NC'},
}

def load_jsons():
    return json.load(open(os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json'))), \
           json.load(open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')))

def fix_label(img_path, IC50, MUT):
    path_lower = img_path.lower()
    src = 'drug' if '/drugs_data/' in path_lower else ('mutant' if '/mutants_data/' in path_lower else None)
    if not src: return 'unknown'
    m = re.search(r'Well(\w\d+)_', os.path.basename(img_path))
    well = m.group(1) if m else None
    if not well: return 'unknown'
    pk = None
    for pn in range(1, 7):
        if f'/p{pn}/' in path_lower: pk = f'P{pn}'; break
    if not pk:
        m2 = re.search(r'/[Pp](\d)/', img_path)
        if m2: pk = f'P{m2.group(1)}'
    if not pk: return 'unknown'
    if src == 'drug':
        if pk in IC50 and well in IC50[pk]:
            info = IC50[pk][well]; ab, ic = info.get('antibiotic', ''), info.get('ic50_multiple', '')
            if ab and ic: return 'control' if ic == 'control' else f"{ab.replace(' ', '_')}_{ic if 'x' in str(ic) else f'{ic}x'}"
    else:
        row, col_raw = well[0], well[1:].lstrip('0') or '0'
        if pk in MUT and row in MUT[pk] and col_raw in MUT[pk][row]:
            return MUT[pk][row][col_raw].get('id', 'unknown')
    return 'unknown'

def ab_name(l): return l.rsplit('_', 1)[0] if '_' in l and l.rsplit('_', 1)[1].endswith('x') else l
def conc_name(l): return l.rsplit('_', 1)[1] if '_' in l and l.rsplit('_', 1)[1].endswith('x') else None
def gene_name(l):
    if l.startswith('WT NC') or l.startswith('NC'): return l
    if '_' in l and l.rsplit('_', 1)[1].replace('.', '').isdigit(): return l.rsplit('_', 1)[0]
    return l
def is_drug(l): return '_' in l and l.rsplit('_', 1)[1].endswith('x')
def get_moa(l): return ANTIBIOTIC_TO_MOA.get(ab_name(l), 'Unknown')
def get_pathway(l): return GENE_TO_PATHWAY.get(gene_name(l), 'Unknown')


def group_centroids_gpu(emb, labels_int, n_groups):
    """Compute L2-normalized centroids per group on GPU."""
    oh = torch.zeros(len(labels_int), n_groups, device=emb.device)
    oh[torch.arange(len(labels_int)), labels_int] = 1.0
    counts = oh.sum(dim=0, keepdim=True).T + 1e-8
    centroids = (oh.T @ emb) / counts
    return F.normalize(centroids, p=2, dim=1)


# ══════════════════════════════════════════════════════════════════════
#  A1: L3 Permutation Test (GPU-accelerated)
# ══════════════════════════════════════════════════════════════════════

def run_l3_permutation_test(drug_emb, drug_lbl, mut_emb, mut_lbl, output_dir, n_perm=5000):
    print("\n═══ A1: L3 Permutation Test (GPU) ═══")
    t0 = time.time()

    drug_moa_strs = np.array([get_moa(l) for l in drug_lbl])
    mut_path_strs = np.array([get_pathway(l) for l in mut_lbl])
    moas = sorted(set(drug_moa_strs))
    pathways = sorted(set(mut_path_strs))
    n_moa, n_path = len(moas), len(pathways)

    moa_to_idx = {m: i for i, m in enumerate(moas)}
    path_to_idx = {p: i for i, p in enumerate(pathways)}

    drug_labels = torch.tensor([moa_to_idx[m] for m in drug_moa_strs], device=DEVICE)
    mut_labels = torch.tensor([path_to_idx[p] for p in mut_path_strs], device=DEVICE)
    drug_emb_t = torch.tensor(drug_emb, device=DEVICE)
    mut_emb_t = torch.tensor(mut_emb, device=DEVICE)

    # Observed centroids and similarity
    drug_centroids = group_centroids_gpu(drug_emb_t, drug_labels, n_moa)
    mut_centroids = group_centroids_gpu(mut_emb_t, mut_labels, n_path)
    obs_sim = drug_centroids @ mut_centroids.T  # (n_moa, n_path)

    # Batched GPU permutation test
    batch_size = 250
    n_batches = (n_perm + batch_size - 1) // batch_size
    p_values = torch.ones((n_moa, n_path), device=DEVICE)

    rng = np.random.RandomState(42)
    perm_indices = np.zeros((n_perm, len(drug_labels)), dtype=np.int64)

    for b in range(n_batches):
        b_start = b * batch_size
        b_end = min(b_start + batch_size, n_perm)
        b_sz = b_end - b_start

        for i in range(b_sz):
            perm = b_start + i
            shuffled = drug_moa_strs.copy()
            rng.shuffle(shuffled)
            perm_indices[perm] = [moa_to_idx[m] for m in shuffled]

        perm_labels = torch.tensor(perm_indices[b_start:b_end], device=DEVICE)
        # (batch, N) labels

        # Compute centroids for all permutations in batch
        # One-hot: (batch, N, n_moa)
        oh = torch.zeros(b_sz, len(drug_labels), n_moa, device=DEVICE)
        oh.scatter_(2, perm_labels.unsqueeze(-1), 1.0)
        counts = oh.sum(dim=1, keepdim=True).transpose(1, 2) + 1e-8
        # (batch, n_moa, N) @ (N, 1280) → (batch, n_moa, 1280)
        centroids_batch = (oh.transpose(1, 2) @ drug_emb_t.unsqueeze(0).expand(b_sz, -1, -1)) / counts
        centroids_batch = F.normalize(centroids_batch, p=2, dim=2)

        # Similarity: (batch, n_moa, n_path)
        sim_batch = centroids_batch @ mut_centroids.T.unsqueeze(0)
        p_values += (sim_batch >= obs_sim.unsqueeze(0)).sum(dim=0)

        if (b + 1) % 4 == 0:
            print(f"  Batch {b+1}/{n_batches} ({b_sz} perms)  "
                  f"elapsed={time.time()-t0:.0f}s")

    p_values /= (n_perm + 1)
    sig_mask = p_values < 0.05

    # Print
    print(f"\n  L3 Similarity (● = p<0.05):")
    header = f"{'MOA':25s} " + " ".join(f"{p[:18]:18s}" for p in pathways)
    print(header)
    print("  " + "-" * min(200, 25 + 20 * n_path))
    for i, m in enumerate(moas):
        row = f"  {m:25s} "
        for j in range(n_path):
            mark = "●" if sig_mask[i, j] else "○"
            row += f"{obs_sim[i,j]:+.3f}{mark} "
        print(row)

    expected_hits = sum(1 for moa in moas for p in EXPECTED_L3.get(moa, set()) if p in pathways)
    sig_expected = sum(1 for i, m in enumerate(moas) for p in EXPECTED_L3.get(m, set())
                       if p in pathways and sig_mask[i, pathways.index(p)])
    print(f"\n  Expected pairs: {expected_hits}, Significant: {sig_expected}/{expected_hits}")
    print(f"  Time: {time.time()-t0:.0f}s")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    for ax, mat, title in zip(axes,
                               [obs_sim.cpu().numpy(), -np.log10(p_values.cpu().numpy() + 1e-10)],
                               ['Cosine Similarity (MOA × Pathway)',
                                f'-log10(p-value) — {n_perm} permutations (GPU)']):
        sns.heatmap(mat, xticklabels=pathways, yticklabels=moas, ax=ax,
                    cmap='RdBu_r' if 'Cosine' in title else 'viridis',
                    center=0 if 'Cosine' in title else None,
                    annot=True, fmt='.2f', annot_kws={'fontsize': 7})
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    fig.suptitle(f'A1: L3 Permutation (GPU, {n_perm} perms) | Significant: {sig_expected}/{expected_hits}',
                 fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, 'phaseA1_l3_permutation.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return {'obs_sim': obs_sim.cpu().numpy(), 'p_values': p_values.cpu().numpy(),
            'sig_mask': sig_mask.cpu().numpy(), 'moas': moas, 'pathways': pathways,
            'n_sig_expected': sig_expected}


# ══════════════════════════════════════════════════════════════════════
#  A2: Concentration Dependence
# ══════════════════════════════════════════════════════════════════════

def run_concentration_analysis(drug_emb, drug_lbl, mut_emb, mut_lbl, output_dir):
    print("\n═══ A2: Concentration Dependence ═══")
    t0 = time.time()

    ab_list = np.array([ab_name(l) for l in drug_lbl])
    conc_list = np.array([conc_name(l) for l in drug_lbl])
    gene_list = np.array([gene_name(l) for l in mut_lbl])
    conc_order = {'0.25x': 1, '0.5x': 2, '1x': 3, '2x': 4}
    concentrations = ['0.25x', '0.5x', '1x', '2x']

    drug_emb_t = torch.tensor(drug_emb, device=DEVICE)
    mut_emb_t = torch.tensor(mut_emb, device=DEVICE)
    antibiotics = sorted(set(ab_list))

    results = []
    for ab in antibiotics:
        expected = EXPECTED_MATCHES.get(ab, set())
        if not expected:
            continue

        conc_centroids = {}
        for c in concentrations:
            mask = (ab_list == ab) & (conc_list == c)
            if mask.sum() > 0:
                cent = drug_emb_t[mask].mean(dim=0)
                cent = F.normalize(cent.unsqueeze(0), p=2, dim=1).squeeze(0)
                conc_centroids[c] = cent

        if len(conc_centroids) < 2:
            continue

        target_mask = torch.zeros(len(mut_emb_t), dtype=torch.bool, device=DEVICE)
        for gene in expected:
            target_mask |= (torch.tensor([g == gene for g in gene_list], device=DEVICE))
        if target_mask.sum() == 0:
            continue
        target_cent = F.normalize(mut_emb_t[target_mask].mean(dim=0).unsqueeze(0), p=2, dim=1).squeeze(0)

        x_vals, y_vals = [], []
        for c in concentrations:
            if c in conc_centroids:
                s = float(conc_centroids[c] @ target_cent)
                x_vals.append(conc_order[c])
                y_vals.append(s)

        if len(x_vals) >= 3:
            rho, p = spearmanr(x_vals, y_vals)
        else:
            rho, p = 0, 1.0

        results.append({
            'antibiotic': ab, 'expected_genes': ','.join(expected),
            'n_conc': len(conc_centroids),
            'sim_0.25x': f"{y_vals[0]:+.3f}" if len(y_vals) > 0 else '—',
            'sim_0.5x': f"{y_vals[1]:+.3f}" if len(y_vals) > 1 else '—',
            'sim_1x': f"{y_vals[2]:+.3f}" if len(y_vals) > 2 else '—',
            'sim_2x': f"{y_vals[3]:+.3f}" if len(y_vals) > 3 else '—',
            'trend': f"{rho:+.2f} (p={p:.2f})",
            'increasing': '✓' if rho > 0.5 else '✗' if rho < -0.5 else '·',
        })

        trend_mark = '↑' if rho > 0.3 else '↓' if rho < -0.3 else '→'
        conc_str = ' '.join(f'{c}={y_vals[j]:+.3f}' for j, c in enumerate(concentrations) if j < len(y_vals))
        print(f"  {ab:20s} | {conc_str} | ρ={rho:+.2f} {trend_mark}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'phaseA2_concentration_dependence.csv'), index=False)
    n_inc = sum(1 for r in results if r.get('increasing') == '✓')
    n_dec = sum(1 for r in results if r.get('increasing') == '✗')
    print(f"  Increasing trend: {n_inc}/{len(results)}, Decreasing: {n_dec}/{len(results)}  "
          f"({time.time()-t0:.0f}s)")
    return results


# ══════════════════════════════════════════════════════════════════════
#  A3: Cross-domain Linear Probes (GPU)
# ══════════════════════════════════════════════════════════════════════

class LinearProbe(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, x):
        return self.fc(x)

def train_probe(X_train, y_train, X_test, y_test, n_classes, lr=1e-3, epochs=200):
    X_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    X_te = torch.tensor(X_test, dtype=torch.float32, device=DEVICE) if X_test is not None else None
    y_te = torch.tensor(y_test, dtype=torch.long, device=DEVICE) if y_test is not None else None

    model = LinearProbe(X_t.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_t)
        loss = F.cross_entropy(logits, y_t)
        loss.backward()
        opt.step()

        if X_te is not None and (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(X_te).argmax(1)
                acc = (preds == y_te).float().mean().item()
                best_acc = max(best_acc, acc)

    model.eval()
    with torch.no_grad():
        train_preds = model(X_t).argmax(1)
        train_acc = (train_preds == y_t).float().mean().item()
        test_acc = None
        if X_te is not None:
            test_preds = model(X_te).argmax(1)
            test_acc = (test_preds == y_te).float().mean().item()

    return train_acc, test_acc or best_acc


def run_linear_probes(drug_emb, drug_lbl, mut_emb, mut_lbl, output_dir):
    print("\n═══ A3: Cross-domain Linear Probes (GPU) ═══")
    t0 = time.time()

    drug_moas = np.array([get_moa(l) for l in drug_lbl])
    drug_pathways = np.array([get_pathway(l) for l in drug_lbl])
    mut_moas = np.array([get_moa(l) if is_drug(l) else 'Unknown' for l in mut_lbl])
    mut_pathways = np.array([get_pathway(l) for l in mut_lbl])

    scaler = StandardScaler()
    drug_emb_s = scaler.fit_transform(drug_emb).astype(np.float32)
    mut_emb_s = scaler.transform(mut_emb).astype(np.float32)

    # Cross-domain probes:
    #   Train on drug emb → MOA. Test on mutant emb → assign pseudo-MOA via EXPECTED_MATCHES.
    #   Train on mutant emb → Pathway. Test on drug emb → assign pseudo-pathway via reverse mapping.
    ab_to_moa_rev = {ab: moa for moa, abx in MOA_GROUPS.items() for ab in abx}
    drug_moa_from_mutant = np.array([
        next((ab_to_moa_rev[ab] for ab, genes in EXPECTED_MATCHES.items() if gene_name(l) in genes), 'Unknown')
        for l in mut_lbl
    ])
    gene_to_pathway_rev = GENE_TO_PATHWAY
    # For drug labels, get which genes they target → pathway
    drug_pseudo_pathway = np.array([
        next((GENE_TO_PATHWAY[g] for ab, genes in EXPECTED_MATCHES.items()
              if ab_name(l) == ab for g in genes if g in GENE_TO_PATHWAY), 'Unknown')
        for l in drug_lbl
    ])

    tasks = [
        ('Drug MOA → self (oracle)', drug_moas, drug_emb_s, drug_emb_s, None, 'self'),
        ('Mutant Pathway → self (oracle)', mut_pathways, mut_emb_s, mut_emb_s, None, 'self'),
        ('Drug MOA → Mutant (cross)', drug_moas, drug_emb_s, mut_emb_s, drug_moa_from_mutant, 'cross'),
        ('Mutant Pathway → Drug (cross)', mut_pathways, mut_emb_s, drug_emb_s, drug_pseudo_pathway, 'cross'),
    ]

    all_results = []
    for name, y_train_str, X_train_raw, X_test_raw, y_test_str, mode in tasks:
        # Filter unknowns from train labels
        train_keep = y_train_str != 'Unknown'
        y_train_f = y_train_str[train_keep]
        unique = sorted(set(y_train_f))
        n_classes = len(unique)
        if n_classes == 0:
            print(f"  {name:40s} | SKIP (no valid labels)")
            continue
        baseline = 1.0 / n_classes
        cls_map = {u: i for i, u in enumerate(unique)}
        y_train = np.array([cls_map[y] for y in y_train_f])
        X_train = X_train_raw[train_keep]

        if mode == 'self':
            train_acc, _ = train_probe(X_train, y_train, None, None, n_classes)
            result = {'probe': name, 'n_train': len(X_train), 'n_classes': n_classes,
                      'random_baseline': f"{baseline:.1%}", 'train_acc': f"{train_acc:.1%}",
                      'test_acc': 'N/A'}

        elif mode == 'cross':
            # Filter test labels to only classes seen in training
            test_keep = y_test_str != 'Unknown'
            y_test_f = y_test_str[test_keep]
            test_keep2 = np.array([y in unique for y in y_test_f])
            X_test = X_test_raw[test_keep][test_keep2]
            y_test = np.array([cls_map[y] for y in y_test_f])[test_keep2]

            if len(X_test) == 0 or len(np.unique(y_test)) < 2:
                print(f"  {name:40s} | SKIP (no overlapping classes)")
                continue

            # Train on full source domain, test on target domain
            _, test_acc = train_probe(X_train, y_train, X_test, y_test, n_classes)
            result = {'probe': name, 'n_train': len(X_train), 'n_test': len(X_test),
                      'n_classes': n_classes, 'random_baseline': f"{baseline:.1%}",
                      'train_acc': 'N/A', 'test_acc': f"{test_acc:.1%}"}

        all_results.append(result)
        if mode == 'self':
            print(f"  {name:40s} | train={result['train_acc']} (baseline={baseline:.1%})")
        else:
            print(f"  {name:40s} | test={result['test_acc']} (baseline={baseline:.1%})")

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, 'phaseA3_linear_probes.csv'), index=False)
    print(f"  Time: {time.time()-t0:.0f}s")
    return all_results


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', default='both/fold_Plate_1/embeddings_Plate_1_mil_n3.npz')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--n_perm', type=int, default=5000, help='Permutations for A1')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.embeddings), 'phase_a')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Using {args.n_perm} permutations for A1")

    data = np.load(args.embeddings)
    paths = data['paths']

    IC50, MUT = load_jsons()
    labels = np.array([fix_label(p, IC50, MUT) for p in paths])
    print(f"Total: {len(labels)}, Unique: {len(np.unique(labels))}")

    d_mask = np.array([is_drug(l) or l == 'control' for l in labels])
    m_mask = np.array([not is_drug(l) and l != 'control' and l != 'unknown' for l in labels])

    drug_emb = data['embeddings'][d_mask].astype(np.float32)
    drug_lbl = labels[d_mask]
    mut_emb = data['embeddings'][m_mask].astype(np.float32)
    mut_lbl = labels[m_mask]
    print(f"Drug: {len(drug_emb)}, Mutant: {len(mut_emb)}")

    a1 = run_l3_permutation_test(drug_emb, drug_lbl, mut_emb, mut_lbl, output_dir, n_perm=args.n_perm)
    a2 = run_concentration_analysis(drug_emb, drug_lbl, mut_emb, mut_lbl, output_dir)
    a3 = run_linear_probes(drug_emb, drug_lbl, mut_emb, mut_lbl, output_dir)

    print("\n" + "=" * 60)
    print("  PHASE A SUMMARY (GPU)")
    print("=" * 60)
    print(f"\n  A1 — L3 Permutation Test ({args.n_perm} perms, GPU):")
    print(f"    Significant expected MOA↔Pathway pairs: {a1['n_sig_expected']}")
    n_inc = sum(1 for r in a2 if r.get('increasing') == '✓')
    print(f"\n  A2 — Concentration Dependence:")
    print(f"    Increasing trend: {n_inc}/{len(a2)}")
    print(f"\n  A3 — Cross-domain Linear Probes (GPU):")
    for r in a3:
        test = f"test={r['test_acc']}" if r.get('test_acc', 'N/A') != 'N/A' else ''
        print(f"    {r['probe']:40s} {test}  (baseline={r['random_baseline']})")
    print(f"\n  Outputs: {output_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()
