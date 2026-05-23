#!/usr/bin/env python3
"""
Harmony batch correction for drug↔mutant embedding alignment.

WHAT HARMONY DOES:
- Harmony (Korsunsky et al. 2019, Nature Methods) is a batch correction method.
- It takes PCA embeddings and adjusts them so samples from different batches
  (here: "drug" = antibiotic-treated WT E. coli, "mutant" = gene knockdown E. coli)
  occupy the same region of PCA space.
- It works by: soft-clustering → adjusting cluster centroids to balance
  batch proportions → shifting each point via its cluster memberships.
- The goal: remove technical domain shifts so drug and mutant embeddings
  become comparable for cross-domain classification.

OUTPUT:
1. harmony_tsne_analysis.png — tSNE before/after, domain overlap, class structure
2. harmony_cosine_heatmap.png — Gene × Drug cosine similarity before/after
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from harmony_corrector import HarmonyCorrector

# ── Helpers ──────────────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def get_gene(lbl):
    g = str(lbl).rsplit('_', 1)[0]
    return g if g not in ('NC', 'WT NC') else None

def get_drug(lbl):
    s = str(lbl)
    parts = s.rsplit('_', 1)
    if len(parts) < 2 or parts[0] in ('NC', 'WT NC'):
        return None
    return parts[0]

# ── Load embeddings ─────────────────────────────────────────────────────────────

data = np.load('both/fold_Plate_1/embeddings_Plate_1_mil_n3_patched.npz', allow_pickle=True)
embs = data['embeddings']          # (4032, 1280)
labels = np.array([str(l) for l in data['labels']])
paths = data['paths']

# Split into drug vs mutant
drug_mask = np.array(['Drugs_Data' in str(p) for p in paths])
mut_mask = ~drug_mask

drug_emb = embs[drug_mask];   drug_labels = labels[drug_mask]
mut_emb  = embs[mut_mask];    mut_labels  = labels[mut_mask]

print(f"Drug: {drug_emb.shape}, {len(set(drug_labels))} classes")
print(f"Mutant: {mut_emb.shape}, {len(set(mut_labels))} classes")

# ── PCA ──────────────────────────────────────────────────────────────────────────

pca_50 = PCA(n_components=50)
pca_embs = pca_50.fit_transform(embs)          # (4032, 50) — BEFORE Harmony

# ── Harmony ──────────────────────────────────────────────────────────────────────
# CRITICAL: domain labels must match the row order of embs.
# The buggy code used ['drug']*N + ['mutant']*M which assumed all drugs came first.
# Drug and mutant samples are INTERLEAVED in the data, so 50% of labels were wrong.

domain_labels = np.full(len(embs), 'mutant', dtype=object)
domain_labels[drug_mask] = 'drug'

hc = HarmonyCorrector(n_pca=50, batch_vars=['domain'])
corrected_pca, _ = hc.fit_transform(embs, {'domain': domain_labels.tolist()})
# corrected_pca = (4032, 50) — AFTER Harmony

drug_pca = pca_embs[drug_mask];  mut_pca = pca_embs[mut_mask]
drug_cor = corrected_pca[drug_mask];  mut_cor = corrected_pca[mut_mask]

# ── tSNE (single joint run, so BEFORE and AFTER share the same 2D space) ────────

print("Running tSNE on combined BEFORE + AFTER...")
combined_pca = np.vstack([pca_embs, corrected_pca])  # (8064, 50)
tsne_joint = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000).fit_transform(combined_pca)
tsne_before = tsne_joint[:len(embs)]
tsne_after  = tsne_joint[len(embs):]

# Compute Harmony shift in PCA space (magnitude per point)
delta_pca = corrected_pca - pca_embs
shift_norms = np.linalg.norm(delta_pca, axis=1)

domain_colors = {'drug': '#1f77b4', 'mutant': '#ff7f0e'}

fig, axes = plt.subplots(2, 3, figsize=(24, 14))

for col, (tsne, title, label) in enumerate([
    (tsne_before, 't-SNE space — BEFORE Harmony', 'BEFORE'),
    (tsne_after,  't-SNE space — AFTER Harmony',  'AFTER'),
]):
    # Row 0: drug (blue) vs mutant (orange)
    ax = axes[0, col]
    for dom, c in domain_colors.items():
        m = drug_mask if dom == 'drug' else mut_mask
        ax.scatter(tsne[m, 0], tsne[m, 1], c=c, label=dom, s=4, alpha=0.4)
    ax.set_title(f'{title}', fontsize=14, fontweight='bold')
    ax.legend(markerscale=8, fontsize=11)
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

    # Row 1: by biological class
    ax = axes[1, col]
    all_labels = list(drug_labels) + list(mut_labels)
    unique_cls = sorted(set(all_labels))
    cls_to_int = {c: j for j, c in enumerate(unique_cls)}
    int_lbls = [cls_to_int[l] for l in all_labels]
    cmap = 'tab20' if len(unique_cls) <= 20 else 'gist_ncar'
    ax.scatter(tsne[:, 0], tsne[:, 1], c=int_lbls, cmap=cmap, s=4, alpha=0.6)
    ax.set_title(f'{title} (by class)', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

# Row 0 col 2: histogram of shift magnitudes in PCA space
ax = axes[0, 2]
ax.hist(shift_norms, bins=50, color='gray', edgecolor='white', alpha=0.7)
ax.axvline(shift_norms.mean(), color='red', linestyle='--', label=f"mean={shift_norms.mean():.4f}")
ax.axvline(np.median(shift_norms), color='blue', linestyle=':', label=f"median={np.median(shift_norms):.4f}")
ax.set_xlabel('|Δ| (PCA-50 L2 norm)')
ax.set_ylabel('Count')
ax.set_title('Harmony shift magnitude\n(in PCA 50-d space)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)

# Row 1 col 2: shift vectors in tSNE space (all points)
ax = axes[1, 2]
# Sample a subset of points for clarity
n_show = 300
idx = np.random.choice(len(embs), n_show, replace=False)
for i in idx:
    ax.plot([tsne_before[i, 0], tsne_after[i, 0]],
            [tsne_before[i, 1], tsne_after[i, 1]],
            c='gray', alpha=0.15, lw=0.4)
# Color start points by domain
drug_i = [i for i in idx if drug_mask[i]]
mut_i  = [i for i in idx if mut_mask[i]]
ax.scatter(tsne_before[drug_i, 0], tsne_before[drug_i, 1],
           c=domain_colors['drug'], s=6, alpha=0.5, label='drug start')
ax.scatter(tsne_before[mut_i, 0], tsne_before[mut_i, 1],
           c=domain_colors['mutant'], s=6, alpha=0.5, label='mutant start')
ax.set_title(f'Shift vectors ({n_show} samples)\nt-SNE joint space', fontsize=13, fontweight='bold')
ax.legend(markerscale=6, fontsize=9)

plt.tight_layout()
plt.savefig('harmony_tsne_analysis.png', dpi=150, bbox_inches='tight')
print("Saved harmony_tsne_analysis.png")
plt.close()

# ── Genes × Drugs cosine similarity BEFORE vs AFTER Harmony ────────────────────

def compute_gene_drug_sim(drug_emb, drug_labels, mut_emb, mut_labels):
    gene_centroids = {}
    for lbl in sorted(set(mut_labels)):
        gene = get_gene(lbl)
        if gene is None: continue
        gene_centroids.setdefault(gene, []).append(mut_emb[mut_labels == lbl].mean(axis=0))
    gene_centroids = {g: np.mean(vs, axis=0) for g, vs in gene_centroids.items()}
    genes = sorted(gene_centroids.keys())

    drug_centroids = {}
    for lbl in sorted(set(drug_labels)):
        d = get_drug(lbl)
        if d is None: continue
        drug_centroids.setdefault(d, []).append(drug_emb[drug_labels == lbl].mean(axis=0))
    drug_centroids = {d: np.mean(vs, axis=0) for d, vs in drug_centroids.items()}
    drugs = sorted(drug_centroids.keys())

    sim = np.zeros((len(genes), len(drugs)))
    for i, g in enumerate(genes):
        for j, d in enumerate(drugs):
            sim[i, j] = cosine_sim(gene_centroids[g], drug_centroids[d])
    return sim, genes, drugs

drug_lbl_str = np.array([str(l) for l in drug_labels])
mut_lbl_str  = np.array([str(l) for l in mut_labels])

sim_before, genes, drugs = compute_gene_drug_sim(drug_pca, drug_lbl_str, mut_pca, mut_lbl_str)
sim_after,  _,     _     = compute_gene_drug_sim(drug_cor,  drug_lbl_str, mut_cor,  mut_lbl_str)

diff = sim_after - sim_before

vlim = max(abs(sim_before.min()), abs(sim_before.max()), abs(sim_after.min()), abs(sim_after.max()))

fig, axes = plt.subplots(1, 3, figsize=(30, 9))

for ax, mat, title, v in [
    (axes[0], sim_before, 'Gene × Drug Cosine Similarity\nBEFORE Harmony', (-vlim, vlim)),
    (axes[1], sim_after,  'Gene × Drug Cosine Similarity\nAFTER Harmony',  (-vlim, vlim)),
    (axes[2], diff,       'Difference (After − Before)', None),
]:
    if v:
        im = ax.imshow(mat, cmap=plt.cm.RdYlBu_r, vmin=v[0], vmax=v[1], aspect='auto')
    else:
        lim = max(abs(diff.min()), abs(diff.max()))
        im = ax.imshow(mat, cmap=plt.cm.RdBu, vmin=-lim, vmax=lim, aspect='auto')
    ax.set_xticks(range(len(drugs)))
    ax.set_yticks(range(len(genes)))
    ax.set_xticklabels(drugs, rotation=90, fontsize=6)
    ax.set_yticklabels(genes, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Cosine Similarity', fontsize=10)
    ax.set_title(title, fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('harmony_cosine_heatmap.png', dpi=200, bbox_inches='tight')
print("Saved harmony_cosine_heatmap.png")
plt.close()

# ── Report ──
print(f"\n{'='*70}")
print("INTERPRETATION")
print(f"{'='*70}")
print(f"Genes × Drugs matrix: {len(genes)} genes × {len(drugs)} drugs")
print(f"BEFORE: similarity range [{sim_before.min():.4f}, {sim_before.max():.4f}]")
print(f"AFTER:  similarity range [{sim_after.min():.4f}, {sim_after.max():.4f}]")
print(f"DIFF:   range [{diff.min():.4f}, {diff.max():.4f}]")
n_inc = (diff > 0.01).sum()
n_dec = (diff < -0.01).sum()
print(f"\nSimilarity increased (>0.01): {n_inc}/{diff.size}")
print(f"Decreased (<-0.01): {n_dec}/{diff.size}")
print(f"Unchanged: {diff.size - n_inc - n_dec}/{diff.size}")
print(f"\nDone!")
