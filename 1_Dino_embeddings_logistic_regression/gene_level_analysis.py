"""
Gene-level analysis: group drug doses → 22 genes, mutant replicates → 28 genes.
Cross-validated accuracy, confusion matrix, cosine similarity, t-SNE.
GPU accelerated via cuML.
"""
import numpy as np, csv, os, json, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

try:
    from cuml.linear_model import LogisticRegression
    from cuml.preprocessing import StandardScaler
    from cuml.manifold import TSNE
    import cupy as cp
    HAS_CUML = True
    print("Using cuML (GPU)")
except ImportError:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    HAS_CUML = False
    print("cuML not available, using sklearn (CPU)")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(BASE_DIR, "features_all.npz")
CSV_PATH = os.path.join(BASE_DIR, "features_metadata.csv")
OUT_DIR = os.path.join(BASE_DIR, "analysis_figures")
SHIFT_PATH = os.path.join(OUT_DIR, 'alignment_mean_shift_vector.npy')
os.makedirs(OUT_DIR, exist_ok=True)

MUTANT_NC = {f'NC_{i}' for i in range(1,7)} | {f'WT NC_{i}' for i in range(1,7)}
PLATES = ['P1','P2','P3','P4','P5','P6']
PLATE_COLORS = {'P1':'#e41a1c','P2':'#377eb8','P3':'#4daf4a','P4':'#984ea3','P5':'#ff7f00','P6':'#a65628'}

data = np.load(NPZ_PATH)
embeddings = data["embeddings"]
with open(CSV_PATH) as f:
    metadata = list(csv.DictReader(f))

sources = np.array([m["source"] for m in metadata])
plates = np.array([m["plate"] for m in metadata])
labels = np.array([m["label"] for m in metadata])

drug_keep = (sources == 'drug') & ~np.array([l == 'drug_control' for l in labels])
mutant_keep = (sources == 'mutant') & ~np.array([l in MUTANT_NC for l in labels])
keep = drug_keep | mutant_keep

emb_all = embeddings[keep].astype(np.float32)
src_all = sources[keep]
plate_all = plates[keep]
label_all = labels[keep]

# --- Gene-level grouping ---
def drug_gene(lbl):
    return lbl.rsplit('_', 1)[0]

def mutant_gene(lbl):
    return lbl.rsplit('_', 1)[0]

def get_gene(lbl, src):
    return drug_gene(lbl) if src == 'drug' else mutant_gene(lbl)

gene_labels = np.array([get_gene(l, s) for l, s in zip(label_all, src_all)])
unique_genes = sorted(set(gene_labels))
gene_to_idx = {g: i for i, g in enumerate(unique_genes)}
gene_idx = np.array([gene_to_idx[g] for g in gene_labels], dtype=np.int32)

drug_genes = sorted(set(gene_labels[src_all == 'drug']))
mutant_genes = sorted(set(gene_labels[src_all == 'mutant']))
drug_gene_idx = [i for i, g in enumerate(unique_genes) if g in drug_genes]
mutant_gene_idx = [i for i, g in enumerate(unique_genes) if g in mutant_genes]

print(f"Drug genes: {len(drug_genes)} ({sum(src_all=='drug')} samples)")
print(f"Mutant genes: {len(mutant_genes)} ({sum(src_all=='mutant')} samples)")
print(f"Total: {len(emb_all)} samples, {len(unique_genes)} gene-level classes")

# --- Alignment ---
shift_vector = np.load(SHIFT_PATH).astype(np.float32)
emb_drug_mask = src_all == 'drug'
emb_aligned = emb_all.copy()
emb_aligned[emb_drug_mask] = emb_all[emb_drug_mask] + shift_vector
print(f"Mean shift vector norm: {np.linalg.norm(shift_vector):.4f}")

# --- t-SNE (subsample to ~12k for speed) ---
N_TSNE = 12000
rng = np.random.RandomState(42)
idx_tsne = rng.choice(len(emb_all), N_TSNE, replace=False)
src_tsne = src_all[idx_tsne]
plate_tsne = plate_all[idx_tsne]

print("Computing t-SNE...")
common_tsne_kw = dict(n_components=2, random_state=42)
tsne_kw = dict(perplexity=30, learning_rate=200.0, n_neighbors=90, **common_tsne_kw)
if HAS_CUML:
    tsne_before = TSNE(**tsne_kw)
    emb_before_2d = tsne_before.fit_transform(cp.asarray(emb_all[idx_tsne]))
    emb_before_2d = cp.asnumpy(emb_before_2d)
    tsne_after = TSNE(**tsne_kw)
    emb_after_2d = tsne_after.fit_transform(cp.asarray(emb_aligned[idx_tsne]))
    emb_after_2d = cp.asnumpy(emb_after_2d)
else:
    tsne_before = TSNE(**tsne_kw)
    emb_before_2d = tsne_before.fit_transform(emb_all[idx_tsne])
    tsne_after = TSNE(**tsne_kw)
    emb_after_2d = tsne_after.fit_transform(emb_aligned[idx_tsne])

from matplotlib.lines import Line2D
LEG_SRC = [Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=8,label=s)
           for s,c in [('Control','#4daf4a'), ('Mutant','#377eb8'), ('Drug','#e41a1c')]]
LEG_PLATE = [Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=8,label=p)
             for p,c in PLATE_COLORS.items()]

for tag, emb2d in [('before', emb_before_2d), ('after', emb_after_2d)]:
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    src_colors = np.where(src_tsne == 'control', '#4daf4a',
                   np.where(src_tsne == 'mutant', '#377eb8', '#e41a1c'))
    axes[0].scatter(emb2d[:,0], emb2d[:,1], c=src_colors, s=3, alpha=0.6, rasterized=True)
    axes[0].set_title(f'by Source — {tag}', fontsize=14, fontweight='bold')
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].legend(handles=LEG_SRC, fontsize=11, loc='best')
    
    plate_colors = np.array([PLATE_COLORS[p] for p in plate_tsne])
    axes[1].scatter(emb2d[:,0], emb2d[:,1], c=plate_colors, s=3, alpha=0.6, rasterized=True)
    axes[1].set_title(f'by Plate — {tag}', fontsize=14, fontweight='bold')
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].legend(handles=LEG_PLATE, fontsize=11, loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'tsne_gene_level_{tag}.png'), dpi=200, bbox_inches='tight')
    plt.close()

# --- Cross-validated LR at gene level ---
if HAS_CUML:
    emb_all_gpu = cp.asarray(emb_all)
    emb_aligned_gpu = cp.asarray(emb_aligned)
    gene_idx_gpu = cp.asarray(gene_idx)

folds = []
for i, test_plate in enumerate(PLATES):
    val_plate = PLATES[(i + 4) % 6]
    train_plates = [p for p in PLATES if p not in (test_plate, val_plate)]
    folds.append((train_plates, val_plate, test_plate))

acc_results = {'before': [], 'after': []}
all_cm_before = []
all_cm_after = []
all_sim_before = []
all_sim_after = []
lr_coefs = {}  # (test_plate, tag) -> (drug_weights, mutant_weights)

pbar = tqdm(total=len(folds) * 2, desc="Gene LR", ncols=80)

for train_plates, val_plate, test_plate in folds:
    train_mask = np.array([p in train_plates for p in plate_all])
    val_mask = plate_all == val_plate
    test_mask = plate_all == test_plate

    for tag, emb_use, emb_gpu in [
        ('before', emb_all, emb_all_gpu if HAS_CUML else None),
        ('after', emb_aligned, emb_aligned_gpu if HAS_CUML else None),
    ]:
        if HAS_CUML:
            X_tr = emb_gpu[train_mask]; y_tr = gene_idx_gpu[train_mask]
            X_va = emb_gpu[val_mask]; y_va = gene_idx_gpu[val_mask]
            X_te = emb_gpu[test_mask]; y_te = gene_idx_gpu[test_mask]
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)
            X_te_s = scaler.transform(X_te)
            clf = LogisticRegression(solver='qn', max_iter=5000, C=1.0,
                                     class_weight='balanced', verbose=0)
            clf.fit(X_tr_s, y_tr)
            preds = clf.predict(X_te_s)
            acc = accuracy_score(cp.asnumpy(y_te), cp.asnumpy(preds))
            preds_np = cp.asnumpy(preds)
            y_te_np = cp.asnumpy(y_te)
            w = cp.asnumpy(clf.coef_)
        else:
            X_tr = emb_use[train_mask]; y_tr = gene_idx[train_mask]
            X_va = emb_use[val_mask]; y_va = gene_idx[val_mask]
            X_te = emb_use[test_mask]; y_te = gene_idx[test_mask]
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)
            X_te_s = scaler.transform(X_te)
            clf = LogisticRegression(solver='lbfgs', max_iter=5000, C=1.0,
                                     class_weight='balanced', random_state=42)
            clf.fit(X_tr_s, y_tr)
            preds = clf.predict(X_te_s)
            acc = accuracy_score(y_te, preds)
            preds_np = preds
            y_te_np = y_te
            w = clf.coef_

        acc_results[tag].append(acc)

        cm = confusion_matrix(y_te_np, preds_np, labels=range(len(unique_genes)))
        if tag == 'before':
            if len(all_cm_before) == 0:
                all_cm_before = cm.astype(np.float32)
            else:
                all_cm_before += cm.astype(np.float32)
        else:
            if len(all_cm_after) == 0:
                all_cm_after = cm.astype(np.float32)
            else:
                all_cm_after += cm.astype(np.float32)

        # Cosine similarity
        dw = w[drug_gene_idx]
        mw = w[mutant_gene_idx]
        dn = dw / np.linalg.norm(dw, axis=1, keepdims=True)
        mn = mw / np.linalg.norm(mw, axis=1, keepdims=True)
        sim = dn @ mn.T
        if tag == 'before':
            all_sim_before.append(sim)
        else:
            all_sim_after.append(sim)

        lr_coefs[(test_plate, tag)] = (dw, mw)
        pbar.update(1)

pbar.close()

# Normalize confusion matrices
all_cm_before = all_cm_before / len(folds)
all_cm_after = all_cm_after / len(folds)

# --- Print accuracy ---
print(f"\n{'='*60}")
print("Gene-level test accuracy:")
for tag, accs in acc_results.items():
    print(f"  {tag}: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

# --- Plot confusion matrices ---
gene_short = [g.replace('_', ' ') for g in unique_genes]
for tag, cm, title in [
    ('before', all_cm_before, 'Before Alignment'),
    ('after', all_cm_after, 'After Mean Shift Alignment'),
]:
    cm_norm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_norm, cmap='Blues', vmin=0, vmax=0.5,
                xticklabels=gene_short, yticklabels=gene_short,
                ax=ax, cbar_kws={'label': 'Fraction (rows normalized)'})
    ax.set_xlabel('Predicted Gene', fontsize=12)
    ax.set_ylabel('True Gene', fontsize=12)
    ax.set_title(f'Gene-Level Confusion Matrix — {title}', fontsize=14, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=5)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'confusion_matrix_gene_{tag}.png'), dpi=200, bbox_inches='tight')
    plt.close()

# --- Cosine similarity mean & std across folds ---
sim_before = np.stack(all_sim_before, axis=0)
sim_after = np.stack(all_sim_after, axis=0)

mean_sim_before = sim_before.mean(axis=0)
std_sim_before = sim_before.std(axis=0)
mean_sim_after = sim_after.mean(axis=0)
std_sim_after = sim_after.std(axis=0)

drug_gene_short = [g.replace('_', ' ') for g in drug_genes]
mutant_gene_short = [g.replace('_', ' ') for g in mutant_genes]

for tag, mean_sim, std_sim, title in [
    ('before', mean_sim_before, std_sim_before, 'Before Alignment'),
    ('after', mean_sim_after, std_sim_after, 'After Mean Shift Alignment'),
]:
    fig, axes = plt.subplots(1, 2, figsize=(22, 16))
    
    sns.heatmap(mean_sim, cmap='RdBu_r', center=0,
                xticklabels=mutant_gene_short, yticklabels=drug_gene_short,
                ax=axes[0], cbar_kws={'label': 'Mean Cosine Similarity'})
    axes[0].set_xlabel('Mutant Genes', fontsize=11)
    axes[0].set_ylabel('Drug Genes', fontsize=11)
    axes[0].set_title(f'Mean — {title}', fontsize=13, fontweight='bold')
    plt.setp(axes[0].get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(axes[0].get_yticklabels(), rotation=0, fontsize=6)

    sns.heatmap(std_sim, cmap='Oranges', vmin=0,
                xticklabels=mutant_gene_short, yticklabels=drug_gene_short,
                ax=axes[1], cbar_kws={'label': 'Std Cosine Similarity'})
    axes[1].set_xlabel('Mutant Genes', fontsize=11)
    axes[1].set_ylabel('Drug Genes', fontsize=11)
    axes[1].set_title(f'Std Dev — {title}', fontsize=13, fontweight='bold')
    plt.setp(axes[1].get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(axes[1].get_yticklabels(), rotation=0, fontsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'cosine_sim_gene_{tag}.png'), dpi=200, bbox_inches='tight')
    plt.close()

# --- Top-5 matches (after alignment, mean) ---
matches = []
for i, dg in enumerate(drug_genes):
    sims = mean_sim_after[i]
    top5 = np.argsort(sims)[::-1][:5]
    for j in top5:
        matches.append({'drug_gene': dg, 'mutant_gene': mutant_genes[j],
                        'cosine_similarity': float(sims[j])})

with open(os.path.join(OUT_DIR, 'drug_mutant_gene_top5.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['drug_gene','mutant_gene','cosine_similarity'])
    w.writeheader(); w.writerows(matches)

# --- Summary metrics ---
summary = {
    'num_samples': int(len(emb_all)),
    'num_drug_genes': len(drug_genes),
    'num_mutant_genes': len(mutant_genes),
    'test_accuracy_before': {
        'mean': float(np.mean(acc_results['before'])),
        'std': float(np.std(acc_results['before'])),
        'per_fold': [float(v) for v in acc_results['before']]
    },
    'test_accuracy_after': {
        'mean': float(np.mean(acc_results['after'])),
        'std': float(np.std(acc_results['after'])),
        'per_fold': [float(v) for v in acc_results['after']]
    },
    'cosine_similarity_before': {
        'mean': float(mean_sim_before.mean()),
        'std': float(mean_sim_before.std())
    },
    'cosine_similarity_after': {
        'mean': float(mean_sim_after.mean()),
        'std': float(mean_sim_after.std())
    },
}

# Print drug→mutant top-1 matches
m1 = {}
for m in matches[:22]:  # first 22 = top-1 for each drug
    m1[m['drug_gene']] = m['mutant_gene']
print("\nDrug → Mutant top-1 matches (after alignment):")
for dg, mg in m1.items():
    sim = next(mm['cosine_similarity'] for mm in matches if mm['drug_gene']==dg and mm['mutant_gene']==mg)
    print(f"  {dg:20s} → {mg:10s}  (cos={sim:.3f})")

with open(os.path.join(OUT_DIR, 'lr_gene_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print(f"Results saved to {OUT_DIR}/")
print("  confusion_matrix_gene_before.png / after.png")
print("  cosine_sim_gene_before.png / after.png")
print("  tsne_gene_level_before.png / after.png")
print("  drug_mutant_gene_top5.csv")
print("  lr_gene_summary.json")
