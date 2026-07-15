"""
Multinomial Logistic Regression (GPU): drug vs mutant class matching via cosine similarity.
6-fold plate-based. Before and after alignment.
"""
import numpy as np, csv, os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from tqdm import tqdm

try:
    from cuml.linear_model import LogisticRegression
    from cuml.preprocessing import StandardScaler
    import cupy as cp
    HAS_CUML = True
    print("Using cuML (GPU)")
except ImportError:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_CUML = False
    print("cuML not available, using sklearn (CPU)")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(BASE_DIR, "features_all.npz")
CSV_PATH = os.path.join(BASE_DIR, "features_metadata.csv")
OUT_DIR = os.path.join(BASE_DIR, "analysis_figures")
SHIFT_PATH = os.path.join(OUT_DIR, 'alignment_mean_shift_vector.npy')
os.makedirs(OUT_DIR, exist_ok=True)

MUTANT_NC_LABELS = {f'NC_{i}' for i in range(1, 7)} | {f'WT NC_{i}' for i in range(1, 7)}
DRUG_EXCLUDE = {'drug_control'}
PLATES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']

print("Loading data...")
data = np.load(NPZ_PATH)
embeddings = data["embeddings"]
with open(CSV_PATH) as f:
    metadata = list(csv.DictReader(f))

sources = np.array([m["source"] for m in metadata])
plates = np.array([m["plate"] for m in metadata])
labels_arr = np.array([m["label"] for m in metadata])

drug_keep = (sources == 'drug') & ~np.array([l in DRUG_EXCLUDE for l in labels_arr])
mutant_keep = (sources == 'mutant') & ~np.array([l in MUTANT_NC_LABELS for l in labels_arr])
keep = drug_keep | mutant_keep

emb_all = embeddings[keep].astype(np.float32)
src_all = sources[keep]
plate_all = plates[keep]
label_all = labels_arr[keep]

unique_labels = sorted(set(label_all))
label_to_idx = {l: i for i, l in enumerate(unique_labels)}
label_idx = np.array([label_to_idx[l] for l in label_all], dtype=np.int32)

drug_class_indices = [i for i, l in enumerate(unique_labels) if src_all[label_all == l][0] == 'drug']
mutant_class_indices = [i for i, l in enumerate(unique_labels) if src_all[label_all == l][0] == 'mutant']
drug_labels = [unique_labels[i] for i in drug_class_indices]
mutant_labels = [unique_labels[i] for i in mutant_class_indices]

print(f"Drug: {len(drug_labels)} classes ({sum(src_all=='drug')} samples)")
print(f"Mutant: {len(mutant_labels)} classes ({sum(src_all=='mutant')} samples)")
print(f"Total: {len(emb_all)} samples, {len(unique_labels)} classes")

shift_vector = np.load(SHIFT_PATH).astype(np.float32)
print(f"Shift vector norm: {np.linalg.norm(shift_vector):.4f}")

emb_drug_mask = src_all == 'drug'
emb_aligned = emb_all.copy()
emb_aligned[emb_drug_mask] = emb_all[emb_drug_mask] + shift_vector

if HAS_CUML:
    emb_all_gpu = cp.asarray(emb_all)
    emb_aligned_gpu = cp.asarray(emb_aligned)
    label_idx_gpu = cp.asarray(label_idx)

results = []
all_sim_before = []
all_sim_after = []

folds = []
for i, test_plate in enumerate(PLATES):
    val_plate = PLATES[(i + 4) % 6]
    train_plates = [p for p in PLATES if p != test_plate and p != val_plate]
    folds.append((train_plates, val_plate, test_plate))

pbar = tqdm(total=len(folds) * 2, desc="Training LR", ncols=80)

for train_plates, val_plate, test_plate in folds:
    train_mask = np.array([p in train_plates for p in plate_all])
    val_mask = plate_all == val_plate
    test_mask = plate_all == test_plate

    for tag, emb_use, emb_gpu in [
        ('before', emb_all, emb_all_gpu if HAS_CUML else None),
        ('after', emb_aligned, emb_aligned_gpu if HAS_CUML else None),
    ]:
        if HAS_CUML:
            X_train = emb_gpu[train_mask]
            y_train = label_idx_gpu[train_mask]
            X_val = emb_gpu[val_mask]
            y_val = label_idx_gpu[val_mask]
            X_test = emb_gpu[test_mask]
            y_test = label_idx_gpu[test_mask]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)

            clf = LogisticRegression(solver='qn', max_iter=5000,
                                     C=1.0, class_weight='balanced',
                                     verbose=0)
            clf.fit(X_train_s, y_train)

            val_preds = clf.predict(X_val_s)
            test_preds = clf.predict(X_test_s)
            val_acc = accuracy_score(cp.asnumpy(y_val), cp.asnumpy(val_preds))
            test_acc = accuracy_score(cp.asnumpy(y_test), cp.asnumpy(test_preds))

            weights = cp.asnumpy(clf.coef_)
        else:
            X_train = emb_use[train_mask]
            y_train = label_idx[train_mask]
            X_val = emb_use[val_mask]
            y_val = label_idx[val_mask]
            X_test = emb_use[test_mask]
            y_test = label_idx[test_mask]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)

            clf = LogisticRegression(solver='lbfgs', max_iter=5000,
                                     C=1.0, class_weight='balanced',
                                     random_state=42)
            clf.fit(X_train_s, y_train)

            val_acc = accuracy_score(y_val, clf.predict(X_val_s))
            test_acc = accuracy_score(y_test, clf.predict(X_test_s))

            weights = clf.coef_

        drug_weights = weights[drug_class_indices]
        mutant_weights = weights[mutant_class_indices]
        drug_norm = drug_weights / np.linalg.norm(drug_weights, axis=1, keepdims=True)
        mutant_norm = mutant_weights / np.linalg.norm(mutant_weights, axis=1, keepdims=True)
        sim_matrix = drug_norm @ mutant_norm.T

        results.append({
            'train': '_'.join(train_plates), 'val': val_plate, 'test': test_plate,
            'tag': tag, 'val_acc': float(val_acc), 'test_acc': float(test_acc),
        })

        if tag == 'before':
            all_sim_before.append(sim_matrix)
        else:
            all_sim_after.append(sim_matrix)

        pbar.set_postfix_str(f"fold={test_plate} {tag} test={test_acc:.3f}")
        pbar.update(1)

pbar.close()

print(f"\n{'='*60}")
print("Test accuracy:")
for tag in ['before', 'after']:
    accs = [r['test_acc'] for r in results if r['tag'] == tag]
    print(f"  {tag}: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

mean_sim_before = np.mean(all_sim_before, axis=0)
mean_sim_after = np.mean(all_sim_after, axis=0)

drug_labels_short = [l.replace('_', ' ') for l in drug_labels]
mutant_labels_short = [l.replace('_', ' ') for l in mutant_labels]

np.savetxt(os.path.join(OUT_DIR, 'cosine_sim_drug_mutant_before.csv'),
           mean_sim_before, delimiter=',', fmt='%.6f',
           header=','.join(mutant_labels), comments='')
np.savetxt(os.path.join(OUT_DIR, 'cosine_sim_drug_mutant_after.csv'),
           mean_sim_after, delimiter=',', fmt='%.6f',
           header=','.join(mutant_labels), comments='')

vmax = max(abs(mean_sim_before).max(), abs(mean_sim_after).max())
for tag, sim_mat, title in [
    ('before', mean_sim_before, 'Before Alignment'),
    ('after', mean_sim_after, 'After Mean Shift Alignment'),
]:
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(sim_mat, cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
                xticklabels=mutant_labels_short, yticklabels=drug_labels_short,
                ax=ax, cbar_kws={'label': 'Cosine Similarity'})
    ax.set_xlabel('Mutant Classes', fontsize=12)
    ax.set_ylabel('Drug Classes', fontsize=12)
    ax.set_title(f'Drug vs Mutant Class Similarity — {title}', fontsize=14, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=5)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'cosine_sim_heatmap_{tag}.png'), dpi=200, bbox_inches='tight')
    plt.close()

matches = []
for i, drug_lbl in enumerate(drug_labels):
    sims = mean_sim_after[i]
    top_idx = np.argsort(sims)[::-1][:5]
    for j in top_idx:
        matches.append({'drug_class': drug_lbl, 'mutant_class': mutant_labels[j], 'cosine_similarity': float(sims[j])})

with open(os.path.join(OUT_DIR, 'drug_mutant_top_matches.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['drug_class', 'mutant_class', 'cosine_similarity'])
    w.writeheader()
    w.writerows(matches)

with open(os.path.join(OUT_DIR, 'lr_fold_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\nDone! Files saved to analysis_figures/")
print("  cosine_sim_heatmap_before.png, cosine_sim_heatmap_after.png")
print("  cosine_sim_drug_mutant_before.csv, cosine_sim_drug_mutant_after.csv")
print("  drug_mutant_top_matches.csv, lr_fold_results.json")
