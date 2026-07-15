"""
Multinomial Logistic Regression on DINOv3 CLS token embeddings (center crop).
Loads from pre-consolidated features_all.npz + features_metadata.csv.
Train on P1-P4, Validate on P5, Test on P6.
Supports filtering by data source: control, mutant, drug, or all.
"""

import numpy as np
import json
import os
import pickle
import csv
from collections import Counter
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm

try:
    import cuml
    from cuml.linear_model import LogisticRegression
    from cuml.preprocessing import StandardScaler
    HAS_CUML = True
    print("Using cuML for GPU-accelerated training")
except ImportError:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_CUML = False
    print("cuML not available, using sklearn")

try:
    import cupy as cp
    HAS_CUPY = True
    print("Using cuPy for GPU-accelerated operations")
except ImportError:
    HAS_CUPY = False
    print("cuPy not available, using numpy")

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    print("joblib not available, will use pickle")

from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(BASE_DIR, "features_all.npz")
CSV_PATH = os.path.join(BASE_DIR, "features_metadata.csv")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "features_label_map.json")

CLS_DIM = 1024
C_VALUES = [0.1, 1.0, 10.0, 100.0]
SEED = 42
np.random.seed(SEED)

# Cache files  (include source filter in name)
TRAIN_EMB_CACHE = os.path.join(BASE_DIR, 'train_embeddings.npy')
VAL_EMB_CACHE = os.path.join(BASE_DIR, 'val_embeddings.npy')
TEST_EMB_CACHE = os.path.join(BASE_DIR, 'test_embeddings.npy')
TRAIN_LBL_CACHE = os.path.join(BASE_DIR, 'train_labels.npy')
VAL_LBL_CACHE = os.path.join(BASE_DIR, 'val_labels.npy')
TEST_LBL_CACHE = os.path.join(BASE_DIR, 'test_labels_raw.npy')
METADATA_CACHE = os.path.join(BASE_DIR, 'metadata_cache.json')

# Source filter: 'control', 'mutant', 'drug', or 'all'
SOURCE_FILTER = os.environ.get("FEATURE_SOURCE", "all")


def load_consolidated():
    """Load consolidated features + metadata, filter by source."""
    print(f"\nLoading consolidated features from {NPZ_PATH}")
    data = np.load(NPZ_PATH)
    embeddings = data["embeddings"]
    label_indices = data["label_indices"]

    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        metadata = list(reader)

    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)

    idx_to_label = {int(k): v for k, v in label_map["idx_to_label"].items()}
    label_to_idx = label_map["label_to_idx"]

    print(f"  Total samples loaded: {len(embeddings)}")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    print(f"  Unique labels in map: {len(idx_to_label)}")

    # Filter by source if not 'all'
    if SOURCE_FILTER != "all":
        valid = [i for i, m in enumerate(metadata) if m["source"] == SOURCE_FILTER]
        embeddings = embeddings[valid]
        label_indices = label_indices[valid]
        metadata = [metadata[i] for i in valid]
        print(f"  Filtered to source='{SOURCE_FILTER}': {len(embeddings)} samples")

    return embeddings, label_indices, metadata, idx_to_label, label_to_idx


def split_by_plate(embeddings, label_indices, metadata,
                   train_plates=("P1", "P2", "P3", "P4"),
                   val_plates=("P5",),
                   test_plates=("P6",)):
    """Split into train/val/test by plate."""
    train_idx = [i for i, m in enumerate(metadata) if m["plate"] in train_plates]
    val_idx = [i for i, m in enumerate(metadata) if m["plate"] in val_plates]
    test_idx = [i for i, m in enumerate(metadata) if m["plate"] in test_plates]

    train_emb = embeddings[train_idx]
    train_lbl = label_indices[train_idx]
    train_meta = [metadata[i] for i in train_idx]

    val_emb = embeddings[val_idx]
    val_lbl = label_indices[val_idx]
    val_meta = [metadata[i] for i in val_idx]

    test_emb = embeddings[test_idx]
    test_lbl = label_indices[test_idx]
    test_meta = [metadata[i] for i in test_idx]

    return train_emb, train_lbl, train_meta, val_emb, val_lbl, val_meta, test_emb, test_lbl, test_meta


def cache_paths():
    """Return cache paths unique to the source filter."""
    suffix = "" if SOURCE_FILTER == "all" else f"_{SOURCE_FILTER}"
    return (
        os.path.join(BASE_DIR, f'train_embeddings{suffix}.npy'),
        os.path.join(BASE_DIR, f'val_embeddings{suffix}.npy'),
        os.path.join(BASE_DIR, f'test_embeddings{suffix}.npy'),
        os.path.join(BASE_DIR, f'train_labels{suffix}.npy'),
        os.path.join(BASE_DIR, f'val_labels{suffix}.npy'),
        os.path.join(BASE_DIR, f'test_labels{suffix}.npy'),
        os.path.join(BASE_DIR, f'metadata_cache{suffix}.json'),
    )


def load_cached():
    paths = cache_paths()
    TE_C, VE_C, TsE_C, TL_C, VL_C, TsL_C, M_C = paths
    if all(os.path.exists(p) for p in paths):
        print("\nLoading cached embeddings...")
        train_emb = np.load(TE_C)
        train_lbl = np.load(TL_C)
        val_emb = np.load(VE_C)
        val_lbl = np.load(VL_C)
        test_emb = np.load(TsE_C)
        test_lbl = np.load(TsL_C)
        with open(M_C) as f:
            meta = json.load(f)
        print(f"  Train: {train_emb.shape}")
        print(f"  Val: {val_emb.shape}")
        print(f"  Test: {test_emb.shape}")
        return train_emb, train_lbl, meta['train'], val_emb, val_lbl, meta['val'], test_emb, test_lbl, meta['test']
    return None


def save_cache(train_emb, train_lbl, train_meta, val_emb, val_lbl, val_meta, test_emb, test_lbl, test_meta):
    paths = cache_paths()
    TE_C, VE_C, TsE_C, TL_C, VL_C, TsL_C, M_C = paths
    print("\nSaving embeddings to cache...")
    np.save(TE_C, train_emb)
    np.save(VE_C, val_emb)
    np.save(TsE_C, test_emb)
    np.save(TL_C, train_lbl)
    np.save(VL_C, val_lbl)
    np.save(TsL_C, test_lbl)
    with open(M_C, 'w') as f:
        json.dump({'train': train_meta, 'val': val_meta, 'test': test_meta}, f)
    print("  Cached!")


def main():
    print("=" * 60)
    print("Multinomial Logistic Regression on DINOv3 CLS Embeddings")
    print("=" * 60)
    print(f"\nSource filter: {SOURCE_FILTER}")

    # Load consolidated features
    embeddings, label_indices, metadata, idx_to_label, label_to_idx = load_consolidated()
    num_classes = len(idx_to_label)

    # Try cache
    cached = load_cached()
    if cached is not None:
        train_emb, train_lbl, train_meta, val_emb, val_lbl, val_meta, test_emb, test_lbl, test_meta = cached
    else:
        # Split by plate
        train_emb, train_lbl, train_meta, val_emb, val_lbl, val_meta, test_emb, test_lbl, test_meta = \
            split_by_plate(embeddings, label_indices, metadata)
        save_cache(train_emb, train_lbl, train_meta, val_emb, val_lbl, val_meta, test_emb, test_lbl, test_meta)

    # Only use CLS token dim
    train_emb = np.array(train_emb, dtype=np.float32)[:, :CLS_DIM]
    train_lbl = np.array(train_lbl, dtype=np.int32)
    val_emb = np.array(val_emb, dtype=np.float32)[:, :CLS_DIM]
    val_lbl = np.array(val_lbl, dtype=np.int32)
    test_emb = np.array(test_emb, dtype=np.float32)[:, :CLS_DIM]
    test_lbl = np.array(test_lbl, dtype=np.int32)

    # Source counts in each split
    for split_name, meta_list in [("Train", train_meta), ("Val", val_meta), ("Test", test_meta)]:
        cnt = Counter(m["source"] for m in meta_list)
        print(f"  {split_name}: {len(meta_list)} samples, sources={dict(cnt)}")

    print(f"\nCLS token only (first {CLS_DIM} dims)")
    print(f"Train: {train_emb.shape}, Val: {val_emb.shape}, Test: {test_emb.shape}")
    print(f"Num classes: {num_classes}")

    # L2 normalize
    print("\nL2 normalization (DINO standard)...")
    if HAS_CUPY:
        train_gpu = cp.asarray(train_emb.astype(np.float32))
        val_gpu = cp.asarray(val_emb.astype(np.float32))
        test_gpu = cp.asarray(test_emb.astype(np.float32))
        train_emb = (train_gpu / cp.linalg.norm(train_gpu, axis=1, keepdims=True)).get()
        val_emb = (val_gpu / cp.linalg.norm(val_gpu, axis=1, keepdims=True)).get()
        test_emb = (test_gpu / cp.linalg.norm(test_gpu, axis=1, keepdims=True)).get()
        del train_gpu, val_gpu, test_gpu
        cp._default_memory_pool.free_all_blocks()
    else:
        from sklearn.preprocessing import normalize
        train_emb = normalize(train_emb, norm='l2')
        val_emb = normalize(val_emb, norm='l2')
        test_emb = normalize(test_emb, norm='l2')

    # Grid search C
    print("\nTraining Multinomial Logistic Regression...")
    best_val_acc = 0
    best_c = 1.0
    best_model = None
    results = []

    for c_val in C_VALUES:
        print(f"  C={c_val} ...", end=" ", flush=True)
        if HAS_CUML:
            model = LogisticRegression(max_iter=2000, C=c_val, class_weight='balanced', verbose=0)
        else:
            model = LogisticRegression(solver='lbfgs', max_iter=2000, C=c_val, class_weight='balanced', random_state=SEED, n_jobs=-1, verbose=0)
        model.fit(train_emb, train_lbl)

        val_preds = model.predict(val_emb)
        val_acc = accuracy_score(val_lbl, val_preds) * 100
        val_bal = balanced_accuracy_score(val_lbl, val_preds) * 100
        results.append({'C': c_val, 'val_acc': val_acc, 'val_bal': val_bal})
        print(f"Val Acc={val_acc:.2f}%, Bal={val_bal:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_c = c_val
            best_model = model

    print(f"\nBest C={best_c} (Val Acc={best_val_acc:.2f}%)")

    model = best_model
    train_preds = model.predict(train_emb)
    val_preds = model.predict(val_emb)
    test_preds = model.predict(test_emb)

    train_acc = accuracy_score(train_lbl, train_preds) * 100
    val_acc = accuracy_score(val_lbl, val_preds) * 100
    test_acc = accuracy_score(test_lbl, test_preds) * 100
    train_bal = balanced_accuracy_score(train_lbl, train_preds) * 100
    val_bal = balanced_accuracy_score(val_lbl, val_preds) * 100
    test_bal = balanced_accuracy_score(test_lbl, test_preds) * 100

    print(f"\n--- Results (source={SOURCE_FILTER}) ---")
    print(f"  Train Acc: {train_acc:.2f}%  Bal: {train_bal:.2f}%")
    print(f"  Val   Acc: {val_acc:.2f}%  Bal: {val_bal:.2f}%")
    print(f"  Test  Acc: {test_acc:.2f}%  Bal: {test_bal:.2f}%")

    # Save predictions and model
    suffix = "" if SOURCE_FILTER == "all" else f"_{SOURCE_FILTER}"
    np.save(os.path.join(BASE_DIR, f'test_preds{suffix}.npy'), test_preds)
    np.save(os.path.join(BASE_DIR, f'test_labels{suffix}.npy'), test_lbl)
    np.save(os.path.join(BASE_DIR, f'train_preds{suffix}.npy'), train_preds)
    np.save(os.path.join(BASE_DIR, f'train_labels{suffix}.npy'), train_lbl)
    np.save(os.path.join(BASE_DIR, f'val_preds{suffix}.npy'), val_preds)
    np.save(os.path.join(BASE_DIR, f'val_labels{suffix}.npy'), val_lbl)

    model_info = {
        'source_filter': SOURCE_FILTER,
        'label_to_idx': label_to_idx,
        'idx_to_label': {str(k): v for k, v in idx_to_label.items()},
        'num_classes': num_classes,
        'has_cuml': HAS_CUML,
        'best_c': best_c,
        'c_search_results': results,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'train_bal': train_bal,
        'val_bal': val_bal,
        'test_bal': test_bal,
        'n_train': len(train_emb),
        'n_val': len(val_emb),
        'n_test': len(test_emb),
    }

    model_path = os.path.join(BASE_DIR, f'logistic_model{suffix}.joblib') if HAS_JOBLIB else os.path.join(BASE_DIR, f'logistic_model{suffix}.pkl')
    if HAS_JOBLIB:
        joblib.dump(model, model_path)
    else:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    with open(os.path.join(BASE_DIR, f'model_info{suffix}.json'), 'w') as f:
        json.dump(model_info, f, indent=2)

    # Save test metadata with predictions
    test_out = []
    for i, meta in enumerate(test_meta):
        test_out.append({
            **meta,
            'predicted_idx': int(test_preds[i]),
            'predicted_label': idx_to_label.get(int(test_preds[i]), 'unknown'),
            'true_idx': int(test_lbl[i]),
            'true_label': idx_to_label.get(int(test_lbl[i]), 'unknown'),
        })
    with open(os.path.join(BASE_DIR, f'test_predictions{suffix}.json'), 'w') as f:
        json.dump(test_out, f, indent=2)

    print(f"\nModel saved: {model_path}")
    print(f"Predictions saved to: test_predictions{suffix}.json")
    print("=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
