"""
Multinomial Logistic Regression on CNN Embeddings using cuML (GPU)
Train on P1-P4, Validate on P5, Test on P6
"""

import numpy as np
import json
import os
import re
import glob
import pickle
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
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
EMBEDDINGS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'embeddings')
PLATE_WELL_FILE = os.path.join(os.path.dirname(BASE_DIR), 'plate_well_id_path.json')

CLS_DIM = 1024  # Use only CLS token (first 1024 dims) for classification

# C values to try for regularization (try higher C since less regularization helped)
C_VALUES = [0.1, 1.0, 10.0, 100.0]

SEED = 42
np.random.seed(SEED)

# Cache file paths
TRAIN_EMB_CACHE = os.path.join(BASE_DIR, 'train_embeddings.npy')
VAL_EMB_CACHE = os.path.join(BASE_DIR, 'val_embeddings.npy')
TEST_EMB_CACHE = os.path.join(BASE_DIR, 'test_embeddings.npy')
TRAIN_LBL_CACHE = os.path.join(BASE_DIR, 'train_labels.npy')
VAL_LBL_CACHE = os.path.join(BASE_DIR, 'val_labels.npy')
TEST_LBL_CACHE = os.path.join(BASE_DIR, 'test_labels_raw.npy')
METADATA_CACHE = os.path.join(BASE_DIR, 'metadata_cache.json')

def load_cached_embeddings():
    """Load cached processed embeddings if available"""
    if os.path.exists(TRAIN_EMB_CACHE) and os.path.exists(VAL_EMB_CACHE) and os.path.exists(TEST_EMB_CACHE):
        print("\n📂 Loading cached embeddings...")
        train_embeddings = np.load(TRAIN_EMB_CACHE)
        train_labels = np.load(TRAIN_LBL_CACHE)
        val_embeddings = np.load(VAL_EMB_CACHE)
        val_labels = np.load(VAL_LBL_CACHE)
        test_embeddings = np.load(TEST_EMB_CACHE)
        test_labels = np.load(TEST_LBL_CACHE)
        
        with open(METADATA_CACHE, 'r') as f:
            metadata = json.load(f)
        train_metadata = metadata['train']
        val_metadata = metadata['val']
        test_metadata = metadata['test']
        
        print(f"  Train: {train_embeddings.shape}")
        print(f"  Val: {val_embeddings.shape}")
        print(f"  Test: {test_embeddings.shape}")
        return train_embeddings, train_labels, train_metadata, val_embeddings, val_labels, val_metadata, test_embeddings, test_labels, test_metadata
    return None

def save_cached_embeddings(train_emb, train_lbl, train_meta, val_emb, val_lbl, val_meta, test_emb, test_lbl, test_meta):
    """Save processed embeddings for future use"""
    print("\n💾 Saving embeddings to cache...")
    np.save(TRAIN_EMB_CACHE, train_emb)
    np.save(VAL_EMB_CACHE, val_emb)
    np.save(TEST_EMB_CACHE, test_emb)
    np.save(TRAIN_LBL_CACHE, train_lbl)
    np.save(VAL_LBL_CACHE, val_lbl)
    np.save(TEST_LBL_CACHE, test_lbl)
    
    metadata = {
        'train': train_meta,
        'val': val_meta,
        'test': test_meta
    }
    with open(METADATA_CACHE, 'w') as f:
        json.dump(metadata, f)
    print("  Cached embeddings saved!")

with open(PLATE_WELL_FILE, 'r') as f:
    plate_data = json.load(f)

plate_maps = {}
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_maps[plate] = {}
    for row, wells in plate_data[plate].items():
        for col, info in wells.items():
            well = f"{row}{int(col):02d}"
            plate_maps[plate][well] = info['id']

all_labels = sorted(set(label for pm in plate_maps.values() for label in pm.values()))
label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
num_classes = len(all_labels)

print(f"Number of classes: {num_classes}")
print(f"Labels: {all_labels[:5]}... (total {len(all_labels)})")

def extract_well_from_filename(filename):
    match = re.search(r'Well(\w\d+)_', filename)
    if match:
        return match.group(1)
    return None

def load_embeddings_for_plate(plate):
    """Load all embeddings and labels for a plate"""
    plate_dir = os.path.join(EMBEDDINGS_DIR, plate)
    if not os.path.exists(plate_dir):
        print(f"Warning: {plate_dir} does not exist")
        return [], [], []
    
    image_dirs = [d for d in os.listdir(plate_dir) if os.path.isdir(os.path.join(plate_dir, d))]
    
    embeddings = []
    labels = []
    metadata = []
    
    for img_dir in tqdm(image_dirs, desc=f"Loading {plate}", leave=False):
        img_path = os.path.join(plate_dir, img_dir)
        well = extract_well_from_filename(img_dir)
        
        if well is None:
            continue
            
        label = plate_maps[plate].get(well, 'WT')
        label_idx = label_to_idx.get(label, 0)
        
        crop_files = sorted(glob.glob(os.path.join(img_path, 'crop_*.npy')))
        
        for crop_file in crop_files:
            try:
                emb = np.load(crop_file)
                embeddings.append(emb)
                labels.append(label_idx)
                metadata.append({
                    'plate': plate,
                    'well': well,
                    'filename': img_dir,
                    'crop_file': os.path.basename(crop_file)
                })
            except Exception as e:
                print(f"Error loading {crop_file}: {e}")
    
    return embeddings, labels, metadata

print("\n" + "="*60)
print("Loading embeddings...")
print("="*60)

# Try to load from cache first
cached = load_cached_embeddings()

if cached is not None:
    print("  ✅ Loaded from cache!")
    train_embeddings, train_labels, train_metadata, val_embeddings, val_labels, val_metadata, test_embeddings, test_labels, test_metadata = cached
    train_labels = train_labels.astype(np.int32)
    val_labels = val_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)
else:
    print("\nLoading training data (P1-P4)...")
    train_embeddings = []
    train_labels = []
    train_metadata = []

    for plate in ['P1', 'P2', 'P3', 'P4']:
        print(f"  Loading {plate}...")
        emb, lbl, meta = load_embeddings_for_plate(plate)
        train_embeddings.extend(emb)
        train_labels.extend(lbl)
        train_metadata.extend(meta)
        print(f"    {plate}: {len(emb)} crops")

    print(f"\nTotal training crops: {len(train_embeddings)}")

    print("\nLoading validation data (P5)...")
    val_embeddings, val_labels, val_metadata = load_embeddings_for_plate('P5')
    print(f"Validation crops: {len(val_embeddings)}")

    print("\nLoading test data (P6)...")
    test_embeddings, test_labels, test_metadata = load_embeddings_for_plate('P6')
    print(f"Test crops: {len(test_embeddings)}")

    train_embeddings = np.array(train_embeddings, dtype=np.float32)[:, :CLS_DIM]
    train_labels = np.array(train_labels, dtype=np.int32)
    val_embeddings = np.array(val_embeddings, dtype=np.float32)[:, :CLS_DIM]
    val_labels = np.array(val_labels, dtype=np.int32)
    test_embeddings = np.array(test_embeddings, dtype=np.float32)[:, :CLS_DIM]
    test_labels = np.array(test_labels, dtype=np.int32)

    # Save to cache
    test_labels_raw = test_labels.copy()
    save_cached_embeddings(
        train_embeddings, train_labels, train_metadata,
        val_embeddings, val_labels, val_metadata,
        test_embeddings, test_labels_raw, test_metadata
    )

print(f"\nUsing CLS token only (first {CLS_DIM} dimensions)")
print(f"Training shape: {train_embeddings.shape}")
print(f"Validation shape: {val_embeddings.shape}")
print(f"Test shape: {test_embeddings.shape}")

print("\n" + "="*60)
print("Preprocessing: L2 Normalization (DINOv2/v3 standard)...")
print("="*60)

# L2 normalize embeddings (DINOv2/v3 standard practice)
if HAS_CUPY:
    print("Using cuPy for GPU-accelerated L2 normalization...")
    train_embeddings_gpu = cp.asarray(train_embeddings.astype(np.float32))
    val_embeddings_gpu = cp.asarray(val_embeddings.astype(np.float32))
    test_embeddings_gpu = cp.asarray(test_embeddings.astype(np.float32))
    
    # L2 normalize on GPU
    train_norms = cp.linalg.norm(train_embeddings_gpu, axis=1, keepdims=True)
    val_norms = cp.linalg.norm(val_embeddings_gpu, axis=1, keepdims=True)
    test_norms = cp.linalg.norm(test_embeddings_gpu, axis=1, keepdims=True)
    
    train_embeddings = (train_embeddings_gpu / train_norms).get()
    val_embeddings = (val_embeddings_gpu / val_norms).get()
    test_embeddings = (test_embeddings_gpu / test_norms).get()
    
    del train_embeddings_gpu, val_embeddings_gpu, test_embeddings_gpu
    cp._default_memory_pool.free_all_blocks()
    print("Applied L2 normalization on GPU!")
else:
    from sklearn.preprocessing import normalize
    train_embeddings = normalize(train_embeddings, norm='l2')
    val_embeddings = normalize(val_embeddings, norm='l2')
    test_embeddings = normalize(test_embeddings, norm='l2')
    print("Applied L2 normalization on CPU")

print("\n" + "="*60)
print("Training Multinomial Logistic Regression (GPU) with C search...")
print("="*60)

# Grid search for best C value
best_val_acc = 0
best_c = 1.0
best_model = None
results = []

for c_val in C_VALUES:
    print(f"\n>>> Training with C={c_val}...")
    
    if HAS_CUML:
        model = LogisticRegression(
            max_iter=2000,
            C=c_val,
            class_weight='balanced',
            verbose=0
        )
    else:
        model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=2000,
            C=c_val,
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1,
            verbose=0
        )
    
    model.fit(train_embeddings, train_labels)
    
    # Evaluate on validation set
    val_preds = model.predict(val_embeddings)
    val_acc = accuracy_score(val_labels, val_preds) * 100
    val_bal_acc = balanced_accuracy_score(val_labels, val_preds) * 100
    
    results.append({
        'C': c_val,
        'val_acc': val_acc,
        'val_bal_acc': val_bal_acc
    })
    
    print(f"    C={c_val}: Val Accuracy = {val_acc:.2f}%, Val Balanced = {val_bal_acc:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_c = c_val
        best_model = model

print(f"\n>>> Best C = {best_c} with Val Accuracy = {best_val_acc:.2f}%")
print(f"\nC values tested: {results}")

print("\n" + "="*60)
print("Evaluating best model...")
print("="*60)

model = best_model

train_preds = model.predict(train_embeddings)
val_preds = model.predict(val_embeddings)
test_preds = model.predict(test_embeddings)

train_probs = model.predict_proba(train_embeddings)
val_probs = model.predict_proba(val_embeddings)
test_probs = model.predict_proba(test_embeddings)

train_acc = accuracy_score(train_labels, train_preds) * 100
val_acc = accuracy_score(val_labels, val_preds) * 100
test_acc = accuracy_score(test_labels, test_preds) * 100

train_bal_acc = balanced_accuracy_score(train_labels, train_preds) * 100
val_bal_acc = balanced_accuracy_score(val_labels, val_preds) * 100
test_bal_acc = balanced_accuracy_score(test_labels, test_preds) * 100

print(f"\n--- Best Model: C={best_c} ---")
print(f"\n--- Accuracy ---")
print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Validation Accuracy: {val_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")

print(f"\n--- Balanced Accuracy ---")
print(f"Train Balanced Accuracy: {train_bal_acc:.2f}%")
print(f"Validation Balanced Accuracy: {val_bal_acc:.2f}%")
print(f"Test Balanced Accuracy: {test_bal_acc:.2f}%")

print("\n" + "="*60)
print("Saving predictions and model...")
print("="*60)

np.save(os.path.join(BASE_DIR, 'test_preds.npy'), test_preds)
np.save(os.path.join(BASE_DIR, 'test_labels.npy'), test_labels)
np.save(os.path.join(BASE_DIR, 'test_probs.npy'), test_probs)
np.save(os.path.join(BASE_DIR, 'train_preds.npy'), train_preds)
np.save(os.path.join(BASE_DIR, 'train_labels.npy'), train_labels)
np.save(os.path.join(BASE_DIR, 'val_preds.npy'), val_preds)
np.save(os.path.join(BASE_DIR, 'val_labels.npy'), val_labels)

with open(os.path.join(BASE_DIR, 'idx_to_label.json'), 'w') as f:
    json.dump(idx_to_label, f)

crop_mapping = {}
for idx, meta in enumerate(test_metadata):
    crop_file = meta.get('crop_file', 'unknown')
    # Extract crop position from filename like "crop_00_01.npy" -> row=0, col=1
    if crop_file.startswith('crop_') and crop_file.endswith('.npy'):
        try:
            parts = crop_file.replace('crop_', '').replace('.npy', '').split('_')
            crop_row = int(parts[0])
            crop_col = int(parts[1])
        except:
            crop_row = -1
            crop_col = -1
    else:
        crop_row = -1
        crop_col = -1
    
    crop_mapping[idx] = {
        'filename': meta['filename'],
        'well': meta['well'],
        'plate': meta['plate'],
        'crop_file': crop_file,
        'crop_row': crop_row,
        'crop_col': crop_col,
        'crop_position': f"row{crop_row}_col{crop_col}"
    }

with open(os.path.join(BASE_DIR, 'crop_to_image_mapping.json'), 'w') as f:
    json.dump(crop_mapping, f, indent=2)

print(f"✅ Saved crop-to-image mapping for {len(crop_mapping)} crops")

model_info = {
    'label_to_idx': label_to_idx,
    'idx_to_label': idx_to_label,
    'num_classes': num_classes,
    'has_cuml': HAS_CUML,
    'best_c': best_c,
    'c_search_results': results
}

if HAS_CUML:
    if HAS_JOBLIB:
        joblib.dump(model, os.path.join(BASE_DIR, 'logistic_model.joblib'))
        print("Saved model using joblib")
    else:
        import pickle
        with open(os.path.join(BASE_DIR, 'logistic_model.pkl'), 'wb') as f:
            pickle.dump(model, f)
        print("Saved model using pickle")
else:
    with open(os.path.join(BASE_DIR, 'logistic_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    print("Saved model using pickle")

with open(os.path.join(BASE_DIR, 'model_info.json'), 'w') as f:
    json.dump(model_info, f, indent=2)

print("Saved predictions and model")

print("\n" + "="*60)
print("DONE!")
print("="*60)
print(f"\nResults saved to: {BASE_DIR}")
