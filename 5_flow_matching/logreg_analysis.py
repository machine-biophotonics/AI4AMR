#!/usr/bin/env python3
"""Post-hoc logistic regression on t10_cond bottleneck latents — single fold."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings, os, json, re, argparse
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='t10_uncond',
                    choices=['t05_cond','t05_uncond','t10_cond','t10_uncond'])
args = parser.parse_args()
MODE = args.mode

DATA = "/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/5_flow_matching/latent_analysis_pacmap"
OUT = os.path.join(DATA, f"logreg_results_{MODE}")
os.makedirs(OUT, exist_ok=True)

labels = np.load(os.path.join(DATA, "labels.npy"))
class_names = np.load(os.path.join(DATA, "class_names.npy"), allow_pickle=True)
n_classes = len(np.unique(labels))
n_images = len(labels)

def infer_group(name):
    name = str(name)
    if name == 'control' or name.startswith('NC_') or name.startswith('WT NC_'):
        return 2
    if re.search(r'_\d+\.?\d*x$', name):
        return 0
    return 1

group_labels = np.array([infer_group(class_names[l]) for l in labels])
results = {}

# Load features
feats = np.load(os.path.join(DATA, f"feats_{MODE}.npy")).astype(np.float64)
scaler = StandardScaler()
feats_scaled = scaler.fit_transform(feats)

# single stratified 80/20 split
tr_idx, val_idx = train_test_split(
    np.arange(n_images), test_size=0.2, stratify=labels, random_state=42
)

# 185-way
print(f"185-WAY | {MODE} | {feats.shape}\n")
clf = LogisticRegression(solver='lbfgs', max_iter=5000, C=1.0, n_jobs=8, random_state=42)
clf.fit(feats_scaled[tr_idx], labels[tr_idx])
y_pred = clf.predict(feats_scaled[val_idx])
acc = accuracy_score(labels[val_idx], y_pred)
f1_m = f1_score(labels[val_idx], y_pred, average='macro')
print(f"  >>> acc={acc:.4f}  f1_macro={f1_m:.4f}\n")
results['185_way'] = {'acc': float(acc), 'f1_macro': float(f1_m)}

# 3-way
print("3-WAY (Drug/Mutant/Control)")
tr_idx3, val_idx3 = train_test_split(
    np.arange(n_images), test_size=0.2, stratify=group_labels, random_state=42
)
clf3 = LogisticRegression(solver='lbfgs', max_iter=5000, C=1.0, n_jobs=8, random_state=42)
clf3.fit(feats_scaled[tr_idx3], group_labels[tr_idx3])
y_pred3 = clf3.predict(feats_scaled[val_idx3])
acc3 = accuracy_score(group_labels[val_idx3], y_pred3)
print(f"  >>> acc={acc3:.4f}\n")
results['3_way'] = {'acc': float(acc3)}

# Per-class accuracy
print("PER-CLASS ACCURACY")
y_pred_all = clf.predict(feats_scaled)
per_class = {}
for c in range(n_classes):
    m = labels == c
    if m.sum() > 0:
        per_class[c] = float(accuracy_score(labels[m], y_pred_all[m]))

class_accs = per_class
sorted_cls = sorted(class_accs.items(), key=lambda x: -x[1])

print(f"  {'Class':<35s} {'Acc':>7s} {'Type':>8s}")
print(f"  {'-'*35} {'-'*7} {'-'*8}")
for cls, acc in sorted_cls[:15]:
    print(f"  {str(class_names[cls]):<35s} {acc:>6.2%}  {['Drug','Mutant','Control'][infer_group(class_names[cls])]:>8s}")
print(f"  {'...':>35s}")
for cls, acc in sorted_cls[-10:]:
    print(f"  {str(class_names[cls]):<35s} {acc:>6.2%}  {['Drug','Mutant','Control'][infer_group(class_names[cls])]:>8s}")

results['per_class'] = {int(k): v for k, v in class_accs.items()}
results['n_images'] = n_images
results['n_classes'] = n_classes
results['feature_dim'] = 256
results['class_names'] = [str(n) for n in class_names]

with open(os.path.join(OUT, "logreg_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nDone -> {OUT}/logreg_results.json")
