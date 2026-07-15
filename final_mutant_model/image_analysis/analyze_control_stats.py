#!/usr/bin/env python3
"""
RF (sklearn) + LR (PyTorch GPU) with class_weight='balanced'.
Leave-one-plate-out across 6 feature sets.
Saves majority-display confusion matrices (like generate_mutant_confusion.py)
+ ROC curves to {output_dir}/{clf}_results/.
"""
import numpy as np
import pandas as pd
import os, json, argparse, warnings, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='control_7stats_41classes.csv')
parser.add_argument('--output_dir', default='control_analyze_results')
parser.add_argument('--rf_trees', type=int, default=500)
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
inpath = os.path.join(SCRIPT_DIR, args.input)
outdir = os.path.join(SCRIPT_DIR, args.output_dir)

df = pd.read_csv(inpath)
PLATES = ['P1','P2','P3','P4','P5','P6']
PLATE_COLORS = {'P1':'#e41a1c','P2':'#377eb8','P3':'#4daf4a','P4':'#984ea3','P5':'#ff7f00','P6':'#a65628'}
PLATE_STYLES = {'P1':'-','P2':'--','P3':'-.','P4':':','P5':'-','P6':'--'}
ALL_STATS = ['mean','std','snr','entropy','p1','p99','median']

FEATURE_SETS = []
for region in ['full','center1128','center224']:
    for ptype in ['raw','mp']:
        cols = [f'{region}_{ptype}_{s}' for s in ALL_STATS]
        if all(c in df.columns for c in cols):
            FEATURE_SETS.append((region, ptype, cols))

all_labels = sorted(df['label'].unique())
label_to_idx = {l:i for i,l in enumerate(all_labels)}
n_classes = len(all_labels)



# ——— Confusion matrix plot (majority display, like generate_mutant_confusion) ———
def plot_majority_cm(cm, labels, title, output_path, mean_acc, std_acc):
    n = len(labels)
    cm_display = np.zeros((n, n))
    for i in range(n):
        if cm[i].sum() > 0:
            max_idx = cm[i].argmax()
            cm_display[i, max_idx] = 100

    n_max_on_diag = sum(1 for i in range(n) if cm[i].sum() > 0 and cm[i].argmax() == i)

    fig, ax = plt.subplots(figsize=(max(16, n*0.45), max(14, n*0.42)))
    sns.heatmap(cm_display, annot=cm, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax,
                vmin=0, vmax=100,
                cbar=False,
                linewidths=0.3, linecolor='white', square=True,
                annot_kws={'fontsize': 5})

    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                    edgecolor='#FF4444', lw=2.5))

    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.set_title(f'{title}  |  Acc={mean_acc:.1f}±{std_acc:.1f}%  |  {n_max_on_diag}/{n} majority-on-diag', fontsize=11)
    ax.set_xticks(np.arange(n)+0.5, labels, rotation=90, fontsize=5)
    ax.set_yticks(np.arange(n)+0.5, labels, rotation=0, fontsize=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return np.trace(cm) / np.sum(cm) * 100, n_max_on_diag


# ——— ROC plot (real FPR/TPR data + mean curve) ———
def plot_roc(fold_roc_data, set_name, clf_key, output_path, mean_acc, std_acc):
    fig, ax = plt.subplots(figsize=(10, 8))
    fold_aucs = []
    for p, (fpr, tpr) in zip(PLATES, fold_roc_data):
        fold_auc = auc(fpr, tpr)
        fold_aucs.append(fold_auc)
        ax.plot(fpr, tpr, color=PLATE_COLORS[p], ls=PLATE_STYLES[p], lw=1.2, alpha=0.6,
                label=f'{p} (AUC={fold_auc:.3f})')
    # Mean ROC curve
    all_fpr = np.sort(np.unique(np.concatenate([fpr for fpr, _ in fold_roc_data])))
    mean_tpr = np.zeros_like(all_fpr)
    for fpr, tpr in fold_roc_data:
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= len(fold_roc_data)
    mean_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color='black', lw=3, label=f'Mean (AUC={mean_auc:.3f})')
    ax.plot([0,1],[0,1],'k--',alpha=0.4,label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{clf_key.upper()} LOOCV ROC — {set_name}\nMean AUC={mean_auc:.3f} | Acc={mean_acc:.1f}±{std_acc:.1f}%')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fold_aucs


# ══════════════════════════════════════════════
# 1. RANDOM FOREST (sklearn)
# ══════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  RANDOM FOREST (class_weight=balanced, {args.rf_trees} trees)")
print(f"{'='*70}")
rf_outdir = os.path.join(outdir, 'rf_results')
os.makedirs(rf_outdir, exist_ok=True)
rf_results = []

for region, ptype, cols in FEATURE_SETS:
    set_name = f"{region}_{ptype}"
    fold_accs, fold_aucs = [], []
    y_true_all, y_pred_all = [], []
    y_score_all, plates_all, labels_all, paths_all, wells_all, images_all = [], [], [], [], [], []

    fold_roc_data = []
    t0 = time.time()
    for held_out in PLATES:
        train_df = df[df['plate'] != held_out]
        test_df = df[df['plate'] == held_out]
        X_train = train_df[cols].values.astype(np.float32)
        y_train = train_df['label'].map(label_to_idx).values
        X_test = test_df[cols].values.astype(np.float32)
        y_test = test_df['label'].map(label_to_idx).values

        clf = RandomForestClassifier(n_estimators=args.rf_trees, max_depth=20,
                                      random_state=42, class_weight='balanced', n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_score = clf.predict_proba(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_score_all.append(y_score)
        plates_all.extend(test_df['plate'].tolist())
        labels_all.extend(test_df['label'].tolist())
        paths_all.extend(test_df['path'].tolist())
        wells_all.extend(test_df['well'].tolist())
        images_all.extend(test_df['image'].tolist())
        fold_accs.append(accuracy_score(y_test, y_pred) * 100)

        classes_present = sorted(set(y_test))
        if len(classes_present) >= 2:
            y_test_bin = label_binarize(y_test, classes=classes_present)
            y_score_sub = y_score[:, classes_present]
            fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score_sub.ravel())
            fold_roc_data.append((fpr, tpr))

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=range(n_classes))

    np.savez(os.path.join(rf_outdir, f'{set_name}_predictions.npz'),
             y_true=np.array(y_true_all, dtype=np.int32),
             y_pred=np.array(y_pred_all, dtype=np.int32),
             y_score=np.vstack(y_score_all),
             plates=np.array(plates_all, dtype=object),
             labels=np.array(labels_all, dtype=object),
             paths=np.array(paths_all, dtype=object),
             wells=np.array(wells_all, dtype=object),
             images=np.array(images_all, dtype=object))
    acc_total, n_maj = plot_majority_cm(
        cm, all_labels,
        f'RF — {set_name}',
        os.path.join(rf_outdir, f'cm_{set_name}.png'),
        mean_acc, std_acc)

    fold_aucs = plot_roc(fold_roc_data, set_name, 'RF', os.path.join(rf_outdir, f'roc_{set_name}.png'), mean_acc, std_acc)

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    mean_auc = np.mean(fold_aucs) if fold_aucs else 0
    rf_results.append({
        'set': set_name, 'mean_auc': float(mean_auc),
        'mean_acc_pct': float(mean_acc), 'acc_std_pct': float(std_acc),
        'total_acc_pct': float(acc_total),
        'majority_on_diagonal': int(n_maj), 'n_classes': n_classes,
        'per_fold_acc_pct': [float(a) for a in fold_accs],
        'per_fold_auc': [float(a) for a in fold_aucs],
    })
    elapsed = time.time() - t0
    print(f"  {set_name:>16}  Acc={mean_acc:.1f}±{std_acc:.1f}%  AUC={mean_auc:.3f}  Maj={n_maj}/{n_classes}  [{elapsed:.0f}s]")

with open(os.path.join(rf_outdir, 'results.json'), 'w') as f:
    json.dump(rf_results, f, indent=2)


# ══════════════════════════════════════════════
# 2. LOGISTIC REGRESSION (sklearn, saga solver)
# ══════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  LOGISTIC REGRESSION (sklearn, solver=saga, class_weight=balanced)")
print(f"{'='*70}")
lr_outdir = os.path.join(outdir, 'lr_results')
os.makedirs(lr_outdir, exist_ok=True)
lr_results = []

for region, ptype, cols in FEATURE_SETS:
    set_name = f"{region}_{ptype}"
    fold_accs, fold_aucs = [], []
    y_true_all, y_pred_all = [], []
    y_score_all, plates_all, labels_all, paths_all, wells_all, images_all = [], [], [], [], [], []

    fold_roc_data = []
    t0 = time.time()
    for held_out in PLATES:
        train_df = df[df['plate'] != held_out]
        test_df = df[df['plate'] == held_out]
        X_train = train_df[cols].values.astype(np.float32)
        y_train = train_df['label'].map(label_to_idx).values
        X_test = test_df[cols].values.astype(np.float32)
        y_test = test_df['label'].map(label_to_idx).values

        clf = LogisticRegression(solver='saga', max_iter=2000, random_state=42,
                                  class_weight='balanced', n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_score = clf.predict_proba(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_score_all.append(y_score)
        plates_all.extend(test_df['plate'].tolist())
        labels_all.extend(test_df['label'].tolist())
        paths_all.extend(test_df['path'].tolist())
        wells_all.extend(test_df['well'].tolist())
        images_all.extend(test_df['image'].tolist())
        fold_accs.append(accuracy_score(y_test, y_pred) * 100)

        classes_present = sorted(set(y_test))
        if len(classes_present) >= 2:
            y_test_bin = label_binarize(y_test, classes=classes_present)
            y_score_sub = y_score[:, classes_present]
            fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score_sub.ravel())
            fold_roc_data.append((fpr, tpr))

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=range(n_classes))

    np.savez(os.path.join(lr_outdir, f'{set_name}_predictions.npz'),
             y_true=np.array(y_true_all, dtype=np.int32),
             y_pred=np.array(y_pred_all, dtype=np.int32),
             y_score=np.vstack(y_score_all),
             plates=np.array(plates_all, dtype=object),
             labels=np.array(labels_all, dtype=object),
             paths=np.array(paths_all, dtype=object),
             wells=np.array(wells_all, dtype=object),
             images=np.array(images_all, dtype=object))
    acc_total, n_maj = plot_majority_cm(
        cm, all_labels,
        f'LR — {set_name}',
        os.path.join(lr_outdir, f'cm_{set_name}.png'),
        mean_acc, std_acc)

    fold_aucs = plot_roc(fold_roc_data, set_name, 'LR', os.path.join(lr_outdir, f'roc_{set_name}.png'), mean_acc, std_acc)
    mean_auc = np.mean(fold_aucs) if fold_aucs else 0
    lr_results.append({
        'set': set_name, 'mean_auc': float(mean_auc),
        'mean_acc_pct': float(mean_acc), 'acc_std_pct': float(std_acc),
        'total_acc_pct': float(acc_total),
        'majority_on_diagonal': int(n_maj), 'n_classes': n_classes,
        'per_fold_acc_pct': [float(a) for a in fold_accs],
        'per_fold_auc': [float(a) for a in fold_aucs],
    })
    elapsed = time.time() - t0
    print(f"  {set_name:>16}  Acc={mean_acc:.1f}±{std_acc:.1f}%  AUC={mean_auc:.3f}  Maj={n_maj}/{n_classes}  [{elapsed:.0f}s]")

with open(os.path.join(lr_outdir, 'results.json'), 'w') as f:
    json.dump(lr_results, f, indent=2)


# ══════════════════════════════════════════════
# COMPARISON
# ══════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  RF vs LR Comparison")
print(f"{'='*70}")
print(f"{'Feature Set':>16} {'RF Acc%':>10} {'RF Std':>8} {'LR Acc%':>10} {'LR Std':>8} {'Best':>8}")
print('-'*62)
for rr, lr in zip(rf_results, lr_results):
    best = 'RF' if rr['mean_acc_pct'] >= lr['mean_acc_pct'] else 'LR'
    print(f"{rr['set']:>16} {rr['mean_acc_pct']:>8.1f} {rr['acc_std_pct']:>8.1f} {lr['mean_acc_pct']:>10.1f} {lr['acc_std_pct']:>8.1f} {best:>8}")
print(f"\nResults saved to:\n  {rf_outdir}/\n  {lr_outdir}/")
