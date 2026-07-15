#!/usr/bin/env python3
"""
Read control mode prediction CSV, generate 41×41 and 7-group confusion matrices.
Row-majority binary coloring, count annotations in all cells.

Usage:
  python3 generate_control_confusion.py <csv_path>
"""
import numpy as np, pandas as pd, sys, os, json, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter

csv_path = sys.argv[1]
outdir = os.path.dirname(csv_path)

df = pd.read_csv(csv_path)

# --- Majority vote per image ---
image_results = []
for img, grp in df.groupby('image_path'):
    gt = grp['ground_truth_label'].iloc[0]
    if pd.isna(gt) or gt == '' or gt == 'None':
        continue
    # Majority vote across positions
    preds = grp['predicted_class_name'].tolist()
    majority_pred = Counter(preds).most_common(1)[0][0]
    image_results.append({'image_path': img, 'ground_truth': gt, 'predicted': majority_pred})

if not image_results:
    print("ERROR: No images with ground truth labels found.")
    sys.exit(1)

df_img = pd.DataFrame(image_results)
all_labels = sorted(df_img['ground_truth'].unique())
label_to_idx = {l:i for i,l in enumerate(all_labels)}
n = len(all_labels)

y_true = df_img['ground_truth'].map(label_to_idx).values
y_pred = df_img['predicted'].map(label_to_idx).values
cm = confusion_matrix(y_true, y_pred, labels=range(n))

# ── Group mapping (same 7 groups as analyze_grouped_cm.py) ──
GROUP_MAP = {}
for l in all_labels:
    if l == 'drug_control':
        GROUP_MAP[l] = 'drug_control'
    elif l.startswith('WT NC'):
        GROUP_MAP[l] = 'WT NC'
    elif l.startswith('NC_'):
        GROUP_MAP[l] = 'NC'
    elif l.startswith('ACE-1_NC') or l == 'ACE-1_plusATC':
        GROUP_MAP[l] = 'ACE-1 +ATC/NC'
    elif l == 'ACE-1_minusATC':
        GROUP_MAP[l] = 'ACE-1 -ATC'
    elif l.startswith('MG1655_NC') or l == 'MG1655_plusATC':
        GROUP_MAP[l] = 'MG1655 +ATC/NC'
    elif l == 'MG1655_minusATC':
        GROUP_MAP[l] = 'MG1655 -ATC'
    else:
        GROUP_MAP[l] = 'other'

group_order = ['ACE-1 -ATC','ACE-1 +ATC/NC','MG1655 -ATC','MG1655 +ATC/NC','NC','WT NC','drug_control']
g_names = [g for g in group_order if g in set(GROUP_MAP.values())]
g2i = {g:i for i,g in enumerate(g_names)}
ng = len(g_names)

y_true_g = np.array([g2i[GROUP_MAP[all_labels[t]]] for t in y_true])
y_pred_g = np.array([g2i[GROUP_MAP[all_labels[p]]] for p in y_pred])
cm_g = confusion_matrix(y_true_g, y_pred_g, labels=range(ng))

total_acc = np.trace(cm) / np.sum(cm) * 100
total_acc_g = np.trace(cm_g) / np.sum(cm_g) * 100

# ── Plot 41×41 CM ──
def plot_cm(cm, labels, title, output_path, mean_acc):
    n = len(labels)
    cm_display = np.zeros((n, n))
    for i in range(n):
        if cm[i].sum() > 0:
            cm_display[i, cm[i].argmax()] = 100

    n_maj = sum(cm[i].argmax() == i for i in range(n) if cm[i].sum() > 0)

    fig, ax = plt.subplots(figsize=(max(16, n*0.45), max(14, n*0.42)))
    sns.heatmap(cm_display, annot=cm, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax,
                vmin=0, vmax=100, cbar=False,
                linewidths=0.3, linecolor='white', square=True,
                annot_kws={'fontsize': 5})

    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                    edgecolor='#FF4444', lw=2.5))

    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.set_title(f'{title}  |  Acc={mean_acc:.1f}%  |  {n_maj}/{n} majority-on-diag', fontsize=11)
    ax.set_xticks(np.arange(n)+0.5, labels, rotation=90, fontsize=5)
    ax.set_yticks(np.arange(n)+0.5, labels, rotation=0, fontsize=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return n_maj

# ── Plot 7-group CM ──
def plot_grouped_cm(cm, labels, title, output_path, mean_acc):
    n = len(labels)
    cm_display = np.zeros((n, n))
    for i in range(n):
        if cm[i].sum() > 0:
            cm_display[i, cm[i].argmax()] = 100

    n_maj = sum(cm[i].argmax() == i for i in range(n) if cm[i].sum() > 0)

    fig, ax = plt.subplots(figsize=(n*1.2+2, n*1.1+1))
    sns.heatmap(cm_display, annot=cm, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax,
                vmin=0, vmax=100, cbar=False, linewidths=1, linecolor='white',
                square=True, annot_kws={'fontsize': 13})

    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                    edgecolor='#FF4444', lw=3))

    ax.set_xlabel('Predicted Group', fontsize=13)
    ax.set_ylabel('True Group', fontsize=13)
    ax.set_title(f'{title}  |  Acc={mean_acc:.1f}%  |  {n_maj}/{n} majority-on-diag', fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=11)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return n_maj

# ── Save 41×41 ──
base = os.path.splitext(os.path.basename(csv_path))[0]
cm_path = os.path.join(outdir, f'{base}_cm41.png')
n_maj41 = plot_cm(cm, all_labels, f'Control — {base}', cm_path, total_acc)
print(f'41×41 CM saved: {cm_path}')
print(f'Acc={total_acc:.1f}%, {n_maj41}/{n} majority-on-diag')

# ── Save grouped ──
cm_g_path = os.path.join(outdir, f'{base}_cm7.png')
n_maj7 = plot_grouped_cm(cm_g, g_names, f'Grouped — {base}', cm_g_path, total_acc_g)
print(f'7-group CM saved: {cm_g_path}')
print(f'Group Acc={total_acc_g:.1f}%, {n_maj7}/{ng} majority-on-diag')

# ── Per-image results CSV ──
results_path = os.path.join(outdir, f'{base}_image_results.csv')
df_img.to_csv(results_path, index=False)
print(f'Image results saved: {results_path}')

# ── Print per-class accuracy ──
print(f'\n{"Class":30s}  Total  Correct  Acc%')
print('-'*55)
for i, l in enumerate(all_labels):
    tot = cm[i].sum()
    corr = cm[i, i]
    print(f'{l:30s}  {tot:5d}  {corr:5d}  {corr/tot*100:5.1f}%')

print(f'\n{"Group":20s}  Total  Correct  Acc%')
print('-'*45)
for i, g in enumerate(g_names):
    tot = cm_g[i].sum()
    corr = cm_g[i, i]
    print(f'{g:20s}  {tot:5d}  {corr:5d}  {corr/tot*100:5.1f}%')
