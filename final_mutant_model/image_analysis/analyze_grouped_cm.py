#!/usr/bin/env python3
"""
Load RF predictions npz, collapse 41 classes into 7 groups, plot grouped CM.
Usage:
  python3 analyze_grouped_cm.py <npz_path>
"""
import numpy as np, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

npz_path = sys.argv[1]
data = np.load(npz_path, allow_pickle=True)
y_true = data['y_true']
y_pred = data['y_pred']
labels = data['labels']
plates = data['plates']

all_labels = sorted(np.unique(labels))
PLATES = ['P1','P2','P3','P4','P5','P6']

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
group_names = [g for g in group_order if g in set(GROUP_MAP.values())]
g2i = {g:i for i,g in enumerate(group_names)}
n_groups = len(group_names)

y_true_g = np.array([g2i[GROUP_MAP[all_labels[t]]] for t in y_true])
y_pred_g = np.array([g2i[GROUP_MAP[all_labels[p]]] for p in y_pred])

# Per-fold group accuracies
fold_group_accs = np.zeros((len(PLATES), n_groups))
fold_total_accs = np.zeros(len(PLATES))
for k, held in enumerate(PLATES):
    mask = plates == held
    yt = y_true_g[mask]
    yp = y_pred_g[mask]
    fold_total_accs[k] = (yt == yp).mean() * 100
    for g in range(n_groups):
        gm = yt == g
        if gm.sum() > 0:
            fold_group_accs[k, g] = (yp[gm] == g).mean() * 100

# Aggregate confusion matrix
cm = confusion_matrix(y_true_g, y_pred_g, labels=range(n_groups))

# Display: binary majority coloring
cm_display = np.zeros((n_groups, n_groups))
for i in range(n_groups):
    idx = cm[i].argmax()
    cm_display[i, idx] = 100

mean_total = fold_total_accs.mean()
std_total = fold_total_accs.std()
mean_group = fold_group_accs.mean(axis=0)
std_group = fold_group_accs.std(axis=0)

fig, ax = plt.subplots(figsize=(n_groups*1.2+2, n_groups*1.1+1))
sns.heatmap(cm_display, annot=cm, fmt='d', cmap='Blues',
            xticklabels=group_names, yticklabels=group_names, ax=ax,
            vmin=0, vmax=100, cbar=False, linewidths=1, linecolor='white',
            square=True, annot_kws={'fontsize': 13})
for i in range(n_groups):
    ax.add_patch(plt.Rectangle((i,i),1,1,fill=False,edgecolor='#FF4444',lw=3))
ax.set_xlabel('Predicted Group', fontsize=13)
ax.set_ylabel('True Group', fontsize=13)
base = os.path.splitext(os.path.basename(npz_path))[0]
n_maj = sum(cm[i].argmax() == i for i in range(n_groups))
ax.set_title(f'{base}  |  Group Acc={mean_total:.1f}±{std_total:.1f}%  |  {n_maj}/{n_groups} majority-on-diag',
             fontsize=14)
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=11)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=11)
plt.tight_layout()
outpath = os.path.join(os.path.dirname(npz_path), f'{base}_grouped_cm.png')
fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Grouped CM saved: {outpath}')
print(f'Group accuracy: {mean_total:.1f}±{std_total:.1f}%')

print(f'\n{"Group":20s}  {"Mean":>8s}  {"Std":>6s}')
print('-'*36)
for i, g in enumerate(group_names):
    print(f'{g:20s}  {mean_group[i]:6.1f}%  {std_group[i]:5.1f}%')

print(f'\nPer-fold group accuracies:')
print(f'{"Fold":6s}', end='')
for g in group_names:
    print(f'  {g:>12s}', end='')
print()
for k, held in enumerate(PLATES):
    print(f'{held:6s}', end='')
    for g in range(n_groups):
        print(f'  {fold_group_accs[k,g]:10.1f}%', end='')
    print()
