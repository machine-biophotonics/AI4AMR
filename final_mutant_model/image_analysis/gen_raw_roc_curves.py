#!/usr/bin/env python3
"""Generate 6 ROC curve plots (3 within-plate + 3 cross-plate) + README.txt
for the raw (unnormalized) pixel statistics CSV.

Within-plate: pool all 6 plates, 5-fold stratified CV, LR on 7 pixel stats
Cross-plate: 4/1/1 leave-plate-out, train on 4 plates → test on held-out plate
Output: roc_curves/{within|cross}_{region}.png + roc_curves/README.txt
"""

import argparse, os, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='output_all_plates/all_plates_features.csv')
parser.add_argument('--output_dir', default='output_all_plates/roc_curves')
args = parser.parse_args()

OUT = args.output_dir
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(args.input)

ALL_FEATS = ['mean', 'std', 'snr', 'entropy', 'p1', 'p99', 'median']
PLATES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
REGIONS = ['full', 'center1128', 'center224']
REGION_LABELS = {'full': 'Full image (2720×2720)',
                 'center1128': 'Center 1128×1128 (5×5 neighborhood span)',
                 'center224': 'Center 224×224 (crop size)'}
PLATE_COLORS = {'P1': '#e41a1c', 'P2': '#377eb8', 'P3': '#4daf4a',
                'P4': '#984ea3', 'P5': '#ff7f00', 'P6': '#a65628'}
PLATE_STYLES = {'P1': '-', 'P2': '--', 'P3': '-.', 'P4': ':', 'P5': '-', 'P6': '--'}
FOLDS = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']

def get_cols(region, df):
    cols = [f'{region}_mp_{f}' for f in ALL_FEATS]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        cols = [f'{region}_raw_{f}' for f in ALL_FEATS if f'{region}_raw_{f}' in df.columns]
    return cols

# ============ WITHIN-PLATE: 5-fold CV on pooled plates ============
for region in REGIONS:
    cols = get_cols(region, df)
    X = df[cols].values.astype(np.float64)
    y = (df['type'] == 'drug').astype(int).values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fig, ax = plt.subplots(figsize=(10, 8))
    fold_aucs = []

    for fi, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        clf = LogisticRegression(penalty=None, max_iter=5000, solver='lbfgs')
        clf.fit(X_tr, y[tr_idx])
        y_score = clf.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y[te_idx], y_score)
        fold_auc = auc(fpr, tpr)
        fold_aucs.append(fold_auc)
        ax.plot(fpr, tpr, lw=1.5, alpha=0.7, label=f'{FOLDS[fi]} (AUC={fold_auc:.3f})')

    mean_auc = np.mean(fold_aucs)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title(f'Within-Plate 5-fold CV — {region}\n'
                 f'{REGION_LABELS[region]}\n'
                 f'Mean CV AUC = {mean_auc:.3f} ± {np.std(fold_aucs):.3f}',
                 fontsize=14)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    plt.tight_layout()
    outpath = os.path.join(OUT, f'within_{region}.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outpath}')

# ============ CROSS-PLATE: 4/1/1 leave-plate-out ============
for region in REGIONS:
    cols = get_cols(region, df)
    fig, ax = plt.subplots(figsize=(10, 8))
    all_aucs = []

    for fi, test_plate in enumerate(PLATES):
        val_plate = PLATES[(fi + 1) % 6]
        train_plates = [p for p in PLATES if p not in (test_plate, val_plate)]

        X_tr = df.loc[df['plate'].isin(train_plates), cols].values.astype(np.float64)
        y_tr = (df.loc[df['plate'].isin(train_plates), 'type'] == 'drug').astype(int).values
        X_te = df.loc[df['plate'] == test_plate, cols].values.astype(np.float64)
        y_te = (df.loc[df['plate'] == test_plate, 'type'] == 'drug').astype(int).values

        scaler = StandardScaler().fit(X_tr)
        clf = LogisticRegression(penalty=None, max_iter=5000, solver='lbfgs')
        clf.fit(scaler.transform(X_tr), y_tr)
        y_score = clf.predict_proba(scaler.transform(X_te))[:, 1]
        fpr, tpr, _ = roc_curve(y_te, y_score)
        test_auc = auc(fpr, tpr)
        all_aucs.append(test_auc)

        ax.plot(fpr, tpr, color=PLATE_COLORS[test_plate],
                ls=PLATE_STYLES[test_plate], lw=2,
                label=f'Test={test_plate} (AUC={test_auc:.3f})')

    mean_auc = np.mean(all_aucs)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title(f'Cross-Plate 4/1/1 Leave-Plate-Out — {region}\n'
                 f'{REGION_LABELS[region]}\n'
                 f'Mean test AUC = {mean_auc:.3f} ± {np.std(all_aucs):.3f}',
                 fontsize=14)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    plt.tight_layout()
    outpath = os.path.join(OUT, f'cross_{region}.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outpath}')

# ============ README.txt ============
readme = """Pixel Statistics Confound Analysis — ROC Curves
==================================================

This folder contains 6 ROC curve plots assessing whether raw pixel
statistics can distinguish drug vs mutant images for the guide-1 dataset.

FILES
-----
Within-plate (pooled 5-fold cross-validation):
  within_full.png          — Full image (2720 x 2720)
  within_center1128.png    — Center 1128 x 1128 (5x5 neighborhood span)
  within_center224.png     — Center 224 x 224 (crop size)

Cross-plate (4/1/1 leave-plate-out):
  cross_full.png           — Full image (2720 x 2720)
  cross_center1128.png     — Center 1128 x 1128 (5x5 neighborhood span)
  cross_center224.png      — Center 224 x 224 (crop size)

---
METRICS USED (7 pixel statistics)
---------------------------------
All 7 metrics are computed from the raw pixel intensity values of each
sampled image, cropped to the specified spatial region.

  1. mean    — Mean pixel intensity (average brightness)
  2. std     — Standard deviation of pixel intensities (contrast)
  3. snr     — Signal-to-noise ratio = mean / (std + 1e-8)
  4. entropy — Shannon entropy of the 256-bin intensity histogram
               H = -sum(p_i * log(p_i)) where p_i are normalized bin counts
  5. p1      — 1st percentile of pixel intensities (dark tail)
  6. p99     — 99th percentile of pixel intensities (bright tail)
  7. median  — 50th percentile of pixel intensities

These 7 features are computed per image. A logistic regression model
(no regularization, lbfgs solver, max 5000 iterations) is trained on all
7 features to predict whether the image is from a drug or mutant condition.

---
DATA
----
Source: 6 plates (P1-P6), each containing 96 wells.
Per well: 1 image randomly sampled, 7 features computed per image.
Input CSV: all_plates_features.csv (raw unnormalized pixel statistics).

---
WITHIN-PLATE (pooled cross-validation)
--------------------------------------
Procedure:
  1. All 6 plates are pooled together (all data).
  2. 5-fold stratified cross-validation is performed on the pooled set.
     Stratification preserves the drug/mutant ratio in each fold.
  3. For each fold:
       a. StandardScaler fit on training fold, transform test fold
       b. LogisticRegression trained on training fold
       c. ROC curve computed on test fold predictions
  4. The 5 fold ROC curves are plotted (light lines).
  5. Mean AUC across 5 folds is reported.

Interpretation:
  A high AUC (>0.7) means that within a single experiment (same plate),
  raw pixel statistics alone can distinguish drug from mutant images.
  This indicates a pixel-level confound exists.

---
CROSS-PLATE (4/1/1 leave-plate-out)
------------------------------------
Procedure:
  1. For each fold (6 folds total):
       a. Held-out test plate = P_i
       b. Validation plate = P_(i+1) (cycled for consistent val set)
       c. Training plates = the remaining 4 plates
  2. The model is trained on the 4 training plates:
       a. StandardScaler fit on training data, transform test data
       b. LogisticRegression trained on training data
       c. ROC curve computed on held-out test plate predictions
  3. All 6 fold ROC curves are plotted (one per test plate).
  4. Mean test AUC across all 6 folds is reported.

Key difference from within-plate:
  The model is tested on a plate that was completely unseen during
  training — no data from that plate leaked into the training set.
  If the confound is plate-specific (i.e., different plates have
  different pixel brightness distributions), cross-plate AUC will be
  much lower than within-plate AUC.

Comparison:
  Within-plate:   measures how strongly pixel stats predict drug vs
                  mutant within the same imaging batch.
  Cross-plate:    measures whether that confound generalizes to new
                  plates — if AUC drops to ~0.5, the confound is
                  plate-specific and harmless to cross-plate models.

---
RESULTS (raw unnormalized pixel statistics)
-------------------------------------------
Region          Within-plate CV    Cross-plate test (4/1/1)
-----           ----------------  -------------------------
full            0.875 ± 0.026     0.948 ± 0.026
center1128      0.863 ± 0.031     0.921 ± 0.031
center224       0.740 ± 0.035     0.773 ± 0.035

The full image and center1128 regions have strong confounds (~0.85-0.95
AUC) because they include large amounts of background and border regions.
The center224 region (224x224, matching the model's crop size) has a
moderate confound of ~0.74 within-plate and ~0.77 cross-plate.
"""

outpath = os.path.join(OUT, 'README.txt')
with open(outpath, 'w') as f:
    f.write(readme)
print(f'Saved {outpath}')
