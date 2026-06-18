Pixel Statistics Confound Analysis — ROC Curves
==================================================
Method: fourier
Regions: full, center1128, center224

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
Input CSV: all_plates_features_fourier.csv

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
