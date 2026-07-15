# Control Classification Analysis Report

## a) Class Distribution (images per label)

| Count | Labels |
|-------|--------|
| 252 | 24 `NC_*` classes (all MG1655_NC_1-6_plusATC, MG1655_NC_1-6_minusATC, ACE-1_NC_1-6_plusATC, ACE-1_NC_1-6_minusATC) |
| 1512 | 4 non-NC classes (MG1655_plusATC, MG1655_minusATC, ACE-1_plusATC, ACE-1_minusATC) |

**Total: 12,096 images across 28 classes**

NC strains have 252 images each (from 2 wells × 6 plates × 21 images per well). Non-NC strains have 1512 images each (from 12 wells × 6 plates × 21 images).

## b) Label Taxonomy

| Group | # Labels | Members |
|-------|----------|---------|
| ATC-induced (`plusATC`) | 14 | 6× MG1655_NC_*, MG1655, 6× ACE-1_NC_*, ACE-1 |
| Non-induced (`minusATC`) | 14 | 6× MG1655_NC_*, MG1655, 6× ACE-1_NC_*, ACE-1 |
| MG1655 (WT strain) | 14 | 6× NC_plusATC, 6× NC_minusATC, plusATC, minusATC |
| ACE-1 (mutant strain) | 14 | 6× NC_plusATC, 6× NC_minusATC, plusATC, minusATC |
| NC_* wells | 24 | All NC_X_plusATC and NC_X_minusATC variants |
| Non-NC wells | 4 | MG1655_plusATC/minusATC, ACE-1_plusATC/minusATC |

## c) Feature Space Analysis (mean vectors per class, full_raw)

### Mutual Nearest Neighbor: Same strain/same NC#, +/-ATC

Only **5/14 (36%)** paired classes (same NC number, opposite induction) are mutual nearest neighbors in the 7-dimensional feature space:

- **Mutual NNs**: ACE-1_NC_1, ACE-1_NC_3, ACE-1_NC_4, MG1655_NC_5, MG1655_NC_6
- **Not mutual NNs**: 9 pairs — typically one member of the pair is closer to a *different NC number* of the same strain than to its own induction counterpart. E.g., MG1655_NC_1_plusATC → MG1655_NC_2_minusATC (not its own minusATC). This suggests within-strain NC-number proximity can outweigh induction signal.

### Cross-Strain vs Cross-Induction Distances

| Comparison | Mean Euclidean Distance |
|-----------|----------------------|
| MG1655_NC_X_plus → MG1655_NC_X_minus (same strain, diff induction) | 2,464 |
| MG1655_NC_X_plus → ACE-1_NC_X_plus (diff strain, same induction) | 14,982 |

**The induction signal is ~6× smaller than the strain signal.** The same-strain-different-induction distance is only 16% of the cross-strain distance. This means:
- Strains (MG1655 vs ACE-1) are easily separable by morphology
- Induction state (ATC +/−) is a much subtler morphological signal
- The classifier must rely on fine-grained features to distinguish +/-ATC

### Non-NC Strain Pairs

| Pair | Distance |
|------|----------|
| MG1655_plusATC ↔ MG1655_minusATC | 2,612 |
| ACE-1_plusATC ↔ ACE-1_minusATC | 2,430 |

Neither non-NC strain pair has mutual NNs. Both are closer to nearby NC classes than to each other.

## d) RF Classifier Confusion Patterns (full_raw, LOOCV)

### Overall Performance (from existing results.json)

| Metric | Value |
|--------|-------|
| Micro-avg AUC (LOOCV) | 0.818 ± 0.022 |
| Accuracy | 22.3% |
| Balanced accuracy | 11.0% |
| Classes with majority-on-diagonal | 5/28 |

The classifier significantly outperforms random (1/28 = 3.6% accuracy, 0.50 AUC) but struggles with 28-way classification.

### Confusion by Induction

| Confusion Type | Errors | % of Off-Diagonal |
|---------------|--------|-------------------|
| Same induction (+↔+ or −↔−) | 4,528 | 48.2% |
| Different induction (+↔−) | 4,868 | 51.8% |

Almost exactly split — the classifier is equally likely to confuse same-induction classes as different-induction classes. Induction state alone does not dominate the confusion structure.

### Confusion by Strain

| Confusion Type | Errors | % of Off-Diagonal |
|---------------|--------|-------------------|
| Same strain, diff +/-ATC | 3,395 | 34.1%* |
| Different strains | 3,585 | 36.0%* |

*These two categories do not sum to 100% because "same strain, same induction" confusions also exist (e.g., MG1655_NC_1_plusATC → MG1655_NC_2_plusATC).

### WT (MG1655) vs Mutant (ACE-1) Confusion

| Prediction Pattern | Errors | % of Off-Diagonal |
|-------------------|--------|-------------------|
| MG1655 → MG1655 (WT↔WT) | 2,875 | 30.6% |
| ACE-1 → ACE-1 (mut↔mut) | 2,936 | 31.2% |
| MG1655 → ACE-1 (WT→mut) | 1,893 | 20.1% |
| ACE-1 → MG1655 (mut→WT) | 1,692 | 18.0% |

**Cross-strain (WT↔mutant) errors account for 38.1%** of all off-diagonal errors, while within-strain errors account for 61.9%. The classifier is more likely to confuse two classes *within the same strain* than to confuse MG1655 with ACE-1.

## Summary of Key Findings

1. **Strain identity is the dominant factor** — MG1655 and ACE-1 are well-separated in feature space (cross-strain distances are ~6× larger than cross-induction distances).

2. **ATC induction produces a subtle but real signal** — Same-strain +/-ATC pairs are closer than cross-strain pairs, but the induction effect is much smaller than the strain effect. Only 5/14 NC pairs are mutual NNs between induction conditions.

3. **The RF classifier struggles at the 28-way classification** (22% accuracy) but is clearly above chance. The low accuracy is expected for 28 fine-grained classes with subtle morphological differences.

4. **Confusions are driven more by NC number similarity than by induction state** — e.g., MG1655_NC_1_plusATC and MG1655_NC_2_minusATC are often closer than MG1655_NC_1_plusATC vs minusATC, suggesting the NC clone number within a strain creates a morphological signature comparable to or stronger than the induction signal.

5. **Recommendation**: For practical use, consider pooling (a) by strain only (MG1655 vs ACE-1 2-class), (b) by induction state only (plusATC vs minusATC 2-class), or (c) using a hierarchical approach — first classify strain, then induction state within each strain.
