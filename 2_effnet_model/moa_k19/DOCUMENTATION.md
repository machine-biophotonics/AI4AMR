# CRISPRi MOA Discovery Analysis - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Data Structure](#data-structure)
3. [Clustering Approach](#clustering-approach)
4. [Metrics Explained Simply](#metrics-explained-simply)
5. [Scripts Used](#scripts-used)
6. [Results Summary](#results-summary)
7. [Files Generated](#files-generated)

---

## Overview

This project discovers **Mechanism of Action (MOA) clusters** from CRISPRi phenotypic embeddings. Instead of using the predefined 85 classes (28 genes × 3 guides + WT), we cluster the embeddings to find natural groupings that might reveal shared biological mechanisms.

**Key Finding:** Using k=19 clusters (optimal from silhouette analysis) provides better separation than k=8, with 9 high-purity clusters (≥80% pure).

---

## Data Structure

```
Level 1: CROPS (290,304 images)
    ↓ aggregate by image
Level 2: IMAGES (2,016 images, ~144 crops each)
    ↓ aggregate by well  
Level 3: WELLS (96 wells, multiple images each)
    ↓ each well = one gene × one guide
Level 4: CLASSES (85 = 28 genes × 3 guides + WT)
```

### File Locations
- **Embeddings:** `effnet_model/eval_results/test_embeddings.npy` (290304 × 1280)
- **Labels:** `effnet_model/eval_results/test_labels.npy` (0-84)
- **Label mapping:** `effnet_model/eval_results/idx_to_label.json`
- **Crop mapping:** `effnet_model/eval_results/crop_to_image_mapping.json`

---

## Clustering Approach

### Step 1: Aggregate to Class Centroids
```
For each of 85 classes:
    centroid = mean(all embeddings for that class)
Result: 85 × 1280 matrix
```

### Step 2: Dimensionality Reduction
```
PCA: 1280 → 84 dimensions (retains 100% variance)
```

### Step 3: K-Means Clustering
```
Tested k = 2 to 19
Best k = 19 (silhouette score = 0.2442)
```

### Step 4: Map Back to All Levels
```
Class → Cluster mapping applied to:
    - Individual crops (290,304)
    - Images (2,016)  
    - Wells (96)
```

---

## Metrics Explained Simply

### 1. Purity

**What it answers:** "How clean is each cluster?"

**How it's calculated:**
```
For each cluster:
    1. Count how many samples from each class are in it
    2. Find the most common class
    3. Purity = (count of dominant class) / (total in cluster)
```

**Example:**
```
Cluster 6 contains:
    - lptA_1, lptA_2, lptA_3  (3 samples)
    - lptC_1, lptC_2, lptC_3  (3 samples)
    Total: 6 samples

Dominant class: lptA (3 samples) OR lptC (3 samples) - tie
Purity = 3/6 = 0.50 (50%)
```

**Our result: 33.7% average purity**
- Means clusters are MIXED (not one class per cluster)
- This is actually INTERESTING - it shows the 85 classes don't map cleanly to phenotypes

---

### 2. Class Accuracy (the 100% number)

**What it answers:** "What's the best-case mapping for each class?"

**How it's calculated:**
```
For each class:
    1. Find which cluster most samples go to
    2. Accuracy = (samples in dominant cluster) / (total samples of that class)
```

**Why it's 100%:**
- Every class WILL have a cluster where most samples go (by definition)
- This is a "best case" metric, not true accuracy

**This is NOT meaningful** - it's just saying each class has a home somewhere.

---

### 3. Better Metrics (not implemented yet)

| Metric | Range | Meaning |
|--------|-------|---------|
| **Silhouette Score** | -1 to 1 | How well-separated clusters are. 0.24 = moderate |
| **NMI** (Normalized Mutual Information) | 0 to 1 | How much clustering agrees with labels |
| **ARI** (Adjusted Rand Index) | -1 to 1 | Similarity of cluster pairs to label pairs |
| **Entropy** | 0 to ∞ | How mixed each cluster is (lower = better) |

---

### 4. Guide Consistency

**What it answers:** "Do all 3 guides for a gene cluster together?"

**Example:**
```
Gene ftsI:
    - ftsI_1 → Cluster 9
    - ftsI_2 → Cluster 9
    - ftsI_3 → Cluster 9
    Result: CONSISTENT ✓

Gene dnaB:
    - dnaB_1 → Cluster 2
    - dnaB_2 → Cluster 1
    - dnaB_3 → Cluster 2
    Result: INCONSISTENT ✗ (guide 2 has different phenotype)
```

**Our result:** 10/28 genes (35.7%) have consistent guides

---

## Scripts Used

### Script 1: `moa_k19_analysis.py`

**Purpose:** Main clustering analysis with k=19

**What it does:**
1. Loads embeddings and labels
2. Aggregates to 85 class centroids
3. Runs K-Means with k=19
4. Maps clusters back to crop/image/well levels
5. Creates t-SNE and UMAP visualizations colored by MOA clusters
6. Creates dendrogram
7. Analyzes guide consistency

**Outputs:**
- `tsne_*_moa.html` - Interactive t-SNE plots (3 levels)
- `tsne_*_pathway.html` - t-SNE colored by original pathways
- `tsne_*_umap.html` - UMAP plots
- `dendrogram.png` - Hierarchical clustering tree
- `moa_cluster_analysis.csv` - Cluster details
- `guide_consistency.csv` - Guide analysis

---

### Script 2: `confusion_3levels_85_19.py`

**Purpose:** Confusion matrices at 3 levels showing 85 classes vs 19 MOA clusters

**What it does:**
1. Uses clustering results from Script 1
2. Builds confusion matrix manually (85 rows × 19 columns)
3. Calculates purity metrics
4. Plots heatmaps with cell counts

**Outputs:**
- `confusion_crop_85_19.png` - Crop level (290K samples)
- `confusion_image_85_19.png` - Image level (2K samples)
- `confusion_well_85_19.png` - Well level (96 samples)
- `accuracy_summary_3levels.csv` - Summary table

**How to read the confusion matrix:**
```
           MOA-0  MOA-1  MOA-2 ... MOA-18
WT           0      0    36288 ...    0      ← WT maps to MOA-2
dnaB_1       0      0    3024  ...    0      ← dnaB_1 maps to MOA-2
dnaB_2       0    3024     0   ...    0      ← dnaB_2 maps to MOA-1
...
```

**Reading the numbers:**
- Each ROW is a true class (85 total)
- Each COLUMN is a discovered MOA cluster (19 total)
- Cell value = how many samples from that class went to that cluster
- Large numbers on the diagonal = good clustering

---

### Script 3: `confusion_85_vs_19.py`

**Purpose:** Single crop-level confusion matrix (simpler version)

**What it does:**
Same as above but only for crop level, with multiple visualization styles

**Outputs:**
- `confusion_85_vs_19_raw.png` - Raw counts
- `confusion_85_vs_19_normalized.png` - Percentage per row
- `confusion_85_vs_19_annotated.png` - Annotated with numbers

---

## Results Summary

### MOA Clusters Found (k=19)

| Cluster | Size | Genes | Notes |
|---------|------|-------|-------|
| C0 | 3 | rpsL, secA | Translation/Secretion |
| C1 | 6 | dnaB, dnaE, mrcA, msbA, parE | Mixed DNA/Cell wall |
| C2 | 8 | WT, dnaB, dnaE, gyrA, lpxA, mrcB, parE | Contains WT (weak phenotypes) |
| C3 | 1 | mrdA_3 | Single outlier |
| C4 | 6 | rplA, rplC, rpoB | Transcription machinery |
| C5 | 9 | dnaE, folA, gyrA, gyrB, mrcB, msbA, parC, parE | DNA/Replication |
| C6 | 6 | lptA, lptC | LPS transport (100% pure genes) |
| C7 | 7 | folP, mrcB, msbA, murA, rpsA | Mixed metabolism |
| C8 | 4 | lpxC, murC | Cell wall/LPS |
| C9 | 5 | ftsI, murC | Septation |
| C10 | 3 | ftsZ_1, ftsZ_2, ftsZ_3 | Cell division (100% pure) |
| C11 | 2 | mrdA_1, mrdA_2 | Cell elongation |
| C12 | 6 | folA, folP, lpxA, parC, rpsA | Mixed |
| C13 | 5 | rplA, rplC, secY | Translation/Secretion |
| C14 | 2 | murA_2, murA_3 | Cell wall |
| C15 | 5 | gyrB, secY, folA, mrcA | Mixed |
| C16 | 3 | rpoA_1, rpoA_2, rpoA_3 | RNA polymerase (100% pure) |
| C17 | 2 | rpsL_3, lpxA_1 | Mixed |
| C18 | 2 | secA_2, secA_3 | Secretion |

### High-Purity MOA Candidates

| Cluster | Genes | Biological Interpretation |
|---------|-------|---------------------------|
| C6 | lptA, lptC | LPS inner membrane transport |
| C10 | ftsZ | Cell division (Z-ring) |
| C16 | rpoA | RNA polymerase α subunit |

### Guide Consistency Summary

| Status | Genes |
|--------|-------|
| **Consistent** (all 3 same cluster) | ftsI, ftsZ, lptA, lptC, lpxC, mrdA, msbA, murC, rpoA, rpoB |
| **Inconsistent** (guides split) | dnaB, dnaE, folA, folP, gyrA, gyrB, lpxA, mrcA, mrcB, murA, parC, parE, rplA, rplC, rpsA, rpsL, secA, secY |

---

## Files Generated in `moa_k19/`

### Visualizations
| File | Description |
|------|-------------|
| `tsne_01_crop_moa.html` | t-SNE: 10K crops colored by MOA cluster |
| `tsne_01_crop_pathway.html` | t-SNE: 10K crops colored by original pathway |
| `tsne_01_crop_umap.html` | UMAP: 10K crops |
| `tsne_02_image_moa.html` | t-SNE: 2K images colored by MOA cluster |
| `tsne_02_image_pathway.html` | t-SNE: 2K images colored by original pathway |
| `tsne_02_image_umap.html` | UMAP: 2K images |
| `tsne_03_well_moa.html` | t-SNE: 96 wells colored by MOA cluster |
| `tsne_03_well_pathway.html` | t-SNE: 96 wells colored by original pathway |
| `tsne_03_well_umap.html` | UMAP: 96 wells |
| `dendrogram.png` | Hierarchical clustering tree |

### Confusion Matrices
| File | Description |
|------|-------------|
| `confusion_crop_85_19.png` | 85 classes vs 19 clusters (crop level) |
| `confusion_image_85_19.png` | 85 classes vs 19 clusters (image level) |
| `confusion_well_85_19.png` | 85 classes vs 19 clusters (well level) |
| `confusion_85_vs_19_raw.png` | Raw counts version |
| `confusion_85_vs_19_normalized.png` | Percentage version |
| `confusion_85_vs_19_annotated.png` | Annotated version |

### Data Tables
| File | Description |
|------|-------------|
| `moa_cluster_analysis.csv` | Cluster details (size, purity, genes) |
| `class_to_cluster_85_19.csv` | Which class → which cluster |
| `guide_consistency.csv` | Guide consistency analysis |
| `accuracy_summary_3levels.csv` | Accuracy/purity at 3 levels |
| `cluster_pathway_crosstab.csv` | MOA clusters vs original pathways |

### Interactive
| File | Description |
|------|-------------|
| `cluster_pathway_heatmap.html` | Heatmap of MOA vs pathways |

---

## Key Takeaways

1. **85 classes don't map cleanly to phenotypes**
   - Average cluster purity is only 33.7%
   - Many classes with same gene but different guides cluster differently

2. **Some genes have robust phenotypes**
   - ftsZ, rpoA, lptA/lptC: all 3 guides cluster together
   - These are high-confidence MOA targets

3. **Some genes show guide-specific effects**
   - folA: guides 1, 2, 3 go to clusters 5, 12, 15
   - May indicate differential knockdown or off-target effects

4. **WT clusters with weak phenotypes**
   - WT is in C2 with DNA metabolism genes
   - Suggests these perturbations have subtle effects

---

## How to Interpret the Results

### For MOA Discovery:
Look for clusters where genes from DIFFERENT original pathways cluster together:
- **C8** (lpxC + murC): LPS + Cell wall → potential shared envelope stress
- **C5** (folA, gyrA, parC): Metabolism + DNA → potential replication stress

### For Quality Control:
Look for genes where guides are INCONSISTENT:
- These may have off-target effects
- Or differential knockdown efficiency
- Worth investigating individually

---

*Last updated: March 2026*
*Analysis performed on EffNet embeddings from CRISPRi reference plate imaging*
