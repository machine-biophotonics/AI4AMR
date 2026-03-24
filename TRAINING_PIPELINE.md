# CRISPRi Reference Plate Imaging - Training Pipeline

## Overview

This pipeline trains a deep learning model to classify bacterial phenotypes from microscopy images. Each image represents a different genetic perturbation (gene knockdown), and the model learns to identify which gene is being perturbed based on cell morphology.

---

## Data Split

| Split | Plates | Samples | Purpose |
|-------|--------|---------|---------|
| **Train** | P1, P2, P3, P4 | ~8,064 images | Learning model weights |
| **Validation** | P5 | ~2,016 images | Hyperparameter tuning, early stopping |
| **Test** | P6 | ~2,016 images | Final evaluation (unseen during training) |

**Total: 6 plates → 97 classes**

---

## Classes

- **97 unique classes** (gene knockdowns)
- **Gene families**: Genes are grouped by name (e.g., `dnaB_1`, `dnaB_2`, `dnaB_3` are variants of the same gene)
- **WT (Wild Type)**: Control samples with no knockdown
- **NC (Negative Control)**: Additional control samples

### Class Imbalance
- WT has ~756 samples (most common)
- Most gene classes have ~84 samples
- Some variants have only ~21 samples

**Solution**: Class weights computed to balance loss contribution.

---

## Image Processing Pipeline

### Original Image Size
- **Typical size**: ~2000x2000 pixels
- **Format**: 16-bit TIFF

### Patch Extraction

#### Step 1: Define Center Region
```
Original Image (2000x2000)
┌────────────────────────────────────┐
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │  ← Edge region (200px margin)
│ ▓┌────────────────────────────┐▓▓ │
│ ▓│                            │▓▓ │  ← Center region
│ ▓│      VALID PATCH AREA      │▓▓ │    where patches are
│ ▓│                            │▓▓ │    extracted from
│ ▓└────────────────────────────┘▓▓ │
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
└────────────────────────────────────┘
        edge_margin = 200 pixels
```

**Why avoid edges?**
- Edges may contain artifacts from plate preparation
- Cell density may be different at edges
- Focus on representative center region

#### Step 2: Extract Patches

| Parameter | Value |
|-----------|-------|
| Patch size | 224 x 224 pixels |
| Edge margin | 200 pixels |
| Model input | ResNet/EfficientNet (224x224x3) |

**Patch extraction varies by split:**

---

## Train / Validation / Test Differences

### Training (P1-P4)

```
For each image:
1. Randomly select 5 patch positions within center region
2. Extract 5 patches (224x224 each)
3. Apply augmentation to each patch
4. Forward through model → get 5 predictions
5. Average predictions → final prediction
6. Compute loss and backpropagate
```

**Data Augmentation (Training only):**
- [x] Random horizontal flip (p=0.5)
- [x] Random vertical flip (p=0.5)
- [x] Color jitter (brightness, contrast, saturation, hue)
- [x] Random erasing (p=0.2) - randomly masks 2-10% of pixels
- [x] Random center crop (from center region only)

**Key point**: Patches are RANDOM each epoch → more training variety

---

### Validation (P5)

```
For each image:
1. Select 5 patch positions (fixed center positions, no randomness)
2. Extract 5 patches
3. NO augmentation
4. Forward through model → get 5 predictions
5. Average predictions → final prediction
6. Compute loss for monitoring
```

**Key point**: Deterministic → consistent evaluation

---

### Test (P6)

```
For each image:
1. Extract 50 patches from center region (more coverage)
2. NO augmentation
3. Forward through model → get 50 predictions
4. Average predictions → final prediction
```

**Key point**: More patches = better representation of cell population in well

---

## Why Multi-Patch?

### Biological Reason
- Each image contains ~hundreds of bacterial cells
- Different cells may show different phenotypes
- Sampling multiple patches captures this heterogeneity

### Statistical Reason
- Averaging reduces prediction variance
- More robust to outliers (bad focus, debris, etc.)

---

## Model Architecture

```
EfficientNet-B0 (pretrained on ImageNet)
├── Conv layers (feature extraction)
├── Global Average Pooling
├── Dropout (p=0.3)
└── Linear (1280 → 97 classes)
```

**Why EfficientNet-B0?**
- Good accuracy/compute tradeoff
- Pretrained weights provide useful features
- 224x224 input matches our patch size

---

## Training Optimizations

| Optimization | Value | Why |
|--------------|-------|-----|
| Optimizer | AdamW | Better weight decay than Adam |
| Learning rate | 1e-3 (max) | OneCycleLR handles schedule |
| Scheduler | OneCycleLR | Warmup + cosine annealing |
| Batch size | 16 | 5 patches × 16 = 80 patches/batch |
| Weight decay | 1e-4 | Regularization |
| Gradient clip | max_norm=1.0 | Prevents explosions |
| Dropout | p=0.3 | Prevents overfitting |

---

## OneCycleLR Schedule

```
LR
 ↑
 │    ╱╲
 │   /  ╲
 │  /    ╲
 │ /      ╲
 │/        ╲________
 └──────────────────→ Epochs
   ↑           ↑
 Warmup    Annealing
 (10%)     to low LR
```

- **Max LR**: 1e-3
- **Warmup**: 10% of training (gradual increase)
- **Annealing**: Cosine curve to near-zero

---

## Early Stopping

| Parameter | Value |
|-----------|-------|
| Patience | 10 epochs |
| Min delta | 0.01 |
| Metric | Validation loss |
| Mode | Minimize |

Training stops if val loss doesn't improve by at least 0.01 for 10 consecutive epochs.

---

## Multi-Patch Averaging

```
Image
  │
  ├── Patch 1 ──→ Model ──→ Logits 1
  ├── Patch 2 ──→ Model ──→ Logits 2
  ├── Patch 3 ──→ Model ──→ Logits 3
  ├── Patch 4 ──→ Model ──→ Logits 4
  └── Patch 5 ──→ Model ──→ Logits 5
                        │
                        ↓ Mean
                  Final Logits ──→ Softmax ──→ Prediction
```

This happens in the model during forward pass - patches are processed in a single batch and averaged.

---

## Output Files

| File | Description |
|------|-------------|
| `best_model.pth` | Best model weights (highest val accuracy) |
| `classes.txt` | Class index to gene name mapping |
| `training_plots.png` | Loss and accuracy curves |
| `roc_curves.png` | ROC curves for best 8 classes |
| `precision_recall_curves.png` | PR curves for best 8 classes |
| `training_results_*.json` | Full results (config, metrics, per-class) |
| `class_metrics_*.csv` | Per-class AUC and AP values |
| `training_log_*.txt` | Human-readable training log |
| `tsne_plot.png` | t-SNE visualization of features |

---

## t-SNE Visualization

- Extracts 1280-dim features from penultimate layer
- Reduces to 2D using t-SNE
- Each gene family has unique color
- Shows how well model separates classes in feature space

---

## Reproducibility

All randomness is controlled via:
- `SEED = 42` (master seed)
- `torch.manual_seed(SEED)`
- `torch.cuda.manual_seed_all(SEED)`
- `np.random.seed(SEED)`
- `random.seed(SEED)`
- `cudnn.deterministic = True`
- `cudnn.benchmark = False`
- `worker_init_fn` for DataLoader workers

---

## Quick Reference Commands

```bash
# Train model (2 epochs for testing)
python train.py

# Full training (50 epochs)
# Edit train.py: num_epochs = 50
python train.py

# Generate t-SNE plot
python generate_tsne.py

# Visualize center crop extraction
python visualize_center_crop.py
```

---

## Expected Performance (97 classes)

| Metric | Random | Good | Target |
|--------|--------|------|--------|
| Accuracy | ~1% | >15% | >30% |
| ROC AUC | 0.5 | >0.7 | >0.85 |

With proper training (50 epochs), expect:
- **Val Accuracy**: 20-35%
- **ROC AUC**: 0.80-0.90

---

## Troubleshooting

**Q: Training is slow**
- Check GPU usage with `nvidia-smi`
- Reduce `num_workers` if CPU-bound
- Consider reducing patch count for faster iteration

**Q: Model overfitting**
- Increase dropout (p=0.5)
- Add more augmentation
- Reduce model capacity
- Use early stopping

**Q: Low accuracy**
- Train for more epochs
- Check data labeling is correct
- Verify class weights are applied
- Try larger model (EfficientNet-B1/B2)
