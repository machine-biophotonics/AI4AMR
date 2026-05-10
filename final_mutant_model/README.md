# final_mutant_model - MIL Training for CRISPRi Reference Plate Imaging

Multiple Instance Learning (MIL) model for classifying CRISPRi guide experiments and antibiotic treatments from plate-based microscopy images.

---

## Project Overview

This project implements several training approaches for bacterial colony classification:

| Approach | Description | Use Case |
|----------|-------------|----------|
| **SC-MIL** | Supervised Contrastive MIL (classification + contrastive loss) | Main model for drug/mutant classification |
| **DANN** | Domain-Adversarial Neural Network (SC-MIL + domain adaptation) | Cross-domain learning (drug ↔ mutant) |
| **SimCLR** | Self-supervised contrastive learning | Pre-training, feature visualization |

---

## Quick Start

### 1. SC-MIL Training (Standard)

**Train on Mutant data (gene classification):**
```bash
cd final_mutant_model
python3 train_mil.py --test_plate Plate_1 --data_mode mutant --use_sc_mil
```

**Train on Drug data (antibiotic classification):**
```bash
python3 train_mil.py --test_plate Plate_1 --data_mode drug --use_sc_mil
```

**Run all 6 folds (cross-validation):**
```bash
python3 train_mil.py --run_all_folds --data_mode mutant --use_sc_mil
```

---

### 2. DANN Training (Domain-Adversarial)

Train on **both** drug and mutant data together with domain adaptation:

```bash
python3 train_mil.py \
  --data_mode both \
  --use_dann \
  --dann_lambda 1.0 \
  --domain_entropy_weight 2.0 \
  --domain_lr_factor 0.1 \
  --test_plate Plate_1
```

This learns **domain-invariant features** that work on both drug AND mutant images.

---

### 3. SimCLR Self-Supervised Training

```bash
cd self_supervised_trial
python3 train_simclr_raw.py --plate P1 --epochs 200
```

---

## DANN (Domain-Adversarial Neural Network) - Detailed

### Theory

DANN learns **domain-invariant features** that work across different domains (drug vs mutant) while maintaining discriminative power for the main task (antibiotic response classification).

```
┌─────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTOR                         │
│                  (EfficientNet-B0 + Attention)               │
│                          ↓                                   │
│         ┌───────────────────┴───────────────────┐           │
│         ↓                                       ↓           │
│    LABEL CLASSIFIER                          DOMAIN CLASSIFIER
│    (185 classes)                            (drug vs mutant)
│         ↓                                       ↓           │
│   CLASSIFICATION LOSS                      DOMAIN LOSS
│   (minimize)                                (MINIMIZE via CE)
└─────────────────────────────────────────────────────────────┘
                          ↑
                    Gradient Reversal Layer (GRL)
                          ↑
                    Feature extractor MAXIMIZES domain loss
```

### Key Components

| Component | Description | Implementation |
|-----------|-------------|----------------|
| **Feature Extractor** | EfficientNet-B0 + Gated Attention | Backbone + attention_pool |
| **Label Classifier** | Predicts 185 classes (drug/mutant + concentration) | Single FC layer |
| **Domain Classifier** | Distinguishes drug vs mutant | 2-layer MLP (1280 → 32 → 2) |
| **GRL** | Gradient Reversal Layer | `grad_output.neg() * alpha` |

### How GRL Works (Per Ganin & Lempitsky, 2015)

```python
# Forward pass: identity function (no change to values)
def forward(ctx, x, alpha):
    ctx.alpha = alpha
    return x

# Backward pass: reverse gradient sign and scale by alpha
def backward(ctx, grad_output):
    return grad_output.neg() * alpha, None
```

This causes:
- **Domain classifier** tries to **minimize** domain loss (learn to distinguish drug/mutant)
- **Feature extractor** via GRL tries to **maximize** domain loss (create features that fool domain classifier)
- Result: Features become **domain-invariant**

### DANN Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_dann` | OFF | Enable Domain-Adversarial training |
| `--dann_lambda` | 1.0 | Weight for domain loss in total loss |
| `--domain_entropy_weight` | 2.0 | Entropy regularization (penalizes confident domain predictions) |
| `--domain_lr_factor` | 0.1 | Domain classifier LR = base_LR × factor (10x lower) |

### GRL Scheduling

The GRL alpha ramps up from 0 → target over the first 20 epochs:

```python
def get_current_grl_alpha(epoch):
    if epoch < grl_warmup_epochs:
        return grl_alpha * (epoch / grl_warmup_epochs)
    return grl_alpha
```

This prevents the domain classifier from overpowering the feature extractor early in training.

### Expected Behavior

| Metric | Target | Meaning |
|--------|--------|---------|
| Domain Accuracy | ~50% | Random guessing - domain classifier confused |
| Domain Loss | ~0.693 | Entropy of random (log(2)) |
| Classification Accuracy | Maintained | Task performance not degraded |

**Training Output Example:**
```
Domain Loss: 0.6948, Domain Acc: 50.0%
SC-MIL Epoch 0: CE Loss=1.2391, SupCon Loss=0.2028, Train Acc=2.08%, Val Acc=7.09%, Val AUC=0.8491
```

### Why Domain Accuracy Should Be ~50%

In a well-trained DANN:
- Domain classifier cannot distinguish drug vs mutant better than random
- Features contain **zero information** about the domain
- All information is in the **main task features** (biological mechanism)

This enables t-SNE visualization colored by **Mechanism of Action (MoA)** to show clustering regardless of whether samples are drugs or mutants.

### Important Notes

1. **DANN requires `--data_mode both`** - Must train on both drug and mutant data
2. **Do NOT use with `--freeze`** - DANN gradients need to flow to backbone
3. **GRL alpha default = 2.0** - Higher than typical (1.0) but works with other regularizations

---

## Training Modes Explained

### SC-MIL (Supervised Contrastive MIL)

| Component | Description |
|-----------|-------------|
| **Backbone** | EfficientNet-B0 (ImageNet/MicroNet pretrained) |
| **Pooling** | Gated Multi-head Attention (4 heads) |
| **Crops** | 3×3 neighborhood = 9 crops per image |
| **Loss** | Weighted focal loss + supervised contrastive loss |
| **Classes** | Mutant: 96 / Drug: 89 |

**How crop extraction works:**
- 100 possible center positions on each plate image
- **1 random position** selected per image per epoch
- At that position: extract **3×3 = 9 crops** (neighborhood)
- Forward pass processes **9 crops** → attention pooling → output
- NOT 100 forward passes!

---

### SimCLR (Self-Supervised)

| Component | Description |
|-----------|-------------|
| **Backbone** | EfficientNet-B0 |
| **Crops** | 1 random position × 9 neighborhood = 9 crops |
| **Augmentation** | Random crop, flip, color jitter, blur |
| **Loss** | SimCLR contrastive loss (NT-Xent) |
| **No labels** | Learns general visual features only |

---

## Data Modes

| Mode | Data Source | Classes | JSON Mapping |
|------|-------------|---------|---------------|
| `mutant` | Mutants_Data/ | 96 genes (e.g., lptA_3, mrdA_2) | plate_well_id_path.json |
| `drug` | Drugs_Data/ | 89 (antibiotic + concentration) | plate_well_ic50_mapping.json |
| `both` | Both | 185 (89 drugs + 96 mutants) | Both JSONs |

---

## Key Arguments

### Training (train_mil.py)

| Argument | Default | Description |
|----------|---------|-------------|
| `--test_plate` | Plate_1 | Test plate (Plate_1 to Plate_6) |
| `--data_mode` | mutant | Data type: mutant, drug, or both |
| `--use_dann` | OFF | Enable Domain-Adversarial training |
| `--dann_lambda` | 1.0 | Domain loss weight (constant) |
| `--domain_entropy_weight` | 2.0 | Entropy regularization weight |
| `--domain_lr_factor` | 0.1 | Domain classifier LR multiplier |
| `--use_sc_mil` | ON | Use SC-MIL (joint contrastive + classification) |
| `--epochs` | 200 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--neighborhood` | 3 | Crop neighborhood (3=3×3=9 crops) |
| `--num_channels` | 1 | Input channels (1=grayscale, 3=RGB) |
| `--pretrained` | micronet | Pretrained weights (imagenet/micronet) |

---

## Crop Extraction Clarification

### Common Misconception: "100 positions = 100 forward passes"

**Correct understanding:**
- 100 positions = 100 **possible center locations** on the plate
- Per image per epoch: **1 random position** is selected
- At that position: extract **3×3 = 9 crops** (neighborhood)
- Forward pass: **9 crops** → attention pooling → 1 output

```
Self-Supervised: 9 crops/forward → 1 output
SC-MIL:           9 crops/forward → 1 output  
DANN:             9 crops/forward → 1 output
```

**All use the same 9 crops per forward pass!**

The speed difference comes from:
- Self-supervised: 1 loss (contrastive)
- SC-MIL: 2 losses (classification + contrastive)
- DANN: 3 losses (classification + contrastive + domain)

---

## Prediction

```bash
# Mutant predictions
python3 predict_all_crops.py --fold Plate_1 --data_mode mutant --checkpoint best_model_acc.pth

# Drug predictions
python3 predict_all_crops.py --fold Plate_1 --data_mode drug --checkpoint best_model_acc.pth
```

---

## Confusion Matrix Generation

```bash
# For mutants (gene hierarchy)
python3 generate_mutant_confusion.py --single_fold Plate_1

# For drugs (MoA grouped + 89-class)
python3 generate_drug_confusion.py --fold Plate_1
```

---

## Cross-Domain Evaluation

**Drug model on Mutant data:**
```bash
python3 predict_all_crops.py --fold Plate_1 --data_mode mutant --drug_on_mutant --checkpoint best_model_acc.pth
```

**Mutant model on Drug data:**
```bash
python3 predict_all_crops.py --fold Plate_1 --data_mode drug --mutant_on_drug --checkpoint best_model_acc.pth
```

---

## Results Summary

### Mutant (Gene Classification)

| Metric | Value |
|--------|-------|
| Image accuracy | ~21-26% |
| Gene-level accuracy | ~46-53% |
| Pathway-level accuracy | ~59-67% |

### Drug (Antibiotic Classification)

| Metric | Value |
|--------|-------|
| Image accuracy | ~43-50% |
| Test AUC | ~0.97 |
| Test AP | ~0.55 |

---

## Project Structure

```
final_mutant_model/
├── train_mil.py                       # Main training script (SC-MIL + DANN)
├── predict_all_crops.py              # Prediction script
├── mil_model.py                      # Model definitions (includes GradientReverse)
├── supcon_loss.py                    # Supervised Contrastive Loss
├── plate_well_id_path.json           # Mutant (gene) mapping
├── plate_well_ic50_mapping.json       # Drug (antibiotic) mapping
│
├── self_supervised_trial/            # Self-supervised experiments
│   ├── train_simclr_raw.py           # SimCLR training
│   ├── extract_embeddings.py         # Extract embeddings
│   ├── plot_tsne.py                  # t-SNE visualization
│   └── antibiotic_mutant_similarity.py # Cross-domain similarity
│
├── generate_mutant_confusion.py      # Mutant confusion matrices
├── generate_drug_confusion.py        # Drug confusion matrices
├── generate_cross_domain_confusion.py # Cross-domain analysis
│
├── mutant/                            # Mutant experiment results
│   └── fold_Plate_1/
│       ├── checkpoint_epoch.pth
│       ├── best_model_acc.pth
│       └── training_sc_mil_*.csv
│
└── drug/                             # Drug experiment results
    └── fold_Plate_1/
        ├── checkpoint_epoch.pth
        └── training_sc_mil_*.csv
```

---

## Requirements

```bash
conda create -n crispri python=3.10
conda activate crispri
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install numpy scikit-learn pandas tqdm albumentations seaborn
```

---

## Checkpoint Types

| File | Description |
|------|-------------|
| `checkpoint_epoch.pth` | Last epoch (199) - use for final predictions |
| `best_model_acc.pth` | Best by validation accuracy |
| `best_model_auc.pth` | Best by validation AUC |
| `best_model_loss.pth` | Best by lowest validation loss |

---

## References

- Ganin, Y., & Lempitsky, V. (2015). Unsupervised Domain Adaptation by Backpropagation. *ICML*.
- Implementation based on: https://github.com/fungtion/DANN