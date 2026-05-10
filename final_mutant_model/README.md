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
  --test_plate Plate_1
```

This learns **domain-invariant features** that work on both drug AND mutant images.

**Parameters:**
- `--data_mode both` - Load both drug and mutant images (185 classes)
- `--use_dann` - Enable Domain-Adversarial training
- `--dann_lambda 1.0` - Weight for domain loss (fixed, no scheduling)

**Expected output:**
```
DANN: Drug classes = 89, Mutant classes = 96
```

---

### 3. SimCLR Self-Supervised Training

```bash
cd self_supervised_trial
python3 train_simclr_raw.py --plate P1 --epochs 200
```

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

### DANN (Domain-Adversarial Neural Network)

Built on top of SC-MIL with additional domain adaptation:

| Component | Description |
|-----------|-------------|
| **Base** | SC-MIL (same backbone, pooling, loss) |
| **Domain Classifier** | Additional binary classifier (drug vs mutant) |
| **GRL** | Gradient Reversal Layer - reverses domain gradients |
| **Goal** | Learn domain-invariant features |

**How GRL works:**
```
Forward:  features → domain_classifier → domain_logits (normal)
Backward: domain_loss gradients → multiply by -1 → features
```

**Effect:** Feature extractor is punished for creating domain-specific features, forcing it to learn features that work on **both** domains.

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
├── mil_model.py                      # Model definitions
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