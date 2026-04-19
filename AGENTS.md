# CRISPRi Reference Plate Imaging - Setup Guide

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/machine-biophotonics/AI4AMR.git
cd AI4AMR
```

### 2. Create Environment
```bash
conda create -n crispri python=3.10
conda activate crispri

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Additional dependencies
pip install pandas scikit-learn tqdm albumentations seaborn --break-system-packages
```

### 3. Download Data
Place plate folders (P1-P6) in the project root directory.

## Model Overview

| Model | Backbone | Pooling | Crops | Description |
|-------|----------|---------|-------|-------------|
| `final_crispr_model` | EfficientNet-B0 | Attention (3x3) | 9 | 3x3 neighborhood MIL |
| `final_mutant_model` | EfficientNet-B0 | Gated Multi-head Attention | 25 | 5x5 neighborhood MIL with BagMix/PseMix |
| `final_max_model` | EfficientNet-B0 | Configurable (max/mean/gmp/certainty/attention) | 3x3 or 5x5 | Versatile MIL with multiple pooling strategies |

---

## Final Mutant Model (Recommended with Data Augmentation)

MIL model with gated multi-head attention pooling and 5x5 neighborhood (25 crops). Supports **BagMix** and **PseMix** data augmentation.

### Architecture
- **Backbone**: EfficientNet-B0 (ImageNet pretrained)
- **Pooling**: Gated Multi-head Attention (4 heads)
- **Crop Neighborhood**: 5×5 (25 crops)
- **Feature Dimension**: 1280-dim

### BagMix Data Augmentation

Simple bag-level augmentation modes:

```bash
# Mixup crops (recommended)
python3 train_mil.py --test_plate P6 --bag_mix mixup_crop --bag_mix_alpha 1.0 --bag_mix_prob 0.5

# Subset sampling
python3 train_mil.py --test_plate P6 --bag_mix subset --bag_mix_subset_size 12

# Dropout
python3 train_mil.py --test_plate P6 --bag_mix dropout --bag_mix_dropout 0.2

# Bootstrap
python3 train_mil.py --test_plate P6 --bag_mix bootstrap
```

### PseMix Data Augmentation (IEEE TMI 2024)

Pseudo-bag mixup from paper: "Pseudo-Bag Mixup Augmentation for MIL-Based WSI Classification"

```bash
# PseMix with k-means clustering (recommended)
python3 train_mil.py --test_plate P6 --use_psemix \
  --psemix_mode psebmix_kmeans \
  --psemix_n_pseb 8 \
  --psemix_n_pheno 4 \
  --psemix_alpha 1.0 \
  --psemix_prob 0.5

# PseMix random
python3 train_mil.py --test_plate P6 --use_psemix \
  --psemix_mode psebmix_random \
  --psemix_n_pseb 8 \
  --psemix_alpha 1.0
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 200 | Training epochs |
| `--batch_size` | 16 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--num_heads` | 4 | Attention heads |
| `--test_plate` | P6 | Test plate |
| `--run_all_folds` | - | Run all 6 folds |

### BagMix Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--bag_mix` | none | Mode: subset, mixup, dropout, shuffle, cutmix, mixup_crop, bootstrap, cluster |
| `--bag_mix_alpha` | 1.0 | Beta distribution parameter |
| `--bag_mix_prob` | 0.5 | Probability of applying augmentation |
| `--bag_mix_subset_size` | - | Size for subset mode |
| `--bag_mix_dropout` | 0.0 | Dropout ratio |

### PseMix Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_psemix` | - | Enable PseMix |
| `--psemix_mode` | psebmix | Mode: psebmix, psebmix_kmeans, psebmix_random |
| `--psemix_n_pseb` | 8 | Number of pseudo-bags per bag |
| `--psemix_n_pheno` | 8 | Number of phenotype clusters |
| `--psemix_alpha` | 1.0 | Beta distribution parameter |
| `--psemix_prob` | 0.5 | Probability of mixing |

### Training

```bash
cd final_mutant_model

# Train single fold with PseMix (recommended)
python3 train_mil.py --test_plate P6 --use_psemix --psemix_mode psebmix_kmeans --psemix_n_pseb 8 --psemix_n_pheno 4

# Train single fold with simple BagMix
python3 train_mil.py --test_plate P6 --bag_mix mixup_crop --bag_mix_alpha 1.0

# Train single fold (no augmentation)
python3 train_mil.py --test_plate P6

# Run all 6 folds
python3 train_mil.py --run_all_folds
```

### Prediction

```bash
cd final_mutant_model
python3 predict_all_crops.py --fold P1 --checkpoint best_model.pth
```

---

## Final Max Model (Recommended for Pooling Experiments)

The most flexible model with multiple pooling strategies and configurable crop neighborhoods.

### Architecture
- **Backbone**: EfficientNet-B0 (ImageNet pretrained)
- **Pooling**: Configurable - attention, max, mean, gmp, certainty
- **Crop Neighborhood**: Configurable 3×3 (9 crops) or 5×5 (25 crops)
- **Feature Dimension**: 1280-dim (attention: 256-dim)

### Pooling Strategies

| Strategy | Description | Paper |
|----------|-------------|-------|
| `attention` | Gated multi-head attention (4 heads × 64-dim) | Ilse et al. 2018 |
| `max` | Max-pooling (FocusMIL) | arXiv:1908.01456 |
| `mean` | Simple average | - |
| `gmp` | Generalized Mean Pooling (learnable p) | arXiv:2008.10548 |
| `certainty` | Model certainty-weighted | arXiv:2008.10548 |

### Training Command

```bash
cd final_max_model

# 5x5 neighborhood with attention pooling (default)
python3 train_mil.py --test_plate P6 --pooling attention --crop_neighborhood 5

# 5x5 with mean pooling
python3 train_mil.py --test_plate P6 --pooling mean --crop_neighborhood 5

# 3x3 neighborhood with attention
python3 train_mil.py --test_plate P1 --pooling attention --crop_neighborhood 3

# 3x3 with max pooling
python3 train_mil.py --test_plate P6 --pooling max --crop_neighborhood 3

# Run all 6 folds
python3 train_mil.py --run_all_folds --pooling attention --crop_neighborhood 5
```

### Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--pooling` | attention, max, mean, gmp, certainty | attention | Pooling strategy |
| `--crop_neighborhood` | 3, 5 | 5 | 3×3 (9 crops) or 5×5 (25 crops) |
| `--test_plate` | P1-P6 | P6 | Test plate |
| `--epochs` | int | 200 | Training epochs |
| `--batch_size` | int | 16 | Batch size |
| `--lr` | float | 1e-4 | Learning rate |
| `--num_heads` | int | 4 | Attention heads |
| `--run_all_folds` | flag | - | Run all 6 folds |

### Prediction

```bash
cd final_max_model
python3 predict_all_crops.py --fold P1 --checkpoint best_model.pth
```

---

## Final CRISPR Model

3×3 neighborhood (9 crops) MIL model with attention pooling.

### Training

```bash
cd final_crispr_model

# Train single fold
python3 train_mil.py --test_plate P6

# Run all 6 folds
python3 train_mil.py --run_all_folds
```

### Confusion Matrix Generation

```bash
cd final_crispr_model

# Generate for single fold
python3 generate_combined_confusion.py --single_fold P1

# Generate for all folds
python3 generate_combined_confusion.py --folds P1,P2,P3,P4,P5,P6
```

---

## Checkpoint Types

Each training run saves 4 best models:
- `best_model.pth` - Best by validation AUC
- `best_model_auc.pth` - Best by validation AUC
- `best_model_acc.pth` - Best by validation accuracy
- `best_model_loss.pth` - Best by lowest validation loss

---

## Project Structure

```
.
├── final_max_model/                 # Flexible MIL with multiple pooling
│   ├── train_mil.py               # Training script
│   ├── mil_model.py                # Model definition
│   ├── predict_all_crops.py        # Prediction script
│   └── fold_P{1-6}/                # Results per fold
│
├── final_mutant_model/             # 5x5 neighborhood MIL with BagMix/PseMix
│   ├── train_mil.py               # Training script
│   ├── mil_model.py                # Model definition
│   ├── bag_mix.py                  # BagMix & PseMix implementation
│   ├── predict_all_crops.py        # Prediction script
│   └── fold_P{1-6}/
│
├── final_crispr_model/             # 3x3 neighborhood MIL
│   ├── train_mil.py
│   ├── mil_model.py
│   ├── generate_combined_confusion.py
│   └── fold_P{1-6}/
│
├── sam_effnet/                    # EfficientNet-B0 + SAM
├── guide_effnet/                  # Guide generalization
├── dinov3-finetune/               # DINOv3 ViT-L fine-tuning
└── plate_fold/                    # Cross-validation experiments
```
