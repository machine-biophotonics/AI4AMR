# CRISPRi Reference Plate Imaging

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

---

## Model Overview

| Model | Backbone | Pooling | Crops | Description |
|-------|----------|---------|-------|-------------|
| `final_max_model` | EfficientNet-B0 | Configurable | 3x3 or 5x5 | **Recommended** - Versatile MIL with multiple pooling |
| `final_mutant_model` | EfficientNet-B0 | Gated Multi-head Attention | 5x5 (25) | 5x5 neighborhood MIL with BagMix/PseMix |
| `final_crispr_model` | EfficientNet-B0 | Attention | 3x3 (9) | 3x3 neighborhood MIL |

---

## Final Mutant Model (Recommended)

5×5 neighborhood (25 crops) MIL with gated multi-head attention pooling. Supports **BagMix** and **PseMix** data augmentation.

### BagMix Augmentation

Simple bag-level augmentation:

```bash
cd final_mutant_model

# Mixup crops (recommended)
python3 train_mil.py --test_plate P6 --bag_mix mixup_crop --bag_mix_alpha 1.0 --bag_mix_prob 0.5

# Subset sampling
python3 train_mil.py --test_plate P6 --bag_mix subset --bag_mix_subset_size 12

# Dropout
python3 train_mil.py --test_plate P6 --bag_mix dropout --bag_mix_dropout 0.2
```

### PseMix Augmentation (IEEE TMI 2024)

Pseudo-bag mixup with clustering:

```bash
cd final_mutant_model

# PseMix with k-means (recommended)
python3 train_mil.py --test_plate P6 \
  --use_psemix \
  --psemix_mode psebmix_kmeans \
  --psemix_n_pseb 8 \
  --psemix_n_pheno 4 \
  --psemix_alpha 1.0 \
  --psemix_prob 0.5

# PseMix random
python3 train_mil.py --test_plate P6 --use_psemix --psemix_mode psebmix_random --psemix_n_pseb 8
```

### Training

```bash
cd final_mutant_model

# Single fold
python3 train_mil.py --test_plate P6

# All folds
python3 train_mil.py --run_all_folds
```

### Prediction

```bash
cd final_mutant_model
python3 predict_all_crops.py --fold P1 --checkpoint best_model.pth
```

---

## Final Max Model (Pooling Experiments)

The most flexible MIL model with configurable pooling strategies.

### Training

```bash
cd final_max_model

# 5x5 with attention (default)
python3 train_mil.py --test_plate P6 --pooling attention --crop_neighborhood 5

# 5x5 with mean pooling
python3 train_mil.py --test_plate P6 --pooling mean --crop_neighborhood 5

# 3x3 with max pooling
python3 train_mil.py --test_plate P6 --pooling max --crop_neighborhood 3

# GMP pooling
python3 train_mil.py --test_plate P6 --pooling gmp

# Run all folds
python3 train_mil.py --run_all_folds --pooling attention
```

### Pooling Strategies

| Strategy | Description |
|----------|-------------|
| `attention` | Gated multi-head attention (Ilse et al. 2018) |
| `max` | Max-pooling (FocusMIL) |
| `mean` | Simple average |
| `gmp` | Generalized Mean Pooling |
| `certainty` | Model certainty-weighted |

---

## Final CRISPR Model

3×3 neighborhood (9 crops) MIL model.

### Training

```bash
cd final_crispr_model

# Single fold
python3 train_mil.py --test_plate P6

# All folds
python3 train_mil.py --run_all_folds
```

### Confusion Matrix

```bash
cd final_crispr_model

# Single fold
python3 generate_combined_confusion.py --single_fold P1

# All folds
python3 generate_combined_confusion.py --folds P1,P2,P3,P4,P5,P6
```

---

## Common Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--epochs` | Training epochs | 200 |
| `--batch_size` | Batch size | 16 |
| `--lr` | Learning rate | 1e-4 |
| `--num_heads` | Attention heads | 4 |
| `--test_plate` | Test plate | P6 |
| `--run_all_folds` | Run all 6 folds | - |

---

## Project Structure

```
.
├── final_max_model/           # Flexible MIL with multiple pooling
├── final_mutant_model/       # 5x5 MIL with BagMix/PseMix
├── final_crispr_model/     # 3x3 MIL
├── sam_effnet/         # EfficientNet-B0 + SAM
├── guide_effnet/       # Guide generalization
├── dinov3-finetune/    # DINOv3 ViT-L
└── plate_fold/        # Cross-validation
```

---

## Model Comparison

| Model | Backbone | Pooling | Crops | Best For |
|-------|----------|--------|------|-------|
| final_mutant_model | EfficientNet-B0 | Gated Attention | 25 | Classification with augmentation |
| final_max_model | EfficientNet-B0 | Configurable | 9/25 | Pooling experiments |
| final_crispr_model | EfficientNet-B0 | Attention | 9 | Fast baseline |