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
| `final_mutant_model` | EfficientNet-B0 | Gated Multi-head Attention | 5x5 (25) | 5x5 neighborhood MIL |
| `final_crispr_model` | EfficientNet-B0 | Attention | 3x3 (9) | 3x3 neighborhood MIL |

---

## Model Comparison

| Model | Backbone | Optimizer | Trainable Params | Feature Dim |
|-------|----------|-----------|------------------|--------------|
| sam_effnet | EfficientNet-B0 | SAM | ~5.3M | 1280 |
| guide_effnet | EfficientNet-B0 | SAM | ~5.3M | 1280 |
| plate_fold | EfficientNet-B0 | AdamW | ~5.3M | 1280 |
| final_crispr_model | EfficientNet-B0 | Adam | ~5.3M | 1280 |
| final_max_model | EfficientNet-B0 | AdamW | ~5.3M | 1280 |
| final_mutant_model | EfficientNet-B0 | AdamW | ~5.3M | 256 |

---

## Final Max Model (Recommended)

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
python3 train_mil.py --test_plate P6 --pooling gmp --epochs 200

# Certainty pooling
python3 train_mil.py --test_plate P6 --pooling certainty --crop_neighborhood 3

# Run all folds
python3 train_mil.py --run_all_folds --pooling attention --crop_neighborhood 5
```

### Prediction

```bash
cd final_max_model
python3 predict_all_crops.py --fold P1 --checkpoint best_model.pth
```

---

## Final Mutant Model

5x5 neighborhood (25 crops) MIL with gated multi-head attention.

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

## Final CRISPR Model

3x3 neighborhood (9 crops) MIL with attention pooling.

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

## MIL Model Architecture

```
Input: (batch, n_crops, 3, 224, 224)
       │
       ▼
EfficientNet-B0 backbone
       │
       ▼ (batch*n_crops, 1280)
Bottleneck projection
       │
       ▼ (batch, n_crops, 256/1280)
Pooling (attention/max/mean/gmp/certainty)
       │
       ▼ (batch, 256/1280)
Classifier: BN → Linear → ReLU → Dropout → Linear(num_classes)
```

### Pooling Strategies

- **attention**: Gated multi-head attention (4 heads × 64-dim)
- **max**: Max-pooling (FocusMIL)
- **mean**: Simple average
- **gmp**: Generalized Mean Pooling (learnable power p)
- **certainty**: Model certainty-weighted

---

## Project Structure

```
.
├── final_max_model/                 # Flexible MIL (Recommended)
│   ├── train_mil.py
│   ├── mil_model.py
│   ├── predict_all_crops.py
│   └── fold_P{1-6}/
│
├── final_mutant_model/             # 5x5 neighborhood MIL
│   ├── train_mil.py
│   ├── mil_model.py
│   ├── predict_all_crops.py
│   └── fold_P{1-6}/
│
├── final_crispr_model/             # 3x3 neighborhood MIL
│   ├── train_mil.py
│   ├── mil_model.py
│   ├── generate_combined_confusion.py
│   └── fold_P{1-6}/
│
├── sam_effnet/                     # EfficientNet-B0 + SAM
├── guide_effnet/                   # Guide generalization
├── dinov3-finetune/                # DINOv3 ViT-L fine-tuning
└── plate_fold/                    # Cross-validation experiments
```

## Citation

If you use this code, please cite:
```bibtex
@article{crispri2025,
  title={CRISPRi Reference Plate Imaging},
  author={Machine Biophotonics Lab},
  year={2025}
}
```