# CRISPRi Reference Plate Imaging

Deep learning classification pipeline for bacterial phenotype classification from CRISPRi microscopy images.

## Project Overview

This project develops image-based machine learning models to classify bacterial phenotypes from CRISPRi (CRISPR interference) microscopy images. Each image represents a different genetic perturbation (gene knockdown), and the model learns to identify which gene is being perturbed based on cell morphology.

## Dataset

| Split | Plates | Images | Purpose |
|-------|--------|--------|---------|
| Training | P1, P2, P3, P4 | 8,064 | Model training |
| Validation | P5 | 2,016 | Hyperparameter tuning |
| Test | P6 | 2,016 | Final evaluation |

### Image Format
- Resolution: 2720×2720 pixels (16-bit TIFF)
- Channel: Brightfield microscopy (channel 4)
- Wells: 96-well plates (96 classes, all with equal samples)

### Labels
Labels are stored in `sam_effnet/plate_well_id_path.json` which maps each well (e.g., "A01", "B12") to the gene perturbation class.

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
```

### 3. Download Data
Place plate folders (P1-P6) in the project root directory.

### 4. Download DINOv3 Weights (for dinov3-finetune)
Download `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` to `dino_weights/`.

## Model Architectures

### 1. sam_effnet (EfficientNet-B0 + SAM)
CNN baseline with Sharpness-Aware Minimization (SAM) optimizer.

```bash
cd sam_effnet
python train.py \
    --epochs 2000 \
    --batch_size 256 \
    --lr 1e-4 \
    --rho 0.1 \
    --warmup_epochs 6 \
    --patience 0 \
    --grid_size 12 \
    --crop_size 224
```

### 2. guide_effnet (Guide Generalization)
Tests generalization to unseen guides - train on 2 guides, test on 1.

```bash
cd guide_effnet
python train.py \
    --epochs 2000 \
    --batch_size 256 \
    --lr 1e-4 \
    --rho 0.1 \
    --warmup_epochs 6 \
    --patience 0 \
    --guide_experiment 1
```

Experiment options:
- `--guide_experiment 1`: Train on guides 1&2, test on guide 3
- `--guide_experiment 2`: Train on guides 1&3, test on guide 2
- `--guide_experiment 3`: Train on guides 2&3, test on guide 1

### 3. dinov3-finetune (DINOv3 ViT-L + Logistic Regression)
Frozen DINOv3 backbone with linear classifier (no LoRA).

```bash
cd dinov3-finetune
python train_plate.py \
    --exp_name dinov3_lr \
    --epochs 2000 \
    --batch_size 64 \
    --lr 1e-4 \
    --rho 0.1 \
    --patience 0 \
    --data_root /path/to/AI4AMR \
    --label_json_path sam_effnet/plate_well_id_path.json
```

### 4. dinov3-finetune (DINOv3 + LoRA Fine-tuning)
DINOv3 with LoRA adapters for fine-tuning.

```bash
cd dinov3-finetune
python train_plate.py \
    --exp_name dinov3_lora \
    --epochs 2000 \
    --batch_size 16 \
    --lr 1e-4 \
    --rho 0.1 \
    --use_lora \
    --patience 0 \
    --data_root /path/to/AI4AMR \
    --label_json_path sam_effnet/plate_well_id_path.json
```

## Optional Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--center_loss` | Use center loss for better feature discrimination | Disabled |
| `--center_loss_weight` | Weight for center loss | 0.001 |
| `--adaptive` | Use Adaptive SAM (ASAM) instead of SAM | Disabled (SAM rho=0.1) |

## Training Details

All models use identical augmentation pipeline:
- **Symmetry**: HorizontalFlip, VerticalFlip, RandomRotate90, Rotate(360°)
- **Affine**: translate ±15%, scale 85-115%, rotate ±20°
- **Lighting**: CLAHE, RandomBrightnessContrast, RandomGamma, Equalize
- **Shadows**: RandomShadow (p=0.2)
- **Geometric**: ElasticTransform, Perspective, GridDistortion, OpticalDistortion (p=0.4)
- **Noise**: GaussNoise, GaussianBlur, MotionBlur (p=0.4)
- **Artifacts**: SaltAndPepper, ISONoise, CoarseDropout, ImageCompression

### sam_effnet & guide_effnet Configuration
- **Optimizer**: SAM (Sharpness-Aware Minimization) wrapping AdamW
- **Loss**: Weighted Focal Loss (α=0.25, γ=2.0)
- **Domain weights**: Per-plate (n_d^-1/2)
- **Crops**: 144 positions per image (12×12 grid), 1 random crop per epoch
- **GradScaler**: For mixed precision training stability

### dinov3-finetune Configuration
- **Backbone**: DINOv3 ViT-L (frozen)
- **Classifier**: Linear(1024 → 96) = Multinomial Logistic Regression
- **Optimizer**: SAM + AdamW
- **Loss**: Focal Loss with class weights

## Model Comparison

| Model | Backbone | Optimizer | Trainable Params | Feature Dim |
|-------|----------|-----------|------------------|--------------|
| sam_effnet | EfficientNet-B0 | SAM | ~5.3M | 1280 |
| guide_effnet | EfficientNet-B0 | SAM | ~5.3M | 1280 |
| plate_fold | EfficientNet-B0 | AdamW | ~5.3M | 1280 |
| dinov3-finetune LR | DINOv3 ViT-L | SAM | ~100K | 1024 |
| dinov3-finetune LoRA | DINOv3 ViT-L + LoRA | SAM | ~3M | 1024 |

## Plate Cross-Validation (plate_fold)

Leave-one-plate-out cross-validation for robust evaluation.

```bash
cd plate_fold

# Run single fold (test on P6)
python train.py \
    --test_plate P6 \
    --epochs 200 \
    --batch_size 256

# Run all 6 folds (each plate as test once)
python train.py \
    --run_all_folds \
    --epochs 200 \
    --batch_size 256

# Generate aggregate confusion matrices
python generate_combined_confusion.py \
    --folds P1,P2,P3,P4,P5,P6 \
    --family
```

### Confusion Matrix Outputs
- `binary_cm_*` - Binary: 1 if accuracy > 50%, 0 otherwise (Blues colormap)
- `raw_cm_*` - Raw prediction counts  
- `percent_cm_*` - Normalized percentages

Each plot shows: `{n_above_50}/{n} > 50%, {n_above_random}/{n} > Random({baseline}%)`

## Project Structure

```
.
├── sam_effnet/                    # EfficientNet-B0 + SAM (CNN)
│   ├── train.py                   # Training script
│   ├── plate_well_id_path.json    # 96-class labels
│   └── trial_1_144_crops/         # Best results (~22.9% well acc)
│
├── guide_effnet/                  # Guide generalization experiments
│   ├── train.py                   # Training with --guide_experiment
│   ├── plate_well_id_path.json
│   └── classes.txt
│
├── plate_fold/                    # Leave-one-plate-out cross-validation
│   ├── train.py                   # Training with fold logic
│   ├── predict_fold.py            # Prediction script
│   ├── generate_combined_confusion.py  # Aggregate confusion matrices
│   ├── classes.txt                # 96 class labels
│   └── fold_P{1-6}/               # Results per fold
│
├── dinov3-finetune/               # DINOv3 ViT-L fine-tuning
│   ├── train_plate.py             # Main training script
│   ├── dino_finetune/
│   │   └── plate_dataset.py       # Dataset with augmentations
│   └── output/                    # Saved models and metrics
│
├── guide_effnet/                  # Guide generalization experiments
│   ├── train.py                   # Training with --guide_experiment
│   ├── plate_well_id_path.json
│   └── classes.txt
│
├── dinov3-finetune/               # DINOv3 ViT-L fine-tuning
│   ├── train_plate.py             # Main training script
│   ├── dino_finetune/
│   │   ├── plate_dataset.py       # Dataset with augmentations
│   │   └── model/
│   │       └── plate_classifier.py
│   └── output/                    # Saved models and metrics
│
├── 1_Dino_embeddings_logistic_regression/  # LR baseline on embeddings
├── plate maps/                    # Label mappings
├── dino_weights/                  # DINOv3 pretrained weights
└── requirements.txt
```

## Hardware Requirements

| Model | GPU Memory | Recommended |
|-------|------------|-------------|
| sam_effnet | ~4GB | 8GB GPU |
| guide_effnet | ~4GB | 8GB GPU |
| dinov3-finetune LR | ~8GB | 16GB GPU |
| dinov3-finetune LoRA | ~16GB | 24GB GPU |

## Output Files

Each training run produces:
- `best_model.pth` / `*_best.pt` - Best model checkpoint
- `training_metrics_*.csv` - Epoch-level loss/accuracy
- `training_metrics_*.json` - Full metrics including per-class
- `classes.txt` - Class label mappings

## License

MIT License
