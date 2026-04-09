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
```

### 3. Download Data
Place plate folders (P1-P6) in the project root directory.

### 4. Download DINOv3 Weights (for dinov3-finetune)
Download the pretrained DINOv3 ViT-L weights and place in `dino_weights/`:
- `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`

## Running Experiments

### 1. sam_effnet (EfficientNet-B0 + SAM)
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
Train on 2 guides, test on 1 (tests generalization to unseen guides).

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

### 3. dinov3-finetune (Logistic Regression - no LoRA)
Frozen DINOv3 backbone + linear classifier.

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

### 4. dinov3-finetune (LoRA Fine-tuning)
DINOv3 + LoRA adapters for fine-tuning.

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

| Flag | Description |
|------|-------------|
| `--center_loss` | Enable center loss |
| `--center_loss_weight` | Center loss weight (default: 0.001) |
| `--adaptive` | Use Adaptive SAM (ASAM) instead of SAM |

## Model Outputs

Each training run produces:
- `best_model.pth` / `*_best.pt` - Best model checkpoint
- `training_metrics_*.csv` - Epoch-level metrics
- `*_metrics.json` - Full metrics

## Project Structure

```
.
├── sam_effnet/                    # EfficientNet-B0 + SAM (CNN)
│   ├── train.py                   # Training script
│   └── plate_well_id_path.json    # 96-class labels
│
├── guide_effnet/                  # Guide generalization (SAM)
│   ├── train.py                   # Training with --guide_experiment
│   └── plate_well_id_path.json
│
├── plate_fold/                    # Cross-validation (AdamW, no SAM)
│   ├── train.py                   # Leave-one-plate-out CV
│   └── fold_P{1-6}/               # Results per fold
│
├── sam_effnet_lr_fix/             # SAM + ReduceLROnPlateau scheduler
│
├── dinov3-finetune/               # DINOv3 ViT-L fine-tuning
│   ├── train_plate.py             # Main training script
│   ├── dino_finetune/
│   │   └── plate_dataset.py       # Dataset with augmentations
│   └── output/                    # Saved models and metrics
│
├── 1_Dino_embeddings_logistic_regression/  # LR baseline
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

## Training Features

All CNN models (sam_effnet, guide_effnet, plate_fold) use:
- **SAM optimizer** (Sharpness-Aware Minimization) - except plate_fold uses basic AdamW
- **Focal Loss** (α=0.25, γ=2.0)
- **Domain weights** (per-plate, using n_d^-1/2)
- **144 crops per image** (12×12 grid, 1 random crop per epoch)
- **Basic augmentations**: Flips, rotation, affine, elastic, blur, noise
- **GradScaler** for mixed precision (sam_effnet, guide_effnet)
- **Center loss** (optional, with separate optimizer)

## Plate Cross-Validation (plate_fold)

Train on 4 plates, validate on 1, test on 1 (leave-one-out CV).

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
```

Each fold saves to `fold_P{1-6}/` subfolder.

## Comparison

| Model | Backbone | Optimizer | Trainable Params | Feature Dim |
|-------|----------|-----------|------------------|--------------|
| sam_effnet | EfficientNet-B0 | SAM | ~5.3M | 1280 |
| guide_effnet | EfficientNet-B0 | SAM | ~5.3M | 1280 |
| plate_fold | EfficientNet-B0 | AdamW | ~5.3M | 1280 |
| dinov3-finetune LR | DINOv3 ViT-L | SAM | ~100K | 1024 |
| dinov3-finetune LoRA | DINOv3 ViT-L + LoRA | SAM | ~3M | 1024 |

## Plate Cross-Validation (plate_fold)

Train on 4 plates, validate on 1, test on 1 (leave-one-out CV).

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
```

Each fold saves to `fold_P{1-6}/` subfolder.

## Aggregate Confusion Matrices

Generate aggregate confusion matrices across all folds:

```bash
cd plate_fold
python generate_combined_confusion.py \
    --folds P1,P2,P3,P4,P5,P6 \
    --family
```

Output files in `aggregate/combined/`:
- `binary_cm_*` - Binary: 1 if accuracy > 50%, 0 otherwise (Blues colormap)
- `raw_cm_*` - Raw prediction counts
- `percent_cm_*` - Normalized percentages

Each plot shows: `{n_above_50}/{n} > 50%, {n_above_random}/{n} > Random({baseline}%)`

Options:
- `--family` - Group by gene families (dnaB+dnaE→dna, secA+secY→sec, etc.)
- `--guide` - Guide-level only
