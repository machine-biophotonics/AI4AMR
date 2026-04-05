# CRISPRi Reference Plate Imaging - Setup Guide

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/machine-biophotonics/AI4AMR.git
cd AI4AMR
```

### 2. Create Environment
```bash
# Create conda environment
conda create -n crispri python=3.10
conda activate crispri

# Install PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt
```

### 3. Download Data
The dataset consists of 6 plates (P1-P6) with 2720×2720 pixel TIFF images:
- P1-P4: Training (8,064 images)
- P5: Validation (2,016 images)
- P6: Test (2,016 images)

Place plates in the project root directory.

### 4. Download DINOv3 Weights
Download the pretrained DINOv3 ViT-L weights and place in `dino_weights/`:
- `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`

## Running Experiments

### EfficientNet-B0 + SAM (CNN)
```bash
cd sam_effnet
python train.py \
    --epochs 200 \
    --batch_size 16 \
    --lr 1e-4 \
    --rho 0.1 \
    --warmup_epochs 6 \
    --patience 10 \
    --grid_size 12 \
    --crop_size 224
```

### DINOv3 ViT-L + LoRA + SAM (Vision Transformer)
```bash
cd dinov3-finetune
python train_plate.py \
    --exp_name dinov3_lora_sam \
    --data_root "/path/to/AI4AMR" \
    --label_json_path "sam_effnet/plate_well_id_path.json" \
    --epochs 200 \
    --batch_size 16 \
    --lr 1e-4 \
    --rho 0.1 \
    --warmup_epochs 6 \
    --patience 10 \
    --weight_decay 0.1 \
    --use_lora
```

### Resume Training
```bash
# sam_effnet
cd sam_effnet
python train.py --resume best_model.pth --epochs 200

# DINOv3 + SAM
cd dinov3-finetune
python train_plate.py --resume output/dinov3_lora_sam_best.pt --epochs 200
```

## Model Outputs

Each training run produces:
- `training_metrics_*.csv` - Epoch-level loss/accuracy
- `training_results_*.json` - Full results with per-class metrics
- `best_model.pth` or `*_best.pt` - Best model checkpoint
- `classes.txt` - Class label mappings

## Project Structure

```
.
├── sam_effnet/                        # EfficientNet-B0 + SAM (CNN)
│   ├── train.py                       # Training script with SAM
│   ├── plate_well_id_path.json        # 96-class labels
│   ├── trial_1_144_crops/             # Best results (~22.9% well acc)
│   └── visualize_augmentations.py
│
├── dinov3-finetune/                   # DINOv3 ViT-L + LoRA + SAM
│   ├── train_plate.py                 # Main training script with SAM
│   ├── dino_finetune/
│   │   └── plate_dataset.py           # Same augmentations as sam_effnet
│   └── output/                        # Saved models and metrics
│
├── 2_effnet_model/                    # EfficientNet-B0 baseline
├── final_effnet_model/                # Final EfficientNet script
├── 1_Dino_embeddings_logistic_regression/  # LR baseline
├── plate maps/                        # Label mappings
├── dino_weights/                      # DINOv3 pretrained weights
└── requirements.txt
```

## Hardware Requirements

| Model | GPU Memory | Recommended |
|-------|------------|-------------|
| EfficientNet + SAM | ~4GB | 8GB GPU |
| DINOv3 ViT-L + LoRA + SAM | ~16GB | 24GB GPU |

## Dataset Distribution

The dataset has 96 balanced classes with equal samples per class:
- All 96 classes: equal number of images
- Training (P1-P4): 8,064 images
- Validation (P5): 2,016 images
- Test (P6): 2,016 images

Both models use:
- **Optimizer**: SAM (Sharpness-Aware Minimization) wrapping AdamW
- **Focal loss** (α=0.25, γ=2.0, label_smoothing=0.1)
- **Domain weights** (per-plate, using n_d^-1/2) to handle plate-level variation
- **144 crops per image** (12×12 grid, 1 random crop per epoch)
- **Same augmentations** with reduced probabilities

## Comparison: sam_effnet vs DINOv3 + SAM

| Parameter | sam_effnet (CNN) | DINOv3 + SAM (ViT) |
|-----------|------------------|-------------------|
| Backbone | EfficientNet-B0 | DINOv3 ViT-L + LoRA |
| Optimizer | SAM (rho=0.1) | SAM (rho=0.1) |
| Epochs | 200 | 200 |
| Batch size | 16 | 16 |
| Learning rate | 1e-4 | 1e-4 |
| Val accuracy (crop) | ~15.2% | ~17% |
| Well-level accuracy | ~22.9% | TBD |

**Purpose**: Compare CNN (EfficientNet) vs Vision Transformer (DINOv3) with identical training configuration to understand which architecture works better for this bacterial phenotype classification task.