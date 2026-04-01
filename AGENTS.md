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

### EfficientNet-B0 Training
```bash
cd 2_effnet_model
python train.py \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --warmup_epochs 6 \
    --patience 10 \
    --seed 42
```

### DINOv3 ViT-L + LoRA Fine-tuning
```bash
cd dinov3-finetune
python train_plate.py \
    --epochs 50 \
    --data_root "/path/to/AI4AMR" \
    --label_json_path "plate maps/plate_well_id_path.json" \
    --stain_augmentation \
    --use_lora \
    --batch_size 16 \
    --lr 1e-4 \
    --seed 42
```

### Resume Training
```bash
# EfficientNet
cd final_effnet_model
python train.py --resume best_model.pth --epochs 50

# DINOv3
cd dinov3-finetune
python train_plate.py --resume output/dinov3_lora_plate_last.pt --epochs 50
```

## Model Outputs

Each training run produces:
- `training_metrics_*.csv` - Epoch-level loss/accuracy
- `training_results_*.json` - Full results with per-class AUC/AP
- `best_model.pth` or `*_best.pt` - Best model checkpoint
- `classes.txt` - Class label mappings

## Project Structure

```
.
├── 2_effnet_model/                  # EfficientNet training
├── final_effnet_model/              # Final EfficientNet script
├── dinov3-finetune/                 # DINOv3 fine-tuning
├── 1_Dino_embeddings_logistic_regression/  # LR baseline
├── plate maps/                      # Label mappings
│   └── plate_well_id_path.json      # Required for training
├── dino_weights/                    # DINOv3 pretrained weights
└── requirements.txt
```

## Hardware Requirements

| Model | GPU Memory | Recommended |
|-------|------------|-------------|
| EfficientNet-B0 | ~4GB | 8GB GPU |
| DINOv3 ViT-L + LoRA | ~16GB | 24GB GPU |
| Logistic Regression | ~2GB | 8GB GPU |

## Class Imbalance

The dataset has significant class imbalance:
- Class 0 (WT): 1,008 samples (12.5%)
- Other 84 classes: 84 samples each (1.04%)

Both models use weighted focal loss with:
- Class weights (inverse frequency, normalized to sum to num_classes)
- Domain weights (per-plate, using n_d^-1/2)
