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
Labels are stored in `plate maps/plate_well_id_path.json` which maps each well (e.g., "A01", "B12") to the gene perturbation class.

## Model Architectures

### 1. EfficientNet-B0 with SAM (`sam_effnet/`)
CNN baseline with Sharpness-Aware Minimization (SAM) optimizer. Best well-level accuracy ~22.9%.

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

### 2. DINOv3 ViT-L + LoRA with SAM (`dinov3-finetune/`)
Vision Transformer with LoRA fine-tuning and SAM optimizer, pretrained on satellite imagery (SAT-493M).

```bash
cd dinov3-finetune
python train_plate.py \
    --exp_name dinov3_lora_sam \
    --data_root "/path/to/data" \
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

### 3. Logistic Regression (`1_Dino_embeddings_logistic_regression/`)
Logistic regression on DINOv3 embeddings for baseline comparison.

## Project Structure

```
.
├── sam_effnet/                        # EfficientNet-B0 + SAM training
│   ├── train.py                       # Training script with SAM
│   ├── plate_well_id_path.json        # 96-class labels
│   ├── trial_1_144_crops/             # Best results (~22.9% well acc)
│   └── visualize_augmentations.py     # Augmentation visualization
│
├── dinov3-finetune/                   # DINOv3 ViT-L + LoRA + SAM
│   ├── train_plate.py                  # Main training script
│   ├── dino_finetune/
│   │   └── plate_dataset.py            # Dataset with sam_effnet augmentations
│   └── output/                         # Saved models and metrics
│
├── 2_effnet_model/                    # EfficientNet-B0 baseline
│
├── final_effnet_model/                # Final EfficientNet training script
│
├── 1_Dino_embeddings_logistic_regression/
│   └── train_logistic_regression.py   # LR on DINO embeddings
│
├── plate maps/                        # Label mappings
│   └── plate_well_id_path.json
│
└── dino_weights/                     # DINOv3 pretrained weights
```

## Training Details

Both sam_effnet and DINOv3 + SAM use identical configuration:
- **Optimizer**: SAM (wrapping AdamW) with rho=0.1
- **Learning rate**: 1e-4
- **Cropping**: 144 positions per image (12×12 grid), 1 random crop per epoch
- **Augmentation**: Albumentations pipeline with reduced probabilities
- **Loss**: Weighted Focal Loss (α=0.25, γ=2.0, label_smoothing=0.1)
- **Domain weights**: Per-plate (n_d^-1/2) for plate-level variation
- **Validation**: Center crop only, no augmentation

### Comparison: sam_effnet vs DINOv3 + SAM

| Parameter | sam_effnet (CNN) | DINOv3 + SAM (ViT) |
|-----------|------------------|-------------------|
| Backbone | EfficientNet-B0 | DINOv3 ViT-L |
| Epochs | 200 | 200 |
| Batch size | 16 | 16 |
| Learning rate | 1e-4 | 1e-4 |
| SAM rho | 0.1 | 0.1 |
| Val accuracy (crop) | ~15.2% | ~17% |
| Well-level accuracy | ~22.9% | TBD |

## Common Options

| Flag | Description | Default |
|------|-------------|---------|
| `--epochs` | Number of epochs | 200 |
| `--batch_size` | Batch size | 16 |
| `--lr` | Learning rate | 1e-4 |
| `--rho` | SAM perturbation radius | 0.1 |
| `--warmup_epochs` | Warmup epochs | 6 |
| `--patience` | Early stopping patience | 10 |
| `--min_delta` | Min improvement for early stopping | 0.001 |
| `--resume` | Resume from checkpoint | None |
| `--seed` | Random seed | 42 |

## Requirements

```bash
pip install torch torchvision albumentations scikit-learn pandas matplotlib tqdm
```

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (16GB recommended for DINOv3, 8GB for EfficientNet)

## Output Files

Results are saved in each model's directory:
- `training_metrics_*.csv` - Epoch-level metrics
- `training_results_*.json` - Full results with class-level metrics
- `best_model.pth` / `*_best.pt` - Best model checkpoint

## License

MIT License
