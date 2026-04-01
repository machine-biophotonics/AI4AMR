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
- Wells: 96-well plates (85 gene knockdowns + 1 WT control = 86 classes, but only 85 have unique perturbations)

### Labels
Labels are stored in `plate maps/plate_well_id_path.json` which maps each well (e.g., "A01", "B12") to the gene perturbation class.

## Model Architectures

### 1. EfficientNet-B0 (`2_effnet_model/`)
Standard CNN baseline with ImageNet pretrained weights. Includes evaluation and MOA clustering analysis.

```bash
cd 2_effnet_model
python train.py --epochs 50 --batch_size 16 --lr 1e-4
```

### 2. DINOv3 ViT-L with LoRA (`dinov3-finetune/`)
Vision Transformer with LoRA fine-tuning, pretrained on satellite imagery (SAT-493M).

```bash
cd dinov3-finetune
python train_plate.py \
    --epochs 50 \
    --data_root "/path/to/data" \
    --label_json_path "plate maps/plate_well_id_path.json" \
    --stain_augmentation \
    --use_lora \
    --batch_size 16 \
    --lr 1e-4
```

### 3. Logistic Regression (`1_Dino_embeddings_logistic_regression/`)
Logistic regression on DINOv3 embeddings for baseline comparison.

## Project Structure

```
.
├── 2_effnet_model/                  # EfficientNet-B0 training
│   ├── train.py                      # Training script
│   ├── evaluate_model.py             # Evaluation
│   ├── extract_embeddings.py         # Embedding extraction
│   ├── eval_results/                 # Evaluation outputs
│   └── moa_k19/                      # MOA clustering analysis (k=19)
│
├── final_effnet_model/               # Final EfficientNet training script
│   └── train.py                      # With focal loss + class/domain weighting
│
├── dinov3-finetune/                 # DINOv3 ViT-L + LoRA fine-tuning
│   ├── train_plate.py                # Main training script
│   ├── dino_finetune/                # Dataset and model code
│   └── output/                       # Saved models and metrics
│
├── 1_Dino_embeddings_logistic_regression/
│   ├── train_logistic_regression.py # LR training on embeddings
│   ├── generate_embeddings.py       # Embedding generation
│   ├── dino_moa/                    # MOA analysis on DINO embeddings
│   └── unsupervised_clustering/     # Clustering analysis
│
├── 1_Embeddings_144_crops_Dino/     # DINOv3 embeddings (144 crops × 2048-dim)
│                                       # Note: Large, not tracked in git
│
├── plate maps/                      # Label mappings
│   └── plate_well_id_path.json      # Well → gene class mapping
│
└── dino_weights/                    # DINOv3 pretrained weights
```

## Training Details

Both EfficientNet and DINOv3 use:
- **Random cropping**: 144 positions per image (12×12 grid), 1 random crop per epoch
- **Augmentation**: Albumentations-based pipeline (grayscale-friendly)
- **Loss**: Weighted focal loss (α=0.25, γ=2.0) with:
  - Class weights (inverse frequency)
  - Domain weights (per-plate, using n_d^-1/2)
- **Validation**: Center crop only, no augmentation

### Class Imbalance
- Class 0 (WT-like): 1,008 samples (12.5%)
- Other 84 classes: 84 samples each (1.04%)

Class weights are normalized to sum to num_classes.

## Common Options

| Flag | Description | Default |
|------|-------------|---------|
| `--epochs` | Number of epochs | 50 |
| `--batch_size` | Batch size | 16 |
| `--lr` | Learning rate | 1e-4 |
| `--warmup_epochs` | Warmup epochs | 6 |
| `--patience` | Early stopping patience | 10 |
| `--resume` | Resume from checkpoint | None |
| `--seed` | Random seed | 42 |

## Requirements

```bash
pip install torch torchvision albumentations scikit-learn pandas matplotlib tqdm
```

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (16GB recommended)

## Output Files

Results are saved in each model's directory:
- `training_metrics_*.csv` - Epoch-level metrics
- `training_results_*.json` - Full results with class-level AUC/AP
- `best_model.pth` / `*_best.pt` - Best model checkpoint

## License

MIT License
