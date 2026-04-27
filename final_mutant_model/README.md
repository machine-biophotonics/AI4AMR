# MIL Training for CRISPRi Reference Plate Imaging

Multiple Instance Learning (MIL) model with attention pooling for classifying CRISPRi guide experiments from plate-based images.

## Architecture

| Component | Description |
|-----------|-------------|
| **Backbone** | EfficientNet-B0 (ImageNet pretrained) |
| **Pooling** | Gated Multi-head Attention (4 heads) |
| **Crops** | 3×3 neighborhood (9 crops per image) |
| **Feature Dim** | 1280 |

## Training Pipeline

### Stage 1: Patch-Level SimCLR Pre-training (Optional)
- Uses `neighborhood=1` (single crop per image)
- InfoNCE contrastive loss between augmented views
- Learns generic patch-level features
- **Controlled by:** `--contrastive_epochs` (default: 50, set to 0 to skip)

### Stage 2: SC-MIL Joint Training
- Uses `neighborhood=3` (9 crops = bag)
- Supervised Contrastive Loss + Focal Cross-Entropy
- Joint optimization of representation and classifier
- **Controlled by:** `--sc_mil` (enabled by default)

## Arguments

### Training
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 200 | Total training epochs for standard mode |
| `--batch_size` | 16 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--num_heads` | 4 | Attention heads |
| `--seed` | 42 | Random seed |

### Data
| Argument | Default | Description |
|----------|---------|-------------|
| `--test_plate` | P6 | Test plate (P1-P6) |
| `--data_root` | None | Path to data directory |
| `--run_all_folds` | - | Run all 6 folds |
| `--neighborhood` | 3 | Crop neighborhood (3=3×3, 5=5×5, etc) |
| `--grid_size` | 12 | Grid size for crop positions |

### Regularization
| Argument | Default | Description |
|----------|---------|-------------|
| `--dropout` | 0.5 | Dropout rate |
| `--weight_decay` | 0.05 | Weight decay |
| `--label_smoothing` | 0.1 | Label smoothing |

### Stage 1: Contrastive Pre-training
| Argument | Default | Description |
|----------|---------|-------------|
| `--contrastive_epochs` | 50 | Epochs for Stage 1 (0 to skip) |
| `--contrastive_batch_size` | 128 | Batch size for contrastive |
| `--contrastive_temp` | 0.1 | Temperature for SimCLR loss |

### Stage 2: SC-MIL
| Argument | Default | Description |
|----------|---------|-------------|
| `--sc_mil` | enabled | Use SC-MIL (default: enabled) |
| `--no_sc_mil` | - | Disable SC-MIL, use standard training |
| `--sc_mil_epochs` | 200 | Epochs for SC-MIL |
| `--sc_mil_weight` | 0.3 | Weight for contrastive loss (0.1-1.0) |
| `--sc_mil_temp` | 0.07 | Temperature for SupCon loss |

## Usage Examples

### Standard SC-MIL Training
```bash
python train_mil.py --test_plate P6
```

### Skip Stage 1 (faster, no contrastive pre-training)
```bash
python train_mil.py --test_plate P6 --contrastive_epochs 0
```

### Skip Stage 2 (use standard attention only)
```bash
python train_mil.py --test_plate P6 --no_sc_mil
```

### Standard training only (no contrastive at all)
```bash
python train_mil.py --test_plate P6 --contrastive_epochs 0 --no_sc_mil
```

### Run all 6 folds
```bash
python train_mil.py --run_all_folds
```

### Custom hyperparameters
```bash
python train_mil.py --test_plate P6 \
    --sc_mil_epochs 300 \
    --sc_mil_weight 0.5 \
    --batch_size 32 \
    --dropout 0.3
```

## Loss Functions

| Function | Purpose | Equation |
|----------|---------|----------|
| `focal_loss` | Handle class imbalance | α(1-p_t)^γ × CE |
| `weighted_focal_loss` | Focal + weights + smoothing | weighted focal |
| `attention_entropy_loss` | Focused attention | -Σ p·log(p) |
| `SupConLoss` | Bag-level supervised contrastive | Supervised SimCLR |

## Output Files

Each fold creates:
```
fold_P6/
���── best_model.pth           # Best by AUC
├── best_model_auc.pth      # Best by AUC
├── best_model_acc.pth     # Best by accuracy
├── best_model_loss.pth     # Best by loss
├── training_contrastive_TIMESTAMP.csv  # Stage 1 loss
├── training_sc_mil_TIMESTAMP.csv   # Stage 2 loss
├── training_results.json  # Final results
└── ...
```

## Requirements

```bash
pip install torch torchvision
pip install numpy scikit-learn pandas tqdm albumentations
pip install -r requirements.txt
```

## Citation

If you use this code, please cite:

```
@software{crispri-mil,
  title={MIL Training for CRISPRi Reference Plate Imaging},
  author={Machine Biophotonics Lab},
  url={https://github.com/machine-biophotonics/AI4AMR}
}
```