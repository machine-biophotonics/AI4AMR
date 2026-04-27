# MIL Training for CRISPRi Reference Plate Imaging

Multiple Instance Learning (MIL) model with attention pooling for classifying CRISPRi guide experiments from plate-based images.

## Architecture

| Component | Description |
|-----------|-------------|
| **Backbone** | EfficientNet-B0 (ImageNet pretrained) |
| **Pooling** | Gated Multi-head Attention (4 heads) |
| **Crops** | 3×3 neighborhood (9 crops per image) |
| **Feature Dim** | 1280 |

## Research Backing

This implementation follows established research papers:

| Component | Paper | Citation |
|-----------|-------|----------|
| Gated Attention Pooling | Attention-based Deep Multiple Instance Learning | Ilse et al., ICML 2018 |
| SC-MIL | SC-MIL: Supervised Contrastive MIL for Imbalanced Classification | Juyal et al., WACV 2024 (arXiv:2303.13405) |
| Focal Loss | Focal Loss for Dense Object Detection | Lin et al., ICCV 2017 |
| Supervised Contrastive | Supervised Contrastive Learning | Khosla et al., arXiv:2004.11362 |
| SimCLR | Simple Framework for Contrastive Learning | Chen et al., ICML 2020 |

### Key Parameters from Papers

| Parameter | This Code | SC-MIL Paper |
|-----------|----------|------------|
| Temperature τ | 1.0 | τ = 1 (fixed) |
| Curriculum β | 1 → 0.3 | β_t transitions 1 → target |
| SC-MIL weight | 0.3 | 0.3 |
| Focal γ | 2.0 | - |
| Focal α | 0.25 | - |

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
| `--lr_decay` | OFF | Enable LR decay (cosine annealing) |
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
| `--weight_decay` | 0.0 | Weight decay (SC-MIL paper: NOT specified) |
| `--label_smoothing` | 0.1 | Label smoothing |
| `--attention_mode` | softmax | Attention: softmax or sigmoid (ASMIL) |

### Stage 1: Contrastive Pre-training
| Argument | Default | Description |
|----------|---------|-------------|
| `--contrastive_epochs` | 0 | Epochs for Stage 1 (0 to skip) |
| `--contrastive_batch_size` | 128 | Batch size for contrastive |
| `--contrastive_temp` | 0.1 | Temperature for SimCLR loss |

### Stage 2: SC-MIL
| Argument | Default | Description |
|----------|---------|-------------|
| `--sc_mil` | enabled | Use SC-MIL (default: enabled) |
| `--no_sc_mil` | - | Disable SC-MIL, use standard training |
| `--sc_mil_epochs` | 200 | Epochs for SC-MIL |
| `--sc_mil_weight` | 0.3 | Weight for contrastive loss (0.1-1.0) |
| `--sc_mil_temp` | 1.0 | Temperature for SupCon loss (SC-MIL paper: τ=1) |

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

| Function | Purpose | Paper |
|----------|---------|-------|
| `focal_loss` | Handle class imbalance | Lin et al., ICCV 2017 |
| `weighted_focal_loss` | Focal + class weights + smoothing | Lin et al. + label smoothing |
| `attention_entropy_loss` | Focused attention (standard mode only) | Regularization |
| `SupConLoss` | Supervised contrastive loss | Khosla et al., 2020 |

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

If you use this code for SC-MIL, please cite:

```
SC-MIL: Supervised Contrastive Multiple Instance Learning for Imbalanced Classification in Pathology
Juyal et al., WACV 2024
arXiv:2303.13405
```

Other key citations:

```
Attention-based Deep Multiple Instance Learning
Ilse et al., ICML 2018

Focal Loss for Dense Object Detection  
Lin et al., ICCV 2017

Supervised Contrastive Learning
Khosla et al., arXiv:2004.11362
```

## License

MIT License