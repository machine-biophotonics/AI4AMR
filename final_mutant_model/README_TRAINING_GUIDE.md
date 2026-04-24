# Final Mutant Model - Training Guide

## Quick Start

### Train with PseMix (Recommended):
```bash
cd final_mutant_model/src
python3 train_mil.py --test_plate P6 --use_psemix --psemix_mode psebmix_kmeans --psemix_n_pseb 30 --psemix_n_pheno 8 --psemix_alpha 1.0 --psemix_prob 0.5
```

### Recommended: With Warmup + Weight Decay:
```bash
python3 train_mil.py --test_plate P6 \
  --use_psemix \
  --warmup_epochs 5 \
  --weight_decay 1e-4 \
  --lr_scheduler cosine
```

---

## Complete Training Arguments

### TRAINING PARAMETERS (Basic)
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 200 | Total training epochs |
| `--batch_size` | 16 | Batch size |
| `--lr` | 1e-4 | Initial learning rate |
| `--weight_decay` | 1e-4 | Weight decay for optimizer |
| `--num_heads` | 4 | Number of attention heads |
| `--seed` | 42 | Random seed for reproducibility |

### LEARNING RATE SCHEDULER
| Argument | Default | Description |
|----------|---------|-------------|
| `--warmup_epochs` | 5 | Number of warmup epochs (linear ramp-up) |
| `--lr_scheduler` | cosine | LR scheduler: cosine/plateau/none |
| `--lr_patience` | 10 | Patience for ReduceLROnPlateau |
| `--lr_min` | 1e-6 | Minimum learning rate |

### DATA PARAMETERS
| Argument | Default | Description |
|----------|---------|-------------|
| `--grid_size` | 12 | Grid size for sampling positions |
| `--crop_size` | 224 | Crop size for each patch |
| `--neighborhood` | 5 | Neighborhood: 1, 3, 5, 7, 9, 11 |
| `--random_neighborhood` | False | Randomize neighborhood each epoch |
| `--neighborhood_range` | [3, 11] | Range for random neighborhood |
| `--max_positions` | None | Max positions to sample |

### CROSS-VALIDATION
| Argument | Default | Description |
|----------|---------|-------------|
| `--test_plate` | P6 | Test plate (P1-P6) |
| `--run_all_folds` | False | Run all 6 folds |
| `--checkpoint_type` | auc | auc/acc/loss |
| `--resume` | None | Resume from checkpoint |

### BAGMIX (Simple Augmentation)
| Argument | Default | Description |
|----------|---------|-------------|
| `--bag_mix` | none | none/subset/mixup/dropout/shuffle/cutmix/mixup_crop/bootstrap/cluster |
| `--bag_mix_ratio` | 0.5 | Mix ratio |
| `--bag_mix_subset_size` | None | Subset size for subset mode |
| `--bag_mix_dropout` | 0.0 | Dropout ratio |
| `--bag_mix_alpha` | 1.0 | Beta distribution parameter |
| `--bag_mix_prob` | 0.5 | Probability |

### PSEMIX (IEEE TMI 2024) - RECOMMENDED
| Argument | Default | Description |
|----------|---------|-------------|
| `--use_psemix` | False | Enable PseMix |
| `--psemix_mode` | psebmix | psebmix/psebmix_kmeans/psebmix_random/proto/kmeans/random |
| `--psemix_n_pseb` | 30 | Number of pseudo-bags per bag (paper default: 30) |
| `--psemix_n_pheno` | 8 | Number of phenotype clusters (paper default: 8) |
| `--psemix_alpha` | 1.0 | Beta distribution parameter |
| `--psemix_prob` | 0.5 | Mixing probability |

### MIL-DROPOUT (ICML 2025)
| Argument | Default | Description |
|----------|---------|-------------|
| `--use_mildropout` | False | Enable MIL-Dropout |
| `--mildropout_topk` | 3 | Top-k instances to drop |

### MAMMOTH (Optional)
| Argument | Default | Description |
|----------|---------|-------------|
| `--use_mammoth` | False | Enable MAMMOTH |
| `--mammoth_num_experts` | 30 | Number of experts |
| `--mammoth_num_slots` | 10 | Number of slots |
| `--mammoth_num_heads` | 16 | Number of heads |
| `--mammoth_dropout` | 0.1 | Dropout rate |

---

## What's Activated by Default

Running: `python3 train_mil.py --test_plate P6`

- ❌ No augmentation (basic training only)
- ❌ No PseMix
- ❌ No MIL-Dropout
- ❌ No MAMMOTH
- ✅ 5×5 neighborhood (25 crops per sample)
- ✅ 30 pseudo-bags (if PseMix enabled)
- ✅ 8 phenotype clusters (if PseMix enabled)

---

## Recommended Training Commands

### 1. Baseline:
```bash
python3 train_mil.py --test_plate P6
```

### 2. PseMix Only (Recommended):
```bash
python3 train_mil.py --test_plate P6 \
  --use_psemix \
  --psemix_mode psebmix_kmeans \
  --psemix_n_pseb 30 \
  --psemix_n_pheno 8 \
  --psemix_alpha 1.0 \
  --psemix_prob 0.5
```

### 3. PseMix + Warmup + Weight Decay (Best for Reproducibility):
```bash
python3 train_mil.py --test_plate P6 \
  --use_psemix \
  --psemix_mode psebmix_kmeans \
  --psemix_n_pseb 30 \
  --psemix_n_pheno 8 \
  --warmup_epochs 5 \
  --weight_decay 1e-4 \
  --lr_scheduler cosine \
  --lr_min 1e-6
```

### 3. PseMix + MIL-Dropout:
```bash
python3 train_mil.py --test_plate P6 \
  --use_psemix \
  --use_mildropout \
  --mildropout_topk 3
```

### 4. BagMix (Simple Alternative):
```bash
python3 train_mil.py --test_plate P6 \
  --bag_mix mixup_crop \
  --bag_mix_alpha 1.0 \
  --bag_mix_prob 0.5
```

### 5. Run All 6 Folds:
```bash
python3 train_mil.py --run_all_folds
```

### 6. Full Experiment with All Features:
```bash
python3 train_mil.py --test_plate P6 \
  --epochs 200 \
  --batch_size 16 \
  --lr 1e-4 \
  --neighborhood 5 \
  --use_psemix \
  --psemix_mode psebmix_kmeans \
  --psemix_n_pseb 30 \
  --psemix_n_pheno 8 \
  --psemix_alpha 1.0 \
  --psemix_prob 0.5 \
  --use_mildropout \
  --mildropout_topk 3
```

---

## Output Structure

After training, results are saved in:
```
final_mutant_model/src/fold_P6/
├── best_model.pth           # Best model (AUC)
├── best_model_acc.pth      # Best accuracy
├── best_model_loss.pth     # Lowest loss
├── best_model_auc.pth      # Best AUC (duplicate)
├── checkpoint_epoch_*.pth # Periodic checkpoints
├── training_results.json  # Final metrics
└── training_metrics_*.csv # Per-epoch metrics
```

---

## Prediction Commands

### Predict Single Fold:
```bash
python3 predict_all_crops.py --fold P6
```

### Predict All Folds:
```bash
python3 predict_all_crops.py --run_all_folds
```

### Use Specific Checkpoint:
```bash
python3 predict_all_crops.py --fold P6 --checkpoint best_model.pth
```

---

## Implementations Verified

### 1. ABMIL Gated Attention (Ilse et al. 2018)
- ✅ Core algorithm verified: `tanh(V(x)) * sigmoid(U(x))`
- Matches official AMLab implementation

### 2. PseMix (IEEE TMI 2024)
- ✅ Pseudo-bag generation: K-means clustering
- ✅ Default n_pseb = 30 (matches paper)
- ✅ Default n_pheno = 8 (matches paper)
- Reference: https://github.com/liupei101/PseMix

### 3. MIL-Dropout (ICML 2025)
- ✅ Top-k instance dropping
- ✅ Gaussian kernel for similarity
- Reference: https://github.com/ChongQingNoSubway/MILDropout

---

## Visualization Commands

### Debug Neighborhoods (Synthetic Image):
```bash
cd ../visualization
python3 debug_neighborhood.py --output_dir ./debug_output
```

### Visualize with Real Image:
```bash
cd ../visualization
python3 visualize_with_real_images.py
```

### Clear Visualizations:
```bash
cd ../visualization
python3 visualize_clear.py
```

Output saved to: `visualization/psemix_debug/`

---

## Key Concepts

### Phenotype vs Pseudo-Bag:
1. **Phenotype** = Grouping similar instances using K-means on features
2. **Pseudo-Bag** = Random sub-division WITHIN each phenotype group
3. **PseMix** = Mix between pseudo-bags from different bags

### MIL-Dropout:
1. Compute importance for each instance
2. Sort by importance (descending)
3. DROP top-k instances
4. Rescale remaining for normalization

### Neighborhood Statistics (grid_size=12, image=2720×2720):
| N | Centers | Crops/Position | Total Crops |
|---|---------|---------------|-------------|
| 1×1 | 144 | 1 | 144 |
| 3×3 | 100 | 9 | 900 |
| 5×5 | 64 | 25 | 1600 |
| 7×7 | 36 | 49 | 1764 |
| 9×9 | 16 | 81 | 1296 |
| 11×11 | 4 | 121 | 484 |

---

## Model Architecture

- **Backbone**: EfficientNet-B0 (ImageNet pretrained)
- **Feature Dimension**: 1280
- **Pooling**: Gated Multi-Head Attention (4 heads × 1280)
- **Classifier**: 5120 → 96 classes
- **5×5 Neighborhood**: 25 crops per sample

---

Version: 2025-12-19 CRISPRi Reference Plate Imaging
Author: AI Assistant