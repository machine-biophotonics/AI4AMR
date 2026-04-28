# MIL Training for CRISPRi Reference Plate Imaging

Multiple Instance Learning (MIL) model with attention pooling for classifying CRISPRi guide experiments from plate-based images.

---

## Architecture

| Component | Description |
|-----------|-------------|
| **Backbone** | EfficientNet-B0 (ImageNet pretrained, 1280-dim features) |
| **Pooling** | Gated Multi-head Attention (4 heads, 256 hidden dim) |
| **Crops** | Configurable: 3×3 (9 crops), 5×5 (25 crops), etc. |
| **Feature Dim** | 1280 (attention pooled: 1280 × 4 heads = 5120 → 1280) |
| **Classifier** | Dropout + Linear(1280 → num_classes) |

### Model Components

1. **backbone_features**: EfficientNet-B0 feature extractor (before pooling)
2. **backbone_pool**: AdaptiveAvgPool2d + Flatten
3. **attention_pool**: Gated attention (V, U, w gates with tanh/sigmoid)
4. **head_proj**: Linear(5120 → 1280)
5. **classifier**: Dropout(p=dropout) → Linear(1280 → num_classes)

---

## Loss Functions

### Primary Losses

| Loss Function | Formula | Purpose | Paper |
|---------------|---------|---------|-------|
| **Weighted Focal Loss** | α × (1-pt)^γ × CE | Handle class imbalance & hard examples | Lin et al., ICCV 2017 |
| **Supervised Contrastive (SupCon)** | -τ × log(exp(z_i·z_j/τ) / Σ) | Learn discriminative representations | Khosla et al., 2020 |
| **Attention Entropy** | -Σ attn × log(attn) | Regularize attention distribution | Ilse et al., 2018 |

### Regularization Losses

| Loss Function | Formula | Purpose | Paper |
|---------------|---------|---------|-------|
| **SNR (Spectral Norm Reg)** | λ × σ(W) | Constrain weight matrix norm | Zhou et al., NeurIPS 2023 |

### Loss Combination (SC-MIL Mode)

```
total_loss = (1 - sc_mil_weight) × total_focal + sc_mil_weight × total_supcon + λ × SNR
```

Where:
- `total_focal = instance_weight × instance_focal + (1-instance_weight) × bag_focal`
- `total_supcon = instance_weight × instance_supcon + (1-instance_weight) × bag_supcon`

---

## Papers & References

### Core Architecture

| Component | Paper | Citation | Why Implemented |
|-----------|-------|----------|-----------------|
| Gated Attention Pooling | Attention-based Deep Multiple Instance Learning | Ilse et al., ICML 2018 | State-of-the-art MIL pooling |
| EfficientNet-B0 | EfficientNet: Rethinking Model Scaling for CNNs | Tan & Le, ICML 2019 | Strong pretrained backbone |

### Training Techniques

| Technique | Paper | Citation | Why Implemented |
|-----------|-------|----------|-----------------|
| **SC-MIL** | SC-MIL: Supervised Contrastive MIL | Juyal et al., WACV 2024 | Joint contrastive + classification |
| **Supervised Contrastive** | Supervised Contrastive Learning | Khosla et al., arXiv:2004.11362 | Better representations than CE alone |
| **Focal Loss** | Focal Loss for Dense Object Detection | Lin et al., ICCV 2017 | Handle class imbalance |
| **Temperature Scheduling** | Temperature Schedules for Contrastive Methods | Kukleva et al., CVPR 2023 | Improve contrastive learning |
| **TempBalance** | Temperature Balancing, Layer-wise LR | Zhou et al., NeurIPS 2023 | Layer-wise LR adaptation |
| **DropBlock** | DropBlock: Structured Dropout for CNNs | Ghiasi et al., 2018 | Better than standard dropout |
| **SNR** | TempBalance paper | Zhou et al., NeurIPS 2023 | Weight regularization |

---

## All Command-Line Arguments

### Training Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 200 | Total training epochs |
| `--batch_size` | 16 | Batch size |
| `--lr` | 1e-4 | Learning rate for attention/classifier |
| `--num_heads` | 4 | Attention heads |
| `--seed` | 42 | Random seed |
| `--warmup_epochs` | 10 (5% of epochs) | LR warmup epochs |
| `--checkpoint_every` | 1 | Save checkpoint every N epochs |

### Data & Cross-Validation

| Argument | Default | Description |
|----------|---------|-------------|
| `--test_plate` | P6 | Test plate (P1-P6) |
| `--data_root` | None | Path to data directory |
| `--run_all_folds` | OFF | Run all 6 folds (cross-validation) |
| `--neighborhood` | 3 | Crop neighborhood (3=3×3=9 crops, 5=5×5=25) |
| `--grid_size` | 12 | Grid size for crop positions |

### Regularization

| Argument | Default | Description |
|----------|---------|-------------|
| `--dropout` | 0.5 | Dropout rate for classifier |
| `--weight_decay` | 0.05 | Weight decay (AdamW) |
| `--label_smoothing` | 0.1 | Label smoothing for cross-entropy |
| `--use_dropblock` | OFF | Enable DropBlock regularization |
| `--dropblock_prob` | 0.1 | DropBlock probability |
| `--dropblock_size` | 3 | DropBlock block size |
| `--dropblock_warmup` | 1000 | DropBlock warmup iterations |
| `--use_snr` | OFF | Enable Spectral Norm Regularization |
| `--snr_lambda` | 0.1 | SNR penalty weight |

### SC-MIL Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_sc_mil` | OFF | Enable SC-MIL (joint contrastive + classification) |
| `--sc_mil_epochs` | 100 | Epochs for SC-MIL training |
| `--sc_mil_weight` | 0.3 | Weight for contrastive loss (0.1-1.0) |
| `--sc_mil_temp` | 0.07 | Base temperature for SupCon loss |
| `--contrastive_level` | bag | Contrastive level: instance, bag, or both |
| `--instance_weight` | 0.5 | Weight for instance-level vs bag-level |

### Temperature Scheduling

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_temp_schedule` | OFF | Enable temperature oscillation schedule |
| `--temp_warmup_epochs` | 10 | Temperature warmup epochs |
| `--temp_schedule_min` | 0.07 | Minimum temperature |
| `--temp_schedule_max` | 0.5 | Maximum temperature |
| `--temp_schedule_period` | 0 | Oscillation period (0=cosine decay) |

### TempBalance (Layer-wise LR)

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_tempbalance` | OFF | Enable TempBalance layer-wise LR |
| `--tb_lr_min_ratio` | 0.5 | Min LR multiplier for undertrained layers |
| `--tb_lr_max_ratio` | 1.5 | Max LR multiplier for overtrained layers |
| `--tb_interval` | 10 | Update TempBalance every N epochs |

### Contrastive Pre-training (Stage 1 - Optional)

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_contrastive` | OFF | Enable patch-level SimCLR pre-training |
| `--contrastive_epochs` | 50 | Epochs for SimCLR pre-training |
| `--contrastive_batch_size` | 128 | Batch size for contrastive |
| `--contrastive_temp` | 0.1 | Temperature for SimCLR loss |

---

## Usage Examples

### Basic SC-MIL Training (Recommended)
```bash
python3 train_mil.py --test_plate P6 --use_sc_mil
```

### Run All Folds with SC-MIL
```bash
python3 train_mil.py --run_all_folds --use_sc_mil
```

### With Temperature Schedule
```bash
python3 train_mil.py --test_plate P6 --use_sc_mil \
  --use_temp_schedule \
  --temp_schedule_min 0.07 \
  --temp_schedule_max 0.5
```

### With TempBalance (Layer-wise LR)
```bash
python3 train_mil.py --test_plate P6 --use_sc_mil \
  --use_tempbalance \
  --tb_lr_min_ratio 0.5 \
  --tb_lr_max_ratio 1.5
```

### With DropBlock (Recommended for stability)
```bash
python3 train_mil.py --test_plate P6 --use_sc_mil \
  --use_dropblock \
  --dropblock_prob 0.1 \
  --dropblock_size 3
```

### With SNR (Spectral Norm Regularization)
```bash
python3 train_mil.py --test_plate P6 --use_sc_mil \
  --use_snr \
  --snr_lambda 0.1
```

### Combined: All Regularizations
```bash
python3 train_mil.py --test_plate P6 --use_sc_mil \
  --use_dropblock --dropblock_prob 0.1 \
  --use_tempbalance \
  --use_temp_schedule \
  --use_snr --snr_lambda 0.1 \
  --batch_size 16 \
  --epochs 200
```

### 3×3 Neighborhood (9 crops) - Default
```bash
python3 train_mil.py --test_plate P6 --use_sc_mil --neighborhood 3
```

### 5×5 Neighborhood (25 crops)
```bash
python3 train_mil.py --test_plate P6 --use_sc_mil --neighborhood 5
```

### Instance + Bag Contrastive
```bash
python3 train_mil.py --test_plate P6 --use_sc_mil \
  --contrastive_level both \
  --instance_weight 0.5
```

---

## Loss Calculation Details

### 1. Weighted Focal Loss
```
Focal = α × (1 - p_t)^γ × CE
where p_t = softmax(logits)[target]
α = 0.25, γ = 2.0 (default)
```

Applied at both instance-level (per crop) and bag-level (pooled), then weighted by `instance_weight`.

### 2. Supervised Contrastive Loss (SupCon)
```
SupCon = -τ × log( exp(z_i·z_j/τ) / Σ_k exp(z_i·z_k/τ) )
```
- Positive pairs: samples with same label
- Negative pairs: samples with different labels
- Temperature τ controls sharpness (lower = more aggressive)

### 3. SNR (Spectral Norm Regularization)
```
SNR = λ × σ(W)
```
Where σ(W) is the largest singular value of weight matrix. Penalizes large weight norms.

### 4. Attention Entropy (Standard mode only)
```
Entropy = -Σ attn × log(attn)
```
Encourages focused attention (optional regularization).

---

## Why Each Feature?

### SC-MIL (Supervised Contrastive MIL)
- **Why**: Combines supervised contrastive learning with MIL
- **Benefit**: Better representations than CE alone, handles imbalance
- **Paper**: Juyal et al., WACV 2024

### Temperature Scheduling
- **Why**: High temp early = explore, low temp late = refine
- **Benefit**: More stable training, better convergence
- **Paper**: Kukleva et al., CVPR 2023

### TempBalance
- **Why**: Different layers need different LR (undertrained vs overtrained)
- **Benefit**: Balances layer training, improves generalization
- **Paper**: Zhou et al., NeurIPS 2023

### DropBlock
- **Why**: Standard dropout ineffective for CNNs (spatial correlation)
- **Benefit**: Drops contiguous regions → forces robust features
- **Paper**: Ghiasi et al., 2018

### SNR
- **Why**: Constrains weight matrix norm for stability
- **Benefit**: Regularization, prevents large weights
- **Paper**: Zhou et al., NeurIPS 2023

---

## Output Files

Each fold creates:
```
fold_P6/
├── best_model.pth           # Best by validation AUC
├── best_model_auc.pth      # Best by validation AUC
├── best_model_acc.pth      # Best by validation accuracy
├── best_model_loss.pth     # Best by validation loss
├── checkpoint_epoch.pth    # Last checkpoint
├── training_sc_mil_TIMESTAMP.csv   # SC-MIL training metrics
├── training_results.json   # Final results
└── ...
```

---

## Requirements

```bash
conda create -n crispri python=3.10
conda activate crispri
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install numpy scikit-learn pandas tqdm albumentations
```

---

## Citation

If you use this code, please cite:

```bibtex
# SC-MIL
@InProceedings{Juyal2024SCMIL,
  title={SC-MIL: Supervised Contrastive Multiple Instance Learning for Imbalanced Classification},
  author={Juyal, Priya et al.},
  booktitle={WACV 2024},
  year={2024}
}

# TempBalance
@InProceedings{Zhou2023TempBalance,
  title={Temperature Balancing, Layer-wise Weight Analysis, and Neural Network Training},
  author={Zhou, Yefan et al.},
  booktitle={NeurIPS 2023},
  year={2023}
}

# DropBlock
@InProceedings{Ghiasi2018DropBlock,
  title={DropBlock: A regularization technique for convolutional neural networks},
  author={Ghiasi, Golnaz et al.},
  booktitle={CVPR 2018},
  year={2018}
}
```

---

## License

MIT License