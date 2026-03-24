# AI4AMR: AI for Antimicrobial Resistance - CRISPRi Reference Plate Imaging

Deep learning classification pipeline for bacterial phenotype classification from CRISPRi microscopy images.

## Project Overview

This project develops image-based machine learning models to classify bacterial phenotypes from CRISPRi (CRISPR interference) microscopy images. Each image represents a different genetic perturbation (gene knockdown), and the model learns to identify which gene is being perturbed based on cell morphology.

### Key Capabilities
- Multi-model support: EfficientNet, BacNet, DINOv3 (Vision Transformer)
- Multi-crop extraction from high-resolution images (2720×2720)
- Multi-class classification (97 gene knockdowns + controls)
- t-SNE/UMAP visualization for feature analysis

---

## Dataset

| Split | Plates | Images | Purpose |
|-------|--------|--------|---------|
| Training | P1, P2, P3, P4 | 8,064 | Model training |
| Validation | P5 | 2,016 | Hyperparameter tuning |
| Test | P6 | 2,016 | Final evaluation |

### Classes
- **97 unique classes**: Gene knockdowns (e.g., `dnaA_1`, `dnaB_2`)
- **WT**: Wild-type control
- **NC**: Negative control

### Image Format
- Resolution: 2720×2720 pixels (16-bit TIFF)
- Channel: Brightfield microscopy (channel 4)

---

## Models

### 1. EfficientNet-B0 (train.py)
Standard CNN baseline with ImageNet pretrained weights.

### 2. BacNet (train_bacnet.py)
Lightweight custom CNN optimized for microscopy images.

### 3. DINOv3 ViT-L (train_dinov3.py)
Vision Transformer with satellite imagery pretraining (SAT-493M).

| Model | Parameters | Best For |
|-------|------------|----------|
| EfficientNet-B0 | 5.3M | Quick experiments |
| BacNet | ~2M | Fast training |
| DINOv3 ViT-L | 300M | Highest accuracy |

---

## Quick Start

### Training EfficientNet
```bash
python train.py --epochs 50 --n_crops 9
```

### Training DINOv3
```bash
python train_dinov3.py --n_crops 25 --batch_size 32 --num_workers 16
```

### Generate Visualizations
```bash
python generate_plots.py
python generate_tsne.py
```

---

## Project Structure

```
.
├── train.py                 # EfficientNet-B0 training
├── train_bacnet.py          # BacNet training (lightweight CNN)
├── train_dinov3.py          # DINOv3 ViT-L + Logistic Regression
├── train_ai4ab.py           # Original AI4AB model
├── train_ai4ab_optimized.py # Optimized AI4AB
├── config.py                # Configuration file
├── plate_well_id_path.json  # Label mappings
├── classes.txt              # Class index to gene name
├── results/                 # Model outputs and visualizations
│   └── dinov3/              # DINOv3 specific results
├── visualizations/          # Visualization scripts
├── P1-P6/                   # Image data by plate
└── dinov3/                  # Facebook Research DINOv3 repo
```

---

## Scripts

### Training Scripts
| Script | Description |
|--------|-------------|
| `train.py` | Main EfficientNet-B0 training with multi-crop inference |
| `train_bacnet.py` | Lightweight CNN with 144-crop extraction |
| `train_dinov3.py` | DINOv3 ViT-L feature extraction + Logistic Regression |
| `train_ai4ab.py` | AI4AB model (custom architecture) |
| `train_ai4ab_optimized.py` | Optimized version with AMP |

### Visualization Scripts
| Script | Description |
|--------|-------------|
| `generate_plots.py` | Training curves, confusion matrix, ROC/PR curves |
| `generate_tsne.py` | t-SNE visualization of embeddings |
| `generate_interactive_plots.py` | Interactive HTML visualizations |
| `plot_umap.py` | UMAP dimensionality reduction |
| `plot_confusion_matrix.py` | Confusion matrix visualization |
| `plot_gradcam.py` | Grad-CAM attention visualization |
| `visualize_crops.py` | Crop extraction visualization |
| `visualize_bacnet.py` | BacNet-specific visualizations |
| `visualize_dinov3_crops.py` | DINOv3 crop grid visualization |
| `visualize_dinov3_fiftyone.py` | Interactive FiftyOne visualization |

### Data Processing Scripts
| Script | Description |
|--------|-------------|
| `update_labels_from_plate_maps.py` | Update labels from Excel plate maps |
| `update_p2_json.py` | Update P2 plate labels |
| `update_p3_json.py` | Update P3 plate labels |
| `get_classes.py` | Extract class names |

### Debugging Scripts
| Script | Description |
|--------|-------------|
| `debug_train.py` | Debug training loop |
| `debug_1batch.py` | Debug single batch processing |
| `debug_bacnet.py` | Debug BacNet model |
| `debug_ai4ab.py` | Debug AI4AB model |
| `debug_optimized.py` | Debug optimized training |
| `test_batch_size.py` | Test different batch sizes |
| `test_gpu_memory.py` | Check GPU memory usage |
| `test_batch_speed.py` | Benchmark batch processing speed |

### Analysis Scripts
| Script | Description |
|--------|-------------|
| `umap_grid_search_cnn.py` | UMAP hyperparameter search |
| `plot_training.py` | Training plot analysis |
| `visualize_center_crop.py` | Center crop visualization |
| `visualize_crop_pipeline.py` | Crop pipeline flow |

---

## Output Files

| File | Description |
|------|-------------|
| `best_model.pth` | Best model weights (highest val accuracy) |
| `classes.txt` | Class index to gene name mapping |
| `training_plots.png` | Loss and accuracy curves |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `tsne_plot.png` | t-SNE visualization |
| `roc_curves.png` | ROC curves for top classes |
| `training_results_*.json` | Full training results |

---

## Configuration

Key hyperparameters in scripts:
- `n_crops`: Number of crops per image (9, 25, 144)
- `batch_size`: Training batch size
- `learning_rate`: Optimizer learning rate
- `epochs`: Number of training epochs

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (24GB recommended for DINOv3)
- 50GB+ disk space

See requirements.txt for dependencies.

---

## Citation

If you use this code, please cite:

```
AI4AMR: AI for Antimicrobial Resistance - CRISPRi Reference Plate Imaging
Machine Biophotonics Lab
```

---

## License

MIT License