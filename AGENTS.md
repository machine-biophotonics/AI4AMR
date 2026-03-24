# AI Agent Guidance - CRISPRi Reference Plate Imaging Project

## Project Overview

This repository contains deep learning models for classifying bacterial phenotypes from CRISPRi microscopy images. The goal is to identify which gene is being knocked down based on cell morphology in microscopy images.

## Dataset Structure

- **Images**: 2720×2720 pixel 16-bit TIFF files (brightfield microscopy)
- **Labels**: 97 classes (gene knockdowns + WT + NC controls)
- **Split**: P1-P4 (train: 8,064), P5 (val: 2,016), P6 (test: 2,016)
- **Mapping**: `plate_well_id_path.json` contains image path to label mapping

## Key Technical Details

### Image Processing
- Extract multiple crops from each image to capture cell population diversity
- Edge margin of 200px to avoid artifacts at plate edges
- Crops resized to 224×224 for CNN models, 256×256 for DINOv3

### Available Models

1. **EfficientNet-B0** (`train.py`)
   - Standard CNN with ImageNet pretrained weights
   - Best for quick experiments

2. **BacNet** (`train_bacnet.py`)
   - Lightweight custom CNN
   - Uses 144 crops per image (12×12 grid)

3. **DINOv3 ViT-L** (`train_dinov3.py`)
   - Vision Transformer with satellite imagery pretraining (SAT-493M)
   - Uses 25 crops (5×5 grid) of 512×512 pixels
   - Extracts CLS token embeddings, averages per image
   - Trains Logistic Regression classifier on embeddings

## Scripts by Category

### Primary Training Scripts
- `train.py` - Main EfficientNet training (start here for new experiments)
- `train_bacnet.py` - BacNet lightweight CNN
- `train_dinov3.py` - DINOv3 Vision Transformer approach
- `train_ai4ab.py` - Original AI4AB architecture

### Visualization & Analysis
- `generate_plots.py` - Training metrics, confusion matrix
- `generate_tsne.py` - t-SNE embedding visualization
- `plot_umap.py` - UMAP dimensionality reduction
- `visualize_crops.py` - Crop extraction visualization

### Data Preparation
- `update_labels_from_plate_maps.py` - Update labels from Excel files
- `get_classes.py` - Extract unique class names

### Debugging
- `debug_train.py` - Debug full training loop
- `test_gpu_memory.py` - Check GPU memory usage
- `test_batch_speed.py` - Benchmark data loading speed

## Common Workflows

### Training a New Model
1. Start with `train.py` for baseline
2. Use `--n_crops 9` for quick iteration
3. Scale up to full crop count for production

### Running DINOv3
```bash
python train_dinov3.py --n_crops 25 --batch_size 32 --num_workers 16 --crop_size 512
```

### Generating Visualizations
```bash
python generate_plots.py
python generate_tsne.py
```

## Important Notes for Agents

### Configuration
- All scripts use seed=42 for reproducibility
- Device is auto-detected (cuda/cpu)
- Class weights are balanced for imbalanced classes

### Output Locations
- Models saved to: `best_model.pth` (auto-overwrites)
- Results in: `results/` subdirectory
- Classes in: `classes.txt`

### Dependencies
- PyTorch 2.0+
- torchvision
- PIL/Pillow
- numpy, scikit-learn
- matplotlib, tqdm

### Known Issues
- Large model checkpoints (DINOv3 ViT-7B is 27GB) may not fit in 24GB GPU
- Use ViT-L (300M params, 1.2GB) instead for RTX 4500 Ada (24GB)
- 16-bit TIFF images require special handling

## Coding Conventions

- Use type hints where possible
- Set random seeds before any operations
- Use `os.path.join()` for path handling
- Device management: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`

## Testing Approach

Before running full training:
1. Use small `--n_crops` (e.g., 9)
2. Use `--epochs 2` for quick validation
3. Check GPU memory with `test_gpu_memory.py`

## Troubleshooting

**Out of memory**: Reduce batch size or crop count
**Slow training**: Increase `num_workers` in DataLoader
**Low accuracy**: 
- Check label mappings in `plate_well_id_path.json`
- Verify class weights are applied
- Train for more epochs

## File Naming Conventions

- Training logs: `training_log_*.txt`
- Results JSON: `training_results_*.json`
- Metrics CSV: `class_metrics_*.csv`
- Timestamps in format: `YYYYMMDD_HHMMSS`