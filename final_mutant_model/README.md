# final_mutant_model - MIL Training for CRISPRi Reference Plate Imaging

Multiple Instance Learning (MIL) model for classifying CRISPRi guide experiments and antibiotic treatments from plate-based microscopy images.

---

## Project Structure

```
final_mutant_model/
├── train_mil.py                       # Training script
├── predict_all_crops.py               # Prediction/evaluation script
├── mil_model.py                       # Model definitions
├── supcon_loss.py                    # Supervised Contrastive Loss
├── plate_well_id_path.json            # Mutant (gene) mapping
├── plate_well_ic50_mapping.json       # Drug (antibiotic) mapping
│
├── generate_combined_confusion.py     # Mutant: gene hierarchy confusion matrices
├── generate_drug_confusion.py         # Drug: MoA grouped confusion matrices
├── generate_89class_confusion.py      # Drug: 89-class confusion matrix
├── generate_gene_confusion.py         # Drug-on-mutant cross-domain analysis
│
├── mutant/                            # Mutant (gene) experiment results
│   └── fold_Plate_1/                  # Fold results (Plate_1 = test plate)
│       ├── checkpoint_epoch.pth       # Last epoch checkpoint
│       ├── best_model_acc.pth         # Best by validation accuracy
│       ├── best_model_auc.pth         # Best by validation AUC
│       ├── best_model_loss.pth        # Best by validation loss
│       ├── training_results.json      # Training metrics summary
│       ├── training_sc_mil_*.csv      # SC-MIL training log
│       ├── predictions_all_crops_mil_checkpoint_epoch_n3.csv
│       ├── predictions_all_crops_mil_best_model_acc_n3.csv
│       └── confusion_matrices/        # Confusion matrix outputs
│           ├── raw_cm_*_guide.png     # Raw counts (guide level)
│           ├── raw_cm_*_gene.png      # Raw counts (gene level)
│           ├── raw_cm_*_pathway.png   # Raw counts (pathway level)
│           ├── raw_cm_*_family.png    # Raw counts (family level)
│           ├── percent_cm_*_*.png     # Percentage normalized
│           ├── binary_cm_*_*.png      # Binary (correct/incorrect)
│           └── combined_metrics.csv   # Summary metrics
│
└── drug/                              # Drug (antibiotic) experiment results
    └── fold_Plate_1/
        ├── checkpoint_epoch.pth
        ├── best_model_acc.pth
        ├── best_model_auc.pth
        ├── training_results.json
        ├── training_sc_mil_*.csv
        ├── predictions_all_crops_mil_best_model_acc_n3.csv
        └── confusion_matrices/        # Drug confusion outputs
            ├── confusion_matrix_antibiotic_0.25x.png
            ├── confusion_matrix_antibiotic_0.5x.png
            ├── confusion_matrix_antibiotic_1x.png
            ├── confusion_matrix_antibiotic_2x.png
            ├── confusion_matrix_moa_0.25x.png
            ├── confusion_matrix_moa_0.5x.png
            ├── confusion_matrix_moa_1x.png
            ├── confusion_matrix_moa_2x.png
            └── combined_metrics.csv
```

---

## Quick Start

### Training

**Train on Mutant data (gene/knockdown classification):**
```bash
cd /media/student/Data_SSD_1-TB/2025_12_19\ CRISPRi\ Reference\ Plate\ Imaging/final_mutant_model
python3 train_mil.py --test_plate Plate_1 --data_mode mutant --use_sc_mil
```

**Train on Drug data (antibiotic + concentration classification):**
```bash
python3 train_mil.py --test_plate Plate_1 --data_mode drug --use_sc_mil
```

**Run all 6 folds (cross-validation):**
```bash
python3 train_mil.py --run_all_folds --data_mode mutant --use_sc_mil
```

### Prediction

**Generate predictions on Mutant test plate:**
```bash
# Using last checkpoint (epoch 199)
python3 predict_all_crops.py \
  --fold Plate_1 \
  --data_mode mutant \
  --checkpoint checkpoint_epoch.pth \
  --crop_neighborhood 3

# Using best accuracy checkpoint
python3 predict_all_crops.py \
  --fold Plate_1 \
  --data_mode mutant \
  --checkpoint best_model_acc.pth \
  --crop_neighborhood 3
```

**Generate predictions on Drug test plate:**
```bash
python3 predict_all_crops.py \
  --fold Plate_1 \
  --data_mode drug \
  --checkpoint checkpoint_epoch.pth \
  --crop_neighborhood 3
```

### Confusion Matrix Generation

**For Mutant predictions (gene hierarchy):**
```bash
# Use predictions from best accuracy checkpoint
python3 generate_combined_confusion.py --single_fold Plate_1

# Or specify prediction CSV explicitly
python3 generate_combined_confusion.py \
  --single_fold Plate_1 \
  --prediction_csv predictions_all_crops_mil_best_model_acc_n3.csv
```

**For Drug predictions (MoA grouped):**
```bash
python3 generate_drug_confusion.py --fold Plate_1
```

**For Drug predictions (89-class):**
```bash
python3 generate_89class_confusion.py --fold Plate_1
```

---

## Data Modes

| Mode | Data Source | Classes | JSON Mapping |
|------|-------------|---------|---------------|
| `mutant` | Mutants_Data/ | 96 genes (e.g., lptA_3, mrdA_2) | plate_well_id_path.json |
| `drug` | Drugs_Data/ | 89 (antibiotic + concentration) | plate_well_ic50_mapping.json |
| `both` | Both | Combined | Both JSONs |

---

## Key Arguments

### Training (train_mil.py)

| Argument | Default | Description |
|----------|---------|-------------|
| `--test_plate` | Plate_1 | Test plate (Plate_1 to Plate_6) |
| `--data_mode` | mutant | Data type: mutant, drug, or both |
| `--use_sc_mil` | ON | Use SC-MIL (joint contrastive + classification) |
| `--epochs` | 200 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--neighborhood` | 3 | Crop neighborhood (3=3x3=9 crops, 5=5x5=25) |
| `--dropout` | 0.5 | Dropout rate |
| `--sc_mil_weight` | 0.3 | Weight for contrastive loss |
| `--sc_mil_temp` | 0.07 | Temperature for SupCon loss |

### Prediction (predict_all_crops.py)

| Argument | Default | Description |
|----------|---------|-------------|
| `--fold` | P6 | Test fold (Plate_1, Plate_2, etc.) |
| `--data_mode` | drug | Data type: mutant, drug, or both |
| `--checkpoint` | best_model_acc.pth | Checkpoint file to use |
| `--crop_neighborhood` | 3 | Crop neighborhood size |
| `--mil_mode` | True | Use MIL mode with attention pooling |

---

## Output Files Explained

### Checkpoints

| File | Description |
|------|-------------|
| `checkpoint_epoch.pth` | Last epoch (199) - use for final predictions |
| `best_model_acc.pth` | Best by validation accuracy |
| `best_model_auc.pth` | Best by validation AUC |
| `best_model_loss.pth` | Best by lowest validation loss |

### Prediction CSV Format

Each row represents one crop prediction:
- `image_path`: Full path to image file
- `well`: Well position (e.g., A01, B12)
- `ground_truth_label`: True label (e.g., lptA_3 or Chloramphenicol_1x)
- `predicted_class_name`: Model prediction
- `position_index`: Crop position in grid (0-99)

### Confusion Matrix Hierarchy (Mutants)

| Level | Classes | Description |
|-------|---------|-------------|
| guide | 96 | Individual CRISPRi guide (e.g., lptA_3) |
| gene | 30 | Gene target (e.g., lptA) |
| pathway | 11 | Biological pathway (e.g., cell wall) |
| family | 16 | Gene family (e.g., lpt) |

### Confusion Matrix Groups (Drugs)

| Group | Antibiotics |
|-------|-------------|
| Cell wall (PBP 2) | Avibactam, Clavulanic Acid, Meropenem, Mecillinam |
| Cell wall (PBP 3) | Aztreonam, Ceftriaxone, Cefepim |
| Cell wall (PBP 1) | Sulbactam, Penicillin, Cefsulodin |
| Ribosome | Doxicyclin, Chloramphenicol, Clarithromycin, Kanamycin |
| Gyrase | Ciprofloxacin, Norfloxacin, Levofloxacin |
| Membrane integrity | Polymyxin B, Colistin |
| RNA polymerase | Rifampicin |
| DNA synthesis | Trimethoprim |
| Control | DMSO |

---

## Cross-Domain Evaluation

**Drug model on Mutant data:**
```bash
python3 predict_all_crops.py \
  --fold Plate_1 \
  --data_mode mutant \
  --drug_on_mutant \
  --checkpoint best_model_acc.pth
```

**Mutant model on Drug data:**
```bash
python3 predict_all_crops.py \
  --fold Plate_1 \
  --data_mode drug \
  --mutant_on_drug \
  --checkpoint best_model_acc.pth
```

---

## Results Summary

### Mutant (Gene Classification)

| Metric | Value |
|--------|-------|
| Image accuracy (best_model_acc) | ~21-26% |
| Image accuracy (checkpoint_epoch) | ~26% |
| Gene-level accuracy | ~46-53% |
| Pathway-level accuracy | ~59-67% |
| Family-level accuracy | ~55-61% |

### Drug (Antibiotic Classification)

| Metric | Value |
|--------|-------|
| Image accuracy (best_model_acc) | ~43-50% |
| Test AUC | ~0.97 |
| Test AP | ~0.55 |

---

## Requirements

```bash
conda create -n crispri python=3.10
conda activate crispri
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install numpy scikit-learn pandas tqdm albumentations seaborn
```

---

## Notes

1. **Ground truth fix**: The prediction script automatically handles plate key conversion (Plate_1 → P1) for ground truth lookup.

2. **Crop extraction**: Training uses 3x3 neighborhood (9 crops per image), prediction uses 10x10 grid (100 crops).

3. **Confusion matrices**: 
   - Use `generate_combined_confusion.py` for **mutants** (gene hierarchy)
   - Use `generate_drug_confusion.py` for **drugs** (MoA grouped)
   - Use `generate_89class_confusion.py` for **drugs** (full 89 classes)

4. **Checkpoint selection**: 
   - Use `checkpoint_epoch.pth` for final/last epoch predictions
   - Use `best_model_acc.pth` for best validation accuracy
   - Use `best_model_auc.pth` for best validation AUC