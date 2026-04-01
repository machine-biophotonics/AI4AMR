import sys
sys.path.insert(0, '.')
from dino_finetune.plate_dataset import create_datasets
import torch

data_root = "/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging"
label_json_path = f"{data_root}/plate maps/plate_well_id_path.json"
train_ds, val_ds, test_ds = create_datasets(data_root, label_json_path, stain_augmentation=False, target_size=(224, 224))
print(f"Train samples: {len(train_ds)}")
print(f"Val samples: {len(val_ds)}")
print(f"Test samples: {len(test_ds)}")

# Load one sample
img, label, plate = train_ds[0]
print(f"Image shape: {img.shape}, Label: {label}, Plate: {plate}")
print(f"Image dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")

# Check that we have many crops
if len(train_ds) > 8064:
    print("Multi-crop enabled: each image yields multiple crops")
else:
    print("Single crop per image")

# Check that stain augmentation is not causing errors
train_ds_stain, _, _ = create_datasets(data_root, label_json_path, stain_augmentation=True, target_size=(224, 224))
print(f"Train samples with stain aug: {len(train_ds_stain)}")