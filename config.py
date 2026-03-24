"""
Training Configuration for CRISPRi Plate Image Classification
============================================================

This file contains all hyperparameters and settings used for training.
Edit this file to customize training parameters.

SEED & REPRODUCIBILITY
----------------------
SEED: Random seed for reproducibility (42 is standard for reproducibility)
"""
SEED = 42

"""
MODEL ARCHITECTURE
------------------
model_name: Model to use (efficientnet_b0, efficientnet_b1, resnet50, etc.)
pretrained: Use pretrained ImageNet weights
num_classes: Number of gene classes (97)
dropout: Dropout rate before final classification layer
"""
model_name = "efficientnet_b0"
pretrained = True
num_classes = 97
dropout = 0.5

"""
OPTIMIZER SETTINGS
------------------
optimizer: Optimizer type (AdamW, SGD, Adam)
learning_rate: Initial learning rate
weight_decay: L2 regularization strength
"""
optimizer_type = "AdamW"
learning_rate = 1e-3
weight_decay = 1e-3

"""
LEARNING RATE SCHEDULER
------------------------
scheduler: Scheduler type (CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau)
T_0: First restart period (for CosineAnnealingWarmRestarts)
T_mult: Period multiplier after each restart (for CosineAnnealingWarmRestarts)
eta_min: Minimum learning rate
"""
scheduler_type = "CosineAnnealingWarmRestarts"
T_0 = 10
T_mult = 2
eta_min = 1e-6

"""
LOSS FUNCTION
-------------
loss: Loss type (CrossEntropyLoss)
label_smoothing: Label smoothing factor (0.1 helps prevent overconfidence)
class_weights: Use inverse class frequency weighting
"""
use_label_smoothing = True
label_smoothing_factor = 0.1
use_class_weights = True

"""
DATA AUGMENTATION
------------------
train_transform: Augmentations applied to training images
- RandomCenterCrop: Random 224x224 crop from center region (200px edge margin)
- RandomHorizontalFlip: 50% chance horizontal flip
- RandomVerticalFlip: 50% chance vertical flip
- RandomRotation: ±15° random rotation
- ColorJitter: Brightness/contrast/saturation/hue jitter
- RandomErasing: Random pixel dropout
- Normalize: ImageNet normalization

val_transform: Minimal augmentation for validation (just center crop + normalize)
"""
# Color jitter parameters (brightness, contrast, saturation, hue)
color_jitter_brightness = 0.3
color_jitter_contrast = 0.3
color_jitter_saturation = 0.3
color_jitter_hue = 0.1

# Random erasing parameters
random_erasing_prob = 0.3
random_erasing_scale_min = 0.02
random_erasing_scale_max = 0.15

# Random rotation
random_rotation_degrees = 15

"""
PATCH SETTINGS
--------------
n_patches: Number of random crops extracted from each image
patch_size: Size of each crop in pixels (224 for EfficientNet)
edge_margin: Margin to avoid plate edges (200 pixels)
random_patches: Whether to use random (True) or fixed center (False) crop positions
"""
n_patches_train = 10
n_patches_val = 10
n_patches_test = 10
patch_size = 224
edge_margin = 200

"""
DATALOADER SETTINGS
-------------------
batch_size: Number of images per batch (reduced from 16 to 8 due to 10 patches)
num_workers: Number of parallel data loading workers
prefetch_factor: Number of batches to prefetch per worker
persistent_workers: Keep workers alive between epochs
pin_memory: Pin memory for faster GPU transfer
"""
batch_size = 8
num_workers = 8
prefetch_factor = 4
persistent_workers = True
pin_memory = True

"""
TRAINING SETTINGS
-----------------
num_epochs: Maximum number of training epochs
gradient_clip_norm: Gradient clipping threshold
early_stopping_patience: Epochs to wait before early stopping
early_stopping_min_delta: Minimum improvement to reset patience
"""
num_epochs = 50
gradient_clip_norm = 1.0
early_stopping_patience = 10
early_stopping_min_delta = 0.01

"""
DATA SPLIT
-----------
train_plates: Plates used for training (P1, P2, P3, P4)
val_plate: Plate used for validation (P5)
test_plate: Plate used for testing (P6)
"""
train_plates = ["P1", "P2", "P3", "P4"]
val_plate = "P5"
test_plate = "P6"

"""
OUTPUT SETTINGS
---------------
save_dir: Directory to save model checkpoints and plots
checkpoint_name: Name of best model checkpoint
"""
save_dir = "."
checkpoint_name = "best_model.pth"

"""
AUGMENTATION SUMMARY
--------------------
Applied to training images:
1. RandomCenterCrop(224, edge_margin=200) - Extract random 224x224 from center
2. RandomHorizontalFlip(p=0.5)
3. RandomVerticalFlip(p=0.5)
4. RandomRotation(degrees=15)
5. ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
6. ToTensor()
7. RandomErasing(p=0.3, scale=(0.02, 0.15))
8. Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

For each image: 10 random patches extracted, each with independent augmentations
"""
