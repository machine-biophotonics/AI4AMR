# Visualizations

This folder contains visualization scripts for augmentation pipelines used in all models.

## Augmentations Comparison

| Augmentation | sam_effnet | guide_effnet | plate_fold |
|--------------|:----------:|:------------:|:----------:|
| HorizontalFlip | ✓ p=0.5 | ✓ (same) | ✓ (same) |
| VerticalFlip | ✓ p=0.5 | ✓ (same) | ✓ (same) |
| RandomRotate90 | ✓ p=0.5 | ✓ (same) | ✓ (same) |
| Affine | ✓ p=0.5 | ✓ (same) | ✓ (same) |
| ElasticTransform | ✓ p=0.5 | ✓ (same) | ✓ (same) |
| Perspective | ✓ p=0.5 | ✓ (same) | ✓ (same) |
| GridDistortion | ✓ p=0.5 | ✓ (same) | ✓ (same) |
| OpticalDistortion | ✓ p=0.5 | ✓ (same) | ✓ (same) |
| GaussNoise | ✓ p=0.5 | ✓ (same) | ✓ (same) |
| GaussianBlur | ✓ p=0.5 | ✓ (same) | ✓ (same) |
| MotionBlur | ✓ p=0.5 | ✓ (same) | ✓ (same) |
| ImageCompression | ✓ p=0.3 | ✓ (same) | ✓ (same) |
| CoarseDropout | ✓ p=0.4 | ✓ (same) | ✓ (same) |

## Optimizer Comparison

| Model | Optimizer |
|-------|----------|
| sam_effnet | SAM (Sharpness-Aware Minimization) |
| guide_effnet | SAM |
| plate_fold | Basic AdamW |

## Key Points

- **sam_effnet**, **guide_effnet**, and **plate_fold** use **IDENTICAL augmentations**
- Only difference is optimizer: SAM vs basic AdamW
- All use basic augmentations (simpler pipeline)

## Files

- `visualize_augmentations.py` - Generate augmentation visualization
- `augmentations.png` - Output image showing each augmentation

Run with: `python visualize_augmentations.py`