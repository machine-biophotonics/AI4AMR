# Visualizations

This folder contains visualization scripts for augmentation pipelines used in all models.

## Augmentations Comparison

| Augmentation | sam_effnet | guide_effnet | dinov3-finetune |
|--------------|:----------:|:------------:|:---------------:|
| HorizontalFlip | ✓ p=0.5 | ✓ (same) | ✓ (same) |
| VerticalFlip | ✓ p=0.5 | ✓ (same) | ✓ (same) |
| RandomRotate90 | ✓ p=0.5 | ✓ (same) | ✓ (same) |
| Rotate 360° | ✓ p=0.5 | ✓ (same) | ✓ (same) |
| Affine | ✓ p=0.5 | ✓ (same) | ✓ (same) |
| CLAHE | ✓ p=0.3 | ✓ (same) | ✓ (same) |
| RandomShadow | ✓ p=0.2 | ✓ (same) | ✓ (same) |
| ElasticTransform | ✓ p=0.4 | ✓ (same) | ✓ (same) |
| Perspective | ✓ p=0.4 | ✓ (same) | ✓ (same) |
| GridDistortion | ✓ p=0.4 | ✓ (same) | ✓ (same) |
| OpticalDistortion | ✓ p=0.4 | ✓ (same) | ✓ (same) |
| GaussNoise | ✓ p=0.4 | ✓ (same) | ✓ (same) |
| GaussianBlur | ✓ p=0.4 | ✓ (same) | ✓ (same) |
| MotionBlur | ✓ p=0.4 | ✓ (same) | ✓ (same) |
| PixelDropout | ✓ p=0.15 | ✓ (same) | ✓ (same) |
| SaltAndPepper | ✓ p=0.2 | ✓ (same) | ✓ (same) |
| ISONoise | ✓ p=0.15 | ✓ (same) | ✓ (same) |
| Erasing | ✓ p=0.2 | ✓ (same) | ✓ (same) |
| ImageCompression | ✓ p=0.3 | ✓ (same) | ✓ (same) |
| CoarseDropout | ✓ p=0.3 | ✓ (same) | ✓ (same) |

## Key Points

- **sam_effnet** and **guide_effnet** share the **exact same train.py file** (MD5: a04f2430cda6b1bec68afebdf44c4680)
- **dinov3-finetune** uses identical augmentations in `dino_finetune/plate_dataset.py`
- All models use the same strong augmentation pipeline with:
  - Symmetry transforms (flips, rotations)
  - Geometric deformations (elastic, perspective, grid/optical distortion)
  - Lighting/contrast adjustments (CLAHE, brightness, gamma)
  - Shadow simulation
  - Noise and blur
  - Quality artifacts

## Files

- `visualize_augmentations.py` - Generate augmentation visualization
- `augmentations.png` - Output image showing each augmentation