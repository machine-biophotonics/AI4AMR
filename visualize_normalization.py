#!/usr/bin/env python3
"""
Visualize comparison of two normalization methods for 16-bit microscopy images:
1. Min-Max (trial_daniel approach)
2. Percentile-based (final_mutant_model approach)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_IMAGE = os.path.join(SCRIPT_DIR, "Drugs_Data/P1/20260428_161449_035/WellA01_PointA01_0000_ChannelCam-DIA DIC Master Screening_Seq0000.tiff")

def load_image_16bit(path):
    """Load 16-bit TIFF image."""
    try:
        img = tifffile.imread(path)
    except Exception:
        img = np.array(Image.open(path))
    return img

def normalize_minmax(img_array, bit_depth=None):
    """trial_daniel approach: Simple min-max normalization.
    Automatically detects bit depth from dtype if not specified."""
    if bit_depth is None:
        if img_array.dtype == np.uint16:
            bit_depth = 16
        elif img_array.dtype == np.uint8:
            bit_depth = 8
        else:
            bit_depth = 16  # default
    normalized = img_array.astype(np.float32) / (2**bit_depth - 1)
    return normalized

def normalize_percentile(img_array, p_low=0.1, p_high=99.9):
    """final_mutant_model approach: Percentile-based normalization."""
    p_bot = np.percentile(img_array, p_low)
    p_top = np.percentile(img_array, p_high)
    normalized = np.clip(img_array, p_bot, p_top)
    normalized = (normalized - p_bot) / (p_top - p_bot + 1e-8)
    return normalized.astype(np.float32)

def normalize_percentile_skimage(img_array, p_low=0.1, p_high=99.9):
    """Using skimage for rescale_intensity (exact final_mutant_model approach)."""
    try:
        from skimage import exposure
        p_bot = np.percentile(img_array, p_low)
        p_top = np.percentile(img_array, p_high)
        normalized = np.clip(img_array, p_bot, p_top)
        normalized = exposure.rescale_intensity(normalized, out_range='uint8')
        return normalized.astype(np.float32) / 255.0
    except ImportError:
        return normalize_percentile(img_array, p_low, p_high)

def main():
    print(f"Loading image: {SAMPLE_IMAGE}")
    img = load_image_16bit(SAMPLE_IMAGE)
    print(f"Original shape: {img.shape}, dtype: {img.dtype}")
    print(f"Original min: {img.min()}, max: {img.max()}")

    img_min = img.min()
    img_max = img.max()
    p01 = np.percentile(img, 0.1)
    p99 = np.percentile(img, 99.9)
    print(f"Percentiles: 0.1% = {p01:.0f}, 99.9% = {p99:.0f}")

    img_minmax = normalize_minmax(img)
    img_percentile = normalize_percentile(img)
    img_percentile_sk = normalize_percentile_skimage(img)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title(f'Original 16-bit\nmin={img_min}, max={img_max}')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_minmax, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Min-Max Normalization\n(trial_daniel)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(img_percentile, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Percentile Normalization\n(final_mutant_model)')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(img_percentile_sk, cmap='gray', vmin=0, vmax=1)
    axes[0, 3].set_title('Percentile + skimage\n(final_mutant_model)')
    axes[0, 3].axis('off')

    axes[1, 0].hist(img.flatten(), bins=100, color='black', alpha=0.7)
    axes[1, 0].axvline(p01, color='red', linestyle='--', label=f'0.1%: {p01:.0f}')
    axes[1, 0].axvline(p99, color='red', linestyle='--', label=f'99.9%: {p99:.0f}')
    axes[1, 0].set_title('Original Histogram')
    axes[1, 0].legend()

    axes[1, 1].hist(img_minmax.flatten(), bins=100, color='blue', alpha=0.7)
    axes[1, 1].set_title('Min-Max Histogram')
    axes[1, 1].set_xlim(0, 1)

    axes[1, 2].hist(img_percentile.flatten(), bins=100, color='green', alpha=0.7)
    axes[1, 2].set_title('Percentile Histogram')
    axes[1, 2].set_xlim(0, 1)

    axes[1, 3].hist(img_percentile_sk.flatten(), bins=100, color='orange', alpha=0.7)
    axes[1, 3].set_title('Percentile + skimage Histogram')
    axes[1, 3].set_xlim(0, 1)

    plt.tight_layout()
    output_path = os.path.join(SCRIPT_DIR, 'visualizations', 'normalization_comparison.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
    vmin, vmax = 0, 1
    
    axes2[0].imshow(img_minmax, cmap='gray', vmin=vmin, vmax=vmax)
    axes2[0].set_title('Min-Max (trial_daniel)\nSimple division by 65535')
    axes2[0].axis('off')
    
    axes2[1].imshow(img_percentile, cmap='gray', vmin=vmin, vmax=vmax)
    axes2[1].set_title('Percentile 0.1-99.9 (final_mutant_model)\nClips outliers, then rescales')
    axes2[1].axis('off')
    
    diff = np.abs(img_percentile - img_minmax)
    axes2[2].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
    axes2[2].set_title(f'Difference\nMax diff: {diff.max():.3f}')
    axes2[2].axis('off')
    
    plt.tight_layout()
    output_path2 = os.path.join(SCRIPT_DIR, 'visualizations', 'normalization_sidebyside.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path2}")

    print("\n=== Summary ===")
    print(f"Original range: {img_min} - {img_max}")
    print(f"Percentile range (0.1-99.9): {p01:.0f} - {p99:.0f}")
    print(f"Outlier range: {p99 - p01:.0f} values ({100*(p99-p01)/(img_max-img_min):.1f}% of total)")
    print(f"\nMax pixel difference between methods: {diff.max():.4f}")
    print(f"Mean pixel difference: {diff.mean():.4f}")

    plt.close('all')
    print("\nDone! Check visualizations/ folder for output.")

if __name__ == '__main__':
    main()