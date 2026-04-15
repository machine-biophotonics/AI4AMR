#!/usr/bin/env python3
"""
Visualize brightness and contrast augmentation levels.
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageEnhance


def visualize(image_path: str, output_dir: str = './') -> str:
    """Visualize brightness/contrast at different levels.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save the output plot
        
    Returns:
        Path to saved output file
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Different brightness/contrast levels
    levels = [
        ('Original', 0.0, 0.0),
        ('Bright +30%', 0.3, 0.0),
        ('Bright -30%', -0.3, 0.0),
        ('Contrast +30%', 0.0, 0.3),
        ('Contrast -30%', 0.0, -0.3),
        ('Both +30%', 0.3, 0.3),
        ('Both -30%', -0.3, -0.3),
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.flatten()
    
    for idx, (name, bright, contrast) in enumerate(levels):
        # Apply brightness and contrast using PIL
        img = image.copy()
        
        if bright != 0.0:
            factor = 1.0 + bright
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
        
        if contrast != 0.0:
            factor = 1.0 + contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)
        
        # Display
        axes_flat[idx].imshow(img)
        axes_flat[idx].set_title(f'{name}\nB={bright}, C={contrast}', fontsize=10)
        axes_flat[idx].axis('off')
    
    # Hide unused subplot
    axes_flat[7].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'brightness_contrast_levels.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize brightness/contrast augmentation levels')
    parser.add_argument('--image_path', type=str, required=True, help='Path to sample image')
    parser.add_argument('--output_dir', type=str, default='./', help='Output directory')
    args = parser.parse_args()
    
    visualize(args.image_path, args.output_dir)