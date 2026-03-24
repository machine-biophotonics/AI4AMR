"""
Visualize Crop Extraction - AI4AB Style
=====================================
Shows:
1. Original image (2720x2720)
2. 3x3 grid overlay showing crop positions
3. Individual crops extracted
4. Resize from 500x500 to 256x256
5. How crops feed into model
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import os
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_SIZE = 2720
CROP_SIZE = 500
RESIZE_SIZE = 256
N_CROPS = 9
GRID_SIZE = int(np.sqrt(N_CROPS))

step = (IMAGE_SIZE - CROP_SIZE) // (GRID_SIZE + 1)
start = step

crop_positions = []
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        left = start + j * step
        top = start + i * step
        crop_positions.append((left, top, CROP_SIZE, CROP_SIZE))

print(f"Configuration:")
print(f"  Image: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"  Crop: {CROP_SIZE}x{CROP_SIZE}")
print(f"  Resize: {RESIZE_SIZE}x{RESIZE_SIZE}")
print(f"  Grid: {GRID_SIZE}x{GRID_SIZE} = {N_CROPS} crops")

sample_path = os.path.join(BASE_DIR, 'P6')
tif_files = glob.glob(os.path.join(sample_path, '*.tif'))
sample_file = tif_files[0] if tif_files else None

if sample_file:
    full_img = Image.open(sample_file).convert('RGB')
    print(f"Loaded: {os.path.basename(sample_file)}")

scale = 500 / IMAGE_SIZE

def scale_coords(coords):
    return tuple(int(c * scale) for c in coords)

crop_positions_display = [scale_coords(c) for c in crop_positions]

fig = plt.figure(figsize=(20, 12))

gs = fig.add_gridspec(3, 6, height_ratios=[1.2, 1, 1], width_ratios=[1,1,1,1,1,1],
                      hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :3])
if sample_file:
    display_img = full_img.resize((500, 500))
    ax1.imshow(display_img)
for idx, (l, t, w, h) in enumerate(crop_positions_display):
    rect = patches.Rectangle((l, t), w, h, linewidth=2, edgecolor='red', facecolor='none')
    ax1.add_patch(rect)
    ax1.text(l + w//2, t + h//2, str(idx), color='white', fontsize=14, ha='center', va='center',
            bbox=dict(boxstyle='circle', facecolor='red', alpha=0.9), fontweight='bold')
ax1.set_xlim(0, 500)
ax1.set_ylim(500, 0)
ax1.set_title(f'Original Image: {IMAGE_SIZE}x{IMAGE_SIZE} px\n3x3 Grid = {N_CROPS} crops', fontsize=14, fontweight='bold')
ax1.set_xlabel('Each numbered box = one crop', fontsize=11)

ax2 = fig.add_subplot(gs[0, 3])
ax2.imshow(np.ones((100, 100, 3)) * 0.9)
rect = patches.Rectangle((15, 15), 70, 70, linewidth=4, edgecolor='red', facecolor='lightyellow')
ax2.add_patch(rect)
ax2.set_xlim(0, 100)
ax2.set_ylim(100, 0)
ax2.set_title(f'Raw Crop\n{CROP_SIZE}x{CROP_SIZE}', fontsize=12, fontweight='bold')
ax2.text(50, 50, f'{CROP_SIZE}px', ha='center', va='center', fontsize=16, fontweight='bold')

ax3 = fig.add_subplot(gs[0, 4])
ax3.imshow(np.ones((60, 60, 3)) * 0.9)
rect = patches.Rectangle((5, 5), 50, 50, linewidth=4, edgecolor='blue', facecolor='lightcyan')
ax3.add_patch(rect)
ax3.set_xlim(0, 60)
ax3.set_ylim(60, 0)
ax3.set_title(f'After Resize\n{RESIZE_SIZE}x{RESIZE_SIZE}', fontsize=12, fontweight='bold', color='blue')
ax3.text(30, 30, f'{RESIZE_SIZE}px', ha='center', va='center', fontsize=14, fontweight='bold')

ax4 = fig.add_subplot(gs[0, 5])
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('Pipeline', fontsize=12, fontweight='bold')
ax4.text(5, 8, f'{IMAGE_SIZE}px', ha='center', fontsize=12, color='gray')
ax4.annotate('', xy=(5, 6), xytext=(5, 7),
            arrowprops=dict(arrowstyle='->', lw=2))
ax4.text(5, 5.5, f'{CROP_SIZE}px', ha='center', fontsize=12, color='red')
ax4.annotate('', xy=(5, 3.5), xytext=(5, 4.5),
            arrowprops=dict(arrowstyle='->', lw=2))
ax4.text(5, 3, f'{RESIZE_SIZE}px', ha='center', fontsize=12, color='blue')
ax4.text(5, 1.5, 'EfficientNet', ha='center', fontsize=11, fontweight='bold')

ax5 = fig.add_subplot(gs[1, :])
ax5.set_title('Individual Crops Extracted (Each 500x500 → Resized to 256x256)', fontsize=14, fontweight='bold')
ax5.axis('off')

if sample_file:
    for idx, (l, t, w, h) in enumerate(crop_positions):
        col = idx % 6
        row = idx // 6
        if row < 1:
            ax = fig.add_subplot(gs[1, col])
            crop = full_img.crop((l, t, l+w, t+h))
            crop_resized = crop.resize((RESIZE_SIZE, RESIZE_SIZE))
            ax.imshow(crop_resized)
            ax.set_title(f'Crop {idx}: {w}x{h} → {RESIZE_SIZE}x{RESIZE_SIZE}', fontsize=10)
            ax.axis('off')
else:
    for idx in range(9):
        col = idx % 6
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(np.random.rand(RESIZE_SIZE, RESIZE_SIZE, 3))
        ax.set_title(f'Crop {idx}', fontsize=10)
        ax.axis('off')

ax6 = fig.add_subplot(gs[2, :3])
ax6.set_title('Model Architecture', fontsize=14, fontweight='bold')
ax6.set_xlim(0, 12)
ax6.set_ylim(0, 10)
ax6.axis('off')

boxes = [
    (0.5, 7.5, f'Input Batch\n[B, {N_CROPS}, 3, {RESIZE_SIZE}, {RESIZE_SIZE}]', 'lightblue'),
    (0.5, 5, f'Reshape\n[B×{N_CROPS}, 3, {RESIZE_SIZE}, {RESIZE_SIZE}]', 'lightyellow'),
    (0.5, 2.5, f'EfficientNet-B0\n[B×{N_CROPS}, 1280]', 'lightgreen'),
    (0.5, 0, f'AvgPool1D → FC\n[B, num_classes]', 'lightsalmon'),
]
for x, y, text, color in boxes:
    rect = patches.FancyBboxPatch((x, y), 11, 2, boxstyle="round,pad=0.1",
                                  facecolor=color, edgecolor='black', linewidth=2)
    ax6.add_patch(rect)
    ax6.text(x+5.5, y+1, text, ha='center', va='center', fontsize=10, fontweight='bold')
    if y > 0:
        ax6.annotate('', xy=(6.5, y-0.1), xytext=(6.5, y+0.1),
                    arrowprops=dict(arrowstyle='->', lw=2))

ax7 = fig.add_subplot(gs[2, 3:])
ax7.set_title('Feature Aggregation (AvgPool1D)', fontsize=14, fontweight='bold')
ax7.set_xlim(0, 14)
ax7.set_ylim(0, 10)
ax7.axis('off')

for i in range(3):
    for j in range(3):
        x = j * 3 + 1
        y = 7 - i * 3
        rect = patches.FancyBboxPatch((x, y), 2.5, 2.5, boxstyle="round,pad=0.1",
                                     facecolor='lightblue', edgecolor='black', linewidth=1)
        ax7.add_patch(rect)
        ax7.text(x+1.25, y+1.25, f'Crop\n{i*3+j}', ha='center', va='center', fontsize=9)

ax7.annotate('', xy=(11, 8), xytext=(9.5, 8),
            arrowprops=dict(arrowstyle='->', lw=3, color='red'))

rect = patches.FancyBboxPatch((11, 6.5), 2.5, 3, boxstyle="round,pad=0.1",
                              facecolor='lightsalmon', edgecolor='red', linewidth=3)
ax7.add_patch(rect)
ax7.text(12.25, 8, 'Avg\n1D', ha='center', va='center', fontsize=10, fontweight='bold')

ax7.annotate('', xy=(11, 5), xytext=(12.25, 6.5),
            arrowprops=dict(arrowstyle='->', lw=2))

rect = patches.FancyBboxPatch((9.5, 2), 5, 2.5, boxstyle="round,pad=0.1",
                              facecolor='lightgreen', edgecolor='black', linewidth=2)
ax7.add_patch(rect)
ax7.text(12, 3.25, 'Averaged Features\n[1280] → FC → [num_classes]', 
        ha='center', va='center', fontsize=9, fontweight='bold')

fig.text(0.5, 0.02, 
         'AI4AB Pipeline: Image → Crop (500x500) → Resize (256x256) → EfficientNet-B0 → AvgPool → Prediction',
         ha='center', fontsize=12, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig(os.path.join(BASE_DIR, 'crop_extraction_visualization.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✅ Saved: crop_extraction_visualization.png")

fig2, axes2 = plt.subplots(1, 5, figsize=(18, 4))

axes2[0].imshow(np.ones((100, 100, 3)) * 0.8)
rect = patches.Rectangle((10, 10), 80, 80, linewidth=4, edgecolor='blue', facecolor='none')
axes2[0].add_patch(rect)
axes2[0].set_title(f'Original\n{IMAGE_SIZE}x{IMAGE_SIZE}', fontsize=12, fontweight='bold')
axes2[0].axis('off')

axes2[1].imshow(np.ones((100, 100, 3)) * 0.9)
for i in range(3):
    for j in range(3):
        l = 15 + j * 25
        t = 15 + i * 25
        rect = patches.Rectangle((l, t), 20, 20, linewidth=2, edgecolor='red', facecolor='lightyellow')
        axes2[1].add_patch(rect)
axes2[1].set_title(f'3x3 Grid\n{N_CROPS} Crops', fontsize=12, fontweight='bold')
axes2[1].axis('off')

axes2[2].imshow(np.ones((80, 80, 3)) * 0.85)
rect = patches.Rectangle((10, 10), 60, 60, linewidth=3, edgecolor='orange', facecolor='lightyellow')
axes2[2].add_patch(rect)
axes2[2].set_title(f'Crop\n{CROP_SIZE}x{CROP_SIZE}', fontsize=12, fontweight='bold')
axes2[2].axis('off')

axes2[3].imshow(np.ones((50, 50, 3)) * 0.85)
rect = patches.Rectangle((5, 5), 40, 40, linewidth=2, edgecolor='blue', facecolor='lightcyan')
axes2[3].add_patch(rect)
axes2[3].set_title(f'Resize\n{RESIZE_SIZE}x{RESIZE_SIZE}', fontsize=12, fontweight='bold', color='blue')
axes2[3].axis('off')

axes2[4].set_xlim(0, 10)
axes2[4].set_ylim(0, 10)
axes2[4].axis('off')
axes2[4].set_title('Model Flow', fontsize=12, fontweight='bold')

axes2[4].text(5, 8, f'{IMAGE_SIZE}px', ha='center', fontsize=11, color='gray')
axes2[4].annotate('', xy=(5, 6), xytext=(5, 7.5), arrowprops=dict(arrowstyle='->', lw=2))
axes2[4].text(5, 5, f'{CROP_SIZE}px', ha='center', fontsize=11, color='orange')
axes2[4].annotate('', xy=(5, 3.5), xytext=(5, 4.5), arrowprops=dict(arrowstyle='->', lw=2))
axes2[4].text(5, 3, f'{RESIZE_SIZE}px', ha='center', fontsize=11, color='blue')
axes2[4].annotate('', xy=(5, 1.5), xytext=(5, 2.5), arrowprops=dict(arrowstyle='->', lw=2))
axes2[4].text(5, 1, 'EfficientNet', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'crop_pipeline_simple.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ Saved: crop_pipeline_simple.png")
print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"1. Original Image: {IMAGE_SIZE}x{IMAGE_SIZE} pixels")
print(f"2. Extract {N_CROPS} crops (3x3 grid)")
print(f"3. Each crop: {CROP_SIZE}x{CROP_SIZE} pixels")
print(f"4. Resize to: {RESIZE_SIZE}x{RESIZE_SIZE} pixels")
print(f"5. Feed into EfficientNet-B0")
print(f"6. Average features across {N_CROPS} crops")
print(f"7. FC layer → prediction")
print("="*60)
