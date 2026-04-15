#!/usr/bin/env python3
"""
Bar plot showing diversity effect at crop, image, and well levels.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from collections import Counter

# Load class mapping
idx_to_label = {}
with open('/home/student/Desktop/CRISPRi_Imaging/plate_fold/classes.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            idx_to_label[int(parts[0])] = parts[1]

results = []
for n in [1, 2, 3, 4]:
    # Crop-level: from predictions_*.csv (144 crops per image)
    df_crop = pd.read_csv(f'increase_{n}_plates/predictions_{n}_plates.csv')
    
    # Crop-level accuracy: compare indices (vectorized for speed)
    crop_correct = (df_crop['ground_truth_idx'].values == df_crop['predicted_class_idx'].values).sum()
    crop_total = len(df_crop)
    crop_acc = crop_correct / crop_total * 100 if crop_total > 0 else 0
    print(f'n={n}: {crop_total} crops, {crop_correct} correct = {crop_acc:.2f}%')
    
    # Image-level: from image_predictions_*.csv (majority vote)
    df = pd.read_csv(f'increase_{n}_plates/image_predictions_{n}_plates.csv')
    
    # Image-level accuracy (majority vote)
    maj_correct = df['correct_majority'].sum()
    total = len(df)
    img_acc = maj_correct / total * 100
    
    # Well-level: aggregate images to wells, compare indices
    well_preds = {}
    for _, row in df.iterrows():
        well = row['well']
        true_idx = row['ground_truth_idx']
        # majority_vote_pred IS an index (int)
        pred_idx = row['majority_vote_pred']
        
        if well not in well_preds:
            well_preds[well] = {'true': true_idx, 'preds': []}
        well_preds[well]['preds'].append(pred_idx)
    
    # Calculate well-level accuracy (compare indices)
    correct = 0
    total_wells = 0
    for well, data in well_preds.items():
        majority_pred = Counter(data['preds']).most_common(1)[0][0]
        if int(majority_pred) == int(data['true']):
            correct += 1
        total_wells += 1
    
    well_acc = correct / total_wells * 100 if total_wells > 0 else 0
    
    results.append({
        'n_plates': n, 
        'crop_acc': crop_acc, 
        'image_acc': img_acc, 
        'well_acc': well_acc
    })
    print(f'n={n}: Crop={crop_acc:.2f}%, Image={img_acc:.2f}%, Well={well_acc:.2f}%')

df_results = pd.DataFrame(results)

print("=== Results ===")
print(df_results)
fig, ax = plt.subplots(figsize=(10, 7))

x = np.arange(len(df_results))
width = 0.25

bars1 = ax.bar(x - width, df_results['crop_acc'], width, label='Crop-level', color='#1f77b4', edgecolor='black')
bars2 = ax.bar(x, df_results['image_acc'], width, label='Image-level', color='#ff7f0e', edgecolor='black')
bars3 = ax.bar(x + width, df_results['well_acc'], width, label='Well-level', color='#2ca02c', edgecolor='black')

ax.set_xlabel('Number of Training Plates', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Plate Diversity Effect on Classification Accuracy\n(Test Plate: P6, 96 classes)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(['1', '2', '3', '4'])
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# Random baseline
random_baseline = 100 / 96  # 1/96 classes
ax.axhline(y=random_baseline, color='red', linestyle='--', alpha=0.7, 
           label=f'Random ({random_baseline:.2f}%)')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('plate_diversity_all_levels.png', dpi=150, bbox_inches='tight')
print(f"Saved: plate_diversity_all_levels.png")

print("\n=== Summary ===")
print(df_results.to_string(index=False))