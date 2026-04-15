#!/usr/bin/env python3
"""
Bar plot showing majority vote accuracy vs number of training plates.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

results = []
for n in [1, 2, 3, 4]:
    df = pd.read_csv(f'increase_{n}_plates/image_predictions_{n}_plates.csv')
    maj_correct = df['correct_majority'].sum()
    total = len(df)
    maj_acc = maj_correct / total
    results.append({'n_plates': n, 'majority_vote_acc': maj_acc, 'total': total})

df_results = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(df_results['n_plates'].astype(str), df_results['majority_vote_acc'] * 100, 
              color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], edgecolor='black', linewidth=1.5)

ax.set_xlabel('Number of Training Plates', fontsize=12)
ax.set_ylabel('Majority Vote Accuracy (%)', fontsize=12)
ax.set_title('Plate Diversity Effect on Classification Accuracy\n(Test Plate: P6, 2016 images, 96 classes)', fontsize=14)

ax.set_ylim(0, max(df_results['majority_vote_acc'] * 100) * 1.2)

for bar, acc in zip(bars, df_results['majority_vote_acc']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{acc*100:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.axhline(y=1.04, color='red', linestyle='--', alpha=0.7, label='Random baseline (1/96 ≈ 1.04%)')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('plate_diversity_majority_vote.png', dpi=150, bbox_inches='tight')
print(f"Saved plot to plate_diversity_majority_vote.png")

print("\nSummary:")
print(df_results.to_string(index=False))