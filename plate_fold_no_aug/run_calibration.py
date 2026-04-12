#!/usr/bin/env python3
"""
Post-hoc calibration using P5 as calibration set, apply to P6.
Learn temperature and bias corrections on P5, apply to P6.
"""

import os
import json
import ast
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    # Load classes
    classes = {}
    with open(os.path.join(SCRIPT_DIR, 'classes.txt'), 'r') as f:
        for line in f:
            idx, name = line.strip().split(',', 1)
            classes[int(idx)] = name
    
    num_classes = len(classes)
    print(f"Loaded {num_classes} classes")
    
    # Load P5 predictions (calibration set)
    print("\n=== Loading P5 predictions (calibration set) ===")
    df_p5 = pd.read_csv(os.path.join(SCRIPT_DIR, 'fold_P6', 'predictions_on_P5.csv'))
    df_p5['probs_parsed'] = df_p5['probs'].apply(ast.literal_eval)
    
    y_true_p5 = df_p5['ground_truth_idx'].values
    y_probs_p5 = np.array(df_p5['probs_parsed'].tolist())
    y_pred_p5 = y_probs_p5.argmax(axis=1)
    acc_p5 = (y_true_p5 == y_pred_p5).mean() * 100
    f1_p5 = f1_score(y_true_p5, y_pred_p5, average='macro', zero_division=0) * 100
    print(f"P5 (calibration): {len(df_p5)} crops, Acc={acc_p5:.2f}%, F1={f1_p5:.2f}%")
    
    # Load P6 predictions (test set)
    print("\n=== Loading P6 predictions (test set) ===")
    df_p6 = pd.read_csv(os.path.join(SCRIPT_DIR, 'fold_P6', 'predictions_all_crops.csv'))
    df_p6['probs_parsed'] = df_p6['probs'].apply(ast.literal_eval)
    
    y_true_p6 = df_p6['ground_truth_idx'].values
    y_probs_p6 = np.array(df_p6['probs_parsed'].tolist())
    y_pred_p6 = y_probs_p6.argmax(axis=1)
    acc_p6 = (y_true_p6 == y_pred_p6).mean() * 100
    f1_p6 = f1_score(y_true_p6, y_pred_p6, average='macro', zero_division=0) * 100
    print(f"P6 (test): {len(df_p6)} crops, Acc={acc_p6:.2f}%, F1={f1_p6:.2f}%")
    
    # Method 1: Temperature Scaling on P5
    print("\n=== Temperature Scaling (learn on P5) ===")
    best_temp = 1.0
    best_f1_p5 = f1_p5
    
    for temp in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
        y_probs_scaled = y_probs_p5.copy()
        y_probs_scaled = y_probs_scaled ** (1.0 / temp)
        y_probs_scaled = y_probs_scaled / y_probs_scaled.sum(axis=1, keepdims=True)
        
        y_pred_scaled = y_probs_scaled.argmax(axis=1)
        f1_scaled = f1_score(y_true_p5, y_pred_scaled, average='macro', zero_division=0) * 100
        
        if f1_scaled > best_f1_p5:
            best_f1_p5 = f1_scaled
            best_temp = temp
    
    print(f"Best temperature: {best_temp}")
    
    # Apply to P6
    y_probs_p6_scaled = y_probs_p6.copy()
    y_probs_p6_scaled = y_probs_p6_scaled ** (1.0 / best_temp)
    y_probs_p6_scaled = y_probs_p6_scaled / y_probs_p6_scaled.sum(axis=1, keepdims=True)
    y_pred_p6_temp = y_probs_p6_scaled.argmax(axis=1)
    acc_p6_temp = (y_true_p6 == y_pred_p6_temp).mean() * 100
    f1_p6_temp = f1_score(y_true_p6, y_pred_p6_temp, average='macro', zero_division=0) * 100
    print(f"After temp scaling: P6 Acc={acc_p6_temp:.2f}%, F1={f1_p6_temp:.2f}%")
    
    # Method 2: Per-class bias correction learned on P5
    print("\n=== Per-class Bias Correction (learn on P5) ===")
    pred_counts_p5 = pd.Series(y_pred_p5).value_counts().sort_index()
    gt_counts_p5 = pd.Series(y_true_p5).value_counts().sort_index()
    
    scale = 0.01
    corrections = np.zeros(num_classes)
    for c in range(num_classes):
        gt = gt_counts_p5.get(c, 0)
        pred = pred_counts_p5.get(c, 0)
        bias = pred - gt
        corrections[c] = -bias * scale
    
    y_probs_p6_corr = y_probs_p6.copy()
    for c in range(num_classes):
        y_probs_p6_corr[:, c] += corrections[c]
    y_pred_p6_corr = y_probs_p6_corr.argmax(axis=1)
    acc_p6_corr = (y_true_p6 == y_pred_p6_corr).mean() * 100
    f1_p6_corr = f1_score(y_true_p6, y_pred_p6_corr, average='macro', zero_division=0) * 100
    print(f"After bias correction: P6 Acc={acc_p6_corr:.2f}%, F1={f1_p6_corr:.2f}%")
    
    # Method 3: Combined
    print("\n=== Combined (Temperature + Bias) ===")
    y_probs_p6_comb = y_probs_p6.copy()
    y_probs_p6_comb = y_probs_p6_comb ** (1.0 / best_temp)
    for c in range(num_classes):
        y_probs_p6_comb[:, c] += corrections[c]
    y_probs_p6_comb = y_probs_p6_comb / y_probs_p6_comb.sum(axis=1, keepdims=True)
    y_pred_p6_comb = y_probs_p6_comb.argmax(axis=1)
    acc_p6_comb = (y_true_p6 == y_pred_p6_comb).mean() * 100
    f1_p6_comb = f1_score(y_true_p6, y_pred_p6_comb, average='macro', zero_division=0) * 100
    print(f"After combined: P6 Acc={acc_p6_comb:.2f}%, F1={f1_p6_comb:.2f}%")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Default P6:      Acc={acc_p6:.2f}%, F1={f1_p6:.2f}%")
    print(f"Temp scale:      Acc={acc_p6_temp:.2f}%, F1={f1_p6_temp:.2f}% (temp={best_temp})")
    print(f"Bias correction: Acc={acc_p6_corr:.2f}%, F1={f1_p6_corr:.2f}%")
    print(f"Combined:        Acc={acc_p6_comb:.2f}%, F1={f1_p6_comb:.2f}%")
    
    # Save calibration params
    calib = {
        'temperature': best_temp,
        'bias_corrections': corrections.tolist()
    }
    with open(os.path.join(SCRIPT_DIR, 'fold_P6', 'calibration_params.json'), 'w') as f:
        json.dump(calib, f)
    print(f"\nSaved calibration params to fold_P6/calibration_params.json")


if __name__ == '__main__':
    main()