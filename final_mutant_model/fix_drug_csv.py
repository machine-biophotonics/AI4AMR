#!/usr/bin/env python3
"""Fix ground truth in drug predictions CSV using plate_well_ic50_mapping.json"""
import pandas as pd
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Fix drug predictions CSV ground truth')
    parser.add_argument('--csv', type=str, 
                        default='drug/fold_Plate_1/predictions_all_crops_mil_best_model_acc_n3.csv',
                        help='Path to predictions CSV')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: overwrite input)')
    args = parser.parse_args()
    
    csv_path = args.csv
    if not csv_path.startswith('/'):
        csv_path = f'/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/final_mutant_model/{csv_path}'
    
    output_path = args.output if args.output else csv_path
    
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Rows: {len(df)}, NaN ground_truth: {df['ground_truth_label'].isna().sum()}")
    
    # Load IC50 mapping
    ic50_path = '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/final_mutant_model/plate_well_ic50_mapping.json'
    with open(ic50_path, 'r') as f:
        ic50_data = json.load(f)
    
    def fix_ground_truth(row):
        if pd.notna(row['ground_truth_label']):
            return row['ground_truth_label']
        
        plate = row.get('plate', '')
        well = row.get('well', '')
        
        if not plate or not well:
            return None
        
        # Convert plate: 'Plate_1' -> 'P1'
        if 'Plate_' in plate:
            plate_key = f"P{plate.split('_')[-1]}"
        else:
            plate_key = plate
        
        # Look up in IC50 data
        if plate_key in ic50_data and well in ic50_data[plate_key]:
            info = ic50_data[plate_key][well]
            antibiotic = info.get('antibiotic', '')
            ic50_multiple = info.get('ic50_multiple', '')
            if antibiotic and ic50_multiple:
                if ic50_multiple == 'control':
                    return 'control'
                ic50_str = ic50_multiple if 'x' in str(ic50_multiple) else f"{ic50_multiple}x"
                antibiotic_clean = antibiotic.replace(' ', '_')
                return f"{antibiotic_clean}_{ic50_str}"
        
        return None
    
    df['ground_truth_label'] = df.apply(fix_ground_truth, axis=1)
    
    print(f"After fix - NaN ground_truth: {df['ground_truth_label'].isna().sum()}")
    print(f"Valid GT: {df['ground_truth_label'].notna().sum()}")
    
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
    
    # Calculate majority voting accuracy
    print("\nCalculating majority voting accuracy...")
    image_votes = {}
    for _, row in df.iterrows():
        img = row['image_path']
        if img not in image_votes:
            image_votes[img] = {}
        pred = row['predicted_class_name']
        image_votes[img][pred] = image_votes[img].get(pred, 0) + 1
    
    correct_images = 0
    total_images = 0
    for img, votes in image_votes.items():
        majority_pred = max(votes, key=votes.get)
        gt = df[df['image_path'] == img]['ground_truth_label'].iloc[0]
        if pd.notna(gt):
            total_images += 1
            if majority_pred == gt:
                correct_images += 1
    
    if total_images > 0:
        maj_accuracy = correct_images / total_images
        print(f"\nMajority voting accuracy: {maj_accuracy:.4f} ({correct_images}/{total_images})")

if __name__ == '__main__':
    main()