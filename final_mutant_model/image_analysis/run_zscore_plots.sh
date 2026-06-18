#!/bin/bash
# Run all z-score analyses and generate plots
set -e

IA="/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/final_mutant_model/image_analysis"
cd "$IA"

ZCSV="output_all_plates/all_plates_features_zscore.csv"
ZOUT="output_all_plates/zscore_plots"
mkdir -p "$ZOUT"

# Step 1: Generate z-score normalized features CSV
echo "=== Step 1: Generate z-score features ==="
python3 save_features_zscore.py

# Step 2: Run plot scripts
echo ""
echo "=== Step 2: roc_center1128 + roc_region_comparison ==="
python3 plot_roc_stats.py --input "$ZCSV" --output "$ZOUT"

echo ""
echo "=== Step 3: roc_mv_center1128 ==="
python3 plot_roc_mv.py --input "$ZCSV" --output "$ZOUT"

echo ""
echo "=== Step 4: roc_crossfold ==="
python3 plot_sig_crossfold.py --input "$ZCSV" --output "$ZOUT"

echo ""
echo "=== Step 5: roc_feature_ablation_error ==="
python3 plot_feature_ablation_error.py --input "$ZCSV" --output "$ZOUT" --region center1128

# Step 6: also run ablation for center224 for comparison
python3 plot_feature_ablation_error.py --input "$ZCSV" --output "$ZOUT" --region center224

echo ""
echo "=== Done. Output files in: $ZOUT ==="
ls -lh "$ZOUT"/*.png 2>/dev/null || echo "No PNG files found"
