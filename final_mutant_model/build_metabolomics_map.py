import csv
import json
import os
import re

DATA_DIR = "/media/student/Data_HDD_12-TB/all of felix data/Metabolomics_Data/Mutants"
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plate_well_metabonomics_mapping.json")

ROW_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]
PLATE_NUMS = [1, 2, 3, 4]
TIMEPOINTS = [1, 2, 3]

def read_scrambled_csv(filepath):
    """Parse 8-row x 12-col CSV into {row_letter: {col_str: {id: value}}}."""
    rows = {}
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= 8:
                break
            row_letter = ROW_LETTERS[i]
            cols = {}
            for j, val in enumerate(row):
                if j >= 12:
                    break
                col_str = str(j + 1)
                cols[col_str] = {"id": val.strip()}
            rows[row_letter] = cols
    return rows

def main():
    result = {}
    for p in PLATE_NUMS:
        csv_path = os.path.join(DATA_DIR, f"scrambled_plate_map_20260717_101617_P{p}.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping plate P{p}")
            continue
        plate_layout = read_scrambled_csv(csv_path)
        for t in TIMEPOINTS:
            key = f"P{p}_T{t}"
            result[key] = plate_layout
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Written {len(result)} plate-timepoint entries to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
