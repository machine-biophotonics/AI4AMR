import json
import csv
import glob
import os
import re

# Load existing JSON
with open('plate_well_id_path.json', 'r') as f:
    plate_data = json.load(f)

# Load new P3 plate map from CSV
p3_map = {}
rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
with open('P3_plate_map.csv', 'r') as f:
    reader = csv.reader(f)
    for row_idx, row in enumerate(reader):
        for col_idx, label in enumerate(row):
            well = f"{rows[row_idx]}{col_idx+1}"
            p3_map[well] = label.strip()

print("P3 plate map loaded:")
for well, label in sorted(p3_map.items()):
    print(f"  {well}: {label}")

# Get P3 image paths
p3_dir = '/media/student/My Passport1/AI4AMR/2025_12_19 CRISPRi Reference Plate Imaging/P3'
p3_images = glob.glob(os.path.join(p3_dir, '*.tif'))

# Group images by well
well_to_images = {}
for img_path in p3_images:
    filename = os.path.basename(img_path)
    match = re.search(r'Well(\w\d+)_', filename)
    if match:
        well = match.group(1)
        if well not in well_to_images:
            well_to_images[well] = []
        well_to_images[well].append(img_path)

print(f"\nFound {len(p3_images)} images in P3")
print(f"Wells with images: {len(well_to_images)}")

# Update P3 in JSON with one image per well
plate_data['P3'] = {}
for well in sorted(p3_map.keys()):
    label = p3_map[well]
    if well in well_to_images and well_to_images[well]:
        # Use first image for this well
        img_path = well_to_images[well][0]
    else:
        img_path = ""
    
    row = well[0]
    col = well[1:]
    
    if row not in plate_data['P3']:
        plate_data['P3'][row] = {}
    plate_data['P3'][row][col] = {
        "id": label,
        "path": img_path
    }

# Save updated JSON
with open('plate_well_id_path.json', 'w') as f:
    json.dump(plate_data, f, indent=4)

print("\nUpdated plate_well_id_path.json with P3 plate map")
print(f"P3 now has {len(p3_map)} wells")
