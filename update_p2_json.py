import json
import csv
import glob
import os
import re

# Load existing JSON
with open('plate_well_id_path.json', 'r') as f:
    plate_data = json.load(f)

# Load new P2 plate map from CSV
p2_map = {}
rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
with open('P2_plate_map.csv', 'r') as f:
    reader = csv.reader(f)
    for row_idx, row in enumerate(reader):
        for col_idx, label in enumerate(row):
            well = f"{rows[row_idx]}{col_idx+1}"
            p2_map[well] = label.strip()

print("P2 plate map loaded:")
for well, label in sorted(p2_map.items()):
    print(f"  {well}: {label}")

# Get P2 image paths
p2_dir = '/media/student/My Passport1/AI4AMR/2025_12_19 CRISPRi Reference Plate Imaging/P2'
p2_images = glob.glob(os.path.join(p2_dir, '*.tif'))

# Group images by well
well_to_images = {}
for img_path in p2_images:
    filename = os.path.basename(img_path)
    match = re.search(r'Well(\w\d+)_', filename)
    if match:
        well = match.group(1)
        if well not in well_to_images:
            well_to_images[well] = []
        well_to_images[well].append(img_path)

print(f"\nFound {len(p2_images)} images in P2")
print(f"Wells with images: {len(well_to_images)}")

# Update P2 in JSON with one image per well
plate_data['P2'] = {}
for well in sorted(p2_map.keys()):
    label = p2_map[well]
    if well in well_to_images and well_to_images[well]:
        img_path = well_to_images[well][0]
    else:
        img_path = ""
    
    row = well[0]
    col = well[1:]
    
    if row not in plate_data['P2']:
        plate_data['P2'][row] = {}
    plate_data['P2'][row][col] = {
        "id": label,
        "path": img_path
    }

# Save updated JSON
with open('plate_well_id_path.json', 'w') as f:
    json.dump(plate_data, f, indent=4)

print("\nUpdated plate_well_id_path.json with P2 plate map")
print(f"P2 now has {len(p2_map)} wells")
