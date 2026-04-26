
import json
import glob
import os
import re

BASE_DIR = '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging'

with open('/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/ItS2CLR_trial/plate_well_id_path.json') as f:
    plate_data = json.load(f)

plate_maps = {}
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_maps[plate] = {}
    for row, wells in plate_data[plate].items():
        for col, info in wells.items():
            well = f'{row}{int(col):02d}'
            plate_maps[plate][well] = info['id']

def extract_well(fn):
    m = re.search(r'Well(\w\d+)_', fn)
    return m.group(1) if m else None

plate_dir = os.path.join(BASE_DIR, 'P2')
paths = glob.glob(os.path.join(plate_dir, '**', '*.tif'), recursive=True)
print('glob found:', len(paths))

sample = [extract_well(os.path.basename(p)) for p in paths[:5]]
print('sample wells:', sample)

pm_keys = list(plate_maps['P2'].keys())[:10]
print('plate_maps[P2] sample keys:', pm_keys)

# Count matches
valid = 0
for p in paths:
    w = extract_well(os.path.basename(p))
    if w and w in plate_maps['P2']:
        valid += 1
print('valid paths:', valid)
