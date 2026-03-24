import json
from typing import Optional

# Load from JSON
with open('plate_well_id_path.json', 'r') as f:
    plate_data = json.load(f)

all_labels = set()

for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    if plate in plate_data:
        for row in plate_data[plate].values():
            for well_data in row.values():
                if 'id' in well_data:
                    all_labels.add(well_data['id'])

all_labels: list[str] = sorted(all_labels)
label_to_idx: dict[str, int] = {label: idx for idx, label in enumerate(all_labels)}
idx_to_label: dict[int, str] = {idx: label for label, idx in label_to_idx.items()}
num_classes: int = len(all_labels)

print(f"Number of classes: {num_classes}")

with open('classes.txt', 'w') as f:
    for i, label in enumerate(all_labels):
        f.write(f"{i},{label}\n")
print("Classes saved to classes.txt")

print("\nClasses:")
for i, label in enumerate(all_labels):
    print(f"  {i}: {label}")
