import torch
from torch.utils.data import Dataset
import numpy as np
import random
import re
import os
import glob
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


def extract_well_from_filename(filename: str) -> str | None:
    match = re.search(r'Well(\w\d+)_', filename)
    return match.group(1) if match else None


def extract_plate_from_path(path: str) -> str:
    match = re.search(r'/P(\d+)(?:/|$)', path)
    if not match:
        match = re.search(r'P(\d+)', path)
    return f"P{match.group(1)}"


def load_labels(project_root: str, data_dir: str) -> tuple[list, dict, list]:
    """Load all images with their 185 class labels (drugs + mutants combined).

    Returns:
        image_list: list of (path, class_id)
        class_names: list of class name strings
        id_to_name: dict[int -> class name]
    """
    ic50_path = os.path.join(data_dir, 'plate_well_ic50_mapping.json')
    mutant_path = os.path.join(data_dir, 'plate_well_id_path.json')

    with open(ic50_path) as f:
        ic50_data = json.load(f)
    with open(mutant_path) as f:
        mutant_data = json.load(f)

    plate_maps = {}
    for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
        plate_maps[plate] = {}

        if plate in ic50_data:
            for well, info in ic50_data[plate].items():
                antibiotic = info.get('antibiotic', '')
                ic50_m = info.get('ic50_multiple', '')
                if antibiotic and ic50_m:
                    if ic50_m == 'control':
                        drug_class = 'control'
                    else:
                        ic50_str = ic50_m if 'x' in ic50_m else f"{ic50_m}x"
                        drug_class = f"{antibiotic.replace(' ', '_')}_{ic50_str}"
                    plate_maps[plate][f"drug_{well}"] = drug_class

        if plate in mutant_data:
            for row, cols in mutant_data[plate].items():
                for col, cinfo in cols.items():
                    if 'id' in cinfo:
                        well = f"{row}{int(col):02d}"
                        plate_maps[plate][f"mutant_{well}"] = cinfo['id']

    all_labels = sorted(set(
        v for pm in plate_maps.values() for v in pm.values()
    ))
    label_to_idx = {l: i for i, l in enumerate(all_labels)}

    image_list = []
    for pi in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
        for cond, stype in [('Drugs_Data', 'drug'), ('Mutants_Data', 'mutant')]:
            d = os.path.join(project_root, cond, pi)
            if not os.path.exists(d):
                continue
            for ext in ('*.tif', '*.tiff', '*.png'):
                for path in glob.glob(os.path.join(d, '**', ext), recursive=True):
                    well = extract_well_from_filename(os.path.basename(path))
                    if well is None:
                        continue
                    composite = f"{stype}_{well}"
                    if composite in plate_maps.get(pi, {}):
                        label = plate_maps[pi][composite]
                        image_list.append((path, label_to_idx[label]))

    return image_list, all_labels, label_to_idx


class FlowCropDataset(Dataset):
    """Single-crop dataset for flow matching.

    Each epoch selects a random 224x224 crop per image.
    Returns (crop_tensor, class_id).
    """
    def __init__(
        self,
        image_items: list,
        crop_size: int = 224,
        grid_size: int = 12,
        augment: bool = True,
        seed: int = 42,
        epoch: int = 0,
    ):
        self.image_items = image_items
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.augment = augment
        self.seed = seed
        self.epoch = epoch

        sample_img = Image.open(image_items[0][0])
        w, h = sample_img.size
        self.image_size = w

        stride = (w - crop_size) // (grid_size - 1)
        self.positions = [(j * stride, i * stride)
                          for i in range(grid_size) for j in range(grid_size)]
        self.num_positions = len(self.positions)

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=1.0),
            ToTensorV2(),
        ]) if augment else A.Compose([
            A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=1.0),
            ToTensorV2(),
        ])

        self.epoch_centers = {}
        self.set_epoch(epoch)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        num_img = len(self.image_items)
        num_pos = self.num_positions

        if not self.augment:
            center = (self.image_size - self.crop_size) // 2
            self.epoch_centers = {i: (center, center) for i in range(num_img)}
            return

        cycle = epoch // num_pos
        pos_in_cycle = epoch % num_pos
        rng = random.Random(self.seed + cycle)
        shuffled = self.positions.copy()
        rng.shuffle(shuffled)

        self.epoch_centers = {}
        for idx in range(num_img):
            assigned_idx = (idx + pos_in_cycle) % num_pos
            self.epoch_centers[idx] = shuffled[assigned_idx]

    def _load_mmap(self, path: str) -> np.ndarray:
        try:
            import tifffile
            return tifffile.memmap(path)
        except Exception:
            return np.array(Image.open(path))

    def __len__(self):
        return len(self.image_items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, class_id = self.image_items[idx]
        mmap = self._load_mmap(path)

        if mmap.ndim == 3:
            mmap = mmap[:, :, 0]

        cx, cy = self.epoch_centers[idx]
        crop = mmap[cy:cy + self.crop_size, cx:cx + self.crop_size]

        if mmap.dtype == np.uint16:
            crop = crop.astype(np.float32) / 65535.0
        elif mmap.dtype == np.uint8:
            crop = crop.astype(np.float32) / 255.0
        else:
            crop = crop.astype(np.float32)

        crop_tensor = self.transform(image=crop)['image']
        return crop_tensor, class_id
