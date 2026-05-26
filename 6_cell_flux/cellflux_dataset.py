"""CellFlux paired dataset — exact CellFlux control→perturbed pairing.

CellFlux (ICML 2025) data loading logic:
    For each perturbed (treated) sample, randomly pair with a control
    from the same plate (batch). This matches `read_files_pert` in the
    official CellFlux repository.

Transform matches CellFlux:
    (X + random_noise) / 255.0 → normalize to [-1, 1] → augment (H/V flip)
"""
import os, re, glob, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from PIL import Image


def load_labels(project_root, data_dir):
    """Load all images with class labels. Same as mil_model.load_labels."""
    from mil_model import load_labels as _load_labels
    return _load_labels(project_root, data_dir)


def _plate_from_path(path: str) -> str:
    m = re.search(r'/[Pp](\d)/', path)
    return f'P{m.group(1)}' if m else ''


CONTROL_NAMES = {
    'control',
    'NC_1', 'NC_2', 'NC_3', 'NC_4', 'NC_5', 'NC_6',
    'WT NC_1', 'WT NC_2', 'WT NC_3', 'WT NC_4', 'WT NC_5', 'WT NC_6',
}


class CellFluxTransform:
    """Exact transform from CellFlux CustomTransform.

    Steps:
        1. (X + random_noise) / 255.0  — add tiny noise to break ties
        2. Normalize to [-1, 1] via mean=0.5, std=0.5
        3. Optional: RandomHorizontalFlip(p=0.3), RandomVerticalFlip(p=0.3)
    """
    def __init__(self, augment=True, normalize=True):
        self.augment = augment
        self.normalize = normalize

    def __call__(self, img_np: np.ndarray) -> torch.Tensor:
        """img_np: [H, W] or [H, W, C] uint8/uint16 array."""
        random_noise = np.random.rand(*img_np.shape).astype(np.float32)
        if img_np.dtype == np.uint16:
            img_np = (img_np.astype(np.float32) + random_noise) / 65535.0
        elif img_np.dtype == np.uint8:
            img_np = (img_np.astype(np.float32) + random_noise) / 255.0
        else:
            img_np = img_np.astype(np.float32) + random_noise

        t = torch.from_numpy(img_np).float()
        if t.ndim == 2:
            t = t.unsqueeze(0)
        else:
            t = t.permute(2, 0, 1)

        if self.normalize:
            t = (t - 0.5) / 0.5

        if self.augment:
            if torch.rand(1).item() < 0.3:
                t = t.flip(-1)
            if torch.rand(1).item() < 0.3:
                t = t.flip(-2)

        return t


class CellFluxPairedDataset(Dataset):
    """Paired control→perturbed dataset — exact CellFlux pairing.

    For each perturbed sample, randomly pairs with a control from the same plate.

    Args:
        perturbed_items: list of (path, class_id) for perturbed samples
        control_items_by_plate: dict[plate -> list of (path, class_id)]
        perturbed_classes: set of perturbed class_ids (used for remapping to 0..N-1)
        augment: whether to apply augmentation
        crop_size: spatial crop size (default 224)
    """
    def __init__(self, perturbed_items, control_items_by_plate, perturbed_classes,
                 augment=True, crop_size=224):
        self.perturbed_items = perturbed_items
        self.ctrl_by_plate = control_items_by_plate
        self.pert2cond = {cid: i for i, cid in enumerate(sorted(perturbed_classes))}
        self.augment = augment
        self.crop_size = crop_size
        self.transform = CellFluxTransform(augment=augment, normalize=True)

    def __len__(self):
        return len(self.perturbed_items)

    def __getitem__(self, idx):
        trt_path, trt_class = self.perturbed_items[idx]
        plate = _plate_from_path(trt_path)

        ctrl_items = self.ctrl_by_plate[plate]
        ctrl_idx = np.random.randint(len(ctrl_items))
        ctrl_path, _ = ctrl_items[ctrl_idx]

        trt_img = self._load_crop(trt_path)
        ctrl_img = self._load_crop(ctrl_path)

        cond = self.pert2cond[trt_class]
        return ctrl_img, trt_img, cond

    def _load_crop(self, path):
        img = Image.open(path)
        w, h = img.size
        # Random crop of crop_size x crop_size
        if w > self.crop_size:
            x = np.random.randint(0, w - self.crop_size + 1)
        else:
            x = 0
        if h > self.crop_size:
            y = np.random.randint(0, h - self.crop_size + 1)
        else:
            y = 0
        crop = img.crop((x, y, x + self.crop_size, y + self.crop_size))
        arr = np.array(crop)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return self.transform(arr)


def build_datasets(
    project_root,
    data_dir,
    train_plates=None,
    val_split=0.2,
    test_plate=None,
    seed=42,
):
    """Build train/val/test datasets with CellFlux-style pairing.

    CellFlux data split:
        - Train: perturbed images from train plates, paired with controls from same plates
        - Val: held-out perturbed images (per-class stratified), paired with controls
        - Test: perturbed images from test_plate (if specified), paired with controls

    Returns:
        train_ds, val_ds, test_ds: CellFluxPairedDataset instances
        num_pert_classes: number of perturbed classes (172)
        class_names: full 185 class names
        pert2cond: mapping from class_id to condition index (0..171)
    """
    image_list, class_names, label_to_idx = load_labels(project_root, data_dir)
    control_ids = {label_to_idx[n] for n in CONTROL_NAMES}
    perturbed_ids = set(range(len(class_names))) - control_ids
    num_pert_classes = len(perturbed_ids)

    rng = np.random.RandomState(seed)

    if test_plate is not None and train_plates is None:
        train_plates = [p for p in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6'] if p != test_plate]

    if train_plates is not None:
        all_train_items = [(p, c) for p, c in image_list if _plate_from_path(p) in train_plates]
        test_items = [(p, c) for p, c in image_list if _plate_from_path(p) == test_plate] if test_plate else []
    else:
        all_train_items = list(image_list)
        test_items = []

    # Separate control/perturbed
    pert_items = [(p, c) for p, c in all_train_items if c in perturbed_ids]
    ctrl_items = [(p, c) for p, c in all_train_items if c in control_ids]

    ctrl_by_plate = defaultdict(list)
    for p, c in ctrl_items:
        ctrl_by_plate[_plate_from_path(p)].append((p, c))

    # Per-class stratified split of perturbed items
    class_to_items = defaultdict(list)
    for p, c in pert_items:
        class_to_items[c].append((p, c))

    train_pert, val_pert = [], []
    for c, items in class_to_items.items():
        perm = rng.permutation(len(items))
        n_val = max(1, int(len(items) * val_split))
        val_pert.extend(items[idx] for idx in perm[:n_val])
        train_pert.extend(items[idx] for idx in perm[n_val:])

    train_ds = CellFluxPairedDataset(train_pert, ctrl_by_plate, perturbed_ids, augment=True)
    val_ds = CellFluxPairedDataset(val_pert, ctrl_by_plate, perturbed_ids, augment=False)

    test_ds = None
    if test_items:
        test_pert = [(p, c) for p, c in test_items if c in perturbed_ids]
        test_ctrl = [(p, c) for p, c in test_items if c in control_ids]
        test_ctrl_by_plate = defaultdict(list)
        for p, c in test_ctrl:
            test_ctrl_by_plate[_plate_from_path(p)].append((p, c))
        test_ds = CellFluxPairedDataset(test_pert, test_ctrl_by_plate, perturbed_ids, augment=False)

    pert2cond = {cid: i for i, cid in enumerate(sorted(perturbed_ids))}

    return train_ds, val_ds, test_ds, num_pert_classes, class_names, pert2cond
