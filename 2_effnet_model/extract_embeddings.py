import argparse
import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn
import torchvision
from torchvision import models
from torchvision.transforms import ToTensor, Normalize, Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomChoice, ColorJitter, Lambda
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import json
import re
import random
import numpy as np
from tqdm import tqdm

BASE_DIR = '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging'

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.set_num_threads(16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument('--n_crops', type=int, default=144, help='Number of crops per image')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
args = parser.parse_args()

EFFNET_DIR = '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging'

with open(os.path.join(BASE_DIR, 'plate maps', 'plate_well_id_path.json'), 'r') as f:
    plate_data = json.load(f)

plate_maps = {}
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_maps[plate] = {}
    for row, wells in plate_data[plate].items():
        for col, info in wells.items():
            well = f"{row}{int(col):02d}"
            plate_maps[plate][well] = info['id']


def extract_well_from_filename(filename: str):
    match = re.search(r'Well(\w\d+)_', filename)
    if match:
        return match.group(1)
    return None


all_labels = sorted(set(label for pm in plate_maps.values() for label in pm.values()))
label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
num_classes = len(all_labels)
print(f"Number of classes: {num_classes}")


def get_label_from_path(img_path: str):
    dirname = os.path.dirname(img_path)
    plate = os.path.basename(dirname)
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if plate in plate_maps and well in plate_maps[plate]:
        return plate_maps[plate][well]
    return None


class ShuffledCropSampler:
    def __init__(self, total_crops: int, shuffle: bool = True):
        self.indices = list(range(total_crops))
        self.shuffle = shuffle
        if shuffle:
            random.shuffle(self.indices)
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)


class MixedCropDataset(Dataset):
    def __init__(self, image_paths: list, labels: list, 
                 crop_size: int = 224, grid_size: int = 12,
                 n_crops_per_image: int = 144, augment: bool = False):
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.augment = augment
        
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        
        self.total_w = w - crop_size
        self.total_h = h - crop_size
        self.step_w = self.total_w / (grid_size - 1) if grid_size > 1 else 0
        self.step_h = self.total_h / (grid_size - 1) if grid_size > 1 else 0
        
        self.crop_positions = []
        for img_idx in range(len(image_paths)):
            for i in range(grid_size):
                for j in range(grid_size):
                    left = int(j * self.step_w)
                    top = int(i * self.step_h)
                    self.crop_positions.append((img_idx, left, top))
        
        self.n_crops_per_image = min(n_crops_per_image, grid_size * grid_size)
        
        if self.n_crops_per_image < grid_size * grid_size:
            if self.n_crops_per_image == 9:
                center_start = grid_size // 2 - 1
                center_end = grid_size // 2 + 2
                indices = []
                for i in range(center_start, center_end):
                    for j in range(center_start, center_end):
                        indices.append(i * grid_size + j)
            else:
                indices = random.sample(range(grid_size * grid_size), self.n_crops_per_image)
            
            self.crop_positions = []
            for img_idx in range(len(image_paths)):
                for pos_idx in indices:
                    i = pos_idx // grid_size
                    j = pos_idx % grid_size
                    left = int(j * self.step_w)
                    top = int(i * self.step_h)
                    self.crop_positions.append((img_idx, left, top))
        
        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.flip_h = RandomHorizontalFlip(p=0.5)
        self.flip_v = RandomVerticalFlip(p=0.5)
        self.rotate_90 = RandomChoice([
            Lambda(lambda x: x.rotate(0)),
            Lambda(lambda x: x.rotate(90)),
            Lambda(lambda x: x.rotate(180)),
            Lambda(lambda x: x.rotate(270)),
        ])
        self.color = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
    
    def __len__(self):
        return len(self.crop_positions)
    
    def __getitem__(self, idx):
        img_idx, left, top = self.crop_positions[idx]
        
        img = Image.open(self.image_paths[img_idx]).convert('RGB')
        crop = img.crop((left, top, left + self.crop_size, top + self.crop_size))
        
        if self.augment:
            crop = self.flip_h(crop)
            crop = self.flip_v(crop)
            crop = self.rotate_90(crop)
            crop = self.color(crop)
        
        crop = self.to_tensor(crop)
        crop = self.normalize(crop)
        
        return crop, self.labels[img_idx], (img_idx, left, top, os.path.basename(self.image_paths[img_idx]))


train_paths = []
for p in ['P1', 'P2', 'P3', 'P4']:
    train_paths.extend(glob.glob(os.path.join(EFFNET_DIR, p, '*.tif')))

val_paths = glob.glob(os.path.join(EFFNET_DIR, 'P5', '*.tif'))
test_paths = glob.glob(os.path.join(EFFNET_DIR, 'P6', '*.tif'))

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")


def get_label_for_path(img_path):
    label_str = get_label_from_path(img_path)
    if label_str is None:
        label_str = "Unknown"
    return label_to_idx.get(label_str, 0)

test_labels = [get_label_for_path(p) for p in test_paths]

test_data = MixedCropDataset(
    test_paths, test_labels,
    crop_size=224, grid_size=12,
    n_crops_per_image=args.n_crops, augment=False
)

print(f"Test: {len(test_data)} crops ({args.n_crops} crops/image)")


def worker_init_fn(worker_id: int) -> None:
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def collate_fn(batch):
    crops = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    metadata = [item[2] for item in batch]
    return crops, labels, metadata

test_loader = DataLoader(
    test_data, 
    batch_size=args.batch_size,
    sampler=ShuffledCropSampler(len(test_data), shuffle=False),
    shuffle=False, 
    num_workers=4, 
    pin_memory=True, 
    prefetch_factor=2, 
    persistent_workers=False,
    worker_init_fn=worker_init_fn,
    collate_fn=collate_fn
)


class EfficientNetWithEmbeddings(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x


model = EfficientNetWithEmbeddings(num_classes=num_classes)
model = model.to(device)

checkpoint = torch.load(os.path.join(EFFNET_DIR, 'effnet_model', 'best_model.pth'), map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded model from best_model.pth")

model.eval()


def extract_embeddings_incremental(model, loader, device, output_dir, save_every=5000):
    model.eval()
    all_embeddings = []
    all_labels = []
    metadata = []
    total_samples = 0
    
    pbar = tqdm(loader, desc="Extracting embeddings", leave=False)
    with torch.no_grad():
        for crops_batch, labels, meta in pbar:
            crops_batch = crops_batch.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                embeddings = model.get_embedding(crops_batch)
            
            embeddings_np = embeddings.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            batch_size = embeddings_np.shape[0]
            for i in range(batch_size):
                meta_info = meta[i]
                if isinstance(meta_info, tuple) and len(meta_info) == 4:
                    img_idx, left, top, filename = meta_info
                    meta_info = {
                        'img_idx': int(img_idx),
                        'left': int(left),
                        'top': int(top),
                        'filename': str(filename)
                    }
                
                all_embeddings.append(embeddings_np[i])
                all_labels.append(labels_np[i])
                metadata.append(meta_info)
                total_samples += 1
            
            if total_samples % save_every < args.batch_size:
                emb_arr = np.array(all_embeddings)
                lbl_arr = np.array(all_labels)
                np.save(os.path.join(output_dir, 'test_embeddings_partial.npy'), emb_arr)
                np.save(os.path.join(output_dir, 'test_labels_partial.npy'), lbl_arr)
                with open(os.path.join(output_dir, 'metadata_partial.json'), 'w') as f:
                    json.dump(metadata, f)
                print(f"\nCheckpoint saved: {total_samples} samples")
    
    return np.array(all_embeddings), np.array(all_labels), metadata


output_dir = os.path.join(BASE_DIR, 'effnet_model', 'eval_results')
os.makedirs(output_dir, exist_ok=True)

print("\nExtracting embeddings from test set...")
test_embeddings, test_labels_arr, image_metadata = extract_embeddings_incremental(
    model, test_loader, device, output_dir, save_every=10000
)

print(f"Embeddings shape: {test_embeddings.shape}")
print(f"Labels shape: {test_labels_arr.shape}")

np.save(os.path.join(output_dir, 'test_embeddings.npy'), test_embeddings)
np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels_arr)
print(f"\nSaved embeddings to {output_dir}/test_embeddings.npy")

with open(os.path.join(output_dir, 'idx_to_label.json'), 'w') as f:
    json.dump(idx_to_label, f)

if os.path.exists(os.path.join(output_dir, 'metadata_partial.json')):
    os.remove(os.path.join(output_dir, 'metadata_partial.json'))
if os.path.exists(os.path.join(output_dir, 'test_embeddings_partial.npy')):
    os.remove(os.path.join(output_dir, 'test_embeddings_partial.npy'))
if os.path.exists(os.path.join(output_dir, 'test_labels_partial.npy')):
    os.remove(os.path.join(output_dir, 'test_labels_partial.npy'))

def extract_well_plate_from_filename(filename):
    match = re.search(r'Well(\w\d+)_', filename)
    if match:
        well = match.group(1)
        row = well[0]
        col = int(well[1:])
        return well, row, col
    return None, None, None

crop_to_image_mapping = {}
for idx, meta in enumerate(image_metadata):
    if isinstance(meta, dict):
        filename = meta.get('filename', '')
    else:
        filename = str(meta)
    
    well, row, col = extract_well_plate_from_filename(filename)
    
    crop_to_image_mapping[idx] = {
        'filename': filename,
        'well': well,
        'row': row,
        'col': col,
        'img_idx': meta.get('img_idx', -1) if isinstance(meta, dict) else -1,
        'crop_position': (meta.get('left', -1), meta.get('top', -1)) if isinstance(meta, dict) else (-1, -1)
    }

with open(os.path.join(output_dir, 'crop_to_image_mapping.json'), 'w') as f:
    json.dump(crop_to_image_mapping, f, indent=2)
print(f'Saved crop-to-image mapping to eval_results/crop_to_image_mapping.json')

print("\nDone!")
