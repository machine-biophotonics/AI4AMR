#!/usr/bin/env python3
# Must be set before any imports to suppress inductor SM warning
import warnings
warnings.filterwarnings("ignore", message=".*Not enough SMs to use max_autotune_gemm.*")

import os
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "ATEN,CPP"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCH_CUDNN_DETERMINISTIC"] = "1"

"""
MIL training with cycle-based crop extraction + neighbors
Training: configurable crops (3x3, 5x5, 7x7, 9x9, 11x11 neighborhood)
Validation/Test: single center crop
Supports --run_all_folds for cross-validation
"""

import argparse
import sys
import time
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import glob
import json
import re
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
import random
from tqdm import tqdm
import csv
from datetime import datetime
from collections import Counter
import multiprocessing
from functools import partial

from mil_model import AttentionMILModel, MILEncoder, MultiCropDataset, get_gene_from_path, extract_well_from_filename
from supcon_loss import SupConLoss, SupConLossMIL
from torch.utils.tensorboard import SummaryWriter

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Disable inductor max_autotune_gemm at runtime to avoid SM warning on small GPUs
import torch._inductor.config as inductor_config
inductor_config.max_autotune_gemm = False
inductor_config.max_autotune_gemm_backends = "ATEN,CPP"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--prefetch_factor', type=int, default=32,
                    help='Number of batches to prefetch per worker (default: same as batch_size)')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--test_plate', type=str, default='Plate_6')
parser.add_argument('--data_root', type=str, default=None, help='Path to folder containing P1-P6 plate folders')
parser.add_argument('--run_all_folds', action='store_true', default=False, help='Run all 6 folds')
parser.add_argument('--neighborhood', type=int, default=3, choices=[3, 5, 7, 9, 11],
                    help='Neighborhood size: 3=(3x3=9 crops), 5=(5x5=25 crops, recommended)')
parser.add_argument('--grid_size', type=int, default=12,
                    help='Grid size for crop positions')
parser.add_argument('--extraction_mode', type=str, default='neighborhood', choices=['neighborhood', 'raster'],
                    help='Crop extraction mode: neighborhood (N×N grids around positions) or raster (all crops in tiling grid)')
parser.add_argument('--raster_crop_size', type=int, default=500,
                    help='Crop size for raster mode extraction (default 500)')
parser.add_argument('--raster_resize_size', type=int, default=256,
                    help='Resize raster crops to this size for model input (default 256)')
parser.add_argument('--raster_num_crops', type=int, default=25,
                    help='Number of crops to extract in raster mode (default 25, 5x5 grid)')
parser.add_argument('--raster_grid_size', type=int, default=2500,
                    help='Grid size for raster mode - centered on image (default 2500)')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate for classifier (default 0.5 for stronger regularization)')
parser.add_argument('--weight_decay', type=float, default=0.05,
                    help='Weight decay (default 0.05 for stronger regularization)')
parser.add_argument('--pooling', type=str, default='attention', choices=['attention', 'simple_attention', 'mean', 'max'],
                    help='MIL pooling method: attention (gated), simple_attention (no gating), mean (average), max (max)')
parser.add_argument('--label_smoothing', type=float, default=0.1,
                    help='Label smoothing (default 0.1, helps with small datasets)')
parser.add_argument('--entropy_loss_weight', type=float, default=0.01,
                    help='Attention entropy loss weight (default 0.01, AEM regularization for MIL)')
parser.add_argument('--use_contrastive', action='store_true',
                    help='Use patch-level contrastive pre-training')
parser.add_argument('--use_sc_mil', action='store_true', default=True,
                    help='Use SC-MIL: supervised contrastive + classification joint training (recommended)')
parser.add_argument('--sc_mil_epochs', type=int, default=200,
                    help='Epochs for SC-MIL joint training (default 200)')
parser.add_argument('--sc_mil_weight', type=float, default=0.3,
                    help='Weight for SC-MIL contrastive loss vs classification (0.1-1.0)')
parser.add_argument('--sc_mil_temp', type=float, default=0.07,
                    help='Temperature for SC-MIL contrastive loss')
parser.add_argument('--contrastive_level', type=str, default='bag', choices=['instance', 'bag', 'both'],
                    help='Contrastive level: instance (crop), bag (pooled), or both')
parser.add_argument('--instance_weight', type=float, default=0.5,
                    help='Weight for instance-level loss vs bag-level (0.0-1.0)')
parser.add_argument('--warmup_epochs', type=int, default=None,
                    help='Warmup epochs (default: 5%% of epochs, i.e. 10 for 200)')
parser.add_argument('--checkpoint_every', type=int, default=1,
                    help='Save checkpoint every N epochs (default: 1)')
parser.add_argument('--num_channels', type=int, default=1,
                    help='Number of input channels (1 for grayscale, 3 for RGB)')
parser.add_argument('--backbone', type=str, default='efficientnet_b0', choices=['efficientnet_b0', 'mobilenet_v3_small', 'mobilenet_v2'],
                    help='Backbone architecture: efficientnet_b0 (default), mobilenet_v3_small, or mobilenet_v2')
parser.add_argument('--pretrained', type=str, default='micronet', choices=['imagenet', 'micronet'], 
                    help='Pretrained weights: imagenet or micronet (NASA microscopy pretrained, default)')
parser.add_argument('--framework', type=str, default='pytorch', choices=['pytorch', 'tensorflow'],
                    help='Framework: pytorch (default) or tensorflow/keras')
parser.add_argument('--data_mode', type=str, default='mutant', choices=['drug', 'mutant', 'both', 'control'],
                    help='Data mode: drug (drug+concentration), mutant (gene/mutant), both (combine)')
parser.add_argument('--drug_no_concentration', action='store_true', default=False,
                    help='Group drugs by antibiotic name only, ignoring concentration levels (e.g., Ciprofloxacin instead of Ciprofloxacin_2x)')
parser.add_argument('--freeze', action='store_true', default=False,
                    help='Freeze backbone, only train attention pool + classifier head')
parser.add_argument('--guide', type=int, default=None,
                    help='Filter to specific guide number (e.g. 1 for guide 1) in mutant mode')
parser.add_argument('--edge_sigma', type=float, default=None,
                    help='Apply Canny edge detection with given sigma before normalization (e.g., 2.0)')
parser.add_argument('--dual_classifier', action='store_true', default=False,
                    help='Use separate classifiers for drug and mutant domains (only with --data_mode both)')
parser.add_argument('--proj_dim', type=int, default=256,
                    help='Projection bottleneck dimension for dual classifier (default: 256)')
parser.add_argument('--sym_consistency_weight', type=float, default=0.5,
                    help='Weight for symmetric consistency loss (pull same + push different, default: 0.5)')
parser.add_argument('--ce_weight', type=float, default=1.0,
                    help='Weight for classification loss (0=disable CE/focal, train with other losses only)')
parser.add_argument('--force', action='store_true', default=False,
                    help='Skip GPU double-launch check (allow multiple training instances)')
parser.add_argument('--resume', action='store_true', default=False,
                    help='Resume training from checkpoint_epoch.pth in output directory')
args = parser.parse_args()

# ── Guard against accidentally launching two training instances ──────────────
if not args.force and torch.cuda.is_available():
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        other_pids = []
        my_pid = os.getpid()
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 2 and 'python' in parts[1].strip().lower():
                pid = parts[0].strip()
                if pid and pid != str(my_pid):
                    # Check if this is a DataLoader child process, not a separate instance
                    try:
                        ppid = int(open(f'/proc/{pid}/stat').read().split()[3])
                    except (IndexError, IOError, ValueError):
                        ppid = -1
                    if ppid != my_pid:  # not a child worker → separate instance
                        other_pids.append(pid)
        if other_pids:
            print(f"\nWARNING: Another Python process (PID {' '.join(other_pids)}) "
                  f"is already using the GPU.")
            print("  If you meant to run two training instances, use --force to skip this check.\n")
            sys.exit(1)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass  # nvidia-smi not available, proceed anyway
# ─────────────────────────────────────────────────────────────────────────────

# Determine folder name for results (drug_noconcentration vs drug)
data_mode_folder = args.data_mode
if args.data_mode == 'drug' and args.drug_no_concentration:
    data_mode_folder = 'drug_noconcentration'
if args.guide is not None:
    data_mode_folder = f"{args.data_mode}_guide_{args.guide}"
if args.edge_sigma is not None:
    data_mode_folder = f"canny_{data_mode_folder}"

if args.warmup_epochs is None:
    args.warmup_epochs = int(args.sc_mil_epochs * 0.05)  # 5% of SC-MIL training

# Set num_workers based on OS
if sys.platform.startswith('win'):
    NUM_WORKERS = 0  # Windows Python 3.14: multiprocessing spawn required
else:
    NUM_WORKERS = 16

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if args.data_root:
    BASE_DIR = args.data_root
elif args.data_mode == 'drug':
    BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Drugs_Data')
elif args.data_mode == 'control':
    BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Controls_Data')
else:
    BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Mutants_Data')

IC50_MAPPING_PATH = os.path.join(os.path.dirname(__file__), 'plate_well_ic50_mapping.json')
MUTANT_MAPPING_PATH = os.path.join(os.path.dirname(__file__), 'plate_well_id_path.json')
CONTROL_MAPPING_PATH = os.path.join(os.path.dirname(__file__), 'plate_well_control_id_path.json')

# Load drug mapping (antibiotic + concentration)
with open(IC50_MAPPING_PATH, 'r') as f:
    ic50_data = json.load(f)

# Load mutant mapping (gene IDs)
with open(MUTANT_MAPPING_PATH, 'r') as f:
    mutant_data = json.load(f)

# Load control mapping (strain + mutant + ATC condition)
with open(CONTROL_MAPPING_PATH, 'r') as f:
    control_data = json.load(f)

# Build plate_maps based on data_mode
# Use prefixes to distinguish drug vs mutant vs control (they share same well positions)
plate_key_map = {f'Plate_{i}': f'P{i}' for i in range(1, 7)}
plate_maps = {}
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_maps[plate] = {}
    if args.data_mode in ['drug', 'both']:
        if plate in ic50_data:
            for well, info in ic50_data[plate].items():
                antibiotic = info.get('antibiotic', '')
                ic50_multiple = info.get('ic50_multiple', '')
                if antibiotic and ic50_multiple:
                    if args.drug_no_concentration:
                        # Group by antibiotic name only (ignore concentration)
                        drug_class = antibiotic.replace(' ', '_')
                    else:
                        # Include concentration in class name
                        if ic50_multiple == 'control':
                            drug_class = 'control'
                        else:
                            ic50_str = ic50_multiple if 'x' in ic50_multiple else f"{ic50_multiple}x"
                            antibiotic_clean = antibiotic.replace(' ', '_')
                            drug_class = f"{antibiotic_clean}_{ic50_str}"
                    # Prefix with 'drug_' to avoid overwriting mutant data in same wells
                    plate_maps[plate][f"drug_{well}"] = drug_class
    
    if args.data_mode in ['mutant', 'both']:
        if plate in mutant_data:
            for row, cols in mutant_data[plate].items():
                for col, info in cols.items():
                    if 'id' in info:
                        well = f"{row}{int(col):02d}"  # Convert A, 1 -> A01 (2-digit format)
                        # Prefix with 'mutant_' to avoid overwriting drug data in same wells
                        plate_maps[plate][f"mutant_{well}"] = info['id']
    
    if args.data_mode in ['control']:
        # (a) Control plate data (Controls_Data/)
        if plate in control_data:
            for row, cols in control_data[plate].items():
                for col, info in cols.items():
                    if 'id' in info:
                        well = f"{row}{int(col):02d}"
                        plate_maps[plate][f"control_{well}"] = info['id']
        # (b) Mutant controls (NC_* and WT NC_* from Mutants_Data/)
        if plate in mutant_data:
            for row, cols in mutant_data[plate].items():
                for col, info in cols.items():
                    if 'id' in info:
                        well = f"{row}{int(col):02d}"
                        mid = info['id']
                        if mid.startswith('NC_') or mid.startswith('WT NC_'):
                            plate_maps[plate][f"mutant_{well}"] = mid
        # (c) Drug controls (DMSO/control wells from Drugs_Data/)
        if plate in ic50_data:
            for well, info in ic50_data[plate].items():
                if info.get('ic50_multiple') == 'control':
                    plate_maps[plate][f"drug_{well}"] = 'drug_control'

all_plates = ['Plate_1', 'Plate_2', 'Plate_3', 'Plate_4', 'Plate_5', 'Plate_6']

# For drug mode, plates are P1, P2, etc. in Drugs_Data folder
def get_image_paths_for_plate(plate: str) -> list[str]:
    # Convert Plate_1 -> P1 for directory lookup
    plate_key: str = f"P{plate.split('_')[-1]}"  # Plate_1 -> P1, P6 -> P6
    
    # Determine which directories to search based on data_mode
    search_dirs = []
    if args.data_mode == 'drug':
        base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Drugs_Data')
        search_dirs.append((os.path.join(base, plate_key), 'drug'))
    elif args.data_mode == 'mutant':
        base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Mutants_Data')
        search_dirs.append((os.path.join(base, plate_key), 'mutant'))
    elif args.data_mode == 'control':
        for base_name, prefix in [('Controls_Data', 'control'), ('Mutants_Data', 'mutant'), ('Drugs_Data', 'drug')]:
            base = os.path.join(os.path.dirname(os.path.dirname(__file__)), base_name)
            search_dirs.append((os.path.join(base, plate_key), prefix))
    else:  # both - search both directories
        drug_base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Drugs_Data')
        mutant_base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Mutants_Data')
        search_dirs.append((os.path.join(drug_base, plate_key), 'drug'))
        search_dirs.append((os.path.join(mutant_base, plate_key), 'mutant'))
    
    valid_paths = []
    for plate_dir, source_type in search_dirs:
        if not os.path.exists(plate_dir):
            continue
        
        paths = []
        for pattern in ['*.tif', '*.tiff', '*.png']:
            paths.extend(glob.glob(os.path.join(plate_dir, '**', pattern), recursive=True))
        
        # Add prefix based on source type (drug vs mutant)
        well_prefix = f"{source_type}_"
        
        for path in paths:
            well = extract_well_from_filename(os.path.basename(path))
            # Use composite key: drug_A01 or mutant_A01
            composite_well = f"{well_prefix}{well}"
            if composite_well and composite_well in plate_maps.get(plate_key, {}):
                valid_paths.append(path)
    
    return valid_paths

def compute_robust_auc(labels: list, probs: list, num_classes: int) -> float:
    """Compute ROC AUC with robust error handling."""
    import warnings
    warnings.filterwarnings('ignore')
    
    labels_np = np.array(labels)
    probs_np = np.array(probs)
    
    # Get unique classes in ground truth
    unique_classes = np.unique(labels_np)
    if len(unique_classes) < 2:
        return float('nan')
    
    # Filter to only classes present in ground truth (not all num_classes)
    labels_bin = label_binarize(labels_np, classes=unique_classes)
    probs_filtered = probs_np[:, unique_classes]
    
    try:
        auc = roc_auc_score(labels_bin, probs_filtered, average='macro')
        return float(auc)
    except Exception:
        try:
            auc = roc_auc_score(labels_bin, probs_filtered, average='weighted')
            return float(auc)
        except Exception:
            return float('nan')

def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    return (alpha * (1 - pt) ** gamma * ce_loss).mean()

def weighted_focal_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0, label_smoothing: float = 0.0) -> torch.Tensor:
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none', label_smoothing=label_smoothing)
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return (focal * weights).mean()

def attention_entropy_loss(attn_weights: torch.Tensor) -> torch.Tensor:
    """Attention entropy regularization - prevents attention over-concentration in MIL.
    Based on AEM paper (2024) - encourages considering more instances/patches."""
    return -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=1).mean()


def symmetric_consistency_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    emb_norm = F.normalize(embeddings, dim=1)
    sim = emb_norm @ emb_norm.T
    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    label_eq = label_eq & ~eye
    label_neq = ~label_eq & ~eye
    same_class_pairs = label_eq.sum()
    diff_class_pairs = label_neq.sum()
    if same_class_pairs < 1 or diff_class_pairs < 1:
        return torch.tensor(0.0, device=embeddings.device)
    pos = (sim * label_eq).sum() / same_class_pairs
    neg = (sim * label_neq).sum() / diff_class_pairs
    return (1.0 - pos) + (neg - margin).clamp(min=0)


def worker_init_fn(worker_id: int, seed: int = 42) -> None:
    """Module-level worker init function for multiprocessing compatibility"""
    import random
    import numpy as np
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


def train_single_fold(test_plate: str) -> None:
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, data_mode_folder, f'fold_{test_plate}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training fold: test_plate={test_plate}")
    print(f"{'='*60}")
    
    # Convert test_plate to Plate_X format for comparison (P6 -> Plate_6)
    if 'P' in test_plate.upper() and test_plate[-1].isdigit():
        test_plate_normalized = f"Plate_{test_plate[-1]}"
    else:
        test_plate_normalized = test_plate
    
    train_val_plates = [p for p in all_plates if p != test_plate_normalized]
    # Use cyclic validation: test plate i → validation plate is previous one
    test_num = int(test_plate_normalized.split('_')[1])
    val_num = (test_num - 2) % 6 + 1  # Previous plate (cyclic), e.g., test=3 → val=2
    val_plate = f"Plate_{val_num}"
    val_plates = [val_plate] if val_plate in train_val_plates else [train_val_plates[0]]
    train_plates = [p for p in train_val_plates if p not in val_plates][:4]
    
    print(f"Train plates: {train_plates}")
    print(f"Val plates: {val_plates}")
    print(f"Data mode: {args.data_mode}")
    
    # Build classes based on data_mode
    if args.data_mode == 'drug':
        # Use drug+concentration from plate_maps (well -> antibiotic + ic50_multiple)
        # Collect all drug classes from plate_maps (including control)
        drug_classes = set()
        for pm_key, label in plate_maps.items():
            for well, drug_label in label.items():
                if drug_label:  # Include all including 'control'
                    drug_classes.add(drug_label)
        all_classes = sorted(drug_classes)
        
        def drug_label_extractor(path, pm=plate_maps, pmap=plate_key_map):
            # Extract plate and well from path
            path_lower = path.lower()
            for plate_num in range(1, 7):
                if f'/p{plate_num}/' in path_lower or f'\\p{plate_num}\\' in path_lower:
                    plate_key = f'P{plate_num}'
                    break
            else:
                return None
            well = extract_well_from_filename(os.path.basename(path))
            # Use composite key: drug_A01
            composite_well = f"drug_{well}"
            if composite_well and plate_key in pm and composite_well in pm[plate_key]:
                return pm[plate_key][composite_well]
            return None
        
        label_extractor = drug_label_extractor
    elif args.data_mode == 'mutant':
        # Use gene/mutant from plate_maps
        all_classes = sorted(set(label for pm in plate_maps.values() for label in pm.values() if label))
        label_extractor = lambda path: get_gene_from_path(path, plate_maps)
    elif args.data_mode == 'control':
        # Combine all labels from control, mutant (NC/WT), and drug (DMSO) plate_maps
        all_classes = sorted(set(label for pm in plate_maps.values() for label in pm.values() if label))
        
        def control_label_extractor(path, pm=plate_maps, pmap=plate_key_map):
            path_lower = path.lower()
            for plate_num in range(1, 7):
                if f'/p{plate_num}/' in path_lower or f'\\p{plate_num}\\' in path_lower:
                    plate_key = f'P{plate_num}'
                    break
            else:
                return None
            well = extract_well_from_filename(os.path.basename(path))
            # Determine source from path
            if '/controls_data/' in path_lower or '\\controls_data\\' in path_lower:
                source_prefix = 'control_'
            elif '/mutants_data/' in path_lower or '\\mutants_data\\' in path_lower:
                source_prefix = 'mutant_'
            elif '/drugs_data/' in path_lower or '\\drugs_data\\' in path_lower:
                source_prefix = 'drug_'
            else:
                source_prefix = 'control_'
            composite_well = f"{source_prefix}{well}"
            if composite_well and plate_key in pm and composite_well in pm[plate_key]:
                return pm[plate_key][composite_well]
            return None
        
        label_extractor = control_label_extractor
    else:  # both - use plate_maps for both drug and mutant
        # Get all labels from plate_maps (both drug and control)
        all_labels = set()
        for pm in plate_maps.values():
            for label in pm.values():
                if label:
                    all_labels.add(label)
        all_classes = sorted(all_labels)
        
        def both_extractor(path, pm=plate_maps, pmap=plate_key_map):
            path_lower = path.lower()
            for plate_num in range(1, 7):
                if f'/p{plate_num}/' in path_lower or f'\\p{plate_num}\\' in path_lower:
                    plate_key = f'P{plate_num}'
                    break
            else:
                return None
            well = extract_well_from_filename(os.path.basename(path))
            
            # Determine source type from path (drug vs mutant vs control directory)
            if '/drugs_data/' in path_lower or '\\drugs_data\\' in path_lower or '/drugs_data/' in path_lower:
                source_prefix = 'drug_'
            elif '/mutants_data/' in path_lower or '\\mutants_data\\' in path_lower:
                source_prefix = 'mutant_'
            elif '/controls_data/' in path_lower or '\\controls_data\\' in path_lower:
                source_prefix = 'control_'
            else:
                source_prefix = 'control_'
            
            # Use composite key: drug_A01 or mutant_A01
            composite_well = f"{source_prefix}{well}"
            if composite_well and plate_key in pm and composite_well in pm[plate_key]:
                return pm[plate_key][composite_well]
            return None
        
        label_extractor = both_extractor
    
    # ============ DUAL CLASSIFIER: separate drug/mutant class lists ============
    if args.dual_classifier and args.data_mode == 'both':
        drug_labels_set = set()
        mutant_labels_set = set()
        for pm in plate_maps.values():
            for well_key, label in pm.items():
                if label:
                    if well_key.startswith('drug_'):
                        drug_labels_set.add(label)
                    elif well_key.startswith('mutant_'):
                        mutant_labels_set.add(label)
        drug_classes = sorted(drug_labels_set)
        mutant_classes = sorted(mutant_labels_set)
        drug_class_to_idx = {c: i for i, c in enumerate(drug_classes)}
        mutant_class_to_idx = {c: i for i, c in enumerate(mutant_classes)}
        num_drug_classes = len(drug_classes)
        num_mutant_classes = len(mutant_classes)
        all_classes = drug_classes + mutant_classes
        num_classes_total = len(all_classes)

        print(f"Dual classifier mode:")
        print(f"  Drug classes: {num_drug_classes}")
        print(f"  Mutant classes: {num_mutant_classes}")
        print(f"  Total: {num_classes_total}")

        def get_domain(path):
            p = path.lower()
            if '/drugs_data/' in p or '\\drugs_data\\' in p:
                return 0
            return 1

        train_paths, train_labels, train_domains = [], [], []
        val_paths, val_labels, val_domains = [], [], []
        test_paths, test_labels, test_domains = [], [], []

        for plate in train_plates:
            for path in get_image_paths_for_plate(plate):
                label_str = label_extractor(path)
                d = get_domain(path)
                if d == 0 and label_str in drug_class_to_idx:
                    train_paths.append(path)
                    train_labels.append(drug_class_to_idx[label_str])
                    train_domains.append(0)
                elif d == 1 and label_str in mutant_class_to_idx:
                    train_paths.append(path)
                    train_labels.append(mutant_class_to_idx[label_str])
                    train_domains.append(1)

        for plate in val_plates:
            for path in get_image_paths_for_plate(plate):
                label_str = label_extractor(path)
                d = get_domain(path)
                if d == 0 and label_str in drug_class_to_idx:
                    val_paths.append(path)
                    val_labels.append(drug_class_to_idx[label_str])
                    val_domains.append(0)
                elif d == 1 and label_str in mutant_class_to_idx:
                    val_paths.append(path)
                    val_labels.append(mutant_class_to_idx[label_str])
                    val_domains.append(1)

        for plate in [test_plate_normalized]:
            for path in get_image_paths_for_plate(plate):
                label_str = label_extractor(path)
                d = get_domain(path)
                if d == 0 and label_str in drug_class_to_idx:
                    test_paths.append(path)
                    test_labels.append(drug_class_to_idx[label_str])
                    test_domains.append(0)
                elif d == 1 and label_str in mutant_class_to_idx:
                    test_paths.append(path)
                    test_labels.append(mutant_class_to_idx[label_str])
                    test_domains.append(1)

        train_labels = np.array(train_labels)
        val_labels = np.array(val_labels)
        test_labels = np.array(test_labels)
        train_domains = np.array(train_domains)
        val_domains = np.array(val_domains)
        test_domains = np.array(test_domains)

        # Compute per-domain class weights
        drug_counts = Counter(train_labels[train_domains == 0].tolist())
        mutant_counts = Counter(train_labels[train_domains == 1].tolist())
        drug_total = max(sum(drug_counts.values()), 1)
        mutant_total = max(sum(mutant_counts.values()), 1)
        drug_weights = torch.tensor([drug_total / (num_drug_classes * max(drug_counts.get(i, 1), 1)) for i in range(num_drug_classes)], device=device)
        mutant_weights = torch.tensor([mutant_total / (num_mutant_classes * max(mutant_counts.get(i, 1), 1)) for i in range(num_mutant_classes)], device=device)
        drug_weights = drug_weights / drug_weights.sum() * num_drug_classes
        mutant_weights = mutant_weights / mutant_weights.sum() * num_mutant_classes

        class_to_idx = {**drug_class_to_idx, **{c: i + num_drug_classes for i, c in enumerate(mutant_classes)}}
        num_classes = num_classes_total
    else:
        class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
        num_classes = len(all_classes)
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {all_classes}")

        train_paths, train_labels = [], []
        val_paths, val_labels = [], []
        test_paths, test_labels = [], []

        for plate in train_plates:
            for path in get_image_paths_for_plate(plate):
                label = label_extractor(path)
                if label in class_to_idx:
                    train_paths.append(path)
                    train_labels.append(class_to_idx[label])

        for plate in val_plates:
            for path in get_image_paths_for_plate(plate):
                label = label_extractor(path)
                if label in class_to_idx:
                    val_paths.append(path)
                    val_labels.append(class_to_idx[label])

        for plate in [test_plate_normalized]:
            for path in get_image_paths_for_plate(plate):
                label = label_extractor(path)
                if label in class_to_idx:
                    test_paths.append(path)
                    test_labels.append(class_to_idx[label])

        train_labels = np.array(train_labels)
        val_labels = np.array(val_labels)
        test_labels = np.array(test_labels)

        # Filter to specific guide if requested (mutant mode)
        if args.guide is not None:
            guide_suffix = f"_{args.guide}"
            train_mask = np.array([lbl.endswith(guide_suffix) for lbl in [all_classes[i] for i in train_labels]])
            val_mask = np.array([lbl.endswith(guide_suffix) for lbl in [all_classes[i] for i in val_labels]])
            test_mask = np.array([lbl.endswith(guide_suffix) for lbl in [all_classes[i] for i in test_labels]])

            train_paths = [p for p, m in zip(train_paths, train_mask) if m]
            train_labels = train_labels[train_mask]
            val_paths = [p for p, m in zip(val_paths, val_mask) if m]
            val_labels = val_labels[val_mask]
            test_paths = [p for p, m in zip(test_paths, test_mask) if m]
            test_labels = test_labels[test_mask]

            remaining_classes = sorted(set(all_classes[i] for i in np.concatenate([train_labels, val_labels, test_labels])))
            old_all_classes = all_classes
            class_to_idx = {cls: idx for idx, cls in enumerate(remaining_classes)}
            num_classes = len(remaining_classes)
            all_classes = remaining_classes
            train_labels = np.array([class_to_idx[old_all_classes[i]] for i in train_labels])
            val_labels = np.array([class_to_idx[old_all_classes[i]] for i in val_labels])
            test_labels = np.array([class_to_idx[old_all_classes[i]] for i in test_labels])

            print(f"Guide {args.guide} filter applied: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}, Classes={num_classes}")

    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    if args.dual_classifier and args.data_mode == 'both':
        pass  # weights already computed above
    else:
        class_counts = Counter(train_labels)
        total = len(train_labels)
        class_weights = torch.tensor([total / (num_classes * max(class_counts[i], 1)) for i in range(num_classes)], device=device)
        class_weights = class_weights / class_weights.sum() * num_classes
    
    if args.dual_classifier and args.data_mode == 'both':
        train_dataset = MultiCropDataset(train_paths, train_labels, None, neighborhood=args.neighborhood, grid_size=args.grid_size, augment=True, seed=SEED, num_channels=args.num_channels, extraction_mode=args.extraction_mode, raster_crop_size=args.raster_crop_size, raster_resize_size=args.raster_resize_size, raster_num_crops=args.raster_num_crops, raster_grid_size=args.raster_grid_size, edge_sigma=args.edge_sigma, domains=train_domains.tolist())
        val_dataset = MultiCropDataset(val_paths, val_labels, None, neighborhood=args.neighborhood, grid_size=args.grid_size, augment=False, seed=SEED, num_channels=args.num_channels, extraction_mode=args.extraction_mode, raster_crop_size=args.raster_crop_size, raster_resize_size=args.raster_resize_size, raster_num_crops=args.raster_num_crops, raster_grid_size=args.raster_grid_size, edge_sigma=args.edge_sigma, domains=val_domains.tolist())
        test_dataset = MultiCropDataset(test_paths, test_labels, None, neighborhood=args.neighborhood, grid_size=args.grid_size, augment=False, seed=SEED, num_channels=args.num_channels, extraction_mode=args.extraction_mode, raster_crop_size=args.raster_crop_size, raster_resize_size=args.raster_resize_size, raster_num_crops=args.raster_num_crops, raster_grid_size=args.raster_grid_size, edge_sigma=args.edge_sigma, domains=test_domains.tolist())
    else:
        train_dataset = MultiCropDataset(train_paths, train_labels, None, neighborhood=args.neighborhood, grid_size=args.grid_size, augment=True, seed=SEED, num_channels=args.num_channels, extraction_mode=args.extraction_mode, raster_crop_size=args.raster_crop_size, raster_resize_size=args.raster_resize_size, raster_num_crops=args.raster_num_crops, raster_grid_size=args.raster_grid_size, edge_sigma=args.edge_sigma)
        val_dataset = MultiCropDataset(val_paths, val_labels, None, neighborhood=args.neighborhood, grid_size=args.grid_size, augment=False, seed=SEED, num_channels=args.num_channels, extraction_mode=args.extraction_mode, raster_crop_size=args.raster_crop_size, raster_resize_size=args.raster_resize_size, raster_num_crops=args.raster_num_crops, raster_grid_size=args.raster_grid_size, edge_sigma=args.edge_sigma)
        test_dataset = MultiCropDataset(test_paths, test_labels, None, neighborhood=args.neighborhood, grid_size=args.grid_size, augment=False, seed=SEED, num_channels=args.num_channels, extraction_mode=args.extraction_mode, raster_crop_size=args.raster_crop_size, raster_resize_size=args.raster_resize_size, raster_num_crops=args.raster_num_crops, raster_grid_size=args.raster_grid_size, edge_sigma=args.edge_sigma)
    
    train_dataset.set_epoch(0)
    val_dataset.set_epoch(0)
    test_dataset.set_epoch(0)
    
    # Windows Python 3.14: MUST use 0 workers due to multiprocessing pickling changes
    if sys.platform.startswith('win'):
        effective_workers = 0
        print(f"Using {effective_workers} workers (Windows Python 3.14 - multiprocessing spawn required)")
    else:
        effective_workers = NUM_WORKERS
        print(f"Using {effective_workers} workers")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=effective_workers, pin_memory=True, 
                              persistent_workers=True if effective_workers > 0 else False, prefetch_factor=args.prefetch_factor, drop_last=True,
                              worker_init_fn=partial(worker_init_fn, seed=SEED))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=effective_workers, pin_memory=True,
                            persistent_workers=True if effective_workers > 0 else False, prefetch_factor=args.prefetch_factor,
                            worker_init_fn=partial(worker_init_fn, seed=SEED))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=effective_workers, pin_memory=True,
                             persistent_workers=True if effective_workers > 0 else False, prefetch_factor=args.prefetch_factor,
                             worker_init_fn=partial(worker_init_fn, seed=SEED))
    
    if args.extraction_mode == 'raster':
        num_crops = args.grid_size * args.grid_size
        print(f"Extraction mode: RASTER - {num_crops} crops per image ({args.grid_size}x{args.grid_size} grid)")
    else:
        print(f"Extraction mode: NEIGHBORHOOD - {args.neighborhood}x{args.neighborhood}={args.neighborhood**2} crops per position")
    
    # Model selection based on flags
    if args.use_sc_mil:
        print(f"Using MILEncoder with SC-MIL supervised contrastive...")
        print(f"Backbone: {args.backbone}")
        print(f"Pooling: {args.pooling}")
        if args.dual_classifier and args.data_mode == 'both':
            print(f"Dual classifier mode: {num_drug_classes} drug + {num_mutant_classes} mutant classes")
            print(f"Projector bottleneck: {args.proj_dim}-dim")
            model = MILEncoder(
                num_classes=num_classes, num_heads=args.num_heads, dropout=args.dropout,
                use_contrastive=True, num_channels=args.num_channels,
                pretrained=args.pretrained, backbone=args.backbone, pooling=args.pooling,
                dual_classifier=True, num_drug_classes=num_drug_classes,
                num_mutant_classes=num_mutant_classes, proj_dim=args.proj_dim
            )
        else:
            print(f"Classifier: single FC layer with dropout={args.dropout}")
            model = MILEncoder(num_classes=num_classes, num_heads=args.num_heads, dropout=args.dropout, use_contrastive=True, num_channels=args.num_channels, pretrained=args.pretrained, backbone=args.backbone, pooling=args.pooling)
    else:
        print(f"Using AttentionMILModel...")
        print(f"Backbone: {args.backbone}")
        print(f"Pooling: {args.pooling}")
        print(f"Classifier: single FC layer with dropout={args.dropout}")
        model = AttentionMILModel(num_classes=num_classes, num_heads=args.num_heads, dropout=args.dropout, num_channels=args.num_channels, pretrained=args.pretrained, backbone=args.backbone, pooling=args.pooling)
    model = model.to(device)
    
    # Resume from checkpoint if requested
    resume_epoch = -1
    if args.resume:
        checkpoint_path = os.path.join(OUTPUT_DIR, 'checkpoint_epoch.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            resume_epoch = checkpoint['epoch']
            print(f"\nResuming from epoch {resume_epoch} (checkpoint: {checkpoint_path})\n")
        else:
            print(f"\nWarning: --resume specified but no checkpoint at {checkpoint_path}. Starting from scratch.\n")
    
    # Freeze backbone if requested
    if args.freeze:
        print("*** FREEZING BACKBONE - ONLY TRAINING ATTENTION + CLASSIFIER ***")
        for param in model.backbone.parameters():
            param.requires_grad = False
        model.backbone.eval()
    
    # Set up optimizer based on whether backbone is frozen
    if args.freeze:
        # Only train attention pool + classifier
        attention_params = [p for n, p in model.named_parameters() if 'attention_pool' in n or 'classifier' in n]
        optimizer = torch.optim.AdamW([
            {'params': attention_params, 'lr': args.lr}
        ], weight_decay=args.weight_decay, fused=True if torch.cuda.is_available() else False)
    else:
        # Original: backbone + attention + classifier with different LRs
        backbone_params = [p for n, p in model.named_parameters() if 'attention_pool' not in n and 'classifier' not in n]
        attention_params = [p for n, p in model.named_parameters() if 'attention_pool' in n or 'classifier' in n]
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * 0.1},
            {'params': attention_params, 'lr': args.lr}
        ], weight_decay=args.weight_decay, fused=True if torch.cuda.is_available() else False)

    # AMP scaler for mixed precision training
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    if args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup_scheduler, scheduler])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f'training_metrics_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auc', 'drug_val_auc', 'mutant_val_auc', 'backbone_lr', 'classifier_lr'])
    
    # TensorBoard writer - 2 cards: (Train Loss, Val Loss) and (Train Acc, Val Acc)
    tb_writer = SummaryWriter(log_dir=OUTPUT_DIR)
    
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
    # Stage 1: Patch-Level SimCLR Pre-training (proven in papers)
    if args.use_contrastive and resume_epoch < 0:
        print(f"\n{'='*60}")
        print(f"Stage 1: Patch-Level SimCLR Pre-training for {args.contrastive_epochs} epochs...")
        print(f"Contrastive batch size: {args.contrastive_batch_size}")
        print(f"{'='*60}")
        
        # Create two augmented views for each image
        crop_dataset_v1 = MultiCropDataset(train_paths, train_labels, None, neighborhood=1, grid_size=args.grid_size, augment=True, seed=SEED, num_channels=args.num_channels, extraction_mode=args.extraction_mode, raster_crop_size=args.raster_crop_size, raster_resize_size=args.raster_resize_size, raster_num_crops=args.raster_num_crops, raster_grid_size=args.raster_grid_size, edge_sigma=args.edge_sigma)
        crop_dataset_v2 = MultiCropDataset(train_paths, train_labels, None, neighborhood=1, grid_size=args.grid_size, augment=True, seed=SEED+1, num_channels=args.num_channels, extraction_mode=args.extraction_mode, raster_crop_size=args.raster_crop_size, raster_resize_size=args.raster_resize_size, raster_num_crops=args.raster_num_crops, raster_grid_size=args.raster_grid_size, edge_sigma=args.edge_sigma)
        
        # Set initial epoch for both
        crop_dataset_v1.set_epoch(0)
        crop_dataset_v2.set_epoch(0)
        
        # Higher batch size for contrastive (more negatives = better learning)
        crop_loader_v1 = DataLoader(crop_dataset_v1, batch_size=args.contrastive_batch_size, shuffle=True, num_workers=4, pin_memory=True, 
                                    persistent_workers=True, prefetch_factor=args.prefetch_factor, drop_last=True)
        crop_loader_v2 = DataLoader(crop_dataset_v2, batch_size=args.contrastive_batch_size, shuffle=True, num_workers=4, pin_memory=True, 
                                    persistent_workers=True, prefetch_factor=args.prefetch_factor, drop_last=True)
        
        # Train encoder + projection head
        contrastive_params = [p for n, p in model.named_parameters() if 'contrastive_head' in n or 'head_proj' in n or 'backbone' in n]
        contrastive_optimizer = torch.optim.Adam(contrastive_params, lr=args.lr, fused=True if torch.cuda.is_available() else False)
        contrastive_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(contrastive_optimizer, T_max=args.contrastive_epochs)
        contrastive_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        for epoch in range(args.contrastive_epochs):
            model.train()
            run_loss = 0.0
            n_batches = 0
            
            iter_v2 = iter(crop_loader_v2)
            for images_v1, _ in tqdm(crop_loader_v1, desc=f'Contrastive Epoch {epoch}', leave=False):
                try:
                    images_v2, _ = next(iter_v2)
                except StopIteration:
                    iter_v2 = iter(crop_loader_v2)
                    images_v2, _ = next(iter_v2)
                
                images_v1 = images_v1.to(device)
                images_v2 = images_v2.to(device)
                contrastive_optimizer.zero_grad()
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    # Get features for both views
                    feat_v1 = model.get_projected_features(images_v1)
                    feat_v2 = model.get_projected_features(images_v2)
                    
                    # Normalize
                    feat_v1 = F.normalize(feat_v1, dim=1)
                    feat_v2 = F.normalize(feat_v2, dim=1)
                    
                    # SimCLR: z1*z2 for positives, z1*z_all for negatives
                    batch_size = feat_v1.shape[0]
                    temp = args.contrastive_temp
                    
                    # Compute similarity matrix
                    z1 = feat_v1 / temp
                    z2 = feat_v2 / temp
                    
                    # Similarity between matching pairs
                    sim_pos = torch.sum(z1 * z2, dim=1)
                    
                    # All pairs (including negatives)
                    similarity = torch.matmul(z1, z2.T)
                    
                    # Labels: diagonal (matching indices)
                    labels = torch.arange(batch_size, device=feat_v1.device)
                    
                    # Compute InfoNCE loss
                    loss = F.cross_entropy(similarity, labels)
                
                contrastive_scaler.scale(loss).backward()
                contrastive_scaler.step(contrastive_optimizer)
                contrastive_scaler.update()
                
                run_loss += loss.item()
                n_batches += 1
            
            contrastive_scheduler.step()
            avg_loss = run_loss / max(n_batches, 1)
            print(f"Contrastive Epoch {epoch}: Loss={avg_loss:.4f}")
        
        print(f"Stage 1 complete! Now training MIL classifier...")
        train_dataset.set_epoch(0)
    
    # SC-MIL: Supervised Bag-Level Contrastive + Classification Joint Training
    if args.use_sc_mil:
        print(f"\n{'='*60}")
        print(f"SC-MIL: Supervised Bag-Level Contrastive Joint Training")
        print(f"SC-MIL epochs: {args.sc_mil_epochs}, Temp: {args.sc_mil_temp}")
        print(f"Contrastive weight: {args.sc_mil_weight}")
        print(f"{'='*60}")
        
        # Batch size for SC-MIL
        effective_batch_size = args.batch_size  # Use full batch size (16)
        print(f"Using batch size: {effective_batch_size}")
        
        # Recreate data loaders with smaller batch size
        train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, num_workers=effective_workers, pin_memory=True,
                                   persistent_workers=True if effective_workers > 0 else False, prefetch_factor=args.prefetch_factor, drop_last=True)
        
        # Train encoder + attention + classifier jointly
        sc_mil_params = [p for n, p in model.named_parameters()]
        sc_mil_optimizer = torch.optim.AdamW(sc_mil_params, lr=args.lr, weight_decay=args.weight_decay, fused=True if torch.cuda.is_available() else False)
        sc_mil_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(sc_mil_optimizer, T_max=args.sc_mil_epochs)
        if args.warmup_epochs > 0:
            sc_mil_warmup = torch.optim.lr_scheduler.LinearLR(
                sc_mil_optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
            )
            sc_mil_scheduler = torch.optim.lr_scheduler.ChainedScheduler([sc_mil_warmup, sc_mil_scheduler])
        sc_mil_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        # Create CSV file for SC-MIL metrics
        timestamp_sc_mil = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path_sc_mil = os.path.join(OUTPUT_DIR, f"training_sc_mil_{timestamp_sc_mil}.csv")
        with open(csv_path_sc_mil, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_ce_loss', 'train_sc_loss', 'train_sym_loss', 'train_acc', 'drug_train_acc', 'mutant_train_acc', 'val_ce_loss', 'val_acc', 'drug_val_acc', 'mutant_val_acc', 'val_auc', 'drug_val_auc', 'mutant_val_auc', 'lr'])
        
        sc_mil_start = resume_epoch + 1 if resume_epoch >= 0 else 0
        for epoch in range(sc_mil_start, args.sc_mil_epochs):
            epoch_start = time.time()
            train_dataset.set_epoch(epoch)
            model.train()
            run_cl_loss, run_sym_loss, run_ce_loss, correct, total = 0.0, 0.0, 0.0, 0, 0
            drug_correct, drug_total, mutant_correct, mutant_total = 0, 0, 0, 0
            
            for batch in tqdm(train_loader, desc=f'SC-MIL Epoch {epoch}', leave=False):
                images = batch[0].to(device)
                labels = batch[1].to(device)
                domains = batch[2].to(device) if len(batch) > 2 else None
                sc_mil_optimizer.zero_grad()
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    # ============ DUAL CLASSIFIER PATH ============
                    if args.dual_classifier and args.data_mode == 'both':
                        proj_emb = model.get_projected_features(images)
                        drug_logits = model.drug_classifier(proj_emb)
                        mutant_logits = model.mutant_classifier(proj_emb)
                        
                        drug_mask = domains == 0
                        mutant_mask = domains == 1
                        
                        # Per-domain classification loss
                        total_focal = 0.0
                        if drug_mask.any():
                            total_focal += weighted_focal_loss(drug_logits[drug_mask], labels[drug_mask], drug_weights[labels[drug_mask]])
                        if mutant_mask.any():
                            total_focal += weighted_focal_loss(mutant_logits[mutant_mask], labels[mutant_mask], mutant_weights[labels[mutant_mask]])
                        
                        # Per-domain bag-level contrastive on projected embeddings
                        bag_sc_loss = 0.0
                        if args.contrastive_level in ['bag', 'both']:
                            if drug_mask.any():
                                bag_emb = F.normalize(proj_emb[drug_mask], p=2, dim=-1).unsqueeze(1)
                                bag_sc_loss += SupConLoss(temperature=args.sc_mil_temp)(bag_emb, labels[drug_mask])
                            if mutant_mask.any():
                                bag_emb = F.normalize(proj_emb[mutant_mask], p=2, dim=-1).unsqueeze(1)
                                bag_sc_loss += SupConLoss(temperature=args.sc_mil_temp)(bag_emb, labels[mutant_mask])
                        
                        # Per-domain symmetric consistency loss on projected embeddings
                        sym_loss = 0.0
                        if drug_mask.any() and drug_mask.sum() >= 2:
                            sym_loss += symmetric_consistency_loss(proj_emb[drug_mask], labels[drug_mask])
                        if mutant_mask.any() and mutant_mask.sum() >= 2:
                            sym_loss += symmetric_consistency_loss(proj_emb[mutant_mask], labels[mutant_mask])

                        total_sc = bag_sc_loss
                        loss = args.ce_weight * (1 - args.sc_mil_weight) * total_focal + args.sc_mil_weight * total_sc + args.sym_consistency_weight * sym_loss
                        
                        # Accuracy per domain
                        if drug_mask.any():
                            drug_pred = drug_logits[drug_mask].argmax(1)
                            correct += drug_pred.eq(labels[drug_mask]).sum().item()
                            drug_correct += drug_pred.eq(labels[drug_mask]).sum().item()
                            drug_total += drug_mask.sum().item()
                        if mutant_mask.any():
                            mutant_pred = mutant_logits[mutant_mask].argmax(1)
                            correct += mutant_pred.eq(labels[mutant_mask]).sum().item()
                            mutant_correct += mutant_pred.eq(labels[mutant_mask]).sum().item()
                            mutant_total += mutant_mask.sum().item()
                        total += labels.size(0)
                    else:
                        # ============ STANDARD SINGLE-CLASSIFIER PATH ============
                        outputs, attn_weights, crop_embeddings, pooled_embeddings, instance_logits = model(
                            images, return_attention=True, return_crop_embeddings=True, 
                            return_pooled_embeddings=True, return_instance_logits=True
                        )
                        
                        # Contrastive losses
                        num_crops = crop_embeddings.shape[1]
                        
                        if args.contrastive_level in ['instance', 'both']:
                            crop_emb_flat = crop_embeddings.view(-1, crop_embeddings.shape[-1]).unsqueeze(1)
                            crop_emb_flat = F.normalize(crop_emb_flat, p=2, dim=-1)
                            instance_labels_exp = labels.repeat_interleave(num_crops)
                            inst_temp = max(args.sc_mil_temp, 0.1)
                            criterion_inst = SupConLoss(temperature=inst_temp, contrast_mode='one')
                            instance_sc_loss = criterion_inst(crop_emb_flat, instance_labels_exp)
                        else:
                            instance_sc_loss = 0.0
                        
                        if args.contrastive_level in ['bag', 'both']:
                            bag_embeddings = F.normalize(pooled_embeddings, p=2, dim=-1).unsqueeze(1)
                            sc_criterion = SupConLoss(temperature=args.sc_mil_temp)
                            bag_sc_loss = sc_criterion(bag_embeddings, labels)
                        else:
                            bag_sc_loss = 0.0
                        
                        # Classification losses
                        num_crops = crop_embeddings.shape[1]
                        instance_labels = labels.repeat_interleave(num_crops)
                        instance_weights = class_weights[instance_labels]
                        instance_focal = weighted_focal_loss(
                            instance_logits.view(-1, num_classes),
                            instance_labels,
                            instance_weights
                        )
                        bag_focal = weighted_focal_loss(outputs, labels, class_weights[labels])
                        
                        w = args.instance_weight
                        total_focal = w * instance_focal + (1 - w) * bag_focal
                        total_sc = w * instance_sc_loss + (1 - w) * bag_sc_loss
                        sym_loss = 0.0
                        loss = args.ce_weight * (1 - args.sc_mil_weight) * total_focal + args.sc_mil_weight * total_sc

                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                
                sc_mil_scaler.scale(loss).backward()
                sc_mil_scaler.step(sc_mil_optimizer)
                sc_mil_scaler.update()
                
                run_cl_loss += total_sc.item() if isinstance(total_sc, torch.Tensor) else total_sc
                run_ce_loss += total_focal.item() if isinstance(total_focal, torch.Tensor) else total_focal
                run_sym_loss += sym_loss.item() if isinstance(sym_loss, torch.Tensor) else sym_loss
            
            sc_mil_scheduler.step()
            
            train_acc = 100. * correct / total
            drug_train_acc = 100. * drug_correct / drug_total if drug_total > 0 else 0.0
            mutant_train_acc = 100. * mutant_correct / mutant_total if mutant_total > 0 else 0.0
            avg_cl_loss = run_cl_loss / len(train_loader)
            avg_ce_loss = run_ce_loss / len(train_loader)
            
            # VALIDATION after each SC-MIL epoch
            model.eval()
            val_cl_loss, val_ce_loss = 0.0, 0.0
            val_ce_count = 0
            val_correct, val_total = 0, 0
            drug_val_correct, drug_val_total, mutant_val_correct, mutant_val_total = 0, 0, 0, 0
            all_val_preds, all_val_probs, all_val_labels = [], [], []
            drug_val_preds, drug_val_probs, drug_val_labels = [], [], []
            mutant_val_preds, mutant_val_probs, mutant_val_labels = [], [], []
            
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
                for batch in tqdm(val_loader, desc='Validating', leave=False):
                    images = batch[0].to(device)
                    labels = batch[1].to(device)
                    domains = batch[2].to(device) if len(batch) > 2 else None
                    
                    if args.dual_classifier and args.data_mode == 'both':
                        proj_emb = model.get_projected_features(images)
                        drug_logits = model.drug_classifier(proj_emb)
                        mutant_logits = model.mutant_classifier(proj_emb)
                        
                        drug_mask = domains == 0
                        mutant_mask = domains == 1
                        
                        if drug_mask.any():
                            drug_probs = torch.softmax(drug_logits[drug_mask], dim=1)
                            drug_pred = drug_logits[drug_mask].argmax(1)
                            drug_val_preds.extend(drug_pred.cpu().numpy())
                            drug_val_probs.extend(drug_probs.cpu().numpy())
                            drug_val_labels.extend(labels[drug_mask].cpu().numpy())
                            val_correct += drug_pred.eq(labels[drug_mask]).sum().item()
                            drug_val_correct += drug_pred.eq(labels[drug_mask]).sum().item()
                            drug_val_total += drug_mask.sum().item()
                            val_loss = weighted_focal_loss(drug_logits[drug_mask], labels[drug_mask], drug_weights[labels[drug_mask]])
                            val_ce_loss += val_loss.item()
                            val_ce_count += 1
                        
                        if mutant_mask.any():
                            mutant_probs = torch.softmax(mutant_logits[mutant_mask], dim=1)
                            mutant_pred = mutant_logits[mutant_mask].argmax(1)
                            mutant_val_preds.extend(mutant_pred.cpu().numpy())
                            mutant_val_probs.extend(mutant_probs.cpu().numpy())
                            mutant_val_labels.extend(labels[mutant_mask].cpu().numpy())
                            val_correct += mutant_pred.eq(labels[mutant_mask]).sum().item()
                            mutant_val_correct += mutant_pred.eq(labels[mutant_mask]).sum().item()
                            mutant_val_total += mutant_mask.sum().item()
                            val_loss = weighted_focal_loss(mutant_logits[mutant_mask], labels[mutant_mask], mutant_weights[labels[mutant_mask]])
                            val_ce_loss += val_loss.item()
                            val_ce_count += 1
                        
                        val_total += labels.size(0)
                    else:
                        outputs, _ = model(images, return_attention=True)
                        probs = torch.softmax(outputs, dim=1)
                        _, predicted = outputs.max(1)
                        all_val_preds.extend(predicted.cpu().numpy())
                        all_val_probs.extend(probs.cpu().numpy())
                        all_val_labels.extend(labels.cpu().numpy())
                        val_loss = weighted_focal_loss(outputs, labels, class_weights[labels])
                        val_ce_loss += val_loss.item()
                        val_ce_count += 1
                        val_correct += predicted.eq(labels).sum().item()
                        val_total += labels.size(0)
            
            val_acc = 100. * val_correct / val_total
            drug_val_acc = 100. * drug_val_correct / drug_val_total if drug_val_total > 0 else 0.0
            mutant_val_acc = 100. * mutant_val_correct / mutant_val_total if mutant_val_total > 0 else 0.0
            avg_val_ce_loss = val_ce_loss / val_ce_count if val_ce_count > 0 else 0.0
            
            # Per-domain AUC for dual classifier (separate prob dimensions)
            if args.dual_classifier and args.data_mode == 'both':
                drug_val_auc = compute_robust_auc(drug_val_labels, drug_val_probs, num_drug_classes)
                mutant_val_auc = compute_robust_auc(mutant_val_labels, mutant_val_probs, num_mutant_classes)
                val_auc = (drug_val_auc + mutant_val_auc) / 2 if not (np.isnan(drug_val_auc) or np.isnan(mutant_val_auc)) else float('nan')
                print(f"SC-MIL Epoch {epoch}: CE={avg_ce_loss:.4f} SC={avg_cl_loss:.4f} Sym={run_sym_loss/len(train_loader):.4f} | Train: {train_acc:.2f}% (drug={drug_train_acc:.2f}% mut={mutant_train_acc:.2f}%) | Val: {val_acc:.2f}% (drug={drug_val_acc:.2f}% mut={mutant_val_acc:.2f}%) | AUC drug={drug_val_auc:.4f} mut={mutant_val_auc:.4f} | {time.time()-epoch_start:.1f}s")
            else:
                val_auc = compute_robust_auc(all_val_labels, all_val_probs, num_classes)
                print(f"SC-MIL Epoch {epoch}: CE={avg_ce_loss:.4f} SC={avg_cl_loss:.4f} Sym={run_sym_loss/len(train_loader):.4f} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | AUC={val_auc:.4f} | {time.time()-epoch_start:.1f}s")
            
            # Save checkpoint every epoch
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'checkpoint_epoch.pth'))
            
            # Save metrics to CSV
            with open(csv_path_sc_mil, 'a', newline='') as f:
                writer = csv.writer(f)
                if args.dual_classifier and args.data_mode == 'both':
                    writer.writerow([epoch, avg_ce_loss, avg_cl_loss, run_sym_loss/len(train_loader), train_acc, drug_train_acc, mutant_train_acc, avg_val_ce_loss, val_acc, drug_val_acc, mutant_val_acc, val_auc, drug_val_auc, mutant_val_auc, sc_mil_optimizer.param_groups[0]['lr']])
                else:
                    writer.writerow([epoch, avg_ce_loss, avg_cl_loss, run_sym_loss/len(train_loader), train_acc, '', '', avg_val_ce_loss, val_acc, '', '', val_auc, '', '', sc_mil_optimizer.param_groups[0]['lr']])
            
            # TensorBoard logging - 2 cards only (SC-MIL)
            # Card 1: Train CE Loss + Val CE Loss
            tb_writer.add_scalars('Loss', {'train': avg_ce_loss, 'val': avg_val_ce_loss}, epoch)
            # Card 2: Train Acc + Val Acc
            tb_writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
            
            # Save best model based on validation AUC
            if not np.isnan(val_auc) and val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model.pth'))
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model_auc.pth'))
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model_acc.pth'))
            
            if avg_val_ce_loss < best_val_loss:
                best_val_loss = avg_val_ce_loss
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model_loss.pth'))
        
        print(f"SC-MIL training complete!")
        # Skip standard training, go directly to evaluation
        epoch = args.sc_mil_epochs  # Mark as complete
    
    else:
        print("Training...")
        epoch = None  # Means standard training
    
    # Standard or SC-MIL training loop
    if epoch is None:
        start_epoch = resume_epoch + 1 if resume_epoch >= 0 else 0
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            train_dataset.set_epoch(epoch)
            model.train()
            run_loss, correct, total = 0.0, 0, 0
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
                images = batch[0].to(device)
                labels = batch[1].to(device)
                domains = batch[2].to(device) if len(batch) > 2 else None
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    if args.dual_classifier and args.data_mode == 'both':
                        proj_emb = model.get_projected_features(images)
                        drug_logits = model.drug_classifier(proj_emb)
                        mutant_logits = model.mutant_classifier(proj_emb)
                        drug_mask = domains == 0
                        mutant_mask = domains == 1
                        main_loss = 0.0
                        if drug_mask.any():
                            main_loss += weighted_focal_loss(drug_logits[drug_mask], labels[drug_mask], drug_weights[labels[drug_mask]], label_smoothing=args.label_smoothing)
                        if mutant_mask.any():
                            main_loss += weighted_focal_loss(mutant_logits[mutant_mask], labels[mutant_mask], mutant_weights[labels[mutant_mask]], label_smoothing=args.label_smoothing)
                        loss = main_loss
                        if drug_mask.any():
                            correct += drug_logits[drug_mask].argmax(1).eq(labels[drug_mask]).sum().item()
                        if mutant_mask.any():
                            correct += mutant_logits[mutant_mask].argmax(1).eq(labels[mutant_mask]).sum().item()
                        total += labels.size(0)
                    else:
                        outputs, attn_weights = model(images, return_attention=True)
                        main_loss = weighted_focal_loss(outputs, labels, class_weights[labels], label_smoothing=args.label_smoothing)
                        if args.entropy_loss_weight > 0:
                            attn_ent_loss = attention_entropy_loss(attn_weights)
                            loss = main_loss + args.entropy_loss_weight * attn_ent_loss
                        else:
                            loss = main_loss
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                run_loss += main_loss.item() if isinstance(main_loss, torch.Tensor) else main_loss
        
        scheduler.step()
        
        train_acc = 100. * correct / total
        avg_train_loss = run_loss / len(train_loader)
        
        model.eval()
        val_loss_total = 0.0
        all_preds, all_probs, all_labels = [], [], []
        drug_val_preds, drug_val_probs, drug_val_labels = [], [], []
        mutant_val_preds, mutant_val_probs, mutant_val_labels = [], [], []
        
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            for batch in tqdm(val_loader, desc='Validating', leave=False):
                images = batch[0].to(device)
                labels = batch[1].to(device)
                domains = batch[2].to(device) if len(batch) > 2 else None
                
                if args.dual_classifier and args.data_mode == 'both':
                    proj_emb = model.get_projected_features(images)
                    drug_logits = model.drug_classifier(proj_emb)
                    mutant_logits = model.mutant_classifier(proj_emb)
                    drug_mask = domains == 0
                    mutant_mask = domains == 1
                    if drug_mask.any():
                        drug_probs = torch.softmax(drug_logits[drug_mask], dim=1)
                        drug_pred = drug_logits[drug_mask].argmax(1)
                        drug_val_preds.extend(drug_pred.cpu().numpy())
                        drug_val_probs.extend(drug_probs.cpu().numpy())
                        drug_val_labels.extend(labels[drug_mask].cpu().numpy())
                        all_preds.extend(drug_pred.cpu().numpy())
                        all_labels.extend(labels[drug_mask].cpu().numpy())
                        val_loss = weighted_focal_loss(drug_logits[drug_mask], labels[drug_mask], drug_weights[labels[drug_mask]], label_smoothing=args.label_smoothing)
                        val_loss_total += val_loss.item()
                    if mutant_mask.any():
                        mutant_probs = torch.softmax(mutant_logits[mutant_mask], dim=1)
                        mutant_pred = mutant_logits[mutant_mask].argmax(1)
                        mutant_val_preds.extend(mutant_pred.cpu().numpy())
                        mutant_val_probs.extend(mutant_probs.cpu().numpy())
                        mutant_val_labels.extend(labels[mutant_mask].cpu().numpy())
                        all_preds.extend(mutant_pred.cpu().numpy())
                        all_labels.extend(labels[mutant_mask].cpu().numpy())
                        val_loss = weighted_focal_loss(mutant_logits[mutant_mask], labels[mutant_mask], mutant_weights[labels[mutant_mask]], label_smoothing=args.label_smoothing)
                        val_loss_total += val_loss.item()
                else:
                    outputs, _ = model(images, return_attention=True)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    val_loss = weighted_focal_loss(outputs, labels, class_weights[labels], label_smoothing=args.label_smoothing)
                    val_loss_total += val_loss.item()
        
        val_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        avg_val_loss = val_loss_total / len(val_loader)
        
        backbone_lr = optimizer.param_groups[0]['lr']
        classifier_lr = optimizer.param_groups[1]['lr']
        
        # Per-domain AUC for dual classifier
        if args.dual_classifier and args.data_mode == 'both':
            drug_val_auc = compute_robust_auc(drug_val_labels, drug_val_probs, num_drug_classes)
            mutant_val_auc = compute_robust_auc(mutant_val_labels, mutant_val_probs, num_mutant_classes)
            val_auc = (drug_val_auc + mutant_val_auc) / 2 if not (np.isnan(drug_val_auc) or np.isnan(mutant_val_auc)) else float('nan')
            print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%, Drug AUC={drug_val_auc:.4f}, Mutant AUC={mutant_val_auc:.4f}, Backbone LR={backbone_lr:.2e}, Classifier LR={classifier_lr:.2e}, Time={time.time()-epoch_start:.1f}s")
        else:
            val_auc = compute_robust_auc(all_labels, all_probs, num_classes)
            print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%, Val AUC={val_auc:.4f}, Backbone LR={backbone_lr:.2e}, Classifier LR={classifier_lr:.2e}, Time={time.time()-epoch_start:.1f}s")
        
        # TensorBoard logging - 2 cards only
        # Card 1: Train Loss + Val Loss
        tb_writer.add_scalars('Loss', {'train': avg_train_loss, 'val': avg_val_loss}, epoch)
        # Card 2: Train Acc + Val Acc
        tb_writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        
        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model.pth'))
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model_auc.pth'))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model_acc.pth'))
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model_loss.pth'))
        
        # Log metrics to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if args.dual_classifier and args.data_mode == 'both':
                writer.writerow([epoch, avg_train_loss, train_acc, avg_val_loss, val_acc, val_auc, drug_val_auc, mutant_val_auc, backbone_lr, classifier_lr])
            else:
                writer.writerow([epoch, avg_train_loss, train_acc, avg_val_loss, val_acc, val_auc, '', '', backbone_lr, classifier_lr])
        
        if (epoch + 1) % args.checkpoint_every == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'checkpoint_epoch.pth'))
    
    print("Testing...")
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_model_acc.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds, all_probs, all_labels = [], [], []
    drug_test_preds, drug_test_probs, drug_test_labels = [], [], []
    mutant_test_preds, mutant_test_probs, mutant_test_labels = [], [], []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for batch in tqdm(test_loader, desc='Testing', leave=False):
            images = batch[0].to(device)
            labels = batch[1].to(device)
            domains = batch[2].to(device) if len(batch) > 2 else None
            
            if args.dual_classifier and args.data_mode == 'both':
                proj_emb = model.get_projected_features(images)
                drug_logits = model.drug_classifier(proj_emb)
                mutant_logits = model.mutant_classifier(proj_emb)
                
                drug_mask = domains == 0
                mutant_mask = domains == 1
                
                if drug_mask.any():
                    drug_probs = torch.softmax(drug_logits[drug_mask], dim=1)
                    drug_pred = drug_logits[drug_mask].argmax(1)
                    drug_test_preds.extend(drug_pred.cpu().numpy())
                    drug_test_probs.extend(drug_probs.cpu().numpy())
                    drug_test_labels.extend(labels[drug_mask].cpu().numpy())
                    all_preds.extend(drug_pred.cpu().numpy())
                    all_labels.extend(labels[drug_mask].cpu().numpy())
                
                if mutant_mask.any():
                    mutant_probs = torch.softmax(mutant_logits[mutant_mask], dim=1)
                    mutant_pred = mutant_logits[mutant_mask].argmax(1)
                    mutant_test_preds.extend(mutant_pred.cpu().numpy())
                    mutant_test_probs.extend(mutant_probs.cpu().numpy())
                    mutant_test_labels.extend(labels[mutant_mask].cpu().numpy())
                    all_preds.extend(mutant_pred.cpu().numpy())
                    all_labels.extend(labels[mutant_mask].cpu().numpy())
            else:
                outputs, _ = model(images, return_attention=True)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    
    if args.dual_classifier and args.data_mode == 'both':
        drug_test_auc = compute_robust_auc(drug_test_labels, drug_test_probs, num_drug_classes)
        mutant_test_auc = compute_robust_auc(mutant_test_labels, mutant_test_probs, num_mutant_classes)
        test_auc = (drug_test_auc + mutant_test_auc) / 2 if not (np.isnan(drug_test_auc) or np.isnan(mutant_test_auc)) else float('nan')
        print(f"Test Acc: {test_acc:.2f}%, Drug AUC: {drug_test_auc:.4f}, Mutant AUC: {mutant_test_auc:.4f}")
    else:
        test_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
        test_auc = roc_auc_score(test_labels_bin, np.array(all_probs), average='macro')
        test_ap = average_precision_score(test_labels_bin, np.array(all_probs), average='macro')
        print(f"Test Acc: {test_acc:.2f}%, Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")
    
    if args.dual_classifier and args.data_mode == 'both':
        results = {
            'timestamp': timestamp,
            'config': {'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr, 'test_plate': test_plate, 'dropout': args.dropout, 'weight_decay': args.weight_decay, 'neighborhood': args.neighborhood, 'dual_classifier': True, 'num_drug_classes': num_drug_classes, 'num_mutant_classes': num_mutant_classes},
            'results': {'best_val_auc': float(best_val_auc), 'test_acc': float(test_acc), 'drug_test_auc': float(drug_test_auc), 'mutant_test_auc': float(mutant_test_auc)}
        }
    else:
        results = {
            'timestamp': timestamp,
            'config': {'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr, 'test_plate': test_plate, 'dropout': args.dropout, 'weight_decay': args.weight_decay, 'neighborhood': args.neighborhood},
            'results': {'best_val_auc': float(best_val_auc), 'test_acc': float(test_acc), 'test_auc': float(test_auc), 'test_ap': float(test_ap)}
        }
    
    with open(os.path.join(OUTPUT_DIR, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Close TensorBoard writer
    tb_writer.close()
    
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    if args.run_all_folds:
        for test_plate in all_plates:
            fold_dir = os.path.join(SCRIPT_DIR, data_mode_folder, f'fold_{test_plate}')
            
            # Check for any checkpoint files to skip trained folds
            checkpoints = [
                os.path.join(fold_dir, 'best_model.pth'),
                os.path.join(fold_dir, 'best_model_acc.pth'),
                os.path.join(fold_dir, 'best_model_auc.pth'),
                os.path.join(fold_dir, 'best_model_loss.pth'),
            ]
            
            if any(os.path.exists(cp) for cp in checkpoints):
                print(f"\nSkipping {test_plate}: already trained (checkpoint exists)")
                continue
            
            train_single_fold(test_plate)
        
        print("All folds completed!")
    else:
        train_single_fold(args.test_plate)
    
    print("Done!")






