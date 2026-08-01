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
parser.add_argument('--num_workers', type=int, default=16,
                    help='Number of DataLoader workers (default 16)')
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
parser.add_argument('--attention_temp', type=float, default=0.5,
                    help='Temperature for attention softmax (default 0.5)')
parser.add_argument('--attn_hidden_dim', type=int, default=256,
                    help='Hidden dimension of the attention pooling network (default 256)')
parser.add_argument('--classifier_hidden_dim', type=int, default=512,
                    help='Hidden dimension of classifier MLP hidden layers (default 512)')
parser.add_argument('--classifier_layers', type=int, default=0,
                    help='Number of hidden layers in classifier MLP (0 = single linear head, default)')
parser.add_argument('--early_stopping_patience', type=int, default=0,
                    help='Stop training if val acc does not improve for N epochs (0 = disabled)')
parser.add_argument('--skip_test', action='store_true', default=False,
                    help='Skip test evaluation (used for HPO runs)')
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
parser.add_argument('--resume', type=str, default=None,
                    help='Path to a checkpoint_epoch.pth file or an output dir containing one. '
                         'Restores model, optimizer, scheduler, scaler and best-metric state and '
                         'continues training from the next epoch.')
parser.add_argument('--num_channels', type=int, default=1,
                    help='Number of input channels (1 for grayscale, 3 for RGB)')
parser.add_argument('--backbone', type=str, default='efficientnet_b0', choices=['efficientnet_b0', 'mobilenet_v3_small', 'mobilenet_v2'],
                    help='Backbone architecture: efficientnet_b0 (default), mobilenet_v3_small, or mobilenet_v2')
parser.add_argument('--pretrained', type=str, default='imagenet', choices=['imagenet', 'micronet'], 
                    help='Pretrained weights: imagenet (default) or micronet (NASA microscopy pretrained)')
parser.add_argument('--framework', type=str, default='pytorch', choices=['pytorch', 'tensorflow'],
                    help='Framework: pytorch (default) or tensorflow/keras')
parser.add_argument('--data_mode', type=str, default='mutant', choices=['drug', 'mutant', 'both', 'metabolomics_mutant'],
                    help='Data mode: drug (drug+concentration), mutant (gene/mutant), both (combine), metabolomics_mutant (Felix metabolomics data)')
parser.add_argument('--timepoint_split', action='store_true', default=False,
                    help='Metabolomics: split by timepoint instead of plate (train T1, val T2, test T3, 96 classes, 1 fold)')
parser.add_argument('--include_timepoint_in_labels', action='store_true', default=False,
                    help='Metabolomics: prepend timepoint to labels (T1_gene, T2_gene, T3_gene, 288 classes, 4 folds)')
parser.add_argument('--drug_no_concentration', action='store_true', default=False,
                    help='Group drugs by antibiotic name only, ignoring concentration levels (e.g., Ciprofloxacin instead of Ciprofloxacin_2x)')
parser.add_argument('--freeze', action='store_true', default=False,
                    help='Freeze backbone, only train attention pool + classifier head')
parser.add_argument('--guide', type=int, default=None,
    help='Filter to specific guide number (e.g. 1 for guide 1) in mutant mode')
parser.add_argument('--output_dir', type=str, default=None,
    help='Output directory for results/models/logs (default: scripts/<data_mode>/fold_<test_plate>)')
args = parser.parse_args()

if args.timepoint_split and args.include_timepoint_in_labels:
    parser.error('--timepoint_split and --include_timepoint_in_labels are mutually exclusive')

# Determine folder name for results (drug_noconcentration vs drug)
data_mode_folder = args.data_mode
if args.data_mode == 'drug' and args.drug_no_concentration:
    data_mode_folder = 'drug_noconcentration'
if args.data_mode == 'metabolomics_mutant':
    data_mode_folder = 'metabolomics_mutant'
    if args.timepoint_split:
        data_mode_folder = 'metabolomics_mutant_timepoint_split'
    if args.include_timepoint_in_labels:
        data_mode_folder = 'metabolomics_mutant_tp_labels'
if args.guide is not None:
    data_mode_folder = f"{args.data_mode}_guide_{args.guide}"

if args.warmup_epochs is None:
    args.warmup_epochs = int(args.sc_mil_epochs * 0.05)  # 5% of SC-MIL training

# Metabolomics default: use micrograph pretrained weights
if args.data_mode == 'metabolomics_mutant':
    args.pretrained = 'micronet'

# Set num_workers based on OS
if sys.platform.startswith('win'):
    NUM_WORKERS = 0  # Windows Python 3.14: multiprocessing spawn required
else:
    NUM_WORKERS = args.num_workers

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
elif args.data_mode == 'metabolomics_mutant':
    BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Metabolomics_Data', 'Mutants')
else:
    BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Mutants_Data')

IC50_MAPPING_PATH = os.path.join(os.path.dirname(__file__), 'plate_well_ic50_mapping.json')
MUTANT_MAPPING_PATH = os.path.join(os.path.dirname(__file__), 'plate_well_id_path.json')
METABOLOMICS_MAPPING_PATH = os.path.join(os.path.dirname(__file__), 'plate_metabolomics_mutant_mapping.json')

# Load drug mapping (antibiotic + concentration)
with open(IC50_MAPPING_PATH, 'r') as f:
    ic50_data = json.load(f)

# Load mutant mapping (gene IDs)
with open(MUTANT_MAPPING_PATH, 'r') as f:
    mutant_data = json.load(f)

# Load metabolomics mapping
with open(METABOLOMICS_MAPPING_PATH, 'r') as f:
    metabolomics_data = json.load(f)

# Build plate_maps based on data_mode
# Use prefixes to distinguish drug vs mutant (they share same well positions)
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

# For metabolomics_mutant mode: build plate_maps from metabolomics_data
# Collapse P{N}_T{M} -> P{N} (all timepoints share the same scrambled layout)
if args.data_mode == 'metabolomics_mutant':
    plate_maps = {}
    for plate_key, rows in metabolomics_data.items():
        physical_plate = plate_key.split('_')[0]  # P1_T1 -> P1
        if physical_plate not in plate_maps:
            plate_maps[physical_plate] = {}
            for row_letter, cols in rows.items():
                for col_num, info in cols.items():
                    well = f"{row_letter}{int(col_num):02d}"
                    plate_maps[physical_plate][well] = info['id']

all_plates = ['Plate_1', 'Plate_2', 'Plate_3', 'Plate_4', 'Plate_5', 'Plate_6']
# Override for metabolomics
if args.data_mode == 'metabolomics_mutant':
    all_plates = ['P1', 'P2', 'P3', 'P4']
if args.timepoint_split:
    all_plates = ['T1', 'T2', 'T3']  # 3-fold timepoint cross-validation

# For drug mode, plates are P1, P2, etc. in Drugs_Data folder
def get_image_paths_for_plate(plate: str) -> list[str]:
    # Metabolomics mode: plate is already P1, P2, etc.
    # Search all 3 timepoints (P1_T1, P1_T2, P1_T3)
    if args.data_mode == 'metabolomics_mutant':
        timepoints = [f"{plate}_T{t}" for t in [1, 2, 3]]
        valid_paths = []
        for tp in timepoints:
            plate_dir = os.path.join(BASE_DIR, tp)
            if not os.path.exists(plate_dir):
                continue
            paths = []
            for pattern in ['*.tif', '*.tiff', '*.png']:
                paths.extend(glob.glob(os.path.join(plate_dir, '**', pattern), recursive=True))
            for path in paths:
                well = extract_well_from_filename(os.path.basename(path))
                if well and well in plate_maps.get(plate, {}):
                    valid_paths.append(path)
        return valid_paths
    
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

def worker_init_fn(worker_id: int, seed: int = 42) -> None:
    """Module-level worker init function for multiprocessing compatibility"""
    import random
    import numpy as np
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


def build_checkpoint_state(epoch, model, stage, best_val_auc=0.0, best_val_acc=0.0,
                           best_val_loss=float('inf'), patience_counter=0,
                           optimizer=None, scheduler=None, scaler=None):
    """Snapshot the full training state so a run can be resumed exactly where it stopped."""
    state = {
        'epoch': int(epoch),
        'stage': stage,
        'model_state_dict': model.state_dict(),
        'best_val_auc': float(best_val_auc),
        'best_val_acc': float(best_val_acc),
        'best_val_loss': float(best_val_loss),
        'patience_counter': int(patience_counter),
        'args': vars(args),
        'saved_at': datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    if stage == 'sc_mil':
        prefix = 'sc_mil_'
    elif stage == 'contrastive':
        prefix = 'contrastive_'
    else:
        prefix = ''
    for name, obj in [('optimizer', optimizer), ('scheduler', scheduler), ('scaler', scaler)]:
        if obj is not None:
            state[f'{prefix}{name}_state_dict'] = obj.state_dict()
    return state


def resolve_resume_path(resume_arg: str) -> str:
    """Accept a checkpoint file or an output directory; return the checkpoint file path."""
    if os.path.isfile(resume_arg):
        return resume_arg
    if os.path.isdir(resume_arg):
        ckpt = os.path.join(resume_arg, 'checkpoint_epoch.pth')
        if os.path.isfile(ckpt):
            return ckpt
        raise FileNotFoundError(f"No checkpoint_epoch.pth found in resume dir: {resume_arg}")
    raise FileNotFoundError(f"Resume path does not exist: {resume_arg}")


def train_single_fold(test_plate: str) -> None:
    resume_ckpt = None
    resume_stage = None
    if args.resume:
        resume_path = resolve_resume_path(args.resume)
        resume_ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        if 'stage' in resume_ckpt:
            resume_stage = resume_ckpt['stage']
        elif 'sc_mil_optimizer_state_dict' in resume_ckpt:
            resume_stage = 'sc_mil'
        elif 'contrastive_optimizer_state_dict' in resume_ckpt:
            resume_stage = 'contrastive'
        elif 'optimizer_state_dict' in resume_ckpt:
            resume_stage = 'standard'
        else:
            resume_stage = None  # old-format checkpoint (model weights only)
        if 'args' in resume_ckpt:
            saved_args = resume_ckpt['args']
            for k, v in saved_args.items():
                setattr(args, k, v)
            args.resume = resume_path
            args.test_plate = test_plate
        print(f"\n=== RESUMING from {resume_path} (stage={resume_stage}, epoch={resume_ckpt.get('epoch')}) ===")
        if resume_stage is None:
            print("WARNING: checkpoint is old-format (no optimizer/stage info). "
                  "Resuming with model weights only; optimizer and LR schedule restart from scratch.")
    contrastive_done = resume_stage in ('sc_mil', 'standard')

    if args.output_dir:
        OUTPUT_DIR = os.path.join(args.output_dir, data_mode_folder, f'fold_{test_plate}')
    else:
        OUTPUT_DIR = os.path.join(SCRIPT_DIR, data_mode_folder, f'fold_{test_plate}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training fold: test_plate={test_plate}")
    print(f"{'='*60}")
    
    # Metabolomics mode: plates are P1, P2, etc. (not Plate_X)
    if args.data_mode == 'metabolomics_mutant' and not args.timepoint_split:
        test_plate_normalized = test_plate
        train_val_plates = [p for p in all_plates if p != test_plate_normalized]
        test_num = int(test_plate_normalized[-1])  # P1 -> 1, P2 -> 2
        val_num = (test_num - 2) % 4 + 1  # Cyclic previous plate (1->4, 2->1, 3->2, 4->3)
        val_plate = f"P{val_num}"
        val_plates = [val_plate] if val_plate in train_val_plates else [train_val_plates[0]]
        train_plates = [p for p in train_val_plates if p not in val_plates][:4]
    elif args.timepoint_split:
        test_plate_normalized = test_plate
        train_val_plates = []
        train_plates = []
        val_plates = []
    else:
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
        # Map Plate_1 -> P1, Plate_2 -> P2, etc.
        plate_key_map = {f'Plate_{i}': f'P{i}' for i in range(1, 7)}
        
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
    elif args.data_mode == 'metabolomics_mutant':
        base_classes = sorted(set(label for pm in plate_maps.values() for label in pm.values() if label))
        if args.include_timepoint_in_labels:
            all_classes = sorted([f"T{t}_{gene}" for t in [1, 2, 3] for gene in base_classes])
        else:
            all_classes = base_classes
        
        def get_timepoint_from_path(path: str) -> str:
            match = re.search(r'_T(\d)', path, re.IGNORECASE)
            return f"T{match.group(1)}" if match else 'T1'
        
        plate_keys_for_match = ['P1', 'P2', 'P3', 'P4'] if args.timepoint_split else list(plate_maps.keys())
        
        def metabolomics_label_extractor(path, pm=plate_maps, all_plates_local=plate_keys_for_match):
            path_lower = path.lower()
            for p in all_plates_local:
                p_lower = p.lower()  # p1, p2, etc.
                if f'/{p_lower}_t' in path_lower or f'\\{p_lower}_t\\' in path_lower:
                    plate_key = p
                    break
            else:
                return None
            well = extract_well_from_filename(os.path.basename(path))
            if well and plate_key in pm and well in pm[plate_key]:
                gene = pm[plate_key][well]
                if args.include_timepoint_in_labels:
                    tp = get_timepoint_from_path(path)
                    return f"{tp}_{gene}"
                return gene
            return None
        label_extractor = metabolomics_label_extractor
    else:  # both - use plate_maps for both drug and mutant
        plate_key_map = {f'Plate_{i}': f'P{i}' for i in range(1, 7)}
        
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
            
            # Determine source type from path (drug vs mutant directory)
            if '/drugs_data/' in path_lower or '\\drugs_data\\' in path_lower or '/drugs_data/' in path_lower:
                source_prefix = 'drug_'
            elif '/mutants_data/' in path_lower or '\\mutants_data\\' in path_lower:
                source_prefix = 'mutant_'
            else:
                # Default to drug if can't determine
                source_prefix = 'drug_'
            
            # Use composite key: drug_A01 or mutant_A01
            composite_well = f"{source_prefix}{well}"
            if composite_well and plate_key in pm and composite_well in pm[plate_key]:
                return pm[plate_key][composite_well]
            return None
        
        label_extractor = both_extractor
    
    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
    num_classes = len(all_classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {all_classes}")
    
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    
    if args.timepoint_split:
        test_tp = int(test_plate[-1])  # 'T1' -> 1, 'T2' -> 2, 'T3' -> 3
        val_tp = (test_tp - 2) % 3 + 1  # Cyclic previous timepoint (3->2, 1->3, 2->1)
        train_tp = 6 - test_tp - val_tp  # Remaining timepoint (1+2+3=6)
        print(f"Timepoint split fold: Train=T{train_tp}, Val=T{val_tp}, Test=T{test_tp} (96 classes)")
        all_timepoint_paths = []
        for plate in ['P1', 'P2', 'P3', 'P4']:
            all_timepoint_paths.extend(get_image_paths_for_plate(plate))
        
        def _get_tp_num(path):
            match = re.search(r'_T(\d)', path, re.IGNORECASE)
            return int(match.group(1)) if match else None
        
        for path in all_timepoint_paths:
            label = label_extractor(path)
            if label not in class_to_idx:
                continue
            tp = _get_tp_num(path)
            if tp == train_tp:
                train_paths.append(path)
                train_labels.append(class_to_idx[label])
            elif tp == val_tp:
                val_paths.append(path)
                val_labels.append(class_to_idx[label])
            elif tp == test_tp:
                test_paths.append(path)
                test_labels.append(class_to_idx[label])
    else:
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
        
        # Recompute class mapping with only remaining classes
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
    
    class_counts = Counter(train_labels)
    total = len(train_labels)
    # Handle classes with zero samples by using minimum count of 1
    class_weights = torch.tensor([total / (num_classes * max(class_counts[i], 1)) for i in range(num_classes)], device=device)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    train_dataset = MultiCropDataset(train_paths, train_labels, None, neighborhood=args.neighborhood, grid_size=args.grid_size, augment=True, seed=SEED, num_channels=args.num_channels, extraction_mode=args.extraction_mode, raster_crop_size=args.raster_crop_size, raster_resize_size=args.raster_resize_size, raster_num_crops=args.raster_num_crops, raster_grid_size=args.raster_grid_size)
    val_dataset = MultiCropDataset(val_paths, val_labels, None, neighborhood=args.neighborhood, grid_size=args.grid_size, augment=False, seed=SEED, num_channels=args.num_channels, extraction_mode=args.extraction_mode, raster_crop_size=args.raster_crop_size, raster_resize_size=args.raster_resize_size, raster_num_crops=args.raster_num_crops, raster_grid_size=args.raster_grid_size)
    test_dataset = MultiCropDataset(test_paths, test_labels, None, neighborhood=args.neighborhood, grid_size=args.grid_size, augment=False, seed=SEED, num_channels=args.num_channels, extraction_mode=args.extraction_mode, raster_crop_size=args.raster_crop_size, raster_resize_size=args.raster_resize_size, raster_num_crops=args.raster_num_crops, raster_grid_size=args.raster_grid_size)
    
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
        print(f"Pooling: {args.pooling}, num_heads={args.num_heads}, attention_temp={args.attention_temp}, attn_hidden_dim={args.attn_hidden_dim}")
        print(f"Classifier: MLP with {args.classifier_layers} hidden layer(s), hidden_dim={args.classifier_hidden_dim}, dropout={args.dropout}")
        model = MILEncoder(num_classes=num_classes, num_heads=args.num_heads, dropout=args.dropout, use_contrastive=True, num_channels=args.num_channels, pretrained=args.pretrained, backbone=args.backbone, pooling=args.pooling, attention_temp=args.attention_temp, attn_hidden_dim=args.attn_hidden_dim, classifier_hidden_dim=args.classifier_hidden_dim, classifier_layers=args.classifier_layers)
    else:
        print(f"Using AttentionMILModel...")
        print(f"Backbone: {args.backbone}")
        print(f"Pooling: {args.pooling}")
        print(f"Classifier: single FC layer with dropout={args.dropout}")
        model = AttentionMILModel(num_classes=num_classes, num_heads=args.num_heads, dropout=args.dropout, num_channels=args.num_channels, pretrained=args.pretrained, backbone=args.backbone, pooling=args.pooling)
    model = model.to(device)
    
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt['model_state_dict'])
        print(f"Model weights restored from checkpoint (epoch {resume_ckpt['epoch']})")
    
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
    existing_std_csvs = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'training_metrics_*.csv')))
    if resume_stage == 'standard' and existing_std_csvs:
        csv_path = existing_std_csvs[-1]
        print(f"Resuming: appending metrics to {csv_path}")
    else:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auc', 'backbone_lr', 'classifier_lr'])
    
    # TensorBoard writer - 2 cards: (Train Loss, Val Loss) and (Train Acc, Val Acc)
    tb_writer = SummaryWriter(log_dir=OUTPUT_DIR)
    
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Stage 1: Patch-Level SimCLR Pre-training (proven in papers)
    if args.use_contrastive and not contrastive_done:
        print(f"\n{'='*60}")
        print(f"Stage 1: Patch-Level SimCLR Pre-training for {args.contrastive_epochs} epochs...")
        print(f"Contrastive batch size: {args.contrastive_batch_size}")
        print(f"{'='*60}")
        
        # Create two augmented views for each image
        crop_dataset_v1 = MultiCropDataset(train_paths, train_labels, None, neighborhood=1, grid_size=args.grid_size, augment=True, seed=SEED, num_channels=args.num_channels, extraction_mode=args.extraction_mode, raster_crop_size=args.raster_crop_size, raster_resize_size=args.raster_resize_size, raster_num_crops=args.raster_num_crops, raster_grid_size=args.raster_grid_size)
        crop_dataset_v2 = MultiCropDataset(train_paths, train_labels, None, neighborhood=1, grid_size=args.grid_size, augment=True, seed=SEED+1, num_channels=args.num_channels, extraction_mode=args.extraction_mode, raster_crop_size=args.raster_crop_size, raster_resize_size=args.raster_resize_size, raster_num_crops=args.raster_num_crops, raster_grid_size=args.raster_grid_size)
        
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
        
        contrastive_start_epoch = 0
        if resume_stage == 'contrastive':
            contrastive_start_epoch = resume_ckpt.get('epoch', -1) + 1
            co = resume_ckpt.get('contrastive_optimizer_state_dict')
            cs = resume_ckpt.get('contrastive_scheduler_state_dict')
            csc = resume_ckpt.get('contrastive_scaler_state_dict')
            if co is not None and cs is not None:
                contrastive_optimizer.load_state_dict(co)
                contrastive_scheduler.load_state_dict(cs)
                if csc is not None:
                    contrastive_scaler.load_state_dict(csc)
            else:
                print("WARNING: contrastive optimizer/scheduler state missing; restarting their schedule")
            print(f"Resuming contrastive stage from epoch {contrastive_start_epoch}")
        
        for epoch in range(contrastive_start_epoch, args.contrastive_epochs):
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
            
            if (epoch + 1) % args.checkpoint_every == 0:
                torch.save(build_checkpoint_state(epoch, model, 'contrastive',
                                                  optimizer=contrastive_optimizer,
                                                  scheduler=contrastive_scheduler,
                                                  scaler=contrastive_scaler),
                           os.path.join(OUTPUT_DIR, 'checkpoint_epoch.pth'))
        
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
        
        sc_mil_start_epoch = 0
        if resume_stage == 'sc_mil':
            sc_mil_start_epoch = resume_ckpt.get('epoch', -1) + 1
            so = resume_ckpt.get('sc_mil_optimizer_state_dict')
            ss = resume_ckpt.get('sc_mil_scheduler_state_dict')
            ssc = resume_ckpt.get('sc_mil_scaler_state_dict')
            if so is not None and ss is not None:
                sc_mil_optimizer.load_state_dict(so)
                sc_mil_scheduler.load_state_dict(ss)
                if ssc is not None:
                    sc_mil_scaler.load_state_dict(ssc)
            else:
                print("WARNING: SC-MIL optimizer/scheduler state missing; restarting their schedule")
            best_val_auc = resume_ckpt.get('best_val_auc', best_val_auc)
            best_val_acc = resume_ckpt.get('best_val_acc', best_val_acc)
            best_val_loss = resume_ckpt.get('best_val_loss', best_val_loss)
            patience_counter = resume_ckpt.get('patience_counter', patience_counter)
            print(f"Resuming SC-MIL from epoch {sc_mil_start_epoch} "
                  f"(best_val_auc={best_val_auc:.4f}, best_val_acc={best_val_acc:.2f}%)")
        
        # Create CSV file for SC-MIL metrics (reuse existing file when resuming)
        existing_sc_mil_csvs = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'training_sc_mil_*.csv')))
        if resume_stage == 'sc_mil' and existing_sc_mil_csvs:
            csv_path_sc_mil = existing_sc_mil_csvs[-1]
            print(f"Resuming: appending metrics to {csv_path_sc_mil}")
        else:
            timestamp_sc_mil = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path_sc_mil = os.path.join(OUTPUT_DIR, f"training_sc_mil_{timestamp_sc_mil}.csv")
            with open(csv_path_sc_mil, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_ce_loss', 'train_sc_loss', 'train_acc', 'val_ce_loss', 'val_acc', 'val_auc', 'lr'])
        
        for epoch in range(sc_mil_start_epoch, args.sc_mil_epochs):
            epoch_start = time.time()
            train_dataset.set_epoch(epoch)
            model.train()
            run_cl_loss, run_ce_loss, correct, total = 0.0, 0.0, 0, 0
            
            for images, labels in tqdm(train_loader, desc=f'SC-MIL Epoch {epoch}', leave=False):
                images, labels = images.to(device), labels.to(device)
                sc_mil_optimizer.zero_grad()
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    # Single forward pass: get outputs, attn, crop, pooled embeddings, and instance logits
                    outputs, attn_weights, crop_embeddings, pooled_embeddings, instance_logits = model(
                        images, return_attention=True, return_crop_embeddings=True, 
                        return_pooled_embeddings=True, return_instance_logits=True
                    )
                    
                    # ============ CONTRASTIVE LOSSES ============
                    num_crops = crop_embeddings.shape[1]
                    
                    # Instance-level contrastive
                    if args.contrastive_level in ['instance', 'both']:
                        crop_emb_flat = crop_embeddings.view(-1, crop_embeddings.shape[-1]).unsqueeze(1)
                        crop_emb_flat = F.normalize(crop_emb_flat, p=2, dim=-1)
                        instance_labels_exp = labels.repeat_interleave(num_crops)
                        inst_temp = max(args.sc_mil_temp, 0.1)  # Higher temp for stability
                        criterion_inst = SupConLoss(temperature=inst_temp, contrast_mode='one')
                        instance_sc_loss = criterion_inst(crop_emb_flat, instance_labels_exp)
                    else:
                        instance_sc_loss = 0.0
                    
                    # Bag-level contrastive
                    if args.contrastive_level in ['bag', 'both']:
                        bag_embeddings = F.normalize(pooled_embeddings, p=2, dim=-1).unsqueeze(1)
                        sc_criterion = SupConLoss(temperature=args.sc_mil_temp)
                        bag_sc_loss = sc_criterion(bag_embeddings, labels)
                    else:
                        bag_sc_loss = 0.0
                    
                    # ============ CLASSIFICATION LOSSES ============
                    num_crops = crop_embeddings.shape[1]
                    instance_labels = labels.repeat_interleave(num_crops)
                    instance_weights = class_weights[instance_labels]
                    # Instance-level focal
                    instance_focal = weighted_focal_loss(
                        instance_logits.view(-1, num_classes),
                        instance_labels,
                        instance_weights
                    )
                    # Bag-level focal
                    bag_focal = weighted_focal_loss(outputs, labels, class_weights[labels])
                    
                    # ============ COMBINE LOSSES ============
                    w = args.instance_weight
                    
                    # Combined focal: instance + bag
                    total_focal = w * instance_focal + (1 - w) * bag_focal
                    # Combined contrastive: based on contrastive_level
                    total_sc = w * instance_sc_loss + (1 - w) * bag_sc_loss
                    
                    # Combined with classification vs contrastive weight
                    loss = (1 - args.sc_mil_weight) * total_focal + args.sc_mil_weight * total_sc
                
                sc_mil_scaler.scale(loss).backward()
                sc_mil_scaler.step(sc_mil_optimizer)
                sc_mil_scaler.update()
                
                run_cl_loss += total_sc.item()
                run_ce_loss += total_focal.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            sc_mil_scheduler.step()
            
            train_acc = 100. * correct / total
            avg_cl_loss = run_cl_loss / len(train_loader)
            avg_ce_loss = run_ce_loss / len(train_loader)
            
            # VALIDATION after each SC-MIL epoch
            model.eval()
            val_cl_loss, val_ce_loss = 0.0, 0.0
            val_correct, val_total = 0, 0
            all_val_preds, all_val_probs, all_val_labels = [], [], []
            
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
                for images, labels in tqdm(val_loader, desc='Validating', leave=False):
                    images, labels = images.to(device), labels.to(device)
                    outputs, _ = model(images, return_attention=True)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)
                    all_val_preds.extend(predicted.cpu().numpy())
                    all_val_probs.extend(probs.cpu().numpy())
                    all_val_labels.extend(labels.cpu().numpy())
                    val_loss = weighted_focal_loss(outputs, labels, class_weights[labels])
                    val_ce_loss += val_loss.item()
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = 100. * val_correct / val_total
            unique_val_classes = np.unique(np.array(all_val_labels))
            val_auc = compute_robust_auc(all_val_labels, all_val_probs, num_classes)
            avg_val_ce_loss = val_ce_loss / len(val_loader)
            
            print(f"SC-MIL Epoch {epoch}: CE Loss={avg_ce_loss:.4f}, SupCon Loss={avg_cl_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, Val AUC={val_auc:.4f}, Time={time.time()-epoch_start:.1f}s")
            
            # Save checkpoint every epoch (full state for resume)
            if (epoch + 1) % args.checkpoint_every == 0:
                torch.save(build_checkpoint_state(epoch, model, 'sc_mil',
                                                  best_val_auc=best_val_auc,
                                                  best_val_acc=best_val_acc,
                                                  best_val_loss=best_val_loss,
                                                  patience_counter=patience_counter,
                                                  optimizer=sc_mil_optimizer,
                                                  scheduler=sc_mil_scheduler,
                                                  scaler=sc_mil_scaler),
                           os.path.join(OUTPUT_DIR, 'checkpoint_epoch.pth'))
            
            # Save metrics to CSV
            with open(csv_path_sc_mil, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_ce_loss, avg_cl_loss, train_acc, avg_val_ce_loss, val_acc, val_auc, sc_mil_optimizer.param_groups[0]['lr']])
            
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
            
            if args.early_stopping_patience > 0:
                if val_acc > best_val_acc:
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= args.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}: val acc not improved for {args.early_stopping_patience} epochs (best={best_val_acc:.2f}%)")
                    break
        
        print(f"SC-MIL training complete!")
        # Skip standard training, go directly to evaluation
        epoch = args.sc_mil_epochs  # Mark as complete
    
    else:
        print("Training...")
        epoch = None  # Means standard training
    
    # Standard or SC-MIL training loop
    if epoch is None:
        start_epoch = 0
        if resume_stage == 'standard':
            start_epoch = resume_ckpt.get('epoch', -1) + 1
            o = resume_ckpt.get('optimizer_state_dict')
            s = resume_ckpt.get('scheduler_state_dict')
            sc = resume_ckpt.get('scaler_state_dict')
            if o is not None and s is not None:
                optimizer.load_state_dict(o)
                scheduler.load_state_dict(s)
                if sc is not None:
                    scaler.load_state_dict(sc)
            else:
                print("WARNING: optimizer/scheduler state missing; restarting their schedule")
            best_val_auc = resume_ckpt.get('best_val_auc', best_val_auc)
            best_val_acc = resume_ckpt.get('best_val_acc', best_val_acc)
            best_val_loss = resume_ckpt.get('best_val_loss', best_val_loss)
            print(f"Resuming standard training from epoch {start_epoch}")
        
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            train_dataset.set_epoch(epoch)
            model.train()
            run_loss, correct, total = 0.0, 0, 0
            
            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs, attn_weights = model(images, return_attention=True)
                    
                    # SC-MIL loss: weighted focal loss + optional entropy regularization
                    main_loss = weighted_focal_loss(outputs, labels, class_weights[labels], label_smoothing=args.label_smoothing)
                    
                    # Add attention entropy loss if specified (AEM - Attention Entropy Maximization)
                    if args.entropy_loss_weight > 0:
                        attn_ent_loss = attention_entropy_loss(attn_weights)
                        loss = main_loss + args.entropy_loss_weight * attn_ent_loss
                    else:
                        loss = main_loss
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                run_loss += main_loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        
        train_acc = 100. * correct / total
        avg_train_loss = run_loss / len(train_loader)
        
        model.eval()
        val_loss_total = 0.0
        all_preds, all_probs, all_labels = [], [], []
        
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            for images, labels in tqdm(val_loader, desc='Validating', leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs, _ = model(images, return_attention=True)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                val_loss = weighted_focal_loss(outputs, labels, class_weights[labels], label_smoothing=args.label_smoothing)
                val_loss_total += val_loss.item()
        
        val_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        unique_val_classes = np.unique(np.array(all_labels))
        val_auc = compute_robust_auc(all_labels, all_probs, num_classes)
        avg_val_loss = val_loss_total / len(val_loader)
        
        backbone_lr = optimizer.param_groups[0]['lr']
        classifier_lr = optimizer.param_groups[1]['lr']
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%, Val AUC={val_auc:.4f}, Backbone LR={backbone_lr:.2e}, Classifier LR={classifier_lr:.2e}, Time={time.time()-epoch_start:.1f}s")
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, train_acc, avg_val_loss, val_acc, val_auc, backbone_lr, classifier_lr])
        
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
        
        if (epoch + 1) % args.checkpoint_every == 0:
            torch.save(build_checkpoint_state(epoch, model, 'standard',
                                              best_val_auc=best_val_auc,
                                              best_val_acc=best_val_acc,
                                              best_val_loss=best_val_loss,
                                              optimizer=optimizer,
                                              scheduler=scheduler,
                                              scaler=scaler),
                       os.path.join(OUTPUT_DIR, 'checkpoint_epoch.pth'))
    
    if not args.skip_test:
        print("Testing...")
        checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_model_acc.pth'), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            for images, labels in tqdm(test_loader, desc='Testing', leave=False):
                images = images.to(device)
                outputs, _ = model(images, return_attention=True)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        test_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        test_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
        test_auc = roc_auc_score(test_labels_bin, np.array(all_probs), average='macro')
        test_ap = average_precision_score(test_labels_bin, np.array(all_probs), average='macro')
        
        print(f"Test Acc: {test_acc:.2f}%, Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")
        
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
            if args.output_dir:
                fold_dir = os.path.join(args.output_dir, data_mode_folder, f'fold_{test_plate}')
            else:
                fold_dir = os.path.join(SCRIPT_DIR, data_mode_folder, f'fold_{test_plate}')
            
            # Check for any checkpoint files to skip trained folds
            checkpoints = [
                os.path.join(fold_dir, 'best_model.pth'),
                os.path.join(fold_dir, 'best_model_acc.pth'),
                os.path.join(fold_dir, 'best_model_auc.pth'),
                os.path.join(fold_dir, 'best_model_loss.pth'),
            ]
            
            if any(os.path.exists(cp) for cp in checkpoints) and not args.resume:
                print(f"\nSkipping {test_plate}: already trained (checkpoint exists)")
                continue
            
            train_single_fold(test_plate)
        
        print("All folds completed!")
    else:
        train_single_fold(args.test_plate)
    
    print("Done!")






