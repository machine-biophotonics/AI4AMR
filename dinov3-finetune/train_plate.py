import json
import os
import logging
import argparse
import random
import csv
import numpy as np
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dino_finetune.plate_dataset import create_datasets
from dino_finetune.model.plate_classifier import DINOEncoderLoRAForClassification


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer wrapper."""
    def __init__(self, params, base_optimizer, rho=0.1, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        if isinstance(base_optimizer, torch.optim.Optimizer):
            self.base_optimizer = base_optimizer
            for group in self.base_optimizer.param_groups:
                group['rho'] = rho
        else:
            self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Focal Loss implementation."""
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


def weighted_focal_loss(logits, targets, weights, alpha=0.25, gamma=2.0, label_smoothing=0.1):
    """Weighted Focal Loss (combined class and domain weights) with label smoothing."""
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none', label_smoothing=label_smoothing)
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    # Apply weights
    weighted = focal * weights
    return weighted.mean()


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    metrics: dict,
    collect_predictions: bool = False,
) -> Optional[List[dict]]:
    val_loss = 0.0
    # For balanced accuracy
    # Determine number of classes from model
    # Assuming model has a decoder attribute with a Linear layer
    if hasattr(model, 'decoder'):
        # decoder is nn.Sequential, last layer is Linear
        last_layer = model.decoder[-1]
        if isinstance(last_layer, nn.Linear):
            n_classes = last_layer.out_features
        else:
            n_classes = 85  # fallback
    else:
        n_classes = 85
    
    correct_per_class = torch.zeros(n_classes, dtype=torch.long, device='cuda')
    total_per_class = torch.zeros(n_classes, dtype=torch.long, device='cuda')
    
    model.eval()
    predictions = [] if collect_predictions else None
    with torch.no_grad():
        for images, labels, plates, metadata in tqdm(val_loader, desc='Validation', leave=False):
            images = images.float().cuda()
            labels = labels.long().cuda()
            
            logits = model(images)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(logits.data, 1)
            for c in range(n_classes):
                correct_per_class[c] += ((predicted == labels) & (labels == c)).sum()
                total_per_class[c] += (labels == c).sum()
            
            if collect_predictions:
                # Store per-sample predictions
                for i in range(len(labels)):
                    pred_dict = {
                        'image_path': metadata['image_path'][i],
                        'crop_x': metadata['crop_x'][i].item() if hasattr(metadata['crop_x'][i], 'item') else metadata['crop_x'][i],
                        'crop_y': metadata['crop_y'][i].item() if hasattr(metadata['crop_y'][i], 'item') else metadata['crop_y'][i],
                        'true_label': labels[i].item(),
                        'predicted_label': predicted[i].item(),
                        'logits': logits[i].detach().cpu().tolist(),
                    }
                    predictions.append(pred_dict)
    
    # Compute balanced accuracy (macro average)
    per_class_acc = correct_per_class.float() / total_per_class.float()
    # Replace NaN with 0 (when total_per_class == 0)
    per_class_acc = torch.nan_to_num(per_class_acc, nan=0.0)
    balanced_acc = per_class_acc.mean().item()
    
    total_correct = correct_per_class.sum().item()
    total_samples = total_per_class.sum().item()
    
    metrics["val_loss"].append(val_loss / len(val_loader))
    metrics["val_acc"].append(total_correct / total_samples if total_samples > 0 else 0.0)
    metrics["val_balanced_acc"].append(balanced_acc)
    return predictions


def finetune_dino(config, encoder):
    # Set random seed for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    model = DINOEncoderLoRAForClassification(
        encoder=encoder,
        r=config.r,
        emb_dim=config.emb_dim,
        n_classes=config.n_classes,
        use_lora=config.use_lora,
        img_dim=config.img_dim,
        dropout=config.dropout,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    ).cuda()

    if config.lora_weights:
        model.load_parameters(config.lora_weights)

    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        data_root=config.data_root,
        label_json_path=config.label_json_path,
        stain_augmentation=config.stain_augmentation,
        target_size=config.img_dim,
        seed=config.seed,
    )
    
    # Compute domain weights per plate (training plates only)
    plate_counts = {}
    for _, _, plate, _, _ in train_dataset.samples:
        plate_counts[plate] = plate_counts.get(plate, 0) + 1
    total_samples = sum(plate_counts.values())
    print(f"Training samples per plate: {plate_counts}")
    
    # Compute weights as per paper: u_d = n_d^{-1/2}, v_d = min(u_d, 3 * min_j u_j), w_d = v_d / sum(v_j)
    u = {p: n ** -0.5 for p, n in plate_counts.items()}
    min_u = min(u.values())
    v = {p: min(val, 3 * min_u) for p, val in u.items()}
    sum_v = sum(v.values())
    domain_weights_dict = {p: w / sum_v for p, w in v.items()}
    print(f"Domain weights: {domain_weights_dict}")
    
    # Compute class weights (inverse frequency)
    class_counts = {}
    for _, class_idx, _, _, _ in train_dataset.samples:
        class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
    total_samples = sum(class_counts.values())
    n_classes = config.n_classes
    class_weights = {}
    for c in range(n_classes):
        count = class_counts.get(c, 1)  # avoid division by zero
        class_weights[c] = total_samples / (n_classes * count)
    # Normalize class weights to sum to n_classes
    sum_class_weights = sum(class_weights.values())
    for c in class_weights:
        class_weights[c] = class_weights[c] * n_classes / sum_class_weights
    print(f"Class weights: {class_weights}")
    
    # Helper to get domain weights tensor for a batch of plates
    def get_domain_weights(plates: List[str]) -> torch.Tensor:
        weights = [domain_weights_dict[p] for p in plates]
        return torch.tensor(weights, dtype=torch.float32, device='cuda')
    
    # Helper to get class weights tensor for a batch of class indices
    def get_class_weights(class_idxs: List[int]) -> torch.Tensor:
        weights = [class_weights[c] for c in class_idxs]
        return torch.tensor(weights, dtype=torch.float32, device='cuda')
    
    # Helper to get combined weights (class * domain) normalized to mean=1
    def get_combined_weights(plates: List[str], class_idxs: List[int]) -> torch.Tensor:
        class_w = get_class_weights(class_idxs)
        domain_w = get_domain_weights(plates)
        combined = class_w * domain_w
        # Normalize to mean=1 for stable loss scale
        mean_weight = combined.mean()
        if mean_weight > 0:
            combined = combined / mean_weight
        return combined
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # Dataset handles shuffling via set_epoch
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Loss function: Focal Loss
    criterion = lambda logits, targets: focal_loss(logits, targets, alpha=config.focal_alpha, gamma=config.focal_gamma)
    
    # Optimizer: AdamW with separate parameters for LoRA and decoder
    param_groups = []
    lora_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if 'linear_a_q' in name or 'linear_b_q' in name or 'linear_a_v' in name or 'linear_b_v' in name:
            lora_params.append(param)
        elif 'decoder' in name:
            decoder_params.append(param)
        else:
            # Freeze all other parameters (encoder non-LoRA)
            param.requires_grad = False
    if config.use_lora and lora_params:
        param_groups.append({'params': lora_params, 'lr': config.lr})
    if decoder_params:
        param_groups.append({'params': decoder_params, 'lr': config.lr * 10})  # higher LR for decoder
    
    # SAM optimizer wrapping AdamW
    base_optimizer = optim.AdamW(param_groups, lr=config.lr, weight_decay=config.weight_decay, eps=1e-7)
    optimizer = SAM(model.parameters(), base_optimizer, rho=config.rho)
    
    # Scheduler - use base_optimizer for SAM
    warmup_epochs = min(config.warmup_epochs, config.epochs)
    remaining_epochs = config.epochs - warmup_epochs
    T_max = max(1, remaining_epochs)  # Avoid division by zero
    warmup_sched = LambdaLR(optimizer.base_optimizer, lambda epoch: min(1.0, (epoch + 1) / warmup_epochs) if warmup_epochs > 0 else 1.0)
    cos_sched = CosineAnnealingLR(optimizer.base_optimizer, T_max=T_max, eta_min=config.min_lr)
    scheduler = SequentialLR(optimizer.base_optimizer, schedulers=[warmup_sched, cos_sched], milestones=[warmup_epochs])
    
    # Checkpoint functions
    def save_checkpoint(epoch, path):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.base_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
        }
        torch.save(checkpoint, path)
        logging.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.base_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        metrics.update(checkpoint['metrics'])
        logging.info(f"Checkpoint loaded: {path}, resuming from epoch {start_epoch}")
        return start_epoch
    
    metrics = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_balanced_acc": [],
    }
    
    # Setup CSV logging
    csv_path = f"output/{config.exp_name}_metrics.csv"
    os.makedirs("output", exist_ok=True)
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_balanced_acc'])
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda')
    gradient_clip_norm = 1.0
    best_balanced_acc = 0.0
    early_stopping_counter = 0
    start_epoch = 0
    
    # Resume from checkpoint if provided
    if config.resume:
        checkpoint = torch.load(config.resume)
        model.load_parameters(config.resume)
        # Try to extract epoch from filename if available
        import re
        match = re.search(r'_e(\d+)\.pt$', config.resume)
        if match:
            start_epoch = int(match.group(1)) + 1
            logging.info(f"Resuming from epoch {start_epoch}")
        else:
            # Try last or best checkpoint
            if 'last' in config.resume:
                start_epoch = len(metrics["train_loss"])
                logging.info(f"Resuming from last checkpoint, epoch {start_epoch}")
            else:
                logging.info(f"Loaded checkpoint but could not determine epoch, starting from 0")
        
        # Load best balanced acc if available in checkpoint
        if 'best_balanced_acc' in checkpoint:
            best_balanced_acc = checkpoint['best_balanced_acc']
        
    for epoch in range(start_epoch, config.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        batch_count = 0
        
        for images, labels, plates, metadata in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            images = images.float().cuda()
            labels = labels.long().cuda()
            combined_weights = get_combined_weights(plates, labels.cpu().tolist())
            
            # SAM: First forward-backward pass
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = weighted_focal_loss(logits, labels, combined_weights, 
                                         alpha=config.focal_alpha, gamma=config.focal_gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            
            # SAM first step
            optimizer.first_step(zero_grad=True)
            
            # SAM: Second forward-backward pass with perturbed weights
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = weighted_focal_loss(logits, labels, combined_weights, 
                                         alpha=config.focal_alpha, gamma=config.focal_gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            
            # SAM second step
            optimizer.second_step()
            scaler.update()
            
            running_loss += loss.item()
            # Compute accuracy
            _, predicted = torch.max(logits, 1)
            running_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            batch_count += 1
            if config.debug and batch_count >= 2:
                break
        
        avg_train_loss = running_loss / batch_count
        train_acc = running_correct / total_samples if total_samples > 0 else 0.0
        metrics["train_loss"].append(avg_train_loss)
        metrics["train_acc"].append(train_acc)
        
        scheduler.step()
        
        if not config.debug:
            validate_epoch(model, val_loader, criterion, metrics)
            # Save last model (overwrite)
            model.save_parameters(f"output/{config.exp_name}_last.pt")
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                model.save_parameters(f"output/{config.exp_name}_e{epoch}.pt")
            
            current_balanced_acc = metrics['val_balanced_acc'][-1]
            if current_balanced_acc > best_balanced_acc + config.min_delta:
                best_balanced_acc = current_balanced_acc
                model.save_parameters(f"output/{config.exp_name}_best.pt")
                logging.info(f"New best model saved with balanced accuracy: {best_balanced_acc:.4f}")
                # Reset early stopping counter
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            logging.info(
                f"Epoch: {epoch} - train loss: {avg_train_loss:.4f} - train acc: {train_acc:.4f} "
                f"- val loss {metrics['val_loss'][-1]:.4f} - val acc: {metrics['val_acc'][-1]:.4f} "
                f"- balanced acc: {current_balanced_acc:.4f} - early stop counter: {early_stopping_counter}/{config.patience}"
            )
            
            # Write to CSV
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch, avg_train_loss, train_acc, 
                                 metrics['val_loss'][-1], metrics['val_acc'][-1], current_balanced_acc])
            
            # Save LAST checkpoint (every epoch) - always save immediately
            model.save_parameters(f"output/{config.exp_name}_last.pt")
            
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                model.save_parameters(f"output/{config.exp_name}_e{epoch}.pt")
            
            # Early stopping check
            if early_stopping_counter >= config.patience:
                logging.info(f"Early stopping triggered at epoch {epoch}. Best balanced accuracy: {best_balanced_acc:.4f}")
                break

    # Save final model
    model.save_parameters(f"output/{config.exp_name}.pt")
    with open(f"output/{config.exp_name}_metrics.json", "w") as f:
        json.dump(metrics, f)
    
    # Load best model for test evaluation if available
    best_model_path = f"output/{config.exp_name}_best.pt"
    if os.path.exists(best_model_path):
        logging.info(f"Loading best model from {best_model_path}")
        model.load_parameters(best_model_path)
    else:
        logging.warning("Best model not found, using final model for test evaluation")
    
    # Evaluate on test set
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_metrics = {
        "val_loss": [],
        "val_acc": [],
        "val_balanced_acc": [],
    }
    test_predictions = validate_epoch(model, test_loader, criterion, test_metrics, collect_predictions=True)
    logging.info(f"Test accuracy: {test_metrics['val_acc'][0]:.4f}")
    with open(f"output/{config.exp_name}_test_metrics.json", "w") as f:
        json.dump(test_metrics, f)
    if test_predictions is not None:
        with open(f"output/{config.exp_name}_test_predictions.json", "w") as f:
            json.dump(test_predictions, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plate Classification with DINOv3 LoRA")
    parser.add_argument("--exp_name", type=str, default="plate_lora", help="Experiment name")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--size", type=str, default="large", help="DINOv3 size: small, base, large, huge")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA")
    parser.add_argument("--img_dim", type=int, nargs=2, default=(224, 224), help="Image dimensions (height width)")
    parser.add_argument("--lora_weights", type=str, default=None, help="Load LoRA weights from file")
    
    # Dataset
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of plate images")
    parser.add_argument("--label_json_path", type=str, required=True, help="Path to plate_well_id_path.json")
    parser.add_argument("--stain_augmentation", action="store_true", help="Use stain augmentation")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument("--warmup_epochs", type=int, default=6, help="Warmup epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout for classification head")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA scaling factor alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--focal_alpha", type=float, default=0.25, help="Focal loss alpha")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--min_delta", type=float, default=0.001, help="Minimum improvement for early stopping")
    parser.add_argument("--rho", type=float, default=0.1, help="SAM perturbation radius")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path (.pt file with model_state_dict)")
    
    config = parser.parse_args()
    
    # Dataset configuration - use final_effnet model JSON for 96 classes
    config.n_classes = 96  # gene perturbation classification (96 classes including NC and WT controls)
    config.label_json_path = "/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/sam_effnet/plate_well_id_path.json"
    
    # Model configuration
    config.patch_size = 16
    # Load encoder
    # We need to map size to model name
    # For ViT-L/16, we have 'dinov3_vitl16' pretrained on satellite imagery
    # We'll load from local weights
    weights_path = '/home/student/Desktop/2025_12_19 CRISPRi Reference Plate Imaging/dino_weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
    import sys
    sys.path.insert(0, '../dinov3')
    import dinov3
    encoder = torch.hub.load('../dinov3', 'dinov3_vitl16', source='local', weights=weights_path).cuda()
    config.emb_dim = encoder.num_features
    
    logging.basicConfig(level=logging.INFO)
    finetune_dino(config, encoder)