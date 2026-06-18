"""
CS-ARM-BN: Control-Stabilized Adaptive Risk Minimization via Batch Normalization

Implements the adaptation method from:
"Closing the Domain Gap in Biomedical Imaging by In-Context Control Samples"
(arXiv:2604.20824, Apr 2026)

Core idea: At inference time, recompute BatchNorm running statistics using the
target domain's data. Model is trained with standard ERM (per-plate batching).
"""

import torch
import torch.nn as nn


def update_bn_with_full_domain(loader, model, device):
    """
    Recompute BatchNorm running_mean/running_var using the entire target domain.

    Matches the official CS-ARM-BN implementation:
    1. model.train() — BN uses batch statistics
    2. momentum=None — running stats update as simple average
    3. For each batch: forward, save weighted stats, reset running stats
    4. Compute global mean/variance as weighted average across all batches

    Args:
        loader: DataLoader yielding (images, labels) where images shape is (B, N, C, H, W)
        model: MILEncoder model with .backbone containing BatchNorm layers
        device: torch device
    """
    model.train()

    curr_batch_mean = {}
    curr_batch_var = {}
    sizes = []

    for nm, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = None
            m.reset_running_stats()

    with torch.no_grad():
        for images, _ in loader:
            batch_size = images.shape[0]
            images = images.to(device)

            _ = model(images)

            sizes.append(batch_size)

            for nm, m in model.named_modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    mean = m.running_mean * batch_size
                    var = m.running_var * batch_size

                    curr_batch_mean.setdefault(nm, []).append(mean)
                    curr_batch_var.setdefault(nm, []).append(var)

                    m.momentum = None
                    m.reset_running_stats()

    sizes = torch.tensor(sizes, device=device)

    for nm, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            curr_mean = torch.stack(curr_batch_mean[nm])
            curr_var = torch.stack(curr_batch_var[nm])

            m.running_mean.data = torch.sum(curr_mean, dim=0) / torch.sum(sizes)
            m.running_var.data = torch.sum(curr_var, dim=0) / torch.sum(sizes)

    model.eval()

