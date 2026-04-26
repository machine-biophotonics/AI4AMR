"""
================================================================================
ItS2CLR SupConLoss - Iterative Self-Paced Supervised Contrastive Learning
================================================================================
Based on: https://arxiv.org/abs/2210.09452 (CVPR 2023)
Official implementation: https://github.com/Kangningthu/ItS2CLR

KEY CONCEPTS FROM PAPER:
-----------------------
1. Pair Mode 2 (negneg): Use negative samples as query
   - Positive: samples from bags with same label AND both are negative (low conf)
   - Used during warmup before positive instance confidence established

2. Pair Mode 1 (pospos): Use positive samples as query
   - Positive: samples from bags with same label AND both are positive (high conf)
   - Used after warmup when positive instance confidence is established

3. mask_uncertain_neg: Mask out uncertain negative instances
   - Instances with pseudo labels but not confident are excluded from neg pairs

4. Two labels:
   - label: instance-level pseudo label
   - bag_label: bag-level ground truth label
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss with Pair Modes (ItS2CLR CVPR 2023).
    
    Supports two modes matching the official ItS2CLR implementation:
    - pair_mode=2 (negneg): Negative samples as query (warmup phase)
    - pair_mode=1 (pospos): Positive samples as query (after warmup)
    
    Args:
        temperature: Controls hardness of softmax (default: 0.07)
        pair_mode: 1 (pospos) or 2 (negneg) - which samples as query
        mask_uncertain_neg: If True, mask uncertain negatives
        base_temperature: Base temperature for loss scaling
    """
    
    def __init__(self, temperature=0.07, pair_mode=2, mask_uncertain_neg=False, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.pair_mode = pair_mode
        self.mask_uncertain_neg = mask_uncertain_neg
        self.base_temperature = base_temperature

    def forward(self, features, labels, bag_labels=None):
        """
        Args:
            features: [batch_size, n_views, feature_dim] - MUST be L2 normalized!
            labels: [batch_size] - instance-level pseudo labels (or bag labels for MIL)
            bag_labels: [batch_size] - bag-level ground truth labels (optional)
            
        Returns:
            loss: scalar contrastive loss
        """
        device = features.device
        
        if len(features.shape) < 3:
            raise ValueError(
                f'`features` needs to be [batch_size, n_views, feature_dim], '
                f'at least 3 dimensions required. Got {features.shape}'
            )
        
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        
        batch_size = features.shape[0]
        n_views = features.shape[1]
        
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match batch size')
        
        mask_positive = torch.eq(labels, labels.T).float().to(device)
        
        if bag_labels is not None:
            bag_labels = bag_labels.contiguous().view(-1, 1)
            mask_same_bag = torch.eq(bag_labels, bag_labels.T).float().to(device)
            mask_positive = mask_positive * mask_same_bag
        
        if self.mask_uncertain_neg and hasattr(self, 'uncertain_mask') and self.uncertain_mask is not None:
            uncertain_mask = self.uncertain_mask.to(device)
            mask_positive = mask_positive * uncertain_mask
        
        features = features.view(batch_size * n_views, -1)
        anchor_feature = features
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, anchor_feature.T),
            self.temperature
        )
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        mask = mask_positive.repeat(n_views, n_views)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            dim=1,
            index=torch.arange(batch_size * n_views).view(-1, 1).to(device),
            value=0
        )
        mask = mask * logits_mask
        
        if self.pair_mode == 2:
            mask = mask
        elif self.pair_mode == 1:
            mask_pos_same_label = torch.eq(labels, labels.T).float().to(device)
            if bag_labels is not None:
                mask_pos_same_label = mask_pos_same_label * mask_same_bag
            mask_pos_same_label = mask_pos_same_label.repeat(n_views, n_views)
            mask = mask * mask_pos_same_label
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 
                                      torch.ones_like(mask_pos_pairs),
                                      mask_pos_pairs)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(n_views, batch_size).mean()
        
        return loss


class ItS2CLRSupConLoss(nn.Module):
    """
    ItS2CLR SupCon Loss combining pair modes 1 and 2.
    
    Based on official ItS2CLR implementation.
    Automatically handles warmup vs iterative phases.
    """
    
    def __init__(self, temperature=0.07, mask_uncertain_neg=False, base_temperature=0.07):
        super(ItS2CLRSupConLoss, self).__init__()
        self.temperature = temperature
        self.mask_uncertain_neg = mask_uncertain_neg
        self.base_temperature = base_temperature
        
        self.criterion_pair2 = SupConLoss(
            temperature=temperature, 
            pair_mode=2, 
            mask_uncertain_neg=mask_uncertain_neg,
            base_temperature=base_temperature
        )
        self.criterion_pair1 = SupConLoss(
            temperature=temperature, 
            pair_mode=1, 
            mask_uncertain_neg=False,
            base_temperature=base_temperature
        )
    
    def forward(self, features, labels, bag_labels=None, pair_mode=2):
        """
        Args:
            features: [batch_size, n_views, feature_dim]
            labels: instance-level pseudo labels
            bag_labels: bag-level ground truth labels (optional)
            pair_mode: 1 (pospos) or 2 (negneg)
        """
        if pair_mode == 2:
            return self.criterion_pair2(features, labels, bag_labels)
        else:
            return self.criterion_pair1(features, labels, bag_labels)
    
    def set_uncertain_mask(self, uncertain_mask):
        """Set the uncertain mask for filtering uncertain negatives."""
        self.criterion_pair2.uncertain_mask = uncertain_mask


class SupConLossMIL(nn.Module):
    """
    Supervised Contrastive Loss for Multiple Instance Learning (SC-MIL style).
    
    For standard SC-MIL (single-stage joint training).
    Uses bag-level labels to create positive pairs across instances.
    """
    
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLossMIL, self).__init__()
        self.temperature = temperature
        self.supcon = SupConLoss(
            temperature=temperature,
            pair_mode=2,
            base_temperature=base_temperature
        )
    
    def forward(self, features, bag_labels, num_crops_per_bag=9):
        """
        Args:
            features: [batch_size, n_crops, feature_dim]
            bag_labels: [batch_size] - bag-level labels
            num_crops_per_bag: crops per bag (default: 9 for 3x3 MIL)
        """
        batch_size = features.shape[0]
        num_crops = features.shape[1]
        
        features_flat = features.view(-1, 1, features.shape[-1])
        instance_labels = bag_labels.repeat_interleave(num_crops)
        
        return self.supcon(features_flat, instance_labels)


def create_supcon_loss(temperature=0.07, mode='its2clr', num_crops=9):
    """
    Factory function to create the appropriate SupCon loss.
    
    Args:
        temperature: Contrastive temperature (default: 0.07)
        mode: 'its2clr' for two-phase ItS2CLR, 'scmil' for single-stage SC-MIL
        num_crops: Number of crops per bag (for MIL mode)
    """
    if mode == 'its2clr':
        return ItS2CLRSupConLoss(temperature=temperature)
    elif mode == 'scmil':
        return SupConLossMIL(temperature=temperature)
    else:
        return SupConLoss(temperature=temperature)


# =============================================================================
# USAGE EXAMPLES
# =============================================================================
# 
# ItS2CLR (CVPR 2023) - Two-phase training:
# 
# from supcon_loss import ItS2CLRSupConLoss
# criterion = ItS2CLRSupConLoss(temperature=0.07)
# 
# # During warmup (pair_mode=2, negneg):
# sc_loss = criterion(embeddings, pseudo_labels, bag_labels, pair_mode=2)
# 
# # After warmup (pair_mode=1, pospos):
# sc_loss = criterion(embeddings, pseudo_labels, bag_labels, pair_mode=1)
# 
# SC-MIL (WACV 2024) - Single-stage:
# 
# from supcon_loss import SupConLossMIL
# criterion = SupConLossMIL(temperature=0.07)
# sc_loss = criterion(embeddings, bag_labels, num_crops_per_bag=9)