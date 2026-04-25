"""
=============================================================================
SupConLoss - Supervised Contrastive Learning Loss
=============================================================================
Based on: https://arxiv.org/abs/2004.11362
Supervised Contrastive Learning (Khosla et al., 2020)
Original implementation: https://github.com/hobbitlong/SupContrast

This module implements the supervised contrastive loss for MIL (Multiple Instance Learning)
where instances/crops within the same bag are treated as positive pairs.

Reference for MIL adaptation: ItS2CLR (CVPR 2023)
https://github.com/Kangningthu/ItS2CLR

KEY CONCEPTS:
-----------
1. Positive pairs: samples with same label (in MIL, instances from same bag)
2. Negative pairs: samples with different labels
3. Temperature: controls sharpness of softmax (lower = more aggressive learning)
   - Typical: 0.07-0.1 for supervised, 0.5 for SimCLR
4. Features MUST be L2 normalized for proper scaling
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    
    Computes contrastive loss where:
    - Positive pairs: samples with the same label
    - Negative pairs: samples with different labels
    
    For MIL, bags with the same label create positive pairs across all their instances.
    
    Args:
        temperature: Controls hardness of softmax. Lower = sharper (more aggressive).
                   Typical: 0.07-0.1 for supervised contrastive.
        contrast_mode: 'all' uses all views as anchors, 'one' uses first view only.
        base_temperature: Base temperature for gradient scaling.
    """
    
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: [batch_size, n_views, feature_dim] - MUST be L2 normalized!
            labels: [batch_size] - class labels (bag labels for MIL)
            mask: [batch_size, batch_size] - optional pre-computed positive mask
            
        Returns:
            loss: scalar contrastive loss
            
        How it works:
        -------------
        1. Create positive mask from labels (mask[i,j]=1 if labels[i]==labels[j])
        2. Compute normalized dot products (cosine similarity)
        3. Apply temperature scaling
        4. Mask out self-contrast (diagonal)
        5. Compute log-softmax over positive pairs
        6. Average and scale by temperature
        """
        device = features.device
        
        # Validate input
        if len(features.shape) < 3:
            raise ValueError(
                f'`features` needs to be [batch_size, n_views, feature_dim], '
                f'at least 3 dimensions required. Got {features.shape}'
            )
        
        # Flatten if needed
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        
        batch_size = features.shape[0]
        n_views = features.shape[1]
        
        # =========================================================================
        # MASK CREATION: Positive pairs from labels
        # =========================================================================
        # If neither labels nor mask provided -> SimCLR mode (self-contrast only)
        # If labels provided -> supervised mode (same label = positive)
        # If mask provided -> use custom mask
        
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # SimCLR: identity mask (only self is positive)
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # Supervised: positive = same label
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match batch size')
            # mask[i,j] = 1 if labels[i] == labels[j]
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            # Use provided mask
            mask = mask.float().to(device)
        
        # =========================================================================
        # ANCHOR SELECTION: Which views to use as anchors
        # =========================================================================
        contrast_count = n_views
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]  # First view only
            anchor_count = 1
        elif self.contrast_mode == 'all':
            # All views as anchors: flatten [B, n_views, D] -> [B*n_views, D]
            anchor_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
            anchor_count = contrast_count
        else:
            raise ValueError(f'Unknown contrast_mode: {self.contrast_mode}')
        
        # =========================================================================
        # COMPUTE LOGITS: Dot products / temperature
        # =========================================================================
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, anchor_feature.T),
            self.temperature
        )
        
        # Numerical stability: subtract max
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # =========================================================================
        # MASK OUT SELF-CONTRAST
        # =========================================================================
        mask = mask.repeat(anchor_count, contrast_count)
        
        # Create diagonal mask (0 on diagonal to exclude self)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            dim=1,
            index=torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            value=0
        )
        mask = mask * logits_mask
        
        # =========================================================================
        # COMPUTE LOG-PROB: Softmax over valid pairs
        # =========================================================================
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # =========================================================================
        # MEAN OVER POSITIVE PAIRS
        # =========================================================================
        mask_pos_pairs = mask.sum(1)
        # Handle edge case: no positives (avoid division by zero)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 
                                  torch.ones_like(mask_pos_pairs),
                                  mask_pos_pairs)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        
        # =========================================================================
        # LOSS: Scale and average
        # =========================================================================
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss


class SupConLossMIL(nn.Module):
    """
    Supervised Contrastive Loss for Multiple Instance Learning.
    
    Adapts SupConLoss for MIL by:
    - Treating all instances within a bag as positive pairs
    - Using bag-level labels expanded to instance level
    
    Usage:
        # features: [num_bags, n_crops_per_bag, feature_dim]
        # bag_labels: [num_bags]
        criterion = SupConLossMIL(temperature=0.07)
        loss = criterion(features, bag_labels, num_crops_per_bag)
    """
    
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLossMIL, self).__init__()
        self.temperature = temperature
        self.supcon = SupConLoss(
            temperature=temperature,
            contrast_mode='all',
            base_temperature=base_temperature
        )
    
    def forward(self, features, bag_labels, num_crops_per_bag=9):
        """
        Args:
            features: [batch_size, n_crops, feature_dim]
            bag_labels: [batch_size] - bag-level labels
            num_crops_per_bag: number of instances/crops per bag (default: 9 for 3x3 MIL)
            
        Returns:
            loss: scalar contrastive loss
        SupCon expects [batch_size * n_crops, n_views, feature_dim]
        Here we treat each crop as a separate view
        """
        batch_size = features.shape[0]
        num_crops = features.shape[1]
        
        # Reshape: [B, n_crops, D] -> [B*n_crops, 1, D]
        features_flat = features.view(-1, 1, features.shape[-1])
        
        # Expand bag labels to instance level
        # Each bag label repeated for its crops: [0,1,0,1] -> [0,0,0,1,1,1,0,0,0] for n_crops=3
        instance_labels = bag_labels.repeat_interleave(num_crops)
        
        # Forward to SupCon
        return self.supcon(features_flat, instance_labels)


# =============================================================================
# HELPER FUNCTION FOR EASY INTEGRATION
# =============================================================================

def create_supcon_loss(temperature=0.07, mil_mode=False, num_crops=9):
    """
    Factory function to create SupConLoss.
    
    Args:
        temperature: Contrastive temperature (default: 0.07)
        mil_mode: If True, return MIL-adapted loss
        num_crops: Number of crops per bag (for MIL mode)
        
    Returns:
        SupConLoss or SupConLossMIL instance
    """
    if mil_mode:
        return SupConLossMIL(temperature=temperature)
    return SupConLoss(temperature=temperature)


# =============================================================================
# USAGE EXAMPLES (stored as comments, not executable)
# =============================================================================
# Example 1: Standard SupCon with classification
# features = torch.randn(4, 2, 128)
# labels = torch.tensor([0, 1, 0, 1])
# features = F.normalize(features, p=2, dim=-1)
# criterion = SupConLoss(temperature=0.07)
# loss = criterion(features, labels)


# =============================================================================
# HOW TO USE IN TRAINING
# =============================================================================
# 
# from supcon_loss import SupConLoss, SupConLossMIL
# import torch.nn.functional as F
# 
# # Create loss criterion
# criterion = SupConLossMIL(temperature=0.07)
# 
# # In training loop:
# embeddings = model.get_mil_embeddings(images)  # [B, n_crops, feature_dim]
# embeddings = F.normalize(embeddings, p=2, dim=-1)
# supcon_loss = criterion(embeddings, labels, num_crops_per_bag=9)
# ce_loss = F.cross_entropy(logits, labels)
# total_loss = ce_loss + 0.3 * supcon_loss  # 0.3 weight