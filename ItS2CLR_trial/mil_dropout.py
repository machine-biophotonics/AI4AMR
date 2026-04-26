"""
MIL-Dropout: ICML 2025
Paper: "How Effective Can Dropout Be in Multiple Instance Learning?"
GitHub: https://github.com/ChongQingNoSubway/MILDropout

This dropout drops the top-k most important instances during training
for better generalization in MIL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MILDropout(nn.Module):
    """
    MIL-Dropout: Drops top-k most important instances during training.
    
    Key idea: Instead of random dropout, drop the most important instances
    (based on attention weights) to improve generalization.
    
    Args:
        topk: Number of top instances to consider for dropping (default: 3)
        kernel: Number of similar instances to remove (default: 7)
    """
    
    def __init__(self, topk=3, kernel=7):
        super(MILDropout, self).__init__()
        self.topk = topk
        self.kernel = kernel
        
    def forward(self, input):
        """
        Args:
            input: [batch, num_instances, feature_dim]
            
        Returns:
            Masked input with top-k instances dropped during training
        """
        if not self.training:
            return input
        
        batch_size, n, feature_dim = input.shape
        
        if n == 1 or self.topk == 0:
            return input
        
        # Compute importance scores (mean across features)
        importances = torch.mean(input, dim=2, keepdim=True)  # [batch, n, 1]
        importances = torch.sigmoid(importances)
        
        # Generate mask for each sample in batch
        masked_input = []
        for b in range(batch_size):
            mask = self.generate_mask_single(importances[b], input[b])  # [n, 1]
            masked_instance = input[b] * mask
            masked_input.append(masked_instance)
        
        return torch.stack(masked_input, dim=0)
    
    def generate_mask_single(self, importance, features):
        """
        Generate mask for a single bag.
        
        Algorithm:
        1. Sort instances by importance (descending)
        2. Take top-k instances as most important
        3. Find similar instances using cosine similarity
        4. Remove similar instances from the top-k set
        5. Return mask where dropped instances = 0
        """
        n, f = features.shape
        
        # Sort by importance
        importance_flat = importance.squeeze()  # [n]
        _, sorted_idx = torch.sort(importance_flat, descending=True)
        
        topk_idx = sorted_idx[:self.topk]
        remain_idx = sorted_idx[self.topk:]
        
        if len(remain_idx) == 0:
            return torch.ones(n, 1, device=features.device)
        
        # Get top-k and remaining features
        topk_features = features[topk_idx]  # [topk, f]
        remain_features = features[remain_idx]  # [n-topk, f]
        
        # Compute cosine similarity between top-k and remaining
        topk_norm = F.normalize(topk_features, dim=1, p=2)  # [topk, f]
        remain_norm = F.normalize(remain_features, dim=1, p=2)  # [n-topk, f]
        
        similarity = torch.mm(topk_norm, remain_norm.T)  # [topk, n-topk]
        
        # Find most similar instances
        _, sim_sorted_idx = torch.sort(similarity, dim=1, descending=True)
        
        # Get indices to delete (most similar to top-k)
        delete_idx = sim_sorted_idx[:, :self.kernel].flatten()
        delete_idx = torch.unique(delete_idx)
        
        # Get original indices in remain set
        idx_remain_original = remain_idx[delete_idx]
        
        # Combine dropped indices
        dropped_idx = torch.cat([topk_idx, idx_remain_original])
        
        # Create mask (1 = keep, 0 = drop)
        mask = torch.ones(n, 1, device=features.device)
        mask[dropped_idx] = 0
        
        return mask


class MILDropoutSimple(nn.Module):
    """
    Simpler MIL-Dropout: Randomly drop instances with probability.
    Less expensive but still effective.
    """
    
    def __init__(self, drop_rate=0.5):
        super(MILDropoutSimple, self).__init__()
        self.drop_rate = drop_rate
        
    def forward(self, input):
        if not self.training:
            return input
        
        batch_size, n, feature_dim = input.shape
        
        # Random dropout mask
        mask = (torch.rand(batch_size, n, 1, device=input.device) > self.drop_rate).float()
        
        # Scale to maintain expected magnitude
        mask = mask / (1 - self.drop_rate)
        
        return input * mask
