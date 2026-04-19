#!/usr/bin/env python3
"""
BagMix: Bag-Level Data Augmentation for Multiple Instance Learning

Two implementations:
1. Simple BagMix (ours) - Simple crop blending
2. PseMix - Pseudo-bag Mixup (from paper: IEEE TMI 2024)

Modes:
- none: No augmentation
- subset: Randomly sample k instances
- mixup: Simple instance-level mixup
- dropout: Random dropout
- shuffle: Shuffle order
- cutmix: Cut and paste regions
- mixup_crop: Blend full crops
- bootstrap: Sample with replacement
- cluster: K-means based selection

PseMix Modes:
- psebmix: Full PseMix (pseudo-bag mixup with clustering)
- psebmix_kmeans: PseMix with k-means clustering
- psebmix_random: PseMix with random pseudo-bags
"""

import torch
import torch.nn as nn
import random
import numpy as np
from typing import Optional, Tuple, List
from sklearn.cluster import KMeans


class BagMixer:
    """Simple BagMix augmentation for MIL"""
    
    def __init__(
        self,
        mode: str = 'none',
        mix_ratio: float = 0.5,
        subset_size: Optional[int] = None,
        dropout_ratio: float = 0.0,
        alpha: float = 1.0,
        seed: int = 42
    ):
        self.mode = mode.lower()
        self.mix_ratio = mix_ratio
        self.subset_size = subset_size
        self.dropout_ratio = dropout_ratio
        self.alpha = alpha
        self.seed = seed
        
        valid_modes = ['none', 'subset', 'mixup', 'dropout', 'shuffle', 
                       'cutmix', 'mixup_crop', 'bootstrap', 'cluster']
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Choose from {valid_modes}")
    
    def __call__(self, crops: torch.Tensor) -> torch.Tensor:
        if self.mode == 'none':
            return crops
        elif self.mode == 'subset':
            return self._subset(crops)
        elif self.mode == 'mixup':
            return self._mixup(crops)
        elif self.mode == 'dropout':
            return self._dropout(crops)
        elif self.mode == 'shuffle':
            return self._shuffle(crops)
        elif self.mode == 'cutmix':
            return self._cutmix(crops)
        elif self.mode == 'mixup_crop':
            return self._mixup_crop(crops)
        elif self.mode == 'bootstrap':
            return self._bootstrap(crops)
        elif self.mode == 'cluster':
            return self._cluster(crops)
        return crops
    
    def _subset(self, crops: torch.Tensor) -> torch.Tensor:
        num_crops = crops.shape[0]
        k = self.subset_size if self.subset_size else num_crops // 2
        k = min(k, num_crops)
        indices = torch.randperm(num_crops)[:k]
        return crops[indices]
    
    def _mixup(self, crops: torch.Tensor) -> torch.Tensor:
        num_crops = crops.shape[0]
        lam = np.random.beta(self.alpha, self.alpha)
        n1 = int(num_crops * lam)
        n2 = num_crops - n1
        indices = torch.randperm(num_crops)
        idx1, idx2 = indices[:n1], indices[n1:]
        
        result = crops.clone()
        for i in range(min(len(idx1), len(idx2))):
            result[idx1[i]] = lam * crops[idx1[i]] + (1 - lam) * crops[idx2[i]]
        return result
    
    def _dropout(self, crops: torch.Tensor) -> torch.Tensor:
        num_crops = crops.shape[0]
        if self.dropout_ratio <= 0:
            return crops
        n_keep = int(num_crops * (1 - self.dropout_ratio))
        n_keep = max(n_keep, 1)
        indices = torch.randperm(num_crops)[:n_keep]
        return crops[indices]
    
    def _shuffle(self, crops: torch.Tensor) -> torch.Tensor:
        num_crops = crops.shape[0]
        indices = torch.randperm(num_crops)
        return crops[indices]
    
    def _cutmix(self, crops: torch.Tensor) -> torch.Tensor:
        num_crops = crops.shape[0]
        if num_crops < 2:
            return crops
        
        idx1, idx2 = torch.randperm(num_crops)[:2]
        _, C, H, W = crops.shape
        
        lam = np.random.beta(self.alpha, self.alpha)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        result = crops.clone()
        result[idx1, :, bby1:bby2, bbx1:bbx2] = crops[idx2, :, bby1:bby2, bbx1:bbx2]
        return result
    
    def _mixup_crop(self, crops: torch.Tensor) -> torch.Tensor:
        num_crops = crops.shape[0]
        if num_crops < 2:
            return crops
        
        lam = np.random.beta(self.alpha, self.alpha)
        idx1, idx2 = torch.randperm(num_crops)[:2]
        
        result = crops.clone()
        result[idx1] = lam * crops[idx1] + (1 - lam) * crops[idx2]
        return result
    
    def _bootstrap(self, crops: torch.Tensor) -> torch.Tensor:
        num_crops = crops.shape[0]
        indices = torch.randint(0, num_crops, (num_crops,))
        return crops[indices]
    
    def _cluster(self, crops: torch.Tensor) -> torch.Tensor:
        num_crops = crops.shape[0]
        flat = crops.reshape(num_crops, -1)
        n_select = self.subset_size if self.subset_size else max(1, num_crops // 2)
        n_select = min(n_select, num_crops)
        
        if n_select >= num_crops:
            return crops
        
        selected = [0]
        remaining = list(range(1, num_crops))
        
        for _ in range(n_select - 1):
            if not remaining:
                break
            selected_tensor = flat[selected]
            remaining_tensor = flat[remaining]
            dists = torch.cdist(remaining_tensor, selected_tensor).min(dim=1)[0]
            farthest_idx = dists.argmax().item()
            selected.append(remaining[farthest_idx])
            remaining.pop(farthest_idx)
        
        return crops[sorted(selected)]


class PseMixer:
    """
    PseMix: Pseudo-Bag Mixup Augmentation
    From paper: "Pseudo-Bag Mixup Augmentation for Multiple Instance Learning 
              Based Whole Slide Image Classification" (IEEE TMI 2024)
    
    Key differences from simple BagMix:
    1. Divides bag into N pseudo-bags (using clustering)
    2. Mix at pseudo-bag level (not instance level)
    3. Uses soft labels (target mixing)
    4. Requires training-time pseudo-bag indicators
    """
    
    def __init__(
        self,
        mode: str = 'psebmix',
        n_pseb: int = 8,
        n_pheno: int = 8,
        alpha: float = 1.0,
        prob_mixup: float = 0.5,
        clustering_method: str = 'kmeans',
        fine_tune_iter: int = 0,
        seed: int = 42
    ):
        self.mode = mode.lower()
        self.n_pseb = n_pseb
        self.n_pheno = n_pheno
        self.alpha = alpha
        self.prob_mixup = prob_mixup
        self.clustering_method = clustering_method
        self.fine_tune_iter = fine_tune_iter
        self.seed = seed
        
        self.pseudo_bag_labels = {}  # store pseudo-bag labels per bag
        
        valid_modes = ['psebmix', 'psebmix_kmeans', 'psebmix_random']
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Choose from {valid_modes}")
    
    def divide_into_pseudo_bags(self, bag_features: torch.Tensor) -> torch.Tensor:
        """
        Divide bag instances into pseudo-bags using clustering.
        
        Args:
            bag_features: Tensor of shape [N, D] (feature dimension)
            
        Returns:
            pseudo_bag_labels: Tensor of shape [N] indicating pseudo-bag assignment
        """
        if len(bag_features.shape) > 2:
            bag_features = bag_features.squeeze(0)
        
        N, D = bag_features.shape
        
        if self.mode == 'psebmix_random':
            labels = torch.randperm(N) % self.n_pseb
            return labels.long()
        
        elif self.mode in ['psebmix', 'psebmix_kmeans']:
            feats = bag_features.cpu().numpy()
            kmeans = KMeans(n_clusters=self.n_pheno, random_state=self.seed).fit(feats)
            pheno_labels = kmeans.labels_.astype(np.int64)
            pheno_labels = torch.LongTensor(pheno_labels).to(bag_features.device)
            
            pseudo_bag_labels = torch.zeros(N, dtype=torch.long, device=bag_features.device)
            for c in range(self.n_pheno):
                c_size = (pheno_labels == c).sum().item()
                if c_size > 0:
                    pseudo_bag_labels[pheno_labels == c] = self._uniform_assign(c_size, self.n_pseb).to(bag_features.device)
            
            return pseudo_bag_labels
    
    def _uniform_assign(self, N: int, num_label: int) -> torch.Tensor:
        L = torch.randperm(N) % num_label
        rlab = torch.randperm(num_label)
        res = rlab[L]
        return res
    
    def __call__(
        self, 
        bags: List[torch.Tensor], 
        labels: torch.Tensor,
        pseudo_bag_indices: Optional[List[torch.Tensor]] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
        """
        Apply PseMix augmentation.
        
        Args:
            bags: List of tensors, each [N_i, C, H, W] or [N_i, D]
            labels: Tensor of shape [B] with class labels
            pseudo_bag_indices: Optional list of pseudo-bag labels for each bag
            
        Returns:
            mixed_bags: List of augmented bags
            mixed_labels_a: List of original labels
            mixed_labels_b: List of mixed labels (for soft labels)
            mix_ratios: List of mix ratios for each bag
        """
        batch_size = len(bags)
        
        if self.mode == 'none':
            return bags, [labels] * batch_size, [labels] * batch_size, [1.0] * batch_size
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        idxs = torch.randperm(batch_size)
        
        mixed_bags = []
        mixed_labels_a = []
        mixed_labels_b = []
        mix_ratios = []
        
        for i in range(batch_size):
            bag_i = bags[i]
            bag_j = bags[idxs[i]]
            
            if bag_i.shape[0] == 0 or bag_j.shape[0] == 0:
                mixed_bags.append(bag_i)
                mixed_labels_a.append(labels[i].item())
                mixed_labels_b.append(labels[idxs[i]].item())
                mix_ratios.append(1.0)
                continue
            
            lam_discrete = int(lam * (self.n_pseb + 1))
            lam_discrete = min(lam_discrete, self.n_pseb)
            
            n_i = self.n_pseb - lam_discrete
            
            if pseudo_bag_indices is not None and i < len(pseudo_bag_indices):
                ind_i = pseudo_bag_indices[i]
                ind_j = pseudo_bag_indices[idxs[i]]
                
                bag_a = self._fetch_pseudo_bags(bag_i, ind_i, self.n_pseb, lam_discrete)
                bag_b = self._fetch_pseudo_bags(bag_j, ind_j, self.n_pseb, n_i)
            else:
                bag_a = self._random_fetch(bag_i, lam_discrete)
                bag_b = self._random_fetch(bag_j, n_i)
            
            if bag_a is None or bag_a.shape[0] == 0:
                bag_ab = bag_b
                area_ratio = 0.0
                cont_ratio = lam_discrete / self.n_pseb
            elif bag_b is None or bag_b.shape[0] == 0:
                bag_ab = bag_a
                area_ratio = 1.0
                cont_ratio = lam_discrete / self.n_pseb
            else:
                if random.random() <= self.prob_mixup:
                    bag_ab = torch.cat([bag_a, bag_b], dim=0)
                    area_ratio = bag_a.shape[0] / bag_ab.shape[0]
                    cont_ratio = lam_discrete / self.n_pseb
                else:
                    bag_ab = bag_a
                    area_ratio = 1.0
                    cont_ratio = 1.0
            
            mixed_bags.append(bag_ab)
            mixed_labels_a.append(labels[i].item())
            mixed_labels_b.append(labels[idxs[i]].item())
            mix_ratios.append(cont_ratio)
        
        return mixed_bags, mixed_labels_a, mixed_labels_b, mix_ratios
    
    def _fetch_pseudo_bags(
        self, 
        bag: torch.Tensor, 
        ind: torch.Tensor, 
        n: int, 
        n_parts: int
    ) -> Optional[torch.Tensor]:
        """Fetch instances belonging to selected pseudo-bags"""
        if n_parts == 0:
            return None
        
        ind_fetched = torch.randperm(n)[:n_parts].to(bag.device)
        mask = torch.zeros(n, dtype=torch.bool, device=bag.device)
        mask[ind_fetched] = True
        
        selected = mask[ind]
        if selected.sum() == 0:
            return None
        
        return bag[selected]
    
    def _random_fetch(self, bag: torch.Tensor, n: int) -> Optional[torch.Tensor]:
        """Randomly sample n instances from bag"""
        if n <= 0:
            return None
        num_instances = bag.shape[0]
        n = min(n, num_instances)
        indices = torch.randperm(num_instances)[:n].to(bag.device)
        return bag[indices]


def create_bag_mixer(
    mode: str = 'none',
    mix_ratio: float = 0.5,
    subset_size: int = None,
    dropout_ratio: float = 0.0,
    alpha: float = 1.0,
    seed: int = 42,
    use_psemix: bool = False,
    **psemix_kwargs
) -> object:
    """Factory function to create BagMixer or PseMixer"""
    if use_psemix:
        return PseMixer(mode=mode, alpha=alpha, **psemix_kwargs)
    else:
        return BagMixer(
            mode=mode,
            mix_ratio=mix_ratio,
            subset_size=subset_size,
            dropout_ratio=dropout_ratio,
            alpha=alpha,
            seed=seed
        )


def add_bagmix_args(parser):
    """Add BagMix arguments to argparse"""
    parser.add_argument(
        '--bag_mix',
        type=str,
        default='none',
        choices=['none', 'subset', 'mixup', 'dropout', 'shuffle', 'cutmix', 'mixup_crop', 'bootstrap', 'cluster'],
        help='Bag mixing strategy (simple BagMix)'
    )
    parser.add_argument(
        '--bag_mix_ratio',
        type=float,
        default=0.5,
        help='Mix ratio for bag mixup'
    )
    parser.add_argument(
        '--bag_mix_subset_size',
        type=int,
        default=None,
        help='Subset size for subset mode'
    )
    parser.add_argument(
        '--bag_mix_dropout',
        type=float,
        default=0.0,
        help='Dropout ratio for dropout mode'
    )
    parser.add_argument(
        '--bag_mix_alpha',
        type=float,
        default=1.0,
        help='Beta distribution parameter'
    )
    parser.add_argument(
        '--bag_mix_prob',
        type=float,
        default=0.5,
        help='Probability of applying bag mix'
    )
    return parser


def add_psemix_args(parser):
    """Add PseMix arguments to argparse"""
    parser.add_argument(
        '--use_psemix',
        action='store_true',
        help='Use PseMix instead of simple BagMix'
    )
    parser.add_argument(
        '--psemix_mode',
        type=str,
        default='psebmix',
        choices=['psebmix', 'psebmix_kmeans', 'psebmix_random'],
        help='PseMix mode'
    )
    parser.add_argument(
        '--psemix_n_pseb',
        type=int,
        default=8,
        help='Number of pseudo-bags per bag'
    )
    parser.add_argument(
        '--psemix_n_pheno',
        type=int,
        default=8,
        help='Number of phenotype clusters'
    )
    parser.add_argument(
        '--psemix_alpha',
        type=float,
        default=1.0,
        help='Beta distribution parameter for PseMix'
    )
    parser.add_argument(
        '--psemix_prob',
        type=float,
        default=0.5,
        help='Probability of random mixing in PseMix'
    )
    return parser


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='BagMix/PseMix Test')
    parser = add_bagmix_args(parser)
    parser = add_psemix_args(parser)
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Simple BagMix Mode: {args.bag_mix}")
    print(f"PseMix Mode: {args.use_psemix}")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    crops = torch.randn(9, 3, 224, 224)
    
    print("\nTesting Simple BagMix modes...")
    modes = ['none', 'subset', 'dropout', 'shuffle', 'mixup_crop']
    for mode in modes:
        bm = BagMixer(mode=mode, subset_size=5, dropout_ratio=0.2)
        result = bm(crops)
        print(f"  {mode}: {crops.shape} -> {result.shape}")
    
    print("\nTesting PseMix...")
    psemix = PseMixer(mode='psebmix', n_pseb=8, n_pheno=8)
    pseudo_labels = psemix.divide_into_pseudo_bags(crops[0].reshape(9, -1))
    print(f"  Pseudo-bag labels: {pseudo_labels.shape}")
    
    bags = [crops[0], crops[1]]
    labels = torch.tensor([0, 1])
    mixed, labels_a, labels_b, ratios = psemix(bags, labels)
    print(f"  Mixed bags: {len(mixed)}, ratios: {ratios}")
    
    print("\nDone!")