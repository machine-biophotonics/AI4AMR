#!/usr/bin/env python3
"""
PseMix wrapper: Pseudo-Bag Mixup Augmentation for MIL.

Uses official PseMix algorithm: https://github.com/liupei101/PseMix
IEEE TMI 2024
"""

from typing import List, Optional, Tuple, Any

import numpy as np
import torch
from sklearn.cluster import KMeans


class PseMixer:
    """
    PseMix: Pseudo-Bag Mixup Augmentation (IEEE TMI 2024).
    
    Divides each bag into N pseudo-bags using K-means clustering, then performs
    mixup at the pseudo-bag level for data augmentation.
    """
    
    def __init__(
        self,
        mode: str = "psebmix",
        n_pseb: int = 8,
        n_pheno: int = 8,
        alpha: float = 1.0,
        prob_mixup: float = 0.5,
        fine_tune_iter: int = 0,
        mixup_lam_from: str = "content",
        seed: int = 42,
    ) -> None:
        """Initialize PseMixer."""
        self.mode = mode.lower()
        self.n_pseb = n_pseb
        self.n_pheno = n_pheno
        self.alpha = alpha
        self.prob_mixup = prob_mixup
        self.fine_tune_iter = fine_tune_iter
        self.mixup_lam_from = mixup_lam_from
        self.seed = seed
    
    def __call__(
        self,
        bags: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[int], List[int], List[float]]:
        """Apply PseMix to batch of bags."""
        return self.apply_psemix(bags, labels)
    
    def apply_psemix(
        self,
        bags: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[int], List[int], List[float]]:
        """Apply PseMix augmentation."""
        batch_size = len(bags)
        
        if batch_size == 1:
            return bags, [labels[0].item()], [labels[0].item()], [1.0]
        
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        idxs = torch.randperm(batch_size)
        
        pseudo_bag_indices = [self.divide_into_pseudo_bags(bag) for bag in bags]
        
        mixed_bags: List[torch.Tensor] = []
        labels_a: List[int] = []
        labels_b: List[int] = []
        mix_ratios: List[float] = []
        
        for i in range(batch_size):
            bag_i = bags[i]
            bag_j = bags[idxs[i]]
            
            lam_temp = lam if lam != 1.0 else lam - 1e-5
            lam_discrete = min(int(lam_temp * (self.n_pseb + 1)), self.n_pseb)
            n_i = self.n_pseb - lam_discrete
            
            ind_i = pseudo_bag_indices[i]
            ind_j = pseudo_bag_indices[idxs[i]]
            
            bag_a = self._fetch_pseudo_bags(bag_i, ind_i, lam_discrete)
            bag_b = self._fetch_pseudo_bags(bag_j, ind_j, n_i)
            
            # Handle empty cases
            if bag_a is None or bag_a.shape[0] == 0:
                bag_ab = bag_b
                area_ratio = 0.0
                cont_ratio = lam_discrete / self.n_pseb
            elif bag_b is None or bag_b.shape[0] == 0:
                bag_ab = bag_a
                area_ratio = 1.0
                cont_ratio = lam_discrete / self.n_pseb
            else:
                if np.random.rand() <= self.prob_mixup:
                    bag_ab = torch.cat([bag_a, bag_b], dim=0)
                    area_ratio = bag_a.shape[0] / bag_ab.shape[0]
                    cont_ratio = lam_discrete / self.n_pseb
                else:
                    bag_ab = bag_a
                    area_ratio = 1.0
                    cont_ratio = 1.0
            
            if self.mixup_lam_from == "area":
                temp_mix_ratio = area_ratio
            elif self.mixup_lam_from == "content":
                temp_mix_ratio = cont_ratio
            else:
                temp_mix_ratio = lam
            
            mixed_bags.append(bag_ab)
            labels_a.append(labels[i].item())
            labels_b.append(labels[idxs[i]].item())
            mix_ratios.append(temp_mix_ratio)
        
        return mixed_bags, labels_a, labels_b, mix_ratios
    
    def divide_into_pseudo_bags(self, bag_features: torch.Tensor) -> torch.Tensor:
        """Divide bag into pseudo-bags using K-means."""
        if len(bag_features.shape) > 2:
            bag_features = bag_features.squeeze(0)
        
        N, D = bag_features.shape
        
        if self.mode == "psebmix_random":
            return (torch.randperm(N) % self.n_pseb).long().to(bag_features.device)
        
        # K-means clustering
        feats = bag_features.cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_pheno, random_state=self.seed).fit(feats)
        pheno_labels = torch.LongTensor(kmeans.labels_).to(bag_features.device)
        
        # Assign pseudo-bags within each phenotype
        pseudo_bag_labels = torch.zeros(N, dtype=torch.long, device=bag_features.device)
        
        for c in range(self.n_pheno):
            c_mask = pheno_labels == c
            c_indices = torch.where(c_mask)[0]
            c_size = c_indices.shape[0]
            
            if c_size > 0:
                pseudo_bag_labels[c_indices] = (torch.randperm(c_size) % self.n_pseb).to(
                    pseudo_bag_labels.device
                )
        
        return pseudo_bag_labels
    
    def _fetch_pseudo_bags(
        self,
        bag: torch.Tensor,
        ind: torch.Tensor,
        n_parts: int,
    ) -> Optional[torch.Tensor]:
        """Fetch instances belonging to selected pseudo-bags."""
        if n_parts == 0:
            return None
        if n_parts >= self.n_pseb:
            return bag
        
        # Select which pseudo-bags to fetch from
        ind_fetched = torch.randperm(self.n_pseb)[:n_parts].to(bag.device)
        
        # Create mask: True where instance belongs to selected pseudo-bag
        mask = torch.zeros_like(ind, dtype=torch.bool)
        for idx in ind_fetched:
            mask = mask | (ind == idx)
        
        if mask.sum() == 0:
            return None
        
        return bag[mask]


def create_psemix(
    mode: str = "psebmix",
    n_pseb: int = 8,
    n_pheno: int = 8,
    alpha: float = 1.0,
    prob_mixup: float = 0.5,
    seed: int = 42,
    **kwargs,
) -> PseMixer:
    """Factory to create PseMixer."""
    return PseMixer(
        mode=mode,
        n_pseb=n_pseb,
        n_pheno=n_pheno,
        alpha=alpha,
        prob_mixup=prob_mixup,
        seed=seed,
    )


def add_psemix_args(parser) -> Any:
    """Add PseMix arguments to parser."""
    import argparse
    
    parser.add_argument("--use_psemix", action="store_true", help="Use PseMix")
    parser.add_argument(
        "--psemix_mode", type=str, default="psebmix",
        choices=["psebmix", "psebmix_random"],
    )
    parser.add_argument("--psemix_n_pseb", type=int, default=8)
    parser.add_argument("--psemix_n_pheno", type=int, default=8)
    parser.add_argument("--psemix_alpha", type=float, default=1.0)
    parser.add_argument("--psemix_prob", type=float, default=0.5)
    
    return parser


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser = add_psemix_args(parser)
    args = parser.parse_args([])
    
    print(f"PseMix: {args.psemix_mode}, {args.psemix_n_pseb} pseudo-bags")
    
    p = PseMixer(n_pseb=8, n_pheno=8)
    bags = [torch.randn(9, 512), torch.randn(9, 512)]
    labels = torch.tensor([0, 1])
    mixed, la, lb, r = p(bags, labels)
    print(f"OK: {len(mixed)} bags, ratios: {r}")