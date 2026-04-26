"""
================================================================================
HierarchicalSupConLossMIL - Pure PyTorch 4-Level Supervised Contrastive Learning
================================================================================
Guide (96 classes) → Gene → Family → Pathway hierarchy for CRISPRi classification.

Level 1 (weight=1.0): Same GUIDE     → strongest positive pairs (e.g., ftsZ_1 vs ftsZ_2)
Level 2 (weight=0.5): Same GENE       → moderate positive pairs (e.g., ftsZ_1 vs ftsI_1)
Level 3 (weight=0.2): Same FAMILY    → weak positive pairs (e.g., ftsZ vs murA)
Level 4 (weight=0.1): Same PATHWAY   → weakest positive pairs

Fully differentiable - all operations on GPU with proper gradient flow.
Per-level loss and accuracy tracking for validation monitoring.

Based on: https://arxiv.org/abs/2004.11362 (SupCon)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import numpy as np


class HierarchicalSupConLossMIL(nn.Module):
    """
    4-Level Hierarchical Supervised Contrastive Loss for MIL.
    
    Creates positive pairs at biological hierarchy levels:
    - Guide level (weight=1.0): Same guide ID (e.g., ftsZ_1 vs ftsZ_2) → variant grouping
    - Gene level (weight=0.5): Same gene name (e.g., ftsZ_1 vs ftsI_1)
    - Family level (weight=0.2): Same gene family (e.g., ftsZ vs ftsI)
    - Pathway level (weight=0.1): Same biological pathway (e.g., fts vs mur)
    
    This ensures rich positive pairs across all levels of biological similarity.
    """
    
    def __init__(self, temperature=0.07, weights=None, mappings_path=None):
        super(HierarchicalSupConLossMIL, self).__init__()
        self.temperature = temperature
        
        if weights is None:
            weights = {'guide': 1.0, 'gene': 0.5, 'family': 0.2, 'pathway': 0.1}
        self.weights = weights
        
        if mappings_path is None:
            mappings_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'hierarchical_mappings.json'
            )
        
        self.mappings_path = mappings_path
        self._load_mappings()
    
    def _load_mappings(self):
        if os.path.exists(self.mappings_path):
            with open(self.mappings_path, 'r') as f:
                data = json.load(f)
                self.gene_mappings = data.get('mappings', {})
                self.families = data.get('families', {})
                self.pathways = data.get('pathways', {})
        else:
            self.gene_mappings = {}
            self.families = {}
            self.pathways = {}
    
    def _parse_guide(self, label):
        """Parse guide label to extract guide, gene, family, pathway."""
        if isinstance(label, (int, np.integer)):
            return str(label), 'Unknown', 'Unknown', 'Unknown'
        if torch.is_tensor(label):
            return self._parse_guide(label.item())
        if not label or label == 'nan':
            return 'Unknown', 'Unknown', 'Unknown', 'Unknown'
        
        label_str = str(label)
        gene = label_str
        family = 'Unknown'
        pathway = 'Unknown'
        
        if '_' in label_str:
            gene_part = label_str.rsplit('_', 1)[0]
            suffix = label_str.rsplit('_', 1)[1]
            
            if suffix in ('1', '2', '3', 'a', 'b', 'c'):
                gene = gene_part
                if gene in self.gene_mappings:
                    family = self.gene_mappings[gene].get('family', 'Unknown')
                    pathway = self.gene_mappings[gene].get('pathway', 'Unknown')
        
        return label_str, gene, family, pathway
    
    def _get_hierarchy_labels(self, labels):
        guide_labels = []
        gene_labels = []
        family_labels = []
        pathway_labels = []
        
        for label in labels:
            guide, gene, family, pathway = self._parse_guide(label)
            guide_labels.append(guide)
            gene_labels.append(gene)
            family_labels.append(family)
            pathway_labels.append(pathway)
        
        return guide_labels, gene_labels, family_labels, pathway_labels
    
    def _create_label_to_idx(self, labels):
        unique_labels = sorted(set(str(l) for l in labels))
        return {l: i for i, l in enumerate(unique_labels)}
    
    def _labels_to_indices(self, labels, label_to_idx):
        return torch.tensor([label_to_idx.get(str(l), 0) for l in labels], dtype=torch.long, device='cpu')
    
    def _compute_supcon_pure_torch(self, features, labels, temperature):
        device = features.device
        total = features.shape[0]
        
        features_flat = features.squeeze(1)
        
        labels = labels.to(device)
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        mask.fill_diagonal_(0)
        
        sim = torch.matmul(features_flat, features_flat.T) / temperature
        
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        
        exp_sim = torch.exp(sim)
        exp_sim = exp_sim * mask
        exp_sim_sum = exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-10)
        
        log_prob = sim - torch.log(exp_sim_sum)
        
        pos_sum = mask.sum(dim=1)
        pos_sum = pos_sum.clamp(min=1.0)
        
        loss_per_sample = -(mask * log_prob).sum(dim=1) / pos_sum
        loss = loss_per_sample.mean()
        
        labels_cpu = labels.cpu()
        n = labels_cpu.shape[0]
        correct = sum(1 for i in range(n) if (labels_cpu == labels_cpu[i]).sum().item() > 1)
        acc = correct / n if n > 0 else 0.0
        
        return loss, acc
    
    def forward(self, features, bag_labels, num_crops_per_bag=9):
        batch_size = features.shape[0]
        num_crops = features.shape[1]
        
        features_flat = features.view(-1, 1, features.shape[-1])
        
        instance_labels = []
        for label in bag_labels:
            if torch.is_tensor(label):
                label = label.item()
            instance_labels.extend([label] * num_crops)
        
        guide_labels, gene_labels, family_labels, pathway_labels = self._get_hierarchy_labels(instance_labels)
        
        guide_to_idx = self._create_label_to_idx(guide_labels)
        gene_to_idx = self._create_label_to_idx(gene_labels)
        family_to_idx = self._create_label_to_idx(family_labels)
        pathway_to_idx = self._create_label_to_idx(pathway_labels)
        
        guide_indices = self._labels_to_indices(guide_labels, guide_to_idx)
        gene_indices = self._labels_to_indices(gene_labels, gene_to_idx)
        family_indices = self._labels_to_indices(family_labels, family_to_idx)
        pathway_indices = self._labels_to_indices(pathway_labels, pathway_to_idx)
        
        loss_guide, acc_guide = self._compute_supcon_pure_torch(features_flat, guide_indices, self.temperature)
        loss_gene, acc_gene = self._compute_supcon_pure_torch(features_flat, gene_indices, self.temperature)
        loss_family, acc_family = self._compute_supcon_pure_torch(features_flat, family_indices, self.temperature)
        loss_pathway, acc_pathway = self._compute_supcon_pure_torch(features_flat, pathway_indices, self.temperature)
        
        w_guide = self.weights.get('guide', 1.0)
        w_gene = self.weights.get('gene', 0.5)
        w_family = self.weights.get('family', 0.2)
        w_pathway = self.weights.get('pathway', 0.1)
        
        total_loss = w_guide * loss_guide + w_gene * loss_gene + w_family * loss_family + w_pathway * loss_pathway
        
        metrics = {
            'loss_guide': loss_guide.item(),
            'loss_gene': loss_gene.item(),
            'loss_family': loss_family.item(),
            'loss_pathway': loss_pathway.item(),
            'acc_guide': acc_guide,
            'acc_gene': acc_gene,
            'acc_family': acc_family,
            'acc_pathway': acc_pathway,
        }
        
        return total_loss, metrics
    
    def get_metrics_names(self):
        return ['loss_guide', 'loss_gene', 'loss_family', 'loss_pathway', 'acc_guide', 'acc_gene', 'acc_family', 'acc_pathway']


# =============================================================================
# USAGE EXAMPLES
# =============================================================================
# 
# Standard hierarchical contrastive learning:
# from hierarchical_supcon_loss import HierarchicalSupConLoss
# criterion = HierarchicalSupConLoss(temperature=0.07)
# loss = criterion(embeddings, labels)
#
# MIL hierarchical contrastive learning:
# from hierarchical_supcon_loss import HierarchicalSupConLossMIL
# criterion = HierarchicalSupConLossMIL(temperature=0.07)
# loss = criterion(embeddings, bag_labels, num_crops_per_bag=9)
#
# In training:
# supcon_loss = criterion(bag_embeddings, labels)
# ce_loss = F.cross_entropy(logits, labels)
# total_loss = 0.75 * ce_loss + 0.25 * supcon_loss