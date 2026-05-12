"""
CPA-MIL: Shared perturbation embedding space.

Architecture:
  MILEncoder(EfficientNet-B0 + Gated Attention) → z_bag [256]
  Classifier weights = PerturbationEmbedding table.transposed
  → logits = cosine_similarity(z_bag, each prototype) × temperature

The PerturbationEmbedding table IS the shared space.
Each row = prototype vector for one drug/mutant/control.
Training forces z_bag ≈ prototype for the correct label.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from mil_model import AttentionPooling


class MILEncoder(nn.Module):
    """Bag encoder: N crops → single 256-dim embedding."""
    def __init__(self, embedding_dim: int = 256, num_heads: int = 4,
                 num_channels: int = 1, feature_dim: int = 1280):
        super().__init__()
        base_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(
            base_model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        if num_channels == 1:
            orig = base_model.features[0][0]
            w = orig.weight.data.sum(dim=1, keepdim=True)
            self.backbone[0][0] = nn.Conv2d(1, 32, kernel_size=3,
                                            stride=2, padding=1, bias=False)
            self.backbone[0][0].weight.data = w
        self.attention_pool = AttentionPooling(feature_dim, num_heads)
        self.fc = nn.Linear(feature_dim * num_heads, embedding_dim)

    def forward(self, x):
        B, N = x.shape[:2]
        feat = self.backbone(x.view(B * N, *x.shape[2:])).view(B, N, -1)
        pooled, attn = self.attention_pool(feat)
        return self.fc(pooled.reshape(B, -1)), attn


class CPAModel(nn.Module):
    """Shared perturbation embedding model.

    The perturbation_embedding.weight matrix IS the shared space:
      row i = prototype vector for perturbation i.
      logits[i] = cosine_sim(z_bag, prototype[i]) * temperature.

    After training:
      - Similar drugs and mutants have similar prototype vectors
      - Query: cosine_sim(prototype[Ciprofloxacin_2x], prototype[gyrA_1])
    """
    def __init__(self, num_perturbations: int, num_classes: int,
                 embedding_dim: int = 256, num_heads: int = 4,
                 dropout: float = 0.5, num_channels: int = 1,
                 temperature: float = 10.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature

        self.encoder = MILEncoder(embedding_dim, num_heads, num_channels)
        self.perturbation_embedding = nn.Embedding(num_perturbations, embedding_dim)
        nn.init.kaiming_uniform_(self.perturbation_embedding.weight, a=np.sqrt(5))

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, x):
        """Forward: encode bag → z_bag + logits.

        Returns:
            z_bag: [B, embedding_dim] — bag embedding
            logits: [B, num_classes] — for class label prediction
            logits_pert: [B, num_perturbations] — cosine similarity to each prototype
        """
        z_bag, _ = self.encoder(x)
        logits = self.classifier(z_bag)

        # Contrastive to prototype space: logits = cos(z_bag, prototype[i])
        z_norm = F.normalize(z_bag, dim=1)
        proto_norm = F.normalize(self.perturbation_embedding.weight, dim=1)
        logits_pert = z_norm @ proto_norm.T * self.temperature

        return z_bag, logits, logits_pert

    def get_prototypes(self):
        """Return prototype vectors [num_perturbations, embedding_dim]."""
        return self.perturbation_embedding.weight.data.clone()
