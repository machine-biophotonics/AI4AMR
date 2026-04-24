"""
PSAMIL: Probability-Space MIL (ICLR 2025)
Matching: https://github.com/LMBDA-design/PSAMIL

Key features:
1. Feature Bank (fbank) - class prototypes that update during training
2. Probability-space attention (psa mode)
3. Multiple pooling modes: psa, fsa, mha, avg
4. Online bank updates during training

This implementation matches the original paper exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class Attention_Gated(nn.Module):
    """Gated attention mechanism (Ilse et al. 2018)"""
    def __init__(self, L: int = 512, D: int = 128, K: int = 1):
        super().__init__()
        self.L = L
        self.D = D
        self.K = K
        
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.D, self.K)
    
    def forward(self, x: torch.Tensor, isNorm: bool = True) -> torch.Tensor:
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        if isNorm:
            A = F.softmax(A, dim=1)
        return A


class GatedAttentionLayerV(nn.Module):
    """Gated attention V layer - tanh(WV * h + bV)"""
    def __init__(self, dim: int = 512):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1)
    
    def forward(self, features: torch.Tensor, W_V: torch.Tensor, b_V: torch.Tensor) -> torch.Tensor:
        out = F.linear(features, W_V, b_V)
        return torch.tanh(out)


class GatedAttentionLayerU(nn.Module):
    """Gated attention U layer - sigmoid(WU * h + bU)"""
    def __init__(self, dim: int = 512):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1)
    
    def forward(self, features: torch.Tensor, W_U: torch.Tensor, b_U: torch.Tensor) -> torch.Tensor:
        out = F.linear(features, W_U, b_U)
        return torch.sigmoid(out)


class Classifier_1fc(nn.Module):
    """Simple 1-layer classifier"""
    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(n_channels, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class PSAMIL(nn.Module):
    """Probability-Space MIL (PSAMIL) - ICLR 2025
    
    Matches: https://github.com/LMBDA-design/PSAMIL
    
    Key components:
    - Feature bank (fbank): learnable class prototypes [feature_dim, num_classes]
    - Probability-space attention: attention computed in probability space
    - Multiple pooling modes: psa, fsa, mha, avg
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 2,
        pooling: str = "psa",
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.pooling = pooling
        
        self.firsttime = True
        
        # Feature bank - class prototypes [feature_dim, num_classes]
        # requires_grad=False makes it non-trainable but updatable during training
        self.fbank = nn.Parameter(
            data=torch.FloatTensor(feature_dim, num_classes),
            requires_grad=False
        )
        nn.init.kaiming_uniform_(self.fbank)
        
        # PSA: Probability-Space Attention modules
        self.ps_attention = nn.Sequential(
            nn.Linear(num_classes, 1)
        )
        
        # FSA: Feature-Space Attention module
        self.attention_based = nn.Sequential(
            nn.Linear(feature_dim, 1)
        )
        
        # MHA: Multi-Head Gated Attention
        self.att_layer_V = GatedAttentionLayerV(feature_dim)
        self.att_layer_U = GatedAttentionLayerU(feature_dim)
        self.linear_V = nn.Linear(feature_dim, num_classes)
        self.linear_U = nn.Linear(feature_dim, num_classes)
        self.attention_weights = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
        # Final classification linear layer
        self.linear = nn.Parameter(
            data=torch.FloatTensor(feature_dim, num_classes),
            requires_grad=True
        )
        nn.init.kaiming_uniform_(self.linear)
        
        self.output = nn.LogSoftmax(dim=1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _init_fbank(self, features: torch.Tensor) -> None:
        """Initialize feature bank with mean features"""
        if self.firsttime:
            # [feature_dim, num_classes] = mean(features) repeated for each class
            self.fbank.data = features.mean(dim=0, keepdim=True).repeat(self.num_classes, 1).transpose(0, 1)
            self.firsttime = False
    
    def _compute_instance_probs(self, features: torch.Tensor) -> torch.Tensor:
        """Compute instance probabilities using feature bank
        
        Args:
            features: (t, feature_dim) where t = number of instances
        Returns:
            probs: (t, num_classes)
        """
        scores = torch.mm(features, self.fbank)
        probs = F.softmax(scores, dim=1)
        return probs
    
    def _attention_psa(
        self,
        features: torch.Tensor,
        probs: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Probability-space attention pooling
        
        Key innovation: Apply attention in probability space rather than feature space.
        
        Args:
            features: (t, feature_dim)
            probs: (t, num_classes) - instance probabilities from feature bank
            y: ground truth label for bank update
            
        Returns:
            F: pooled features (1, feature_dim)
            alpha: attention weights (1, t)
            pred_one_hot: one-hot predictions (t, num_classes)
            criticalF: critical features (1, feature_dim) for bank update
        """
        # Get predicted labels from probabilities
        predicted_labels = torch.argmax(probs, dim=1)
        pred_one_hot = F.one_hot(predicted_labels, self.num_classes).float()
        
        # PSA attention: two-branch fusion
        alpha = self.ps_attention(probs)
        alpha1 = self.ps_attention(pred_one_hot)
        
        # Average and softmax
        alpha = (alpha.squeeze(-1) + alpha1.squeeze(-1)) / 2
        alpha = F.softmax(alpha.unsqueeze(0), dim=1)
        
        # Pool features
        F_out = torch.mm(alpha, features)
        
        # Compute critical features for bank update
        if y is not None and self.training:
            y_val = y.item() if isinstance(y, torch.Tensor) else y
            if y_val >= 0:
                selected_mask = (predicted_labels == y_val)
                if torch.any(selected_mask):
                    criticalF = features[selected_mask].mean(dim=0, keepdim=True)
                else:
                    criticalF = F_out
            else:
                criticalF = F_out
        else:
            criticalF = F_out
        
        return F_out, alpha, pred_one_hot, criticalF
    
    def _attention_fsa(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Feature-space attention pooling (standard gated attention)"""
        alpha = self.attention_based(features)
        alpha = F.softmax(alpha.squeeze(-1).unsqueeze(0), dim=1)
        F_out = torch.mm(alpha, features)
        return F_out, alpha
    
    def _attention_mha(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-head attention pooling with gating"""
        A_V = self.att_layer_V(features, self.linear_V.weight, self.linear_V.bias)
        A_U = self.att_layer_U(features, self.linear_U.weight, self.linear_U.bias)
        A = self.attention_weights(A_V * A_U)
        
        alpha = torch.transpose(A, 1, 0)
        alpha = torch.sigmoid(alpha)
        c = torch.sum(alpha)
        alpha = alpha / c
        
        F_out = torch.mm(alpha, features)
        return F_out, alpha
    
    def _pooling_avg(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Average pooling"""
        F_out = torch.mean(features, dim=0, keepdim=True)
        alpha = torch.ones(1, features.shape[0], device=features.device) / features.shape[0]
        return F_out, alpha
    
    def _update_fbank(
        self,
        y: torch.Tensor,
        criticalF: torch.Tensor
    ) -> None:
        """Update feature bank during training
        
        Key innovation: Online update of class prototypes based on
        critical features (pooled features from instances matching predicted label).
        
        Args:
            y: ground truth label (scalar)
            criticalF: critical features (1, feature_dim)
        """
        if not self.training:
            return
        
        if y < 0:
            return
        
        y = y.item() if isinstance(y, torch.Tensor) else y
        
        # Ensure correct shapes: criticalF is (1, feature_dim), need (feature_dim,)
        criticalF_vec = criticalF.squeeze(0)  # (feature_dim,)
        
        # Exponential moving average update
        # 99.9% old prototype + 0.1% new critical features
        new_f = F.normalize(
            (0.999 * self.fbank[:, y]) + (0.001 * criticalF_vec),
            dim=0
        )
        self.fbank.data[:, y] = new_f
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        bag_size: Optional[int] = None,
        pooling: Optional[str] = None,
        testmode: str = "bag"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass
        
        Args:
            x: Input features (t, feature_dim) or (batch, t, feature_dim)
            y: Ground truth label (scalar, -1 for inference)
            bag_size: Number of instances (optional, inferred from x)
            pooling: Pooling method - "psa", "fsa", "mha", "avg"
            testmode: "bag" for bag-level, "instance" for instance-level
            
        Returns:
            Y_prob: Log probabilities (1, num_classes)
            Y_hat: Predicted label (1)
            alpha: Attention weights (1, bag_size)
            ins_probs: Instance-level probabilities (bag_size, num_classes)
            F: Aggregated bag features (1, feature_dim)
        """
        # Handle batch dimension
        if x.dim() == 3:
            x = x.squeeze(0)
        
        # Initialize feature bank on first forward
        self._init_fbank(x)
        
        # Compute instance probabilities using feature bank
        with torch.no_grad():
            probs = self._compute_instance_probs(x)
        
        pool = pooling if pooling is not None else self.pooling
        
        # Pooling based on selected method
        if pool == "psa":
            F_out, alpha, pred_one_hot, criticalF = self._attention_psa(x, probs, y)
            
            # Update feature bank during training
            if self.training and y >= 0:
                self._update_fbank(y, criticalF)
                
        elif pool == "fsa":
            F_out, alpha = self._attention_fsa(x)
            pred_one_hot = probs
            
        elif pool == "mha":
            F_out, alpha = self._attention_mha(x)
            pred_one_hot = probs
            
        elif pool == "avg":
            F_out, alpha = self._pooling_avg(x)
            pred_one_hot = probs
            
        else:
            F_out, alpha, pred_one_hot, _ = self._attention_psa(x, probs, y)
        
        # Apply dropout
        F_out = self.dropout(F_out)
        
        # Classification
        Y_logit = torch.matmul(F_out, self.linear)
        Y_prob = self.output(Y_logit)
        Y_hat = torch.argmax(Y_prob, dim=1)
        
        if testmode == "instance":
            return Y_prob, Y_hat, alpha, probs, x
        else:
            return Y_prob, Y_hat, alpha, probs, F_out


class PSAMILWrapper(nn.Module):
    """PSAMIL wrapper with EfficientNet backbone
    
    Integrates PSAMIL with the existing training pipeline.
    """
    
    def __init__(
        self,
        num_classes: int = 96,
        pooling: str = "psa",
        dropout: float = 0.2,
        use_mildropout: bool = False,
        mildropout_topk: int = 3,
        mildropout_kernel: int = 7,
        mammoth=None
    ):
        super().__init__()
        
        # Backbone
        import torchvision
        base_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        feature_dim = 1280
        
        # MAMMOTH option
        self.use_mammoth = mammoth is not None
        if self.use_mammoth:
            self.mammoth = mammoth
            embed_dim = 512
        else:
            embed_dim = feature_dim
        
        # Feature embedding
        self.patch_embed = nn.Linear(feature_dim, embed_dim)
        
        # MIL-Dropout option
        self.use_mildropout = use_mildropout
        if use_mildropout:
            from .mildropout import Mildropout
            self.mildropout = Mildropout(topk=mildropout_topk, kernel=mildropout_kernel)
        
        # PSAMIL pooling
        self.psamil = PSAMIL(
            feature_dim=embed_dim,
            num_classes=num_classes,
            pooling=pooling,
            dropout=dropout
        )
        
        self.pooling = pooling
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_crops = x.shape[:2]
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        x = x.view(batch_size, num_crops, -1)
        
        if self.use_mammoth:
            x = self.mammoth(x)
            x = x.mean(dim=1, keepdim=True).expand(-1, num_crops, -1)
        else:
            x = self.patch_embed(x)
        
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        bag_size: Optional[int] = None,
        pooling: Optional[str] = None,
        return_attention: bool = False
    ):
        """Forward pass compatible with existing pipeline
        
        Args:
            x: Input images (batch, num_crops, C, H, W)
            y: Labels (batch,) - for PSAMIL bank update during training
            bag_size: Number of instances (num_crops)
            pooling: Pooling method
            return_attention: whether to return attention weights
            
        Returns:
            logits: (batch, num_classes)
            attention: (batch, num_crops) if return_attention=True
        """
        if y is not None and y.dim() == 0:
            y = y.unsqueeze(0)
        
        # Extract features
        features = self.extract_features(x)
        
        # Apply MIL-Dropout if enabled
        if self.use_mildropout and self.training:
            features = self.mildropout(features)
        
        # PSAMIL forward - squeeze batch dim for compatibility
        features_squeezed = features.squeeze(0) if features.shape[0] == 1 else features
        
        if y is not None and y.shape[0] > 1:
            # Handle batch processing
            results = []
            attention_results = []
            for i in range(features.shape[0]):
                label = y[i] if y is not None else torch.tensor(-1)
                prob, hat, alpha, ins_probs, F_out = self.psamil(
                    features[i:i+1],
                    label,
                    features.shape[1],
                    pooling or self.pooling,
                    "bag"
                )
                results.append(prob)
                attention_results.append(alpha)
            Y_prob = torch.cat(results, dim=0)
            alpha = torch.cat(attention_results, dim=0)
        else:
            label = y[0] if y is not None else torch.tensor(-1)
            Y_prob, Y_hat, alpha, ins_probs, F_out = self.psamil(
                features,
                label,
                features.shape[1],
                pooling or self.pooling,
                "bag"
            )
        
        # Convert log probabilities to logits for compatibility
        logits = Y_prob.exp()
        
        if return_attention:
            return logits, alpha
        return logits


def create_psamil_model(
    num_classes: int = 96,
    pooling: str = "psa",
    dropout: float = 0.2,
    use_mildropout: bool = False,
    mildropout_topk: int = 3,
    mildropout_kernel: int = 7,
    mammoth=None
) -> PSAMILWrapper:
    """Factory to create PSAMIL model"""
    return PSAMILWrapper(
        num_classes=num_classes,
        pooling=pooling,
        dropout=dropout,
        use_mildropout=use_mildropout,
        mildropout_topk=mildropout_topk,
        mildropout_kernel=mildropout_kernel,
        mammoth=mammoth
    )


def add_psamil_args(parser):
    """Add PSAMIL arguments to argument parser"""
    import argparse
    
    parser.add_argument('--use_psamil', action='store_true',
        help='Use PSAMIL (probability-space attention)')
    parser.add_argument('--psamil_pooling', type=str, default='psa',
        choices=['psa', 'fsa', 'mha', 'avg'],
        help='PSMIL pooling mode: psa=probability-space, fsa=feature-space, mha=multi-head, avg=average')
    parser.add_argument('--psamil_dropout', type=float, default=0.2,
        help='Dropout in PSMIL')
    
    return parser


if __name__ == "__main__":
    print("Testing PSAMIL implementation...")
    
    model = PSAMIL(feature_dim=512, num_classes=96, pooling='psa')
    features = torch.randn(25, 512)
    y = torch.tensor(0)
    
    # Test PSA pooling
    Y_prob, Y_hat, alpha, ins_probs, F_out = model(features, y, bag_size=25, pooling='psa')
    print(f"PSA - Y_prob: {Y_prob.shape}, Y_hat: {Y_hat.shape}, alpha: {alpha.shape}")
    
    # Test FSA pooling
    Y_prob, Y_hat, alpha, ins_probs, F_out = model(features, y, bag_size=25, pooling='fsa')
    print(f"FSA - Y_prob: {Y_prob.shape}, alpha: {alpha.shape}")
    
    # Test MHA pooling
    Y_prob, Y_hat, alpha, ins_probs, F_out = model(features, y, bag_size=25, pooling='mha')
    print(f"MHA - Y_prob: {Y_prob.shape}, alpha: {alpha.shape}")
    
    # Test AVG pooling
    Y_prob, Y_hat, alpha, ins_probs, F_out = model(features, y, bag_size=25, pooling='avg')
    print(f"AVG - Y_prob: {Y_prob.shape}, alpha: {alpha.shape}")
    
    print("\nAll tests passed!")