import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dino import DINOEncoderLoRA


class DINOEncoderLoRAForClassification(DINOEncoderLoRA):
    """DINOEncoderLoRA adapted for classification using the class token."""
    def __init__(
        self,
        encoder,
        r: int = 8,
        emb_dim: int = 1024,
        n_classes: int = 85,
        use_lora: bool = True,
        img_dim: tuple[int, int] = (224, 224),
        dropout: float = 0.2,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        # We ignore use_fpn and set to False
        super().__init__(
            encoder=encoder,
            r=r,
            emb_dim=emb_dim,
            n_classes=n_classes,
            use_lora=use_lora,
            use_fpn=False,
            img_dim=img_dim,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        # Replace the decoder with a classification head
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_dim, n_classes),
        )
        # The parent class already froze encoder parameters before adding LoRA,
        # so LoRA parameters are trainable. Decoder parameters are trainable by default.
        pass
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get encoder features
        if self.use_fpn:
            # This shouldn't happen as we set use_fpn=False
            raise ValueError("FPN not supported for classification")
        else:
            feature = self.encoder.forward_features(x)
            # DINOv3 returns dict with 'x_norm_clstoken' (class token)
            if isinstance(feature, dict):
                cls_token = feature['x_norm_clstoken']
            else:
                # Fallback: assume feature is (B, seq_len, embed_dim) and class token is first token
                cls_token = feature[:, 0, :]
            logits = self.decoder(cls_token)
        return logits
    
    def save_parameters(self, filename: str) -> None:
        """Save LoRA and classification head weights."""
        w_a, w_b = {}, {}
        if self.use_lora:
            w_a = {f"w_a_{i:03d}": self.w_a[i].weight for i in range(len(self.w_a))}
            w_b = {f"w_b_{i:03d}": self.w_b[i].weight for i in range(len(self.w_a))}
        decoder_weights = self.decoder.state_dict()
        torch.save({**w_a, **w_b, **decoder_weights}, filename)
        
    def load_parameters(self, filename: str) -> None:
        """Load LoRA and classification head weights."""
        state_dict = torch.load(filename)
        # Load LoRA parameters
        if self.use_lora:
            for i, w_A_linear in enumerate(self.w_a):
                saved_key = f"w_a_{i:03d}"
                if saved_key in state_dict:
                    w_A_linear.weight = nn.Parameter(state_dict[saved_key])
            for i, w_B_linear in enumerate(self.w_b):
                saved_key = f"w_b_{i:03d}"
                if saved_key in state_dict:
                    w_B_linear.weight = nn.Parameter(state_dict[saved_key])
        # Load decoder parameters
        decoder_head_dict = self.decoder.state_dict()
        decoder_head_keys = [k for k in decoder_head_dict.keys()]
        decoder_state_dict = {k: state_dict[k] for k in decoder_head_keys if k in state_dict}
        self.decoder.load_state_dict(decoder_state_dict)