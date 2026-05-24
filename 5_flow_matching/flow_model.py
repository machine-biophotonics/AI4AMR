import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel


class FlowUNet(nn.Module):
    """UNet2DModel wrapper for Conditional Flow Matching.

    Flow matching (CFM):
        - Linear OT path: x_t = (1 - t) * x_0 + t * x_1
        - Target velocity: u_t = x_1 - x_0
        - Loss: MSE(v_pred, u_t) where v_pred = model(x_t, t, class_labels)
    """
    def __init__(
        self,
        in_channels: int = 1,
        sample_size: int = 224,
        block_out_channels: tuple = (64, 128, 256, 512),
        layers_per_block: int = 2,
        num_class_embeds: int = 50,
    ):
        super().__init__()

        down_block_types = (
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        )
        up_block_types = (
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        )

        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=in_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            num_class_embeds=num_class_embeds,
            class_embed_type=None,
            act_fn="silu",
            norm_num_groups=32,
            dropout=0.1,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Enable gradient checkpointing to save memory
        self.unet.enable_gradient_checkpointing()
        output = self.unet(x_t, timestep=t, class_labels=class_labels, return_dict=True)
        return output.sample


def compute_flow_loss(
    model: FlowUNet,
    x_1: torch.Tensor,
    class_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    B = x_1.shape[0]
    device = x_1.device

    t = torch.rand(B, device=device)
    t = t.view(B, *([1] * (x_1.ndim - 1)))

    x_0 = torch.randn_like(x_1)

    x_t = (1 - t) * x_0 + t * x_1
    u_t = x_1 - x_0

    t_flat = t.view(B)
    v_pred = model(x_t, t_flat, class_labels=class_labels)

    loss = F.mse_loss(v_pred, u_t)
    return loss


@torch.no_grad()
def sample(
    model: FlowUNet,
    num_samples: int,
    num_steps: int = 100,
    class_labels: torch.Tensor | None = None,
    device: str = 'cuda',
) -> torch.Tensor:
    """Generate samples via Euler integration of the ODE.

    Args:
        model: FlowUNet
        num_samples: number of samples to generate
        num_steps: number of Euler steps
        class_labels: (num_samples,) optional class labels
        device: device

    Returns:
        samples: (num_samples, C, H, W) in [-1, 1]
    """
    model.eval()
    C = model.unet.config.in_channels
    H = W = model.unet.config.sample_size
    device = next(model.parameters()).device

    x = torch.randn(num_samples, C, H, W, device=device)
    if class_labels is not None:
        class_labels = class_labels.to(device)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((num_samples,), i * dt, device=device)
        v = model(x, t, class_labels=class_labels)
        x = x + v * dt

    return x.clamp(-1, 1)
