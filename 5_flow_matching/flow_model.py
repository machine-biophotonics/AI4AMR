import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import fft
from diffusers import UNet2DModel


def gaussian_filter_low_pass(fshift, D):
    D = D * 2
    b, c, h, w = fshift.shape
    x = torch.arange(0, h, device=fshift.device)
    y = torch.arange(0, w, device=fshift.device)
    x, y = torch.meshgrid(x, y, indexing='ij')
    center = (int((h - 1) / 2), int((w - 1) / 2))
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
    template = torch.exp(-dis_square / (2 * D ** 2))
    return template.unsqueeze(0).unsqueeze(0).repeat(b, c, 1, 1) * fshift


def gaussian_filter_high_pass(fshift, D):
    D = D / 8.
    b, c, h, w = fshift.shape
    x = torch.arange(0, h, device=fshift.device)
    y = torch.arange(0, w, device=fshift.device)
    x, y = torch.meshgrid(x, y, indexing='ij')
    center = (int((h - 1) / 2), int((w - 1) / 2))
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
    template = 1 - torch.exp(-dis_square / (2 * D ** 2))
    return template.unsqueeze(0).unsqueeze(0).repeat(b, c, 1, 1) * fshift


def Fourier_filter(x, D):
    max_x, min_x = x.max(), x.min()
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    x_high = gaussian_filter_high_pass(x_freq, D)
    x_low = gaussian_filter_low_pass(x_freq, D)
    x_high = fft.ifftshift(x_high, dim=(-2, -1))
    x_high = fft.ifftn(x_high, dim=(-2, -1)).real
    x_low = fft.ifftshift(x_low, dim=(-2, -1))
    x_low = fft.ifftn(x_low, dim=(-2, -1)).real
    return torch.clamp(x_low, min_x, max_x), torch.clamp(x_high, min_x, max_x)


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
        self.unet.enable_gradient_checkpointing()

        self._mid_bottleneck = None
        self._mid_hook_handle = self.unet.mid_block.register_forward_hook(
            self._mid_bottleneck_hook
        )

    def _mid_bottleneck_hook(self, module, input, output):
        self._mid_bottleneck = output.mean(dim=[2, 3])

    def get_mid_bottleneck(self):
        return self._mid_bottleneck

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.unet(x_t, timestep=t, class_labels=class_labels, return_dict=True)
        return output.sample


class SemanticPrototype(nn.Module):
    """Learns class-specific prototypes η = F_φ(Y) for AuxPath-FM.

    Paper: AuxPath-FM (arXiv:2605.06364, May 2026).
    Maps class labels to a prototype image (same spatial dims as data).
    Lightweight: embed → MLP → 1×16×16 → upsample to 224×224.
    """
    def __init__(self, num_classes: int, latent_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(num_classes, latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
        )

    def forward(self, class_labels: torch.Tensor) -> torch.Tensor:
        B = class_labels.shape[0]
        h = self.embed(class_labels)
        h = self.net(h)
        h = h.view(B, 1, 16, 16)
        h = F.interpolate(h, size=(224, 224), mode='bilinear', align_corners=False)
        return h


class AuxProjectionHead(nn.Module):
    """Linear probe on GAP-pooled bottleneck features for auxiliary CE regularization."""
    def __init__(self, bottleneck_dim: int = 256, num_classes: int = 185):
        super().__init__()
        self.fc = nn.Linear(bottleneck_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class FreqFlowUNet(nn.Module):
    """Two-branch FreqFlow: spatial UNet + frequency UNet with FFT decomposition.

    Core insight (Ren et al., CVPR 2026): FMs generate low frequencies first,
    high frequencies (texture/details) later → model them separately.

    Spatial branch (1ch → 1ch): predicts full velocity u_t = x_1 - x_0.
    Frequency branch (1ch → 1ch): predicts high-frequency velocity only,
    takes high-pass filtered x_t as input. Forces model to learn frequency-aware
    representations for sharper details.

    Returns (v_freq, v_spatial): output[0]=freq, output[1]=spatial.
    """
    def __init__(
        self,
        in_channels: int = 1,
        sample_size: int = 224,
        block_out_channels: tuple = (64, 128, 256, 512),
        freq_block_out_channels: tuple = (32, 64, 128, 256),
        layers_per_block: int = 2,
        num_class_embeds: int = 50,
        freq_filter_D: float = 8.0,
    ):
        super().__init__()
        self.freq_filter_D = freq_filter_D

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

        self.spatial_unet = UNet2DModel(
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
        self.spatial_unet.enable_gradient_checkpointing()

        self._mid_bottleneck = None
        self._mid_hook_handle = self.spatial_unet.mid_block.register_forward_hook(
            self._mid_bottleneck_hook
        )

        # Frequency branch: 1ch high-pass input → 1ch high-pass velocity
        self.freq_unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=in_channels,
            block_out_channels=freq_block_out_channels,
            layers_per_block=layers_per_block,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            num_class_embeds=num_class_embeds,
            class_embed_type=None,
            act_fn="silu",
            norm_num_groups=32,
            dropout=0.1,
        )
        self.freq_unet.enable_gradient_checkpointing()

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v_spatial = self.spatial_unet(
            x_t, timestep=t, class_labels=class_labels, return_dict=True
        ).sample

        # Frequency branch: predict high-frequency velocity from high-pass x_t
        _, x_high = Fourier_filter(x_t, self.freq_filter_D)
        v_freq = self.freq_unet(
            x_high, timestep=t, class_labels=class_labels, return_dict=True
        ).sample

        return v_freq, v_spatial

    def _mid_bottleneck_hook(self, module, input, output):
        self._mid_bottleneck = output.mean(dim=[2, 3])

    def get_mid_bottleneck(self):
        return self._mid_bottleneck




def _sample_different_class(labels: torch.Tensor) -> torch.Tensor:
    """For each index, sample a random different index from a different class."""
    B = labels.shape[0]
    device = labels.device
    mask = labels[None, :] != labels[:, None]
    mask.fill_diagonal_(False)
    weights = mask.float()
    weights_sum = weights.sum(dim=1)
    if (weights_sum == 0).any():
        return torch.randint(0, B, (B,), device=device)
    choices = torch.multinomial(weights, 1).squeeze(1)
    return choices


def compute_flow_loss(
    model: FlowUNet | FreqFlowUNet,
    x_1: torch.Tensor,
    class_labels: torch.Tensor | None = None,
    freq_flow: bool = False,
    freq_filter_D: float = 8.0,
    freq_loss_weight: float = 0.25,
    delta_fm_lambda: float = 0.0,
    aux_path: bool = False,
    prototype: SemanticPrototype | None = None,
    aux_path_weight: float = 0.0,
    aux_ce_head: AuxProjectionHead | None = None,
    aux_ce_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    B = x_1.shape[0]
    device = x_1.device

    t = torch.rand(B, device=device)
    t = t.view(B, *([1] * (x_1.ndim - 1)))

    x_0 = torch.randn_like(x_1)

    # AuxPath-FM (arXiv:2605.06364, Algorithm 3):
    # X_t = Interpolant(X1, X0) + c(t)·η   where η = F_φ(Y)
    # Our convention: noise→data direction (for forward Euler sampling)
    # x_0=noise, x_1=data, a(t)=1-t, b(t)=t
    # X_t = (1-t)·X0 + t·X1 + t·(1-t)·η,  v_t = X1 - X0
    # (Equivalent to paper's repo with X0↔X1 and sign flip)
    eta = None
    if aux_path and prototype is not None:
        eta = prototype(class_labels)
        x_t = (1 - t) * x_0 + t * x_1 + t * (1 - t) * eta
    else:
        x_t = (1 - t) * x_0 + t * x_1

    # Target velocity: data - noise
    u_t = x_1 - x_0

    t_flat = t.view(B)

    # Capture bottleneck features via local hook (reliable in train AND eval)
    mid_bottleneck_container = []
    hook_handle = None
    if aux_ce_head is not None and aux_ce_weight > 0.0 and class_labels is not None:
        mid_block = (model.spatial_unet.mid_block if freq_flow else model.unet.mid_block)
        hook_handle = mid_block.register_forward_hook(
            lambda m, i, o: mid_bottleneck_container.append(o.mean(dim=[2, 3]))
        )

    output = model(x_t, t_flat, class_labels=class_labels)

    if hook_handle is not None:
        hook_handle.remove()

    components = {}

    if freq_flow:
        v_freq, v_spatial = output

        loss_spatial = F.mse_loss(v_spatial, u_t)
        _, u_high = Fourier_filter(u_t, freq_filter_D)
        loss_freq = F.mse_loss(v_freq, u_high) * freq_loss_weight

        loss = loss_spatial + loss_freq
        v_pred = v_spatial
        components['spatial'] = loss_spatial.item()
        components['freq'] = loss_freq.item()
    else:
        v_pred = output
        loss = F.mse_loss(v_pred, u_t)
        components['spatial'] = loss.item()
        components['freq'] = 0.0

    components['neg'] = 0.0
    if delta_fm_lambda > 0.0 and class_labels is not None:
        neg_idxs = _sample_different_class(class_labels)
        x_neg = x_1[neg_idxs]
        u_neg = x_neg - x_0[neg_idxs]
        loss_neg = F.mse_loss(v_pred, u_neg)
        loss = loss - delta_fm_lambda * loss_neg
        components['neg'] = loss_neg.item()

    # AuxPath-FM: L_F = MSE(F_φ(Y), X_1)  (paper Algorithm 3, Stage 1)
    components['aux'] = 0.0
    if aux_path and prototype is not None and aux_path_weight > 0.0 and eta is not None:
        loss_aux = F.mse_loss(eta, x_1.detach()) * aux_path_weight
        loss = loss + loss_aux
        components['aux'] = loss_aux.item()

    # Auxiliary CE loss on bottleneck features
    components['ce'] = 0.0
    if aux_ce_head is not None and aux_ce_weight > 0.0 and class_labels is not None and mid_bottleneck_container:
        mid_feats = mid_bottleneck_container[0]
        logits = aux_ce_head(mid_feats)
        loss_ce = F.cross_entropy(logits, class_labels) * aux_ce_weight
        loss = loss + loss_ce
        components['ce'] = loss_ce.item()

    return loss, components


@torch.no_grad()
def sample(
    model: FlowUNet | FreqFlowUNet,
    num_samples: int,
    num_steps: int = 100,
    class_labels: torch.Tensor | None = None,
    device: str = 'cuda',
    freq_flow: bool = False,
    aux_path: bool = False,
    prototype: SemanticPrototype | None = None,
) -> torch.Tensor:
    """Generate samples via Euler integration of the ODE.

    For FreqFlowUNet, uses output[1] (spatial branch) as the velocity.
    For AuxPath-FM, adds ċ(t)*η = (1-2t)*η to velocity (η from prototype).
    """
    model.eval()
    C = model.spatial_unet.config.in_channels if freq_flow else model.unet.config.in_channels
    H = W = model.spatial_unet.config.sample_size if freq_flow else model.unet.config.sample_size
    device = next(model.parameters()).device

    x = torch.randn(num_samples, C, H, W, device=device)
    if class_labels is not None:
        class_labels = class_labels.to(device)

    eta = None
    if aux_path and prototype is not None:
        eta = prototype(class_labels)

    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((num_samples,), t_val, device=device)
        out = model(x, t, class_labels=class_labels)
        v = out[1] if freq_flow else out
        if eta is not None:
            v = v + (1 - 2 * t_val) * eta
        x = x + v * dt

    return x.clamp(-1, 1)



