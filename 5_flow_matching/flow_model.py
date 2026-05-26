import math
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

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.unet(x_t, timestep=t, class_labels=class_labels, return_dict=True)
        return output.sample


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


class StructFlowUNet(nn.Module):
    """Structured Coupling for Flow Matching (SCFM).

    Augments the standard noise source x_0 with a structured latent z.
    A shared encoder extracts z from x_t; at t=1 it acts as VAE posterior,
    at t<1 it informs the flow velocity. A decoder maps z → image for
    the structured component of the source.

    Core idea (Sumba et al., arXiv 2026): x_0 = decoder(z) + ε,
    where z ~ q(z|x_1) captures semantic structure and ε is exogenous noise.
    """
    def __init__(
        self,
        in_channels: int = 1,
        sample_size: int = 224,
        block_out_channels: tuple = (32, 64, 128, 256),
        layers_per_block: int = 2,
        num_class_embeds: int = 50,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.sample_size = sample_size
        self.in_channels = in_channels

        down_block_types = (
            "DownBlock2D", "DownBlock2D",
            "AttnDownBlock2D", "AttnDownBlock2D",
        )
        up_block_types = (
            "AttnUpBlock2D", "AttnUpBlock2D",
            "UpBlock2D", "UpBlock2D",
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

        # Encoder: mid-block features → (μ_z, logvar_z)
        mid_dim = block_out_channels[-1]
        self.encoder_head = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, latent_dim * 2),
        )

        # Decoder: z → pixel-space reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.SiLU(),
            nn.Linear(512, in_channels * sample_size * sample_size),
        )

        # Hook to capture mid-block features
        self._mid_feat = None
        self._mid_handle = self.unet.mid_block.register_forward_hook(self._mid_hook)

    def _mid_hook(self, module, input, output):
        self._mid_feat = output[0] if isinstance(output, tuple) else output

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.unet(x_t, timestep=t, class_labels=class_labels, return_dict=True).sample

    def encode(self, x: torch.Tensor, t: torch.Tensor,
               class_labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode x at timestep t into structured latent parameters."""
        _ = self.forward(x, t, class_labels)
        feat = self._mid_feat
        pooled = feat.flatten(2).mean(dim=2)
        params = self.encoder_head(pooled)
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode structured latent into pixel reconstruction."""
        img = self.decoder(z)
        return img.reshape(-1, self.in_channels, self.sample_size, self.sample_size)

    @torch.no_grad()
    def encode_at_t1(self, x: torch.Tensor,
                     class_labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience: encode at t=1 (VAE posterior)."""
        t = torch.full((x.shape[0],), 1.0, device=x.device)
        return self.encode(x, t, class_labels)


class CombinedFlowUNet(nn.Module):
    """Unified model supporting FreqFlow + StructFlow + base FM in one class.

    Supports all combinations:
    - Base FM (only main_unet)
    - FreqFlow (main_unet + freq_unet for high-frequency branch)
    - StructFlow (main_unet + encoder_head + decoder + mid-block hook)
    - FreqFlow + StructFlow (all of the above combined)
    - DeltaFM + any of the above (loss-time only)

    Forward signature is compatible with existing classes:
    - Base/Struct only: returns single tensor v_pred (like FlowUNet/StructFlowUNet)
    - Freq (with/without Struct): returns tuple (v_freq, v_pred) (like FreqFlowUNet)
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
        use_freq: bool = False,
        use_struct: bool = False,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.freq_filter_D = freq_filter_D
        self.use_freq = use_freq
        self.use_struct = use_struct
        self.latent_dim = latent_dim
        self.sample_size = sample_size
        self.in_ch = in_channels

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

        self.main_unet = UNet2DModel(
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
        self.main_unet.enable_gradient_checkpointing()

        if use_freq:
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

        if use_struct:
            mid_dim = block_out_channels[-1]
            self.encoder_head = nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                nn.SiLU(),
                nn.Linear(mid_dim, latent_dim * 2),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 512),
                nn.SiLU(),
                nn.Linear(512, in_channels * sample_size * sample_size),
            )
            self._mid_feat = None
            self._mid_handle = self.main_unet.mid_block.register_forward_hook(self._mid_hook)

    def _mid_hook(self, module, input, output):
        if self.use_struct:
            self._mid_feat = output[0] if isinstance(output, tuple) else output

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        v_pred = self.main_unet(
            x_t, timestep=t, class_labels=class_labels, return_dict=True
        ).sample

        if self.use_freq:
            _, x_high = Fourier_filter(x_t, self.freq_filter_D)
            v_freq = self.freq_unet(
                x_high, timestep=t, class_labels=class_labels, return_dict=True
            ).sample
            return v_freq, v_pred

        return v_pred

    def encode(self, x: torch.Tensor, t: torch.Tensor,
               class_labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        _ = self.forward(x, t, class_labels)
        feat = self._mid_feat
        pooled = feat.flatten(2).mean(dim=2)
        params = self.encoder_head(pooled)
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        img = self.decoder(z)
        return img.reshape(-1, self.in_ch, self.sample_size, self.sample_size)

    @torch.no_grad()
    def encode_at_t1(self, x: torch.Tensor,
                     class_labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.full((x.shape[0],), 1.0, device=x.device)
        return self.encode(x, t, class_labels)


def compute_struct_flow_loss(
    model: StructFlowUNet | CombinedFlowUNet,
    x_1: torch.Tensor,
    class_labels: torch.Tensor | None = None,
    kl_weight: float = 0.001,
    recon_weight: float = 0.1,
    delta_fm_lambda: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """StructFlow loss: flow matching + VAE-style KL + reconstruction.

    Constructs structured source: x_0 = decode(z) + ε, where
    z ~ q_φ(z|x_1) captures semantic content and ε is random noise.
    """
    B = x_1.shape[0]
    device = x_1.device

    # 1. Encode: structured latent z at t=1
    t_enc = torch.full((B,), 1.0, device=device)
    mu_z, logvar_z = model.encode(x_1, t_enc, class_labels)

    std = torch.exp(0.5 * logvar_z)
    eps_z = torch.randn_like(std)
    z = mu_z + eps_z * std

    # 2. KL divergence: KL(q(z|x_1) || N(0,I))
    kl_loss = -0.5 * (1 + logvar_z - mu_z.pow(2) - logvar_z.exp()).sum(dim=1).mean()

    # 3. Reconstruction loss
    x_recon = model.decode(z)
    recon_loss = F.mse_loss(x_recon, x_1)

    # 4. Flow matching with structured source
    t = torch.rand(B, device=device)
    t_b = t.view(B, *([1] * (x_1.ndim - 1)))

    x_z = model.decode(z).detach()  # structured part, stop grad to avoid trivial solution
    noise = torch.randn_like(x_1)
    x_0 = x_z + noise  # structured + exogenous

    x_t = (1 - t_b) * x_0 + t_b * x_1
    u_t = x_1 - x_0

    v_pred = model(x_t, t, class_labels=class_labels)
    flow_loss = F.mse_loss(v_pred, u_t)

    total = flow_loss + kl_weight * kl_loss + recon_weight * recon_loss

    comp: dict[str, float] = {}
    comp['flow'] = flow_loss.item()
    comp['kl'] = kl_loss.item()
    comp['recon'] = recon_loss.item()
    comp['neg'] = 0.0

    if delta_fm_lambda > 0.0 and class_labels is not None:
        neg_idxs = _sample_different_class(class_labels)
        x_neg = x_1[neg_idxs]
        x_0_neg = torch.randn_like(x_1)
        u_neg = x_neg - x_0_neg
        loss_neg = F.mse_loss(v_pred, u_neg)
        total = total - delta_fm_lambda * loss_neg
        comp['neg'] = loss_neg.item()

    return total, comp


@torch.no_grad()
def sample_struct(
    model: StructFlowUNet,
    num_samples: int,
    num_steps: int = 100,
    class_labels: torch.Tensor | None = None,
    device: str = 'cuda',
) -> torch.Tensor:
    """Generate samples via structured prior + ODE refinement.

    1. Sample z ~ N(0, I) (structured prior)
    2. Decode: x_z = decoder(z)
    3. Add exogenous noise: x_0 = x_z + ε
    4. Integrate ODE from x_0 → x_1
    """
    model.eval()
    latent_dim = model.latent_dim
    device = next(model.parameters()).device

    z = torch.randn(num_samples, latent_dim, device=device)
    x_z = model.decode(z)

    noise = torch.randn_like(x_z)
    x = x_z + noise

    if class_labels is not None:
        class_labels = class_labels.to(device)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((num_samples,), i * dt, device=device)
        v = model(x, t, class_labels=class_labels)
        x = x + v * dt

    return x.clamp(-1, 1)


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
    model: FlowUNet | FreqFlowUNet | StructFlowUNet,
    x_1: torch.Tensor,
    class_labels: torch.Tensor | None = None,
    freq_flow: bool = False,
    freq_filter_D: float = 8.0,
    freq_loss_weight: float = 0.25,
    delta_fm_lambda: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    B = x_1.shape[0]
    device = x_1.device

    t = torch.rand(B, device=device)
    t = t.view(B, *([1] * (x_1.ndim - 1)))

    x_0 = torch.randn_like(x_1)

    x_t = (1 - t) * x_0 + t * x_1
    u_t = x_1 - x_0

    t_flat = t.view(B)
    output = model(x_t, t_flat, class_labels=class_labels)

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
        x_0_neg = torch.randn_like(x_1)
        u_neg = x_neg - x_0_neg
        loss_neg = F.mse_loss(v_pred, u_neg)
        loss = loss - delta_fm_lambda * loss_neg
        components['neg'] = loss_neg.item()

    return loss, components


@torch.no_grad()
def sample(
    model: FlowUNet | FreqFlowUNet,
    num_samples: int,
    num_steps: int = 100,
    class_labels: torch.Tensor | None = None,
    device: str = 'cuda',
    freq_flow: bool = False,
) -> torch.Tensor:
    """Generate samples via Euler integration of the ODE.

    For FreqFlowUNet, uses output[1] (spatial branch) as the velocity.
    """
    model.eval()
    C = model.spatial_unet.config.in_channels if freq_flow else model.unet.config.in_channels
    H = W = model.spatial_unet.config.sample_size if freq_flow else model.unet.config.sample_size
    device = next(model.parameters()).device

    x = torch.randn(num_samples, C, H, W, device=device)
    if class_labels is not None:
        class_labels = class_labels.to(device)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((num_samples,), i * dt, device=device)
        out = model(x, t, class_labels=class_labels)
        v = out[1] if freq_flow else out
        x = x + v * dt

    return x.clamp(-1, 1)


def compute_unified_loss(
    model: CombinedFlowUNet,
    x_1: torch.Tensor,
    class_labels: torch.Tensor | None = None,
    use_freq: bool = False,
    use_struct: bool = False,
    freq_filter_D: float = 8.0,
    freq_loss_weight: float = 0.25,
    kl_weight: float = 0.001,
    recon_weight: float = 0.1,
    delta_fm_lambda: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Unified loss supporting all FreqFlow + StructFlow + DeltaFM combinations."""
    B = x_1.shape[0]
    device = x_1.device
    comp: dict[str, float] = {}

    kl_loss = torch.tensor(0.0, device=device)
    recon_loss = torch.tensor(0.0, device=device)

    if use_struct:
        t_enc = torch.full((B,), 1.0, device=device)
        mu_z, logvar_z = model.encode(x_1, t_enc, class_labels)

        std = torch.exp(0.5 * logvar_z)
        eps_z = torch.randn_like(std)
        z = mu_z + eps_z * std

        kl_loss = -0.5 * (1 + logvar_z - mu_z.pow(2) - logvar_z.exp()).sum(dim=1).mean()
        recon_loss = F.mse_loss(model.decode(z), x_1)
        x_z = model.decode(z).detach()
        noise = torch.randn_like(x_1)
        x_0 = x_z + noise
        comp['kl'] = kl_loss.item()
        comp['recon'] = recon_loss.item()
    else:
        x_0 = torch.randn_like(x_1)
        comp['kl'] = 0.0
        comp['recon'] = 0.0

    t = torch.rand(B, device=device)
    t_b = t.view(B, *([1] * (x_1.ndim - 1)))
    x_t = (1 - t_b) * x_0 + t_b * x_1
    u_t = x_1 - x_0

    output = model(x_t, t, class_labels=class_labels)

    if use_freq:
        v_freq, v_pred = output
        loss_spatial = F.mse_loss(v_pred, u_t)
        _, u_high = Fourier_filter(u_t, freq_filter_D)
        loss_freq = F.mse_loss(v_freq, u_high) * freq_loss_weight
        flow_loss = loss_spatial + loss_freq
        comp['spatial'] = loss_spatial.item()
        comp['freq'] = loss_freq.item()
    else:
        v_pred = output
        flow_loss = F.mse_loss(v_pred, u_t)
        comp['spatial'] = flow_loss.item()
        comp['freq'] = 0.0

    total = flow_loss + kl_weight * kl_loss + recon_weight * recon_loss
    comp['flow'] = flow_loss.item()
    comp['neg'] = 0.0

    if delta_fm_lambda > 0.0 and class_labels is not None:
        neg_idxs = _sample_different_class(class_labels)
        x_neg = x_1[neg_idxs]
        x_0_neg = torch.randn_like(x_1)
        u_neg = x_neg - x_0_neg
        loss_neg = F.mse_loss(v_pred, u_neg)
        total = total - delta_fm_lambda * loss_neg
        comp['neg'] = loss_neg.item()

    return total, comp


@torch.no_grad()
def sample_combined(
    model: CombinedFlowUNet,
    num_samples: int,
    num_steps: int = 100,
    class_labels: torch.Tensor | None = None,
    device: str = 'cuda',
    use_freq: bool = False,
    use_struct: bool = False,
) -> torch.Tensor:
    """Generate samples supporting FreqFlow + StructFlow combinations."""
    model.eval()
    C = model.main_unet.config.in_channels
    H = W = model.main_unet.config.sample_size
    device = next(model.parameters()).device

    if use_struct:
        z = torch.randn(num_samples, model.latent_dim, device=device)
        x_z = model.decode(z)
        x = x_z + torch.randn_like(x_z)
    else:
        x = torch.randn(num_samples, C, H, W, device=device)

    if class_labels is not None:
        class_labels = class_labels.to(device)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((num_samples,), i * dt, device=device)
        out = model(x, t, class_labels=class_labels)
        v = out[1] if use_freq else out
        x = x + v * dt

    return x.clamp(-1, 1)
