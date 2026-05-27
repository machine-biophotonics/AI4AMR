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


class GaussianMixture(nn.Module):
    """Learnable GMM prior for structured latent (VaDE, SCFM).

    p_psi(z) = sum_k pi_k * N(z | mu_k, diag(sigma_k^2))

    Unsupervised KL (VaDE decomposition):
      KL(q(z,c|x) || p(z,c))
        = KL(q(c|x) || p(c)) + sum_c q(c|x) * KL(N(mu_z, sigma^2_z) || N(mu_c, sigma^2_c))

    q(c|x) computed via closed-form Bayes rule (NOT a learned head):
      q(c|x) = p(c|z) = pi_c * N(mu_z|mu_c, sigma^2_c) / sum_j pi_j * N(mu_z|mu_j, sigma^2_j)
    """
    def __init__(self, n_components: int, latent_dim: int, unsupervised: bool = False):
        super().__init__()
        self.n_components = n_components
        self.latent_dim = latent_dim
        self.unsupervised = unsupervised
        self.register_parameter('means', nn.Parameter(torch.randn(n_components, latent_dim) * 0.1))
        self.register_parameter('logvars', nn.Parameter(torch.zeros(n_components, latent_dim)))
        self.register_parameter('logits', nn.Parameter(torch.zeros(n_components)))

    def get_component(self, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.means[labels], self.logvars[labels]

    def kl(self, mu_z: torch.Tensor, logvar_z: torch.Tensor,
           labels: torch.Tensor | None = None) -> torch.Tensor:
        if self.unsupervised:
            return self._kl_unsupervised(mu_z, logvar_z)
        mu_k, logvar_k = self.get_component(labels)
        return 0.5 * (
            logvar_k - logvar_z
            + (logvar_z.exp() + (mu_z - mu_k).pow(2)) / logvar_k.exp()
            - 1
        ).sum(dim=1).mean()

    def _kl_unsupervised(self, mu_z: torch.Tensor,
                         logvar_z: torch.Tensor) -> torch.Tensor:
        """VaDE KL with q(c|x) via Bayes rule."""
        B, D = mu_z.shape
        log_pi = F.log_softmax(self.logits, dim=-1)

        mu_z_exp = mu_z.unsqueeze(1)
        mu_k = self.means.unsqueeze(0)
        logvar_k = self.logvars.unsqueeze(0)

        log_N_z_given_c = -0.5 * (
            math.log(2 * math.pi) + logvar_k
            + (mu_z_exp - mu_k).pow(2) / logvar_k.exp()
        ).sum(dim=-1)

        log_p_c_z = log_pi.unsqueeze(0) + log_N_z_given_c
        q_c = F.softmax(log_p_c_z, dim=-1)
        log_q_c = F.log_softmax(log_p_c_z, dim=-1)

        kl_cat = (q_c * (log_q_c - log_pi.unsqueeze(0))).sum(dim=1)

        logvar_z_exp = logvar_z.unsqueeze(1)
        kl_gauss = 0.5 * (
            logvar_k - logvar_z_exp
            + (logvar_z_exp.exp() + (mu_z_exp - mu_k).pow(2)) / logvar_k.exp()
            - 1
        ).sum(dim=-1)

        kl_weighted = (q_c * kl_gauss).sum(dim=1)
        return (kl_cat + kl_weighted).mean()

    @torch.no_grad()
    def responsibilities(self, mu_z: torch.Tensor) -> torch.Tensor:
        """q(c|x) for diagnostics. Returns [B, K] soft assignments."""
        log_pi = F.log_softmax(self.logits, dim=-1)
        mu_z_exp = mu_z.unsqueeze(1)
        mu_k = self.means.unsqueeze(0)
        logvar_k = self.logvars.unsqueeze(0)
        log_N = -0.5 * (
            math.log(2 * math.pi) + logvar_k
            + (mu_z_exp - mu_k).pow(2) / logvar_k.exp()
        ).sum(dim=-1)
        return F.softmax(log_pi.unsqueeze(0) + log_N, dim=-1)

    @torch.no_grad()
    def diagnostics(self, mu_z: torch.Tensor) -> dict[str, float]:
        """Full GMM health report."""
        K = self.n_components
        pi = F.softmax(self.logits, dim=-1)
        var = self.logvars.exp()
        q = self.responsibilities(mu_z)
        hard = q.argmax(dim=-1)
        active = hard.unique().numel()
        pairwise = torch.cdist(self.means, self.means)
        triu = pairwise[~torch.eye(K, dtype=bool, device=pairwise.device)]
        assignment_ent = -(q * torch.log(q.clamp(1e-10, 1))).sum(dim=-1)
        return dict(
            pi_entropy=(-(pi * torch.log(pi + 1e-10)).sum()).item(),
            pi_max=pi.max().item(),
            pi_min=pi.min().item(),
            var_min=var.min().item(),
            var_max=var.max().item(),
            var_mean=var.mean().item(),
            mean_norm=self.means.norm(dim=-1).mean().item(),
            mean_pair_dist=triu.mean().item(),
            active_ratio=active / K,
            assignment_perplexity=assignment_ent.exp().mean().item(),
        )

    @torch.no_grad()
    def sample(self, n: int, labels: torch.Tensor | None = None) -> torch.Tensor:
        device = self.means.device
        if labels is not None and not self.unsupervised:
            mu = self.means[labels]
            logvar = self.logvars[labels]
        else:
            cats = torch.distributions.Categorical(logits=self.logits)
            idx = cats.sample((n,))
            mu = self.means[idx]
            logvar = self.logvars[idx]
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()


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
        def _unet_kwargs(unet):
            k = dict(timestep=t, return_dict=True)
            if class_labels is not None and unet.config.num_class_embeds is not None:
                k['class_labels'] = class_labels
            return k

        v_spatial = self.spatial_unet(x_t, **_unet_kwargs(self.spatial_unet)).sample

        _, x_high = Fourier_filter(x_t, self.freq_filter_D)
        v_freq = self.freq_unet(x_high, **_unet_kwargs(self.freq_unet)).sample

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
        predict_mu: bool = False,
        exogenous_dim: int = 64,
        use_gmm: bool = False,
        gmm_components: int = 30,
        unsupervised_gmm: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.predict_mu = predict_mu
        self.exogenous_dim = exogenous_dim

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

        # Encoder: mid-block features → (μ_z, logvar_z, [μ_ε])
        mid_dim = block_out_channels[-1]
        if predict_mu:
            self.mu_z_head = nn.Linear(mid_dim, latent_dim)
            self.logvar_z_head = nn.Linear(mid_dim, latent_dim)
            self.mu_eps_head = nn.Linear(mid_dim, exogenous_dim)
        else:
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

        # GMM prior (SCFM VaDE-style)
        if use_gmm:
            self.gmm = GaussianMixture(gmm_components, latent_dim,
                                       unsupervised=unsupervised_gmm)

        # Hook to capture mid-block features
        self._mid_feat = None
        self._mid_handle = self.unet.mid_block.register_forward_hook(self._mid_hook)

    def remove_hooks(self):
        if self._mid_handle is not None:
            self._mid_handle.remove()
            self._mid_handle = None

    def _mid_hook(self, module, input, output):
        self._mid_feat = output[0] if isinstance(output, tuple) else output

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs = dict(timestep=t, return_dict=True)
        if class_labels is not None and self.unet.config.num_class_embeds is not None:
            kwargs['class_labels'] = class_labels
        return self.unet(x_t, **kwargs).sample

    def encode(self, x: torch.Tensor, t: torch.Tensor,
               class_labels: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
        """Encode x at timestep t into structured latent parameters.

        Returns (μ_z, logvar_z) in velocity mode,
        or (μ_z, logvar_z, μ_ε) in predict_mu mode.
        """
        _ = self.forward(x, t, class_labels)
        feat = self._mid_feat
        pooled = feat.flatten(2).mean(dim=2)
        if self.predict_mu:
            mu_z = self.mu_z_head(pooled)
            logvar_z = self.logvar_z_head(pooled)
            mu_eps = self.mu_eps_head(pooled)
            return mu_z, logvar_z, mu_eps
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

    @torch.no_grad()
    def sample_prior_z(self, n: int) -> torch.Tensor:
        """Sample z from GMM prior or N(0,I)."""
        if hasattr(self, 'gmm') and self.gmm is not None:
            return self.gmm.sample(n)
        device = next(self.parameters()).device if list(self.parameters()) else 'cpu'
        return torch.randn(n, self.latent_dim, device=device)


class CombinedFlowUNet(nn.Module):
    """Unified model supporting FreqFlow + StructFlow + base FM in one class.

    Supports all combinations:
    - Base FM (only main_unet)
    - FreqFlow (main_unet + freq_unet for high-frequency branch)
    - StructFlow (main_unet + encoder_head + decoder + mid-block hook)
    - FreqFlow + StructFlow (all of the above combined)
    - DeltaFM + any of the above (loss-time only)

    When dual_predict_mu=True:
      - main_unet outputs 2 channels: [v_pred, mu_pred]
      - encoder has 3 independent heads: mu_z, logvar_z, mu_eps
      - dual loss: CFM(v_pred) + VFM(mu_pred)
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
        predict_mu: bool = False,
        exogenous_dim: int = 64,
        use_gmm: bool = False,
        gmm_components: int = 30,
        unsupervised_gmm: bool = False,
        dual_predict_mu: bool = False,
    ):
        super().__init__()
        self.freq_filter_D = freq_filter_D
        self.use_freq = use_freq
        self.use_struct = use_struct
        self.latent_dim = latent_dim
        self.sample_size = sample_size
        self.in_ch = in_channels
        self.predict_mu = predict_mu
        self.dual_predict_mu = dual_predict_mu
        self.exogenous_dim = exogenous_dim
        effective_predict_mu = predict_mu or dual_predict_mu

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
            out_channels=2 if dual_predict_mu else in_channels,
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
            freq_norm_groups = min(32, freq_block_out_channels[0])
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
                norm_num_groups=freq_norm_groups,
                dropout=0.1,
            )
            self.freq_unet.enable_gradient_checkpointing()

        if use_struct:
            mid_dim = block_out_channels[-1]
            if effective_predict_mu:
                self.mu_z_head = nn.Linear(mid_dim, latent_dim)
                self.logvar_z_head = nn.Linear(mid_dim, latent_dim)
                self.mu_eps_head = nn.Linear(mid_dim, exogenous_dim)
            else:
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
            if use_gmm:
                self.gmm = GaussianMixture(gmm_components, latent_dim,
                                           unsupervised=unsupervised_gmm)
            self._mid_feat = None
            self._mid_handle = self.main_unet.mid_block.register_forward_hook(self._mid_hook)

    def remove_hooks(self):
        if hasattr(self, '_mid_handle') and self._mid_handle is not None:
            self._mid_handle.remove()
            self._mid_handle = None

    def _mid_hook(self, module, input, output):
        if self.use_struct:
            self._mid_feat = output[0] if isinstance(output, tuple) else output

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def _unet_kwargs(unet):
            k = dict(timestep=t, return_dict=True)
            if class_labels is not None and unet.config.num_class_embeds is not None:
                k['class_labels'] = class_labels
            return k

        raw = self.main_unet(x_t, **_unet_kwargs(self.main_unet)).sample

        if self.dual_predict_mu:
            v_pred = raw[:, 0:1]
            mu_pred = raw[:, 1:2]
        else:
            v_pred = raw
            mu_pred = None

        if self.use_freq:
            _, x_high = Fourier_filter(x_t, self.freq_filter_D)
            v_freq = self.freq_unet(x_high, **_unet_kwargs(self.freq_unet)).sample
            if self.dual_predict_mu:
                return v_freq, v_pred, mu_pred
            return v_freq, v_pred

        if self.dual_predict_mu:
            return v_pred, mu_pred
        return v_pred

    def encode(self, x: torch.Tensor, t: torch.Tensor,
               class_labels: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
        _ = self.forward(x, t, class_labels)
        feat = self._mid_feat
        pooled = feat.flatten(2).mean(dim=2)
        if self.predict_mu or self.dual_predict_mu:
            mu_z = self.mu_z_head(pooled)
            logvar_z = self.logvar_z_head(pooled)
            mu_eps = self.mu_eps_head(pooled)
            return mu_z, logvar_z, mu_eps
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

    @torch.no_grad()
    def sample_prior_z(self, n: int) -> torch.Tensor:
        """Sample z from GMM prior or N(0,I)."""
        if hasattr(self, 'gmm') and self.gmm is not None:
            return self.gmm.sample(n)
        device = next(self.parameters()).device if list(self.parameters()) else 'cpu'
        return torch.randn(n, self.latent_dim, device=device)


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """SupCon loss (Khosla et al., NeurIPS 2020) on L2-normalized features.

    All same-class pairs are positives, different-class pairs are negatives.
    Anchors with no positives in batch → loss = 0.
    """
    B = features.shape[0]
    device = features.device
    features = F.normalize(features, dim=1)
    sim = features @ features.T
    sim = sim / temperature
    labels = labels.view(-1, 1)
    eye_mask = torch.eye(B, dtype=torch.bool, device=device)
    mask_pos = (labels == labels.T).float()
    mask_pos = mask_pos.masked_fill(eye_mask, 0.0)
    exp_sim = torch.exp(sim).masked_fill(eye_mask, 0.0)
    denom = exp_sim.sum(dim=1, keepdim=True)
    pos_exp = exp_sim * mask_pos
    pos_sum = pos_exp.sum(dim=1)
    loss = -torch.log(pos_sum / (denom.squeeze() + 1e-8) + 1e-8)
    loss = loss.masked_fill(pos_sum < 1e-8, 0.0)
    return loss.mean()


def compute_struct_flow_loss(
    model: StructFlowUNet | CombinedFlowUNet,
    x_1: torch.Tensor,
    class_labels: torch.Tensor | None = None,
    kl_weight: float = 0.001,
    recon_weight: float = 0.1,
    delta_fm_lambda: float = 0.0,
    supcon_weight: float = 0.0,
    supcon_temperature: float = 0.1,
    predict_mu: bool = False,
    beta: float | None = None,
    use_gmm: bool = False,
    unsupervised_gmm: bool = False,
    r_eps_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """StructFlow loss: flow matching + VAE-style KL + reconstruction + SupCon.

    Two modes:
      - Standard (predict_mu=False): UNet predicts velocity v, loss = MSE(v, u_t).
      - SCFM (predict_mu=True): UNet predicts posterior mean μ, loss = MSE(μ, sg(x₀)).
        Velocity derived: v = (x_t - μ) / t. Adds R_ε = ½·||μ_ε||².

    When predict_mu=True, `beta` overrides `kl_weight` for the effective β-VAE ratio.
    """
    B = x_1.shape[0]
    device = x_1.device

    # 1. Encode: structured latent z at t=1
    t_enc = torch.full((B,), 1.0, device=device)
    if predict_mu:
        mu_z, logvar_z, mu_eps = model.encode(x_1, t_enc, class_labels)
    else:
        mu_z, logvar_z = model.encode(x_1, t_enc, class_labels)

    std = torch.exp(0.5 * logvar_z)
    eps_z = torch.randn_like(std)
    z = mu_z + eps_z * std

    # 2. KL divergence
    if predict_mu and use_gmm and hasattr(model, 'gmm'):
        kl_labels = None if unsupervised_gmm else class_labels
        kl_loss = model.gmm.kl(mu_z, logvar_z, kl_labels)
    else:
        kl_loss = -0.5 * (1 + logvar_z - mu_z.pow(2) - logvar_z.exp()).sum(dim=1).mean()

    # 3. Reconstruction loss + structured source (single decode call)
    x_z = model.decode(z)
    recon_loss = F.mse_loss(x_z, x_1)

    # 4. Flow matching with structured source
    t = torch.rand(B, device=device)
    t_b = t.view(B, *([1] * (x_1.ndim - 1)))

    noise = torch.randn_like(x_1)
    x_0 = x_z + noise  # structured + exogenous (grad flows to encoder/GMM)

    x_t = (1 - t_b) * x_0 + t_b * x_1
    u_t = x_1 - x_0

    model_out = model(x_t, t, class_labels=class_labels)

    if predict_mu:
        mu_pred = model_out
        flow_loss = F.mse_loss(mu_pred, x_0.detach())
        v_pred = (x_t - mu_pred) / t_b.clamp(min=1e-5)
    else:
        v_pred = model_out
        flow_loss = F.mse_loss(v_pred, u_t)

    effective_kl_weight = beta if (predict_mu and beta is not None) else kl_weight
    total = flow_loss + effective_kl_weight * kl_loss + recon_weight * recon_loss

    comp: dict[str, float] = {}
    comp['flow'] = flow_loss.item()
    comp['kl'] = kl_loss.item()
    comp['recon'] = recon_loss.item()
    comp['neg'] = 0.0
    comp['supcon'] = 0.0
    comp['r_eps'] = 0.0

    # Exogenous regularization R_ε (Eq 15)
    if predict_mu:
        r_eps = 0.5 * mu_eps.pow(2).sum(dim=1).mean()
        total = total + r_eps_weight * r_eps
        comp['r_eps'] = r_eps.item()

    if supcon_weight > 0.0 and class_labels is not None:
        supcon_loss = supervised_contrastive_loss(mu_z, class_labels, temperature=supcon_temperature)
        total = total + supcon_weight * supcon_loss
        comp['supcon'] = supcon_loss.item()

    if delta_fm_lambda > 0.0 and class_labels is not None:
        neg_idxs = _sample_different_class(class_labels)
        x_neg = x_1[neg_idxs]
        u_neg = x_neg - x_0  # same source (x_0), different target
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
    predict_mu: bool = False,
) -> torch.Tensor:
    """Generate samples via structured prior + ODE refinement.

    1. Sample z ~ N(0, I) or GMM (structured prior)
    2. Decode: x_z = decoder(z)
    3. Add exogenous noise: x_0 = x_z + ε
    4. Integrate ODE from x_0 → x_1
    """
    model.eval()
    latent_dim = model.latent_dim

    if predict_mu and hasattr(model, 'sample_prior_z'):
        z = model.sample_prior_z(num_samples)
    else:
        z = torch.randn(num_samples, latent_dim, device=device)
    x_z = model.decode(z)

    noise = torch.randn_like(x_z)
    x = x_z + noise

    if class_labels is not None:
        class_labels = class_labels.to(device)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((num_samples,), i * dt, device=device)
        out = model(x, t, class_labels=class_labels)
        if predict_mu:
            v = (x - out) / t.view(-1, *([1] * (x.ndim - 1))).clamp(min=1e-5)
        else:
            v = out
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
        u_neg = x_neg - x_0  # same source (x_0), different target
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
    supcon_weight: float = 0.0,
    supcon_temperature: float = 0.1,
    predict_mu: bool = False,
    beta: float | None = None,
    use_gmm: bool = False,
    unsupervised_gmm: bool = False,
    dual_predict_mu: bool = False,
    r_eps_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Unified loss supporting all FreqFlow + StructFlow + DeltaFM + SupCon combinations.

    When dual_predict_mu=True:
      - UNet outputs both v_pred and mu_pred (2-channel head).
      - Dual loss: CFM(v_pred, u_t) + VFM(mu_pred, sg(x₀)).
      - DeltaFM operates on v_pred directly (stable).
    When predict_mu=True (and dual_predict_mu=False):
      - SCFM: only posterior mean, velocity derived as (x_t - μ) / t.
    """
    B = x_1.shape[0]
    device = x_1.device
    comp: dict[str, float] = {}

    effective_predict_mu = predict_mu or dual_predict_mu

    kl_loss = torch.tensor(0.0, device=device)
    recon_loss = torch.tensor(0.0, device=device)
    r_eps = torch.tensor(0.0, device=device)
    mu_z = None

    if use_struct:
        t_enc = torch.full((B,), 1.0, device=device)
        if effective_predict_mu:
            mu_z, logvar_z, mu_eps = model.encode(x_1, t_enc, class_labels)
        else:
            mu_z, logvar_z = model.encode(x_1, t_enc, class_labels)

        std = torch.exp(0.5 * logvar_z)
        eps_z = torch.randn_like(std)
        z = mu_z + eps_z * std

        if effective_predict_mu and use_gmm and hasattr(model, 'gmm'):
            kl_labels = None if unsupervised_gmm else class_labels
            kl_loss = model.gmm.kl(mu_z, logvar_z, kl_labels)
        else:
            kl_loss = -0.5 * (1 + logvar_z - mu_z.pow(2) - logvar_z.exp()).sum(dim=1).mean()

        x_z = model.decode(z)
        recon_loss = F.mse_loss(x_z, x_1)
        noise = torch.randn_like(x_1)
        x_0 = x_z + noise

        if effective_predict_mu:
            r_eps = 0.5 * mu_eps.pow(2).sum(dim=1).mean()

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

    if dual_predict_mu and use_freq:
        v_freq, v_pred, mu_pred = output
        x_0_sg = x_0.detach()
        loss_v = F.mse_loss(v_pred, u_t)
        loss_mu = F.mse_loss(mu_pred, x_0_sg)
        _, u_high = Fourier_filter(u_t, freq_filter_D)
        loss_freq = F.mse_loss(v_freq, u_high) * freq_loss_weight
        flow_loss = loss_v + loss_mu + loss_freq
        comp['spatial'] = loss_v.item()
        comp['mu_pred'] = loss_mu.item()
        comp['freq'] = loss_freq.item()
    elif dual_predict_mu and not use_freq:
        v_pred, mu_pred = output
        x_0_sg = x_0.detach()
        loss_v = F.mse_loss(v_pred, u_t)
        loss_mu = F.mse_loss(mu_pred, x_0_sg)
        flow_loss = loss_v + loss_mu
        comp['spatial'] = loss_v.item()
        comp['mu_pred'] = loss_mu.item()
        comp['freq'] = 0.0
    elif predict_mu and use_freq:
        mu_high, mu_main = output
        x_0_sg = x_0.detach()
        loss_spatial = F.mse_loss(mu_main, x_0_sg)
        _, u_high = Fourier_filter(u_t, freq_filter_D)
        loss_freq = F.mse_loss(mu_high, u_high) * freq_loss_weight
        flow_loss = loss_spatial + loss_freq
        v_pred = (x_t - mu_main) / t_b.clamp(min=1e-5)
        comp['spatial'] = loss_spatial.item()
        comp['freq'] = loss_freq.item()
    elif predict_mu and not use_freq:
        mu_pred = output
        x_0_sg = x_0.detach()
        flow_loss = F.mse_loss(mu_pred, x_0_sg)
        v_pred = (x_t - mu_pred) / t_b.clamp(min=1e-5)
        comp['spatial'] = flow_loss.item()
        comp['freq'] = 0.0
    elif not predict_mu and use_freq:
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

    effective_kl_weight = beta if (effective_predict_mu and beta is not None) else kl_weight
    total = flow_loss + effective_kl_weight * kl_loss + recon_weight * recon_loss + r_eps_weight * r_eps
    comp['flow'] = flow_loss.item()
    comp['neg'] = 0.0
    comp['supcon'] = 0.0
    comp['r_eps'] = r_eps.item()

    if supcon_weight > 0.0 and class_labels is not None and mu_z is not None and use_struct:
        supcon_loss = supervised_contrastive_loss(mu_z, class_labels, temperature=supcon_temperature)
        total = total + supcon_weight * supcon_loss
        comp['supcon'] = supcon_loss.item()

    if delta_fm_lambda > 0.0 and class_labels is not None:
        if dual_predict_mu:
            # DeltaFM on v_pred (direct UNet output, stable)
            v_pred_for_delta = v_pred
        elif predict_mu:
            v_pred_for_delta = v_pred  # derived velocity, may have 1/t² issues
        else:
            v_pred_for_delta = v_pred
        neg_idxs = _sample_different_class(class_labels)
        x_neg = x_1[neg_idxs]
        u_neg = x_neg - x_0  # same source (x_0), different target
        loss_neg = F.mse_loss(v_pred_for_delta, u_neg)
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
    predict_mu: bool = False,
    dual_predict_mu: bool = False,
) -> torch.Tensor:
    """Generate samples supporting FreqFlow + StructFlow combinations.

    When dual_predict_mu=True: uses v_pred from UNet directly (stable,
    class-conditional, matches old eed9ef9 behavior).
    When predict_mu=True (alone): velocity derived from posterior mean.
    """
    model.eval()
    C = model.main_unet.config.in_channels
    H = W = model.main_unet.config.sample_size

    has_struct_prior = predict_mu or dual_predict_mu
    if use_struct:
        if has_struct_prior and hasattr(model, 'sample_prior_z'):
            z = model.sample_prior_z(num_samples)
        else:
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
        if dual_predict_mu:
            # Direct v_pred from UNet channel 0 (stable, class-conditional)
            v = out[1] if use_freq else out[0]
        elif predict_mu:
            mu_main = out[1] if use_freq else out
            v = (x - mu_main) / t.view(-1, *([1] * (x.ndim - 1))).clamp(min=1e-5)
        else:
            v = out[1] if use_freq else out
        x = x + v * dt

    return x.clamp(-1, 1)
