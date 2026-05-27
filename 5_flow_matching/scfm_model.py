"""SCFM: Structured Coupling for Flow Matching (Sumba et al., arXiv:2605.07676, 2026).

Exact paper implementation (Sec 3.1-3.3, Eq 11-16, Algorithm 1).

Key architectural features:
  1. Shared UNet predicts μ_ϕ(x_t, t) = E[x₀|x_t] (posterior mean of source, NOT velocity)
  2. Velocity derived: v_ϕ,t(x_t) = (x_t - μ_ϕ(x_t,t))/t  (Eq 11)
  3. At t=1: same UNet + variance head → q_ϕ(z|x₁) = N(μ_z, diag(σ²_z))  (Eq 14)
  4. Exogenous regularization: R_ε = ½·E[||μ^ε_ϕ(x₁)||²]  (Eq 15)
  5. GMM prior p_ψ(z) via VaDE-style KL (BCFN from VaDE paper, Eq 16)
  6. Source: x₀ = decoder(z) + ε (structured + exogenous)

Total loss (Eq 16):
  L_SCFM = L_VFM + β·L_KL + L_rec + R_ε
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel


class GaussianMixture(nn.Module):
    """Learnable GMM prior for structured latent (VaDE, SCFM).

    p_psi(z) = sum_k pi_k * N(z | mu_k, diag(sigma_k^2))

    Unsupervised KL (VaDE decomposition):
      KL(q(z,c|x) || p(z,c))
        = KL(q(c|x) || p(c)) + Σ_c q(c|x) · KL(N(μ_z, σ²_z) || N(μ_c, σ²_c))

    q(c|x) computed via closed-form Bayes rule (NOT a learned head):
      q(c|x) = p(c|z) = π_c · N(μ_z|μ_c, σ²_c) / Σ_j π_j · N(μ_z|μ_j, σ²_j)
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
        """VaDE KL with q(c|x) via Bayes rule (SCFM Sec 3.2)."""
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
        """Full GMM health report.

        Args:
            mu_z: [N, d_z] encoder means from a representative sample.

        Returns dict with:
            pi_entropy, pi_max, pi_min (component weight health),
            var_min, var_max, var_mean (component variance health),
            mean_norm, mean_pair_dist (component separation),
            active_ratio (fraction of components used),
            assignment_perplexity (confidence of assignments).
        """
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


class SCFM(nn.Module):
    """SCFM: Structured Coupling for Flow Matching (Sumba et al., 2026).

    Paper reference:
      - Sec 3.1: Encoder-induced coupling Γ_enc, source x₀ = (z, ε)
      - Sec 3.2: Shared UNet predicts μ_ϕ(x_t, t), velocity derived via Eq 11
                Endpoint VAE encoder (Eq 14), exogenous regularization R_ε (Eq 15)
                Total loss L_SCFM = L_VFM + β·L_KL + L_rec + R_ε (Eq 16)
      - Sec 3.3: Sampling via full ODE or decoder-initialized refinement

    The UNet ALWAYS predicts μ_ϕ(x_t, t) = E[x₀|x_t] (posterior mean of the source),
    regardless of t. At t=1, the UNet features are pooled and separate heads
    extract (μ_z, logvar_z, μ_ε). At t<1, the velocity is derived from μ_ϕ.
    """
    def __init__(
        self,
        in_channels: int = 1,
        sample_size: int = 224,
        block_out_channels: tuple = (64, 128, 256, 512),
        layers_per_block: int = 2,
        latent_dim: int = 64,
        use_gmm: bool = False,
        gmm_components: int = 30,
        unsupervised_gmm: bool = False,
        exogenous_dim: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.exogenous_dim = exogenous_dim
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

        # Shared UNet: predicts μ_ϕ(x_t, t) = E[x₀|x_t]  (Eq 11, Proposition 3.1)
        # Same architecture as standard FM UNet, but output is posterior mean
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=in_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            act_fn="silu",
            norm_num_groups=32,
            dropout=0.1,
        )
        self.unet.enable_gradient_checkpointing()

        # Endpoint encoder heads (t=1 only): pooled features → (μ_z, logvar_z, μ_ε)
        # Paper Sec 3.2, Eq 14: q_ϕ(z|x₁) = N(μ^z_ϕ(x₁), diag(σ²_ϕ(x₁)))
        # Separate variance head branches from the mean network
        mid_dim = block_out_channels[-1]
        self.mu_z_head = nn.Linear(mid_dim, latent_dim)
        self.logvar_z_head = nn.Linear(mid_dim, latent_dim)
        self.mu_eps_head = nn.Linear(mid_dim, exogenous_dim)  # μ^ε_ϕ(x₁) for R_ε

        # Decoder p_θ(x₁|z): z → pixel-space reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.SiLU(),
            nn.Linear(512, in_channels * sample_size * sample_size),
        )

        # GMM prior p_ψ(z) (VaDE-style, Sec 3.2)
        if use_gmm:
            self.gmm = GaussianMixture(gmm_components, latent_dim,
                                       unsupervised=unsupervised_gmm)

        # Hook to capture mid-block features for t=1 encoding
        self._mid_feat = None
        self._mid_handle = self.unet.mid_block.register_forward_hook(self._mid_hook)

    def _mid_hook(self, module, input, output):
        self._mid_feat = output[0] if isinstance(output, tuple) else output

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict μ_ϕ(x_t, t) = E[x₀|x_t] (Eq 11, Proposition 3.1).

        Returns the posterior mean of the source x₀ given x_t.
        For t < 1: used to derive velocity.
        For t = 1: features are pooled for VAE encoder.
        """
        return self.unet(x_t, timestep=t, return_dict=True).sample

    def get_velocity(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Derive velocity from posterior mean (Eq 11).

        v_ϕ,t(x_t) = (x_t - μ_ϕ(x_t, t)) / t

        With f(t) = 1-t, ∂_t f = -1, 1-f(t) = t:
          v = (∂_t f)/(1-f(t)) · (μ_ϕ - x_t) = (-1/t)(μ_ϕ - x_t) = (x_t - μ_ϕ)/t
        """
        mu = self.forward(x_t, t)
        t_safe = t.clamp(min=1e-5).view(-1, *([1] * (x_t.ndim - 1)))
        return (x_t - mu) / t_safe

    def encode(self, x_1: torch.Tensor, return_all: bool = False) -> tuple[torch.Tensor, ...]:
        """VAE encoder at t=1 (Sec 3.2, Eq 14).

        q_ϕ(z|x₁) = N(μ^z_ϕ(x₁), diag(σ²_ϕ(x₁)))
        q_ϕ^ε(ε|x₁) = N(μ^ε_ϕ(x₁), I)

        Returns (μ_z, logvar_z) or (μ_z, logvar_z, μ_ε).
        """
        t = torch.full((x_1.shape[0],), 1.0, device=x_1.device)
        _ = self.forward(x_1, t)
        feat = self._mid_feat
        pooled = feat.flatten(2).mean(dim=2)

        mu_z = self.mu_z_head(pooled)
        logvar_z = self.logvar_z_head(pooled)
        mu_eps = self.mu_eps_head(pooled)

        if return_all:
            return mu_z, logvar_z, mu_eps
        return mu_z, logvar_z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decoder p_θ(x₁|z): z → pixel reconstruction."""
        img = self.decoder(z)
        return img.reshape(-1, self.in_channels, self.sample_size, self.sample_size)

    @torch.no_grad()
    def sample_prior_z(self, n: int) -> torch.Tensor:
        """Sample z from GMM prior or N(0,I)."""
        if hasattr(self, 'gmm') and self.gmm is not None:
            return self.gmm.sample(n)
        return torch.randn(n, self.latent_dim, device=next(self.parameters()).device)


def scfm_loss(
    model: SCFM,
    x_1: torch.Tensor,
    beta: float = 0.01,
    recon_weight: float = 0.1,
    use_gmm: bool = False,
    unsupervised_gmm: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """SCFM total loss (Eq 16).

    L_SCFM(θ, ϕ, ψ) = L_VFM(ϕ) + β·L_KL(ϕ, ψ) + L_rec(θ, ϕ) + R_ε(ϕ)

    Where:
      - L_VFM: posterior mean regression (Eq 13): ||μ_ϕ(x_t, t) - sg(x₀)||²
      - L_KL: VaDE-style KL (GMM) or standard VAE KL (N(0,I))
      - L_rec: MSE reconstruction ||decoder(z) - x₁||²
      - R_ε: exogenous regularization (Eq 15): ½·||μ_ε||²
    """
    B = x_1.shape[0]
    device = x_1.device
    comp: dict[str, float] = {}

    # 1. Encode at t=1 (Eq 14): q_ϕ(z|x₁) = N(μ_z, diag(σ²_z))
    mu_z, logvar_z, mu_eps = model.encode(x_1, return_all=True)

    std = torch.exp(0.5 * logvar_z)
    eps_z = torch.randn_like(std)
    z = mu_z + eps_z * std

    # 2. Reconstruction loss: L_rec = ||p_θ(x₁|z) - x₁||²
    x_recon = model.decode(z)
    recon_loss = F.mse_loss(x_recon, x_1)
    comp['recon'] = recon_loss.item()

    # 3. KL divergence: L_KL (Eq 16)
    #    GMM prior (VaDE) or standard N(0,I)
    if use_gmm and hasattr(model, 'gmm'):
        if unsupervised_gmm:
            kl_loss = model.gmm.kl(mu_z, logvar_z)
        else:
            kl_loss = -0.5 * (1 + logvar_z - mu_z.pow(2) - logvar_z.exp()).sum(dim=1).mean()
    else:
        kl_loss = -0.5 * (1 + logvar_z - mu_z.pow(2) - logvar_z.exp()).sum(dim=1).mean()
    comp['kl'] = kl_loss.item()

    # 4. Construct source: x₀ = decoder(z) + ε  (Sec 3.1, encoder-induced coupling)
    #    stop-grad decoder to avoid trivial solution
    x_z = model.decode(z).detach()
    noise = torch.randn_like(x_1)
    x_0 = x_z + noise

    # 5. Flow matching (VFM objective, Eq 13): L_VFM = ||μ_ϕ(x_t, t) - sg(x₀)||²
    t = torch.rand(B, device=device)
    t_b = t.view(-1, *([1] * (x_1.ndim - 1)))

    x_t = (1 - t_b) * x_0 + t_b * x_1
    mu_pred = model(x_t, t)
    flow_loss = F.mse_loss(mu_pred, x_0.detach())
    comp['flow'] = flow_loss.item()

    # 6. Exogenous regularization (Eq 15): R_ε(ϕ) = ½·E[||μ_ε||²]
    r_eps = 0.5 * mu_eps.pow(2).sum(dim=1).mean()
    comp['r_eps'] = r_eps.item()

    total = flow_loss + beta * kl_loss + recon_weight * recon_loss + r_eps

    return total, comp


@torch.no_grad()
def sample_scfm(
    model: SCFM,
    num_samples: int,
    num_steps: int = 100,
    device: str = 'cuda',
) -> torch.Tensor:
    """Sample via full ODE integration (Sec 3.3, standard mode).

    1. z ∼ p_ψ(z) (GMM prior or N(0,I))
    2. ε ∼ N(0, I)
    3. x₀ = decoder(z) + ε
    4. ODE: x = x₀, integrate v_ϕ,t from t=0 to t=1
    """
    model.eval()
    device = next(model.parameters()).device

    z = model.sample_prior_z(num_samples)
    x_z = model.decode(z)
    noise = torch.randn_like(x_z)
    x = x_z + noise

    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((num_samples,), i * dt, device=device)
        v = model.get_velocity(x, t)
        x = x + v * dt

    return x.clamp(-1, 1)


@torch.no_grad()
def sample_scfm_decoder_refinement(
    model: SCFM,
    num_samples: int,
    num_steps: int = 20,
    t0: float = 0.8,
    device: str = 'cuda',
) -> torch.Tensor:
    """Decoder-initialized refinement sampling (Sec 3.3, Algorithm in appendix).

    1. z ∼ p_ψ(z)
    2. x̂₁ = decoder(z)  (decoder proposal)
    3. ε ∼ N(0, I)
    4. x₀ = decoder(z) + ε
    5. x_{t₀} = (1-t₀)·x₀ + t₀·x̂₁
    6. ODE from t=t₀ to t=1
    """
    model.eval()
    device = next(model.parameters()).device

    z = model.sample_prior_z(num_samples)
    x_hat = model.decode(z)
    noise = torch.randn_like(x_hat)
    x_0 = model.decode(z).detach() + noise
    x = (1 - t0) * x_0 + t0 * x_hat

    dt = (1.0 - t0) / num_steps

    for i in range(num_steps):
        t = torch.full((num_samples,), t0 + i * dt, device=device)
        v = model.get_velocity(x, t)
        x = x + v * dt

    return x.clamp(-1, 1)
