import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel


class AuxProjectionHead(nn.Module):
    """185-way linear classifier on pooled bottleneck features.

    Pooled bottleneck (feat_dim) → LayerNorm → Linear(feat_dim, num_classes)
    Trained with CrossEntropy alongside the flow matching objective.
    Removed after training — the backbone retains the structured features.
    """
    def __init__(self, feat_dim: int = 256, num_classes: int = 185):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(x))


class RepaProjector(nn.Module):
    """Projects UNet bottleneck features to DINOv2 representation space.

    Takes (B, C_mid, H, W) conv features and projects to (B, N, D_dinov2)
    patch-level features using a 1x1 conv + reshape.
    """
    def __init__(self, in_channels: int, out_dim: int = 384):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_dim, kernel_size=1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.conv(x)            # (B, D, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D) where N = H*W
        x = self.norm(x)
        return x


class FlowUNet(nn.Module):
    """UNet2DModel wrapper for Conditional Flow Matching.

    Supports:
        - ΔFM contrastive regularization
        - REPA representation alignment with DINOv2
        - Classifier-free guidance (CFG)
        - Midpoint ODE solver

    Note: timesteps are scaled by 1000 before feeding to UNet2DModel
    since its sinusoidal time embedding is designed for t in [0, 1000].
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

        # +1 for null class embedding (used during CFG label dropout)
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=in_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            num_class_embeds=num_class_embeds + 1,
            class_embed_type=None,
            act_fn="silu",
            norm_num_groups=32,
            dropout=0.1,
        )
        self.unet.enable_gradient_checkpointing()

        self._mid_feat = None
        for name, module in self.unet.named_modules():
            if name == 'up_blocks.0':
                self._mid_handle = module.register_forward_hook(self._mid_hook)
                break

    def _mid_hook(self, module, input, output):
        self._mid_feat = output[0] if isinstance(output, tuple) else output

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        t_scaled = t * 1000.0
        self._mid_feat = None
        safe_labels = class_labels.clamp(min=0, max=self.unet.config.num_class_embeds - 1) if class_labels is not None else None
        output = self.unet(x_t, timestep=t_scaled, class_labels=safe_labels, return_dict=True)

        if return_features:
            feat = self._mid_feat
            if feat is None:
                feat = x_t
            return output.sample, feat

        return output.sample

    def forward_with_cfg(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor | None,
        cfg_scale: float = 0.0,
        null_label: int = -1,
    ) -> torch.Tensor:
        if cfg_scale <= 0.0 or class_labels is None:
            return self.forward(x_t, t, class_labels=class_labels)

        v_cond = self.forward(x_t, t, class_labels=class_labels)
        null_labels = torch.full_like(class_labels, self.unet.config.num_class_embeds - 1)
        v_uncond = self.forward(x_t, t, class_labels=null_labels)

        return (1.0 + cfg_scale) * v_cond - cfg_scale * v_uncond


class ContrastiveProjection(nn.Module):
    """Projection head for Supervised Contrastive Loss (CORAL-style).

    Pooled bottleneck (feat_dim) → Linear → LayerNorm → ReLU → Linear(proj_dim)
    Discarded after training — only the backbone retains the structured features.
    """
    def __init__(self, feat_dim: int = 256, proj_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, feat_dim)
        self.norm = nn.LayerNorm(feat_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(feat_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al., 2020).

    Pulls together samples with the same label, pushes apart different labels.
    All controls share label 0 → they attract.
    Each non-control class gets a unique label → they repel from controls and each other.

    Args:
        features: (B, D) — L2-normalized on the fly
        labels: (B,) integer labels (same = positive pair)
        temperature: temperature scaling
    Returns:
        scalar loss
    """
    B = features.shape[0]
    features = F.normalize(features, dim=1)
    sim = features @ features.T / temperature

    label_mat = labels.unsqueeze(0)
    pos_mask = (label_mat == label_mat.T).float()
    pos_mask = pos_mask - torch.eye(B, device=features.device)

    exp_sim = torch.exp(sim)
    pos_sum = (exp_sim * pos_mask).sum(dim=1)
    all_sum = exp_sim.sum(dim=1) - exp_sim.diag()

    log_all = torch.log(all_sum + 1e-8)
    log_pos = torch.log(pos_sum + 1e-8)
    loss = log_all - log_pos

    valid = pos_sum > 1e-8
    if valid.any():
        loss = loss[valid].mean()
    else:
        loss = torch.tensor(0.0, device=features.device)

    return loss


def _lognormal_timesteps(B: int, device: torch.device, mean: float = -1.0, std: float = 1.0):
    """Sample timesteps from log-normal distribution (REPA-style).

    Samples t ~ LogNormal(mean, std), clipped to [0, 1].
    Puts more samples at low-noise (t near 1) for fine-detail learning.
    """
    t = torch.exp(torch.randn(B, device=device) * std + mean)
    return t.clamp(0.0, 1.0)


@torch.no_grad()
def _get_dinov2_features(
    dinov2_model: nn.Module,
    x_1: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Extract DINOv2 patch features for REPA alignment.

    Uses forward_features() to get dict with 'x_norm_patchtokens' (B, N, D).
    Falls back to forward() for models without forward_features().
    """
    x_rgb = x_1.repeat(1, 3, 1, 1)
    x_01 = (x_rgb + 1) / 2
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x_norm = (x_01 - mean) / std

    with torch.amp.autocast('cuda', enabled=True):
        if hasattr(dinov2_model, 'forward_features'):
            raw = dinov2_model.forward_features(x_norm)
            if isinstance(raw, dict):
                for key in ('x_norm_patchtokens', 'patch_tokens', 'x_norm'):
                    if key in raw:
                        return raw[key]
                return list(raw.values())[0]
        raw = dinov2_model(x_norm)

    if isinstance(raw, tuple):
        feats = raw[1] if raw[1].dim() == 3 else raw[0]
    elif isinstance(raw, dict):
        for key in ('x_norm_patchtokens', 'patch_tokens', 'x_norm'):
            if key in raw:
                feats = raw[key]
                break
        else:
            feats = list(raw.values())[0]
    elif raw.dim() == 3:
        feats = raw[:, 1:, :] if raw.shape[1] > 100 else raw
    elif raw.dim() == 2:
        feats = raw.unsqueeze(1)
    else:
        feats = raw

    return feats


def compute_flow_loss(
    model: FlowUNet,
    x_1: torch.Tensor,
    class_labels: torch.Tensor | None = None,
    delta_fm_weight: float = 0.0,
    repa_weight: float = 0.0,
    repa_projector: nn.Module | None = None,
    dinov2_model: nn.Module | None = None,
    label_dropout_prob: float = 0.0,
    null_label: int = -1,
    lognormal_sampling: bool = False,
    aux_head: nn.Module | None = None,
    aux_weight: float = 0.0,
    num_classes: int = 185,
    contrastive_weight: float = 0.0,
    contrastive_projector: nn.Module | None = None,
    contrastive_temperature: float = 0.1,
    control_indices: set | None = None,
) -> tuple[torch.Tensor, dict]:
    """Compute Conditional Flow Matching loss with ΔFM + REPA + lognormal sampling + aux CE + contrastive.

    Loss = MSE(v_pred, v_pos) - λ * MSE(v_pred, v_neg) + α * L_REPA + γ * L_CE + κ * L_contrastive

    Args:
        model: FlowUNet
        x_1: clean images (B, C, H, W) in [-1, 1]
        class_labels: (B,) class indices
        delta_fm_weight: λ for ΔFM repulsive term (0 = disabled)
        repa_weight: α for REPA alignment loss (0 = disabled)
        repa_projector: projects model features to DINOv2 dim
        dinov2_model: frozen DINOv2 encoder
        label_dropout_prob: probability to drop class label for CFG
        null_label: label index for unconditional prediction
        lognormal_sampling: use lognormal timestep distribution (vs uniform)
        aux_head: optional 185-way linear classifier on bottleneck features
        aux_weight: weight γ for auxiliary CE loss
        num_classes: total number of classes (for class-balanced weighting)
        contrastive_weight: κ for Supervised Contrastive loss (0 = disabled)
        contrastive_projector: MLP that projects pooled features to contrastive space
        contrastive_temperature: temperature for SupCon loss
        control_indices: set of class ids that are controls (attract together)
    """
    B = x_1.shape[0]
    device = x_1.device

    if lognormal_sampling:
        t = _lognormal_timesteps(B, device)
    else:
        t = torch.rand(B, device=device)
    t = t.view(B, *([1] * (x_1.ndim - 1)))

    x_0 = torch.randn_like(x_1)
    x_t = (1 - t) * x_0 + t * x_1
    u_pos = x_1 - x_0
    t_flat = t.view(B)

    labels_for_model = class_labels
    if label_dropout_prob > 0.0 and class_labels is not None and model.training:
        drop_mask = torch.rand(B, device=device) < label_dropout_prob
        if drop_mask.any():
            labels_for_model = class_labels.clone()
            labels_for_model[drop_mask] = null_label

    need_features = (
        (repa_weight > 0.0 and repa_projector is not None and dinov2_model is not None)
        or (aux_weight > 0.0 and aux_head is not None)
        or (contrastive_weight > 0.0 and contrastive_projector is not None and control_indices is not None)
    )
    if need_features:
        v_pred, mid_feat = model(x_t, t_flat, class_labels=labels_for_model, return_features=True)
    else:
        v_pred = model(x_t, t_flat, class_labels=labels_for_model)

    flow_loss = F.mse_loss(v_pred, u_pos, reduction='none').mean(dim=(1, 2, 3))

    mask_low = t_flat < 0.3
    mask_mid = (t_flat >= 0.3) & (t_flat <= 0.7)
    mask_high = t_flat > 0.7
    info = {'flow_loss': flow_loss.mean().item()}
    for mask, name in [(mask_low, 'low'), (mask_mid, 'mid'), (mask_high, 'high')]:
        if mask.any():
            info[f'flow_loss_t_{name}'] = flow_loss[mask].mean().item()

    total_loss = flow_loss.mean()

    if delta_fm_weight > 0.0 and B > 1:
        neg_idx = torch.randperm(B, device=device)
        same_mask = neg_idx == torch.arange(B, device=device)
        if same_mask.any():
            neg_idx[same_mask] = (neg_idx[same_mask] + 1) % B
        u_neg = x_1[neg_idx] - x_0[neg_idx]
        repulsive_loss = F.mse_loss(v_pred, u_neg)
        total_loss = total_loss - delta_fm_weight * repulsive_loss
        info['delta_fm_repulsive'] = repulsive_loss.item()

    repa_avail = repa_weight > 0.0 and repa_projector is not None and dinov2_model is not None

    if repa_avail:
        with torch.no_grad():
            dinov2_feats = _get_dinov2_features(dinov2_model, x_1, device)

        proj_feats = repa_projector(mid_feat)

        if proj_feats.shape[1] != dinov2_feats.shape[1]:
            if dinov2_feats.shape[1] == 1:
                proj_feats = proj_feats.mean(dim=1, keepdim=True)
            else:
                n_proj = proj_feats.shape[1]
                n_dino = dinov2_feats.shape[1]
                D = proj_feats.shape[-1]
                h_proj = int(round(n_proj ** 0.5))
                w_proj = h_proj
                h_dino = int(round(n_dino ** 0.5))
                w_dino = h_dino
                B_actual = proj_feats.shape[0]
                proj_2d = proj_feats.transpose(1, 2).reshape(B_actual, D, h_proj, w_proj)
                proj_2d = F.interpolate(proj_2d, size=(h_dino, w_dino),
                                        mode='bilinear', align_corners=False)
                proj_feats = proj_2d.flatten(2).transpose(1, 2)

        repa_loss = (1 - F.cosine_similarity(proj_feats, dinov2_feats, dim=-1)).mean()
        total_loss = total_loss + repa_weight * repa_loss
        info['repa_loss'] = repa_loss.item()

    if aux_weight > 0.0 and aux_head is not None and need_features and class_labels is not None:
        pooled = mid_feat.flatten(2).mean(dim=2)
        aux_logits = aux_head(pooled)  # (B, num_classes)
        aux_loss = F.cross_entropy(aux_logits, class_labels)
        total_loss = total_loss + aux_weight * aux_loss
        info['aux_loss'] = aux_loss.item()
        with torch.no_grad():
            preds = aux_logits.argmax(dim=1)
            acc = (preds == class_labels).float().mean().item()
            info['aux_acc'] = acc

    if contrastive_weight > 0.0 and contrastive_projector is not None and control_indices is not None and class_labels is not None:
        pooled = mid_feat.flatten(2).mean(dim=2)
        proj = contrastive_projector(pooled)
        ctrl_set = torch.tensor(list(control_indices), device=device)
        is_control = torch.isin(class_labels, ctrl_set)
        contrastive_labels = torch.where(is_control, torch.zeros_like(class_labels), class_labels + 1)
        c_loss = supervised_contrastive_loss(proj, contrastive_labels, temperature=contrastive_temperature)
        total_loss = total_loss + contrastive_weight * c_loss
        info['contrastive_loss'] = c_loss.item()

    info['loss'] = total_loss.item()
    return total_loss, info


@torch.no_grad()
def sample(
    model: FlowUNet,
    num_samples: int,
    num_steps: int = 100,
    class_labels: torch.Tensor | None = None,
    cfg_scale: float = 0.0,
    null_label: int = -1,
    solver: str = 'euler',
    cfg_zero_steps: int = 3,
) -> torch.Tensor:
    model.eval()
    if hasattr(model, 'unet'):
        C = model.unet.config.in_channels
        H = W = model.unet.config.sample_size
    else:
        C = model.in_channels
        H = W = model.img_size
    device = next(model.parameters()).device

    x = torch.randn(num_samples, C, H, W, device=device)
    if class_labels is not None:
        class_labels = class_labels.to(device)
    dt = 1.0 / num_steps

    if solver == 'midpoint':
        for i in range(num_steps):
            t = torch.full((num_samples,), i * dt, device=device)
            v1 = model.forward_with_cfg(x, t, class_labels, cfg_scale, null_label)
            if cfg_scale > 0.0 and i < cfg_zero_steps:
                v1 = torch.zeros_like(v1)
            t_half = torch.full((num_samples,), (i + 0.5) * dt, device=device)
            x_mid = x + v1 * (dt * 0.5)
            v2 = model.forward_with_cfg(x_mid, t_half, class_labels, cfg_scale, null_label)
            if cfg_scale > 0.0 and i < cfg_zero_steps:
                v2 = torch.zeros_like(v2)
            x = x + v2 * dt
    else:
        for i in range(num_steps):
            t = torch.full((num_samples,), i * dt, device=device)
            v = model.forward_with_cfg(x, t, class_labels, cfg_scale, null_label)
            if cfg_scale > 0.0 and i < cfg_zero_steps:
                v = torch.zeros_like(v)
            x = x + v * dt

    return x.clamp(-1, 1)
