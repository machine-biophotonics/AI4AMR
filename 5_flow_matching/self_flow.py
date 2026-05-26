import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel


class SelfFlowUNet(nn.Module):
    """Self-Flow UNet with per-pixel timestep conditioning.

    Architecture:
        - UNet2DModel with in_channels=2, out_channels=1
        - Channel 0: noised image x_t
        - Channel 1: per-pixel timestep map t_map ∈ [0,1]
        - Standard scalar timestep for time embedding (t*1000)
        - Mid-block hook for feature extraction (feature alignment)

    Dual-timestep scheduling (training only):
        - 25% of spatial positions at high noise (t_high = sampled t)
        - 75% of spatial positions at low noise (t_low = t * low_noise_scale)
        - EMA teacher sees fully low-noise input
        - Student aligns features to teacher via cosine similarity

    Inference:
        - Standard forward with uniform t_map (same t for all pixels)
    """

    def __init__(
        self,
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

        combined_embeds = num_class_embeds + 1

        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=2,
            out_channels=1,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            num_class_embeds=combined_embeds,
            class_embed_type=None,
            act_fn="silu",
            norm_num_groups=32,
            dropout=0.1,
        )
        self.unet.enable_gradient_checkpointing()

        self._mid_feat = None
        self._mid_handle = self.unet.mid_block.register_forward_hook(self._mid_hook)

    def _mid_hook(self, module, input, output):
        self._mid_feat = output[0] if isinstance(output, tuple) else output

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = x_t.shape
        device = x_t.device

        t_map = t.view(B, 1, 1, 1).expand(-1, 1, H, W)
        x_input = torch.cat([x_t, t_map], dim=1)

        t_scaled = t * 1000.0
        self._mid_feat = None

        safe_labels = class_labels.clamp(min=0, max=self.unet.config.num_class_embeds - 1) if class_labels is not None else None
        output = self.unet(x_input, timestep=t_scaled, class_labels=safe_labels, return_dict=True)

        if return_features:
            feat = self._mid_feat
            if feat is None:
                feat = x_t
            return output.sample, feat

        return output.sample

    @torch.no_grad()
    def forward_teacher(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher forward: extracts mid-block features without gradients.

        Unlike forward(return_features=True), this does NOT set self._mid_feat
        on the student model — it uses a temporary hook to avoid state conflict
        between student and teacher on the same instance.

        For separate teacher instances (EMA copy), use forward(return_features=True)
        on the teacher directly.
        """
        B, _, H, W = x_t.shape
        t_map = t.view(B, 1, 1, 1).expand(-1, 1, H, W)
        x_input = torch.cat([x_t, t_map], dim=1)
        t_scaled = t * 1000.0
        safe_labels = class_labels.clamp(min=0, max=self.unet.config.num_class_embeds - 1) if class_labels is not None else None

        feat_container = []

        def hook_fn(module, input, output):
            f = output[0] if isinstance(output, tuple) else output
            feat_container.append(f)

        handle = self.unet.mid_block.register_forward_hook(hook_fn)
        output = self.unet(x_input, timestep=t_scaled, class_labels=safe_labels, return_dict=True)
        handle.remove()

        feat = feat_container[0] if feat_container else x_t
        return output.sample, feat

    def forward_with_cfg(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        cfg_scale: float = 0.0,
        null_label: int = -1,
        plate_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cfg_scale <= 0.0 or class_labels is None:
            return self.forward(x_t, t, class_labels=class_labels)

        v_cond = self.forward(x_t, t, class_labels=class_labels)
        null_labels = torch.full_like(class_labels, null_label)
        v_uncond = self.forward(x_t, t, class_labels=null_labels)

        return (1.0 + cfg_scale) * v_cond - cfg_scale * v_uncond


def _lognormal_timesteps(B: int, device: torch.device, mean: float = -1.0, std: float = 1.0):
    t = torch.exp(torch.randn(B, device=device) * std + mean)
    return t.clamp(0.0, 1.0)


def _dual_timestep_map(
    t: torch.Tensor,
    B: int, H: int, W: int, device: torch.device,
    mask_ratio: float = 0.25,
    low_noise_scale: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create dual-timestep per-pixel map.

    Returns:
        t_map: (B, 1, H, W) per-pixel timestep values
        mask: (B, 1, H, W) boolean, True = high-noise pixel
    """
    mask = torch.rand(B, 1, H, W, device=device) < mask_ratio
    t_high = t.view(B, 1, 1, 1)
    t_low = t_high * low_noise_scale
    t_map = torch.where(mask, t_high, t_low)
    return t_map, mask


@torch.no_grad()
def _teacher_forward(
    teacher: SelfFlowUNet,
    x_t_low: torch.Tensor,
    t_low: torch.Tensor,
    class_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Get teacher mid-block features.

    Uses a temporary hook to avoid interfering with the student's
    _mid_feat attribute when teacher == student (same instance).
    """
    B, _, H, W = x_t_low.shape
    device = x_t_low.device

    t_map = t_low.view(B, 1, 1, 1).expand(-1, 1, H, W)
    x_input = torch.cat([x_t_low, t_map], dim=1)
    t_scaled = t_low * 1000.0
    safe_labels = class_labels.clamp(min=0, max=teacher.unet.config.num_class_embeds - 1) if class_labels is not None else None

    feat_container = []

    def hook_fn(module, input, output):
        f = output[0] if isinstance(output, tuple) else output
        feat_container.append(f)

    handle = teacher.unet.mid_block.register_forward_hook(hook_fn)
    teacher.unet(x_input, timestep=t_scaled, class_labels=safe_labels, return_dict=True)
    handle.remove()

    return feat_container[0] if feat_container else x_t_low


@torch.no_grad()
def _student_forward_with_feats(
    student: SelfFlowUNet,
    x_t: torch.Tensor,
    t: torch.Tensor,
    t_map: torch.Tensor,
    class_labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get student velocity and mid-block features.

    Uses a temporary hook to avoid state issues with _mid_feat.

    Args:
        student: SelfFlowUNet model
        x_t: (B, 1, H, W) noised image
        t: (B,) scalar timesteps for time embedding
        t_map: (B, 1, H, W) per-pixel timestep map (noise level at each position)
        class_labels: (B,) optional class labels
    """
    B, _, H, W = x_t.shape
    x_input = torch.cat([x_t, t_map], dim=1)
    t_scaled = t * 1000.0
    safe_labels = class_labels.clamp(min=0, max=student.unet.config.num_class_embeds - 1) if class_labels is not None else None

    feat_container = []

    def hook_fn(module, input, output):
        f = output[0] if isinstance(output, tuple) else output
        feat_container.append(f)

    handle = student.unet.mid_block.register_forward_hook(hook_fn)
    output = student.unet(x_input, timestep=t_scaled, class_labels=safe_labels, return_dict=True)
    handle.remove()

    v_pred = output.sample
    feat = feat_container[0] if feat_container else x_t
    return v_pred, feat


def compute_self_flow_loss(
    model: SelfFlowUNet,
    x_1: torch.Tensor,
    class_labels: torch.Tensor | None = None,
    ema_model: torch.nn.Module | None = None,
    mask_ratio: float = 0.25,
    low_noise_scale: float = 0.1,
    rep_weight: float = 1.0,
    label_dropout_prob: float = 0.0,
    null_label: int = -1,
    lognormal_sampling: bool = False,
    uniform_map: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Self-Flow loss: flow matching + feature alignment via dual-timestep scheduling.

    L = L_flow + λ * L_rep

    where:
        - L_flow = MSE(v_pred, x_1 - x_0)  (standard conditional FM)
        - L_rep = 1 - cos_sim(student_feat, teacher_feat)  (feature alignment)
        - Student sees 25% high-noise / 75% low-noise pixels
        - Teacher (EMA) sees fully low-noise input

    Args:
        model: SelfFlowUNet student (online model)
        x_1: clean images (B, 1, H, W) in [-1, 1]
        class_labels: (B,) class indices
        ema_model: EMA copy of SelfFlowUNet (teacher), or nn.Module with .model attr
        mask_ratio: fraction of high-noise pixels (default 0.25 = 25%)
        low_noise_scale: multiplier for low-noise timesteps (default 0.1)
        rep_weight: weight λ for feature alignment loss
        label_dropout_prob: probability to drop class label for CFG
        null_label: label index for unconditional prediction
        lognormal_sampling: use lognormal timestep distribution
        uniform_map: if True, use standard scalar t instead of dual-timestep (for val)

    Returns:
        (total_loss, info_dict)
    """
    B, C, H, W = x_1.shape
    device = x_1.device

    if lognormal_sampling:
        t = _lognormal_timesteps(B, device)
    else:
        t = torch.rand(B, device=device)

    x_0 = torch.randn_like(x_1)
    u_pos = x_1 - x_0

    labels_for_model = class_labels
    if label_dropout_prob > 0.0 and class_labels is not None and model.training:
        drop_mask = torch.rand(B, device=device) < label_dropout_prob
        if drop_mask.any():
            labels_for_model = class_labels.clone()
            labels_for_model[drop_mask] = null_label

    t_flat = t.view(B)

    if uniform_map:
        t_map = t.view(B, 1, 1, 1).expand(-1, 1, H, W)
        x_t = (1 - t_map) * x_0 + t_map * x_1
        v_pred, student_feat = _student_forward_with_feats(model, x_t, t_flat, t_map, labels_for_model)
        rep_loss = torch.tensor(0.0, device=device)
    else:
        t_map, _ = _dual_timestep_map(t, B, H, W, device, mask_ratio, low_noise_scale)
        x_t = (1 - t_map) * x_0 + t_map * x_1
        v_pred, student_feat = _student_forward_with_feats(model, x_t, t_flat, t_map, labels_for_model)

        # Teacher forward: low-noise input (cleaner features)
        rep_loss = torch.tensor(0.0, device=device)
        if rep_weight > 0.0 and ema_model is not None and model.training:
            teacher_model = ema_model.model if hasattr(ema_model, 'model') else ema_model
            t_low = t * low_noise_scale
            t_low_flat = t_low.view(B)
            t_map_low = t_low.view(B, 1, 1, 1).expand(-1, 1, H, W)
            x_t_low = (1 - t_map_low) * x_0 + t_map_low * x_1

            teacher_feat = _teacher_forward(teacher_model, x_t_low, t_low_flat, labels_for_model)

            sf = student_feat.flatten(2).mean(dim=2)
            tf = teacher_feat.flatten(2).mean(dim=2)

            sf_norm = F.normalize(sf, dim=1)
            tf_norm = F.normalize(tf, dim=1)
            cos_sim = (sf_norm * tf_norm).sum(dim=1)
            rep_loss = (1.0 - cos_sim).mean()

    flow_loss = F.mse_loss(v_pred, u_pos)
    total_loss = flow_loss + rep_weight * rep_loss

    info = {
        'flow_loss': flow_loss.item(),
        'rep_loss': rep_loss.item(),
        'loss': total_loss.item(),
    }

    return total_loss, info
