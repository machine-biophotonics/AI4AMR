import torch
import torch.nn as nn
import torch.nn.functional as F

from mil_model import MILEncoder, AttentionPooling, SimpleAttentionPooling


class FeatureDecoder(nn.Module):
    def __init__(self, latent_dim: int = 32, feature_dim: int = 1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class PixelDecoder(nn.Module):
    def __init__(self, latent_dim: int = 32, img_size: int = 224):
        super().__init__()
        self.img_size = img_size
        final_size = img_size // 32

        self.fc = nn.Linear(latent_dim, 512 * final_size * final_size)
        self.unflatten = nn.Unflatten(1, (512, final_size, final_size))

        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = self.unflatten(h)
        h = F.relu(self.bn1(self.deconv1(h)))
        h = F.relu(self.bn2(self.deconv2(h)))
        h = F.relu(self.bn3(self.deconv3(h)))
        h = F.relu(self.bn4(self.deconv4(h)))
        recon = torch.tanh(self.deconv5(h))
        return recon


class MILVAE(nn.Module):
    def __init__(
        self,
        num_classes: int,
        latent_dim: int = 32,
        beta: float = 0.1,
        num_heads: int = 4,
        dropout: float = 0.5,
        use_contrastive: bool = True,
        num_channels: int = 1,
        pretrained: str = "imagenet",
        backbone: str = "efficientnet_b0",
        pooling: str = "attention",
        img_size: int = 224,
        feature_decoder: bool = True,
        pixel_decoder: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.img_size = img_size
        self.feature_dim = 1280

        self.encoder = MILEncoder(
            num_classes=num_classes,
            num_heads=num_heads,
            attention_temp=0.5,
            dropout=dropout,
            use_contrastive=use_contrastive,
            num_channels=num_channels,
            pretrained=pretrained,
            backbone=backbone,
            pooling=pooling,
        )

        self.vae_mu = nn.Linear(self.feature_dim, latent_dim)
        self.vae_logvar = nn.Linear(self.feature_dim, latent_dim)

        self.feature_decoder = FeatureDecoder(latent_dim, self.feature_dim) if feature_decoder else None
        self.pixel_decoder = PixelDecoder(latent_dim, img_size) if pixel_decoder else None

    @property
    def backbone(self):
        return self.encoder.backbone

    @property
    def attention_pool(self):
        return self.encoder.attention_pool

    @property
    def head_proj(self):
        return self.encoder.head_proj

    @property
    def classifier(self):
        return self.encoder.classifier

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_bag(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.get_mil_embeddings(x)

    def encode_to_latent(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bag = self.encode_bag(x)
        mu = self.vae_mu(bag)
        logvar = self.vae_logvar(bag)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar, bag

    def forward(self, x, return_attention=False, return_pooled=False):
        batch_size, num_crops = x.shape[:2]
        x_flat = x.view(batch_size * num_crops, *x.shape[2:])
        feats = self.encoder.backbone(x_flat)
        crop_embeddings = feats.view(batch_size, num_crops, -1)

        pooled, attn_weights = self.encoder.attention_pool(
            crop_embeddings, temperature=self.encoder.attention_temp
        )
        pooled = pooled.reshape(batch_size, -1)
        bag = self.encoder.head_proj(pooled)

        mu = self.vae_mu(bag)
        logvar = self.vae_logvar(bag)
        z = self.reparameterize(mu, logvar)

        logits = self.encoder.classifier(bag)

        results = {
            'logits': logits,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'bag': bag,
            'attn_weights': attn_weights,
            'crop_embeddings': crop_embeddings,
        }

        if self.feature_decoder is not None:
            results['bag_recon'] = self.feature_decoder(z)

        if self.pixel_decoder is not None:
            results['img_recon'] = self.pixel_decoder(z)

        return results

    def loss_fn(self, results, x, labels, class_weights=None):
        total_loss = 0.0
        losses = {}

        if class_weights is not None:
            ce_loss = F.cross_entropy(results['logits'], labels, weight=class_weights)
        else:
            ce_loss = F.cross_entropy(results['logits'], labels)
        losses['ce'] = ce_loss
        total_loss = total_loss + ce_loss

        kl_loss = -0.5 * torch.mean(1 + results['logvar'] - results['mu'].pow(2) - results['logvar'].exp())
        losses['kl'] = kl_loss * self.beta
        total_loss = total_loss + self.beta * kl_loss

        if 'bag_recon' in results:
            bag_recon_loss = F.mse_loss(results['bag_recon'], results['bag'])
            losses['bag_recon'] = bag_recon_loss
            total_loss = total_loss + bag_recon_loss

        if 'img_recon' in results:
            img_recon_loss = F.mse_loss(results['img_recon'], x[:, x.shape[1] // 2])
            losses['img_recon'] = img_recon_loss
            total_loss = total_loss + img_recon_loss

        losses['total'] = total_loss
        return total_loss, losses

    def compute_vae_loss(self, results, x):
        total_loss = 0.0
        losses = {}

        kl_loss = -0.5 * torch.mean(1 + results['logvar'] - results['mu'].pow(2) - results['logvar'].exp())
        losses['kl'] = kl_loss * self.beta
        total_loss = total_loss + self.beta * kl_loss

        if 'bag_recon' in results:
            bag_recon_loss = F.mse_loss(results['bag_recon'], results['bag'])
            losses['bag_recon'] = bag_recon_loss
            total_loss = total_loss + bag_recon_loss

        if 'img_recon' in results:
            center_idx = x.shape[1] // 2
            img_recon_loss = F.mse_loss(results['img_recon'], x[:, center_idx])
            losses['img_recon'] = img_recon_loss
            total_loss = total_loss + img_recon_loss

        losses['total'] = total_loss
        return total_loss, losses

    def encode_deterministic(self, x):
        bag = self.encode_bag(x)
        return self.vae_mu(bag)

    def decode_bag(self, z):
        if self.feature_decoder is not None:
            return self.feature_decoder(z)
        return None

    def decode_img(self, z):
        if self.pixel_decoder is not None:
            return self.pixel_decoder(z)
        return None
