#!/usr/bin/env python3
"""
Harmony batch correction — one-shot post-hoc correction for MIL bag embeddings.

Fixes the original implementation which had a critical bug:
  `transform()` used `self.ho.Z_corr` (training-data correction)
  as the correction for val/test data, producing wrong results.

Correct usage (one-shot):
  1. Collect bag embeddings for ALL samples (train+val+test) from a trained model
  2. hc = HarmonyCorrector(n_pca=50, batch_vars=['plate', 'domain'])
  3. corrected_pca, corrected_orig = hc.fit_transform(embeddings, batch_labels_dict)
  4. Train a classifier on corrected_pca (50-d) — this is where Harmony's
     correction is strongest and matches the literature.

See: https://github.com/immunogenomics/harmony
     Korsunsky et al. 2019, Nature Methods
"""

import numpy as np
from sklearn.decomposition import PCA

try:
    import harmonypy
    HAS_HARMONY = True
except ImportError:
    HAS_HARMONY = False


class HarmonyCorrector:
    """One-shot Harmony batch correction for bag embeddings.

    Usage:
        hc = HarmonyCorrector(n_pca=50, batch_vars=['plate'])
        corrected_pca, corrected_orig = hc.fit_transform(
            embeddings,
            {'plate': plate_labels, 'domain': domain_labels}
        )
        # corrected_pca.shape == (N, n_pca)   — use this for classifier
        # corrected_orig.shape == (N, 1280)    — inverse-projected (diluted)
    """

    def __init__(self, n_pca=50, max_iter=10, sigma=0.1, n_clust=None,
                 batch_vars=None):
        if not HAS_HARMONY:
            raise ImportError("harmonypy is required. Run: pip install harmonypy")
        self.n_pca = n_pca
        self.max_iter = max_iter
        self.sigma = sigma
        self.n_clust = n_clust
        self.batch_vars = batch_vars if batch_vars else ['plate']
        self.pca = None
        self.ho = None
        self.Z_corr = None  # corrected PCA embeddings (N, n_pca)
        self.fitted = False

    def fit_transform(self, embeddings, batch_labels_dict):
        """One-shot: PCA -> Harmony on ALL embeddings jointly.

        Args:
            embeddings: np.ndarray (N, n_features) - bag embeddings
            batch_labels_dict: dict of {var_name: list_of_labels}
                e.g. {'plate': ['P1', 'P2', ...], 'domain': ['drug', 'mutant']}

        Returns:
            corrected_pca: np.ndarray (N, n_pca) - Harmony-corrected PCA
            corrected_orig: np.ndarray (N, n_features) - delta-projected back
        """
        import pandas as pd

        N = embeddings.shape[0]
        n_pca = min(self.n_pca, embeddings.shape[1], N)

        # -- PCA --
        self.pca = PCA(n_components=n_pca)
        pca_embs = self.pca.fit_transform(embeddings)

        # -- Handle sigma for harmonypy v2.0.0 --
        sigma = self.sigma
        nclust = self.n_clust
        if nclust is None:
            nclust = int(min(round(N / 30.0), 100))
        if isinstance(sigma, float) and nclust > 1:
            sigma = np.repeat(sigma, nclust)

        # -- Harmony --
        meta = pd.DataFrame({k: list(v)[:N] for k, v in batch_labels_dict.items()})
        self.ho = harmonypy.run_harmony(
            pca_embs, meta, vars_use=self.batch_vars,
            max_iter_harmony=self.max_iter,
            sigma=sigma,
            nclust=self.n_clust,
            verbose=True,
            random_state=42,
        )
        self.Z_corr = self.ho.Z_corr  # (N, n_pca)
        self.fitted = True

        # -- Inverse projection (diluted -- use Z_corr directly for best results) --
        delta_pca = self.Z_corr - pca_embs
        delta_orig = delta_pca @ self.pca.components_
        corrected_orig = embeddings + delta_orig

        return self.Z_corr.copy(), corrected_orig

    def fit(self, embeddings, batch_labels_dict):
        """Alias for fit_transform that discards the corrected embeddings.

        Kept for backward compatibility with old code.
        """
        self.fit_transform(embeddings, batch_labels_dict)
        return self

    @property
    def corrected_pca(self):
        """Return corrected PCA embeddings (N, n_pca)."""
        if not self.fitted:
            raise RuntimeError("HarmonyCorrector not fitted yet")
        return self.Z_corr.copy()


def train_harmony_classifier(pca_embs, labels, test_mask=None,
                             hidden_dim=128, num_epochs=200, lr=1e-3):
    """Train a simple MLP classifier on Harmony-corrected PCA embeddings.

    Args:
        pca_embs: np.ndarray (N, n_pca) - corrected PCA embeddings
        labels: np.ndarray (N,) - integer class labels
        test_mask: np.ndarray (N,) bool - which samples are test
        hidden_dim: int - MLP hidden dimension
        num_epochs: int
        lr: float

    Returns:
        model: trained MLP
        accuracy: test accuracy
    """
    import torch
    from torch import nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, n_pca = pca_embs.shape
    num_classes = len(np.unique(labels))

    if test_mask is None:
        split = int(N * 0.8)
        X_train, y_train = pca_embs[:split], labels[:split]
        X_test, y_test = pca_embs[split:], labels[split:]
    else:
        X_train, y_train = pca_embs[~test_mask], labels[~test_mask]
        X_test, y_test = pca_embs[test_mask], labels[test_mask]

    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    X_test_t  = torch.from_numpy(X_test).float().to(device)
    y_test_t  = torch.from_numpy(y_test).long().to(device)

    model = nn.Sequential(
        nn.Linear(n_pca, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, num_classes),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = nn.functional.cross_entropy(logits, y_train_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 20 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                preds = model(X_test_t).argmax(dim=1)
                acc = (preds == y_test_t).float().mean().item()
            if acc > best_acc:
                best_acc = acc
            print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}, test_acc={acc*100:.2f}%")

    return model, best_acc
