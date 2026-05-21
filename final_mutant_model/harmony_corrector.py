#!/usr/bin/env python3
"""
Harmony batch correction wrapper for MIL bag embeddings.
Wraps the harmonypy library with PCA pre-/post-processing.
"""

import numpy as np
from sklearn.decomposition import PCA

try:
    import harmonypy
    HAS_HARMONY = True
except ImportError:
    HAS_HARMONY = False


class HarmonyCorrector:
    """Fit/transform Harmony on bag embeddings.

    Usage:
        hc = HarmonyCorrector(n_pca=50, batch_vars=['plate', 'domain'])
        hc.fit(embeddings_np, {'plate': ['P1',..], 'domain': ['drug',..]})
        corrected = hc.transform(embeddings_np)
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
        self.fitted = False

    def fit(self, embeddings, batch_labels_dict):
        """Fit Harmony on bag embeddings.

        Args:
            embeddings: np.ndarray (n_samples, n_features)
            batch_labels_dict: dict of {var_name: list_of_labels}
                e.g. {'plate': ['P1', 'P2', ...], 'domain': ['drug', 'mutant', ...]}
        """
        import pandas as pd

        n = embeddings.shape[0]
        n_pca = min(self.n_pca, embeddings.shape[1], n)

        self.pca = PCA(n_components=n_pca)
        pca_embs = self.pca.fit_transform(embeddings)

        meta = pd.DataFrame({k: list(v)[:n] for k, v in batch_labels_dict.items()})

        self.ho = harmonypy.run_harmony(
            pca_embs, meta, vars_use=self.batch_vars,
            max_iter_harmony=self.max_iter,
            sigma=self.sigma,
            nclust=self.n_clust,
            verbose=True,
            random_state=42,
        )
        self.fitted = True
        return self

    def transform(self, embeddings):
        """Apply fitted Harmony to embeddings.

        Corrects batch effects in PCA space then adds the delta back
        to the original features, preserving all original variation.
        """
        assert self.fitted, "HarmonyCorrector not fitted yet"
        pca_embs = self.pca.transform(embeddings)          # (N, n_pca)
        corrected_pca = self.ho.Z_corr                     # (N, n_pca)
        # Delta in PCA space
        delta_pca = corrected_pca - pca_embs
        # Project delta back to original feature space
        delta_orig = delta_pca @ self.pca.components_      # (N, n_features)
        return embeddings + delta_orig

    def fit_transform(self, embeddings, batch_labels_dict):
        self.fit(embeddings, batch_labels_dict)
        return self.transform(embeddings)
