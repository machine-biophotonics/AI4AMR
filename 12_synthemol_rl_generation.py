#!/usr/bin/env python3
"""
Script 12: SyntheMol RL Generation (Paper-Matching)

Architecture matches Swanson et al. 2026:
  - Two independent RLModels (Z1 for activity, Z2 for log solubility)
  - Each: Chemprop MPNN (BondMessagePassing + MeanAggregation + RegressionFFN)
  - Multi-molecule input via disconnected SMILES (".".join(bbs))
  - Per-node trajectory buffer (both {BB1} and {BB1+BB2} nodes)
  - Dynamic temperature: tau0=0.1, gamma=0.98, target lambda=0.6
  - Dynamic weight updates matching paper Eq. 8-9
  - Thresholds: activity >= 0.5, log solubility >= -4.0

Usage:
  python3 12_synthemol_rl_generation.py --source real --n_rollout 10000
  python3 12_synthemol_rl_generation.py --source commercial --n_rollout 5000 --load_checkpoint
"""

import os, sys, json, pickle, random, time, argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
import logging
logging.getLogger('cheminfo_ensemble').setLevel(logging.WARNING)

# Silence RDKit deprecation warnings from library internals (chemprop/MolE)
from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')

# Singleton Morgan generator (avoids deprecation warning per-call)
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def _morgan_fp(mol):
    """Fast Morgan fingerprint using cached generator."""
    return _MORGAN_GEN.GetFingerprint(mol)

import math

def _normal_cdf(x: float) -> float:
    """Standard normal CDF using math.erfc (no scipy dependency)."""
    return 0.5 * math.erfc(-x / math.sqrt(2))

def _geom_mean_exceed(per_species_vals: List[float], threshold: float = 0.5) -> float:
    """Geometric mean of threshold exceedances.

    For each species i: exceedance_i = max(s_i - threshold + eps, eps)
    Returns: exp(mean(log(exceedance_i)))

    This enforces broad-spectrum activity: any species at threshold pulls
    the geometric mean down to ~eps. See C-MORAL (arXiv 2604.23061, 2026),
    MolRGen (arXiv 2603.18256, 2026), Uncertainty-Aware RL (NeurIPS 2025).
    """
    exceedances = [max(s - threshold + GEOM_MEAN_EPS, GEOM_MEAN_EPS) for s in per_species_vals]
    return float(np.exp(np.mean(np.log(exceedances))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit.Chem import Crippen, Lipinski, Descriptors

from chemprop.nn import BondMessagePassing, MeanAggregation
from chemprop.data import BatchMolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer, MoleculeFeaturizerRegistry
from chemprop.models import load_model as cp_load_model

# Global MolGraph cache (shared across models, like official SMILES_TO_MOL_GRAPH)
SMILES_TO_MOL_GRAPH = {}

# ── Constants ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/media/student/Data_SSD_1-TB/Master Thesis/A IMPACT WP 8 Work package 8 – IPP/IPK")
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SYNTHEMOL_DIR = DATA_DIR / "synthemol_rl"
CHECKPOINT_DIR = SYNTHEMOL_DIR / "checkpoints"
CACHE_DIR = SYNTHEMOL_DIR
SOLUBILITY_CACHE_PATH = SYNTHEMOL_DIR / "solubility_cache.npz"

for d in [SYNTHEMOL_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Paper default hyperparameters
TAU0 = 0.1          # Initial temperature
TAU_MIN = 0.01       # Minimum temperature floor (preserves ~40% exploration mass for top candidate)
TAU_MAX = 10.0      # Maximum temperature
GAMMA = 0.98        # Temperature decay per step
TARGET_LAMBDA = 0.6 # Target Tanimoto similarity lambda (paper: 0.6)
W_ACTIVITY_INIT = 0.5   # Paper: w_k = 1/L
W_SOLUBILITY_INIT = 0.5
SOLUBILITY_MODEL_DIR = MODELS_DIR / "chemprop" / "aqsoildb_solubility_tdc_scaffold"
THRESHOLD_ACTIVITY = 0.5
THRESHOLD_SOLUBILITY = -4.0
N_ROLLOUT_DEFAULT = 10000
SCORE_BATCH_SIZE = 4096  # Mini-batch size for _score_bbs (vectorized, large is fine)
BATCH_SIZE = 32          # Training batch size for RL replay
LEARNING_RATE = 1e-4
N_EPOCHS = 5           # Training epochs per rollout batch (paper: 5)
HIDDEN_DIM = 300       # MPNN hidden dimension (paper: 300)
DEPTH = 3              # MPNN message-passing depth (paper: 3)
MOLFEAT_DIM = 768      # MolE molecular feature dimension
Z1_N_TASKS = 1         # Z1 scalar output (broad-spectrum activity)
RDKIT_FEAT_DIM = 200   # RDKit molecular descriptor dimension (matchee chemprop default)

MAX_NODES = 50000      # Max nodes in trajectory buffer
REPLAY_INTERVAL = 10   # Train every N rollouts (paper: 10)
PATIENCE = 10          # Early stopping patience for training
CLIP_GRAD = 5.0        # Gradient clipping

GEOM_MEAN_EPS = 1e-6           # Small constant to avoid zero in geometric mean

# ── Building Block Loading ─────────────────────────────────────────────────────

def load_building_blocks(source: str = "real") -> Dict[int, Dict[str, List[str]]]:
    """Load ALL 13 REAL reaction building block pools from SyntheMol pickle.

    Args:
        source: 'real' for main Enamine data, 'commercial' for Wuxi data.
    Returns:
        Dict mapping reaction_id -> {pos_key: [smiles_list]}
        where pos_key is 'BB1', 'BB2', or 'BB3'.
    """
    if source == "commercial":
        bb_csv = DATA_DIR / "synthemol" / "wuxi" / "building_blocks.csv"
        rxn_pkl = DATA_DIR / "synthemol" / "wuxi" / "reaction_to_building_blocks.pkl"
    else:
        bb_csv = DATA_DIR / "synthemol" / "building_blocks.csv"
        rxn_pkl = DATA_DIR / "synthemol" / "reaction_to_building_blocks.pkl"

    if not rxn_pkl.exists():
        print(f"  No reaction pickle at {rxn_pkl}, using fallback")
        return _fallback_reactions()

    # Load reaction-to-BB mapping
    with open(rxn_pkl, "rb") as f:
        rxn_map = pickle.load(f)
    print(f"  Loaded {len(rxn_map)} reactions from {rxn_pkl.name}")

    # Load all BBs for validation
    all_smiles_set = set()
    if bb_csv.exists():
        df = pd.read_csv(bb_csv)
        all_smiles_set = set(df["smiles"].dropna().tolist())
        print(f"  Loaded {len(all_smiles_set)} unique building blocks from {bb_csv.name}")

    reactions = {}
    pos_map = {0: "BB1", 1: "BB2", 2: "BB3"}

    for rxn_id, bbs_dict in rxn_map.items():
        bb_sets = {}
        for key_idx in sorted(bbs_dict.keys()):
            if key_idx > 2:
                continue
            pos_key = pos_map[key_idx]
            smiles_raw = bbs_dict[key_idx]
            if isinstance(smiles_raw, set):
                smiles_raw = list(smiles_raw)
            smiles_list = [s for s in smiles_raw if s in all_smiles_set] if all_smiles_set else list(smiles_raw)
            bb_sets[pos_key] = smiles_list
            print(f"  Rxn {rxn_id} {pos_key}: {len(smiles_list)} blocks")

        if bb_sets:
            reactions[rxn_id] = bb_sets

    print(f"  Total: {len(reactions)} reaction schemes loaded")
    return reactions



def _fallback_reactions() -> Dict[int, Dict[str, List[str]]]:
    """Fallback with synthetic reactions for testing."""
    print("  Using fallback synthetic reactions")
    valid_smiles = [
        "Cc1ccccc1", "c1ccccc1", "CC(=O)O", "CN", "CCO", "CCN",
        "CC(=O)N", "c1ccncc1", "c1ccc2ccccc2c1", "CC(C)(C)C(=O)O",
        "CNC(=O)c1ccccc1", "O=C(O)c1ccccc1", "Nc1ccccc1",
        "OC(=O)c1ccccc1", "CCOC(=O)c1ccccc1", "CCCC(=O)O",
        "c1csc2c1cccc2", "c1ccsc1", "c1cnc2c(c1)cccc2",
        "CCOC(=O)C=C", "CC#N", "C=CC(=O)O",
    ]
    import random as _r
    _r.seed(42)
    return {
        1: {"BB1": _r.choices(valid_smiles, k=500), "BB2": _r.choices(valid_smiles, k=2000)},
        2: {"BB1": _r.choices(valid_smiles, k=500), "BB2": _r.choices(valid_smiles, k=2000)},
        275592: {"BB1": _r.choices(valid_smiles, k=500), "BB2": _r.choices(valid_smiles, k=2000), "BB3": _r.choices(valid_smiles, k=2000)},
    }



# ── MolE Features ──────────────────────────────────────────────────────────────
# MolE 768-dim molecular embeddings replace the paper's RDKit 200-dim features.
# Pre-computed MolE embeddings for all BBs are loaded from cache during RLGenerator init.
# MolE (Molecular Embeddings) from DeBERTa-v3-base: 768-dim contextualized
# embeddings that capture molecular semantics via contrastive pretraining.

def get_reaction_schemes(reactions: Dict[int, Dict[str, List[str]]]) -> List[Dict]:
    """Generate reaction scheme dicts from loaded reaction data.

    Creates one scheme per reaction, using the actual BB pool structure
    from the SyntheMol pickle. n_components is inferred from position count.
    """
    schemes = []
    for rxn_id, bb_sets in reactions.items():
        n_comp = len(bb_sets)
        bb_keys = list(bb_sets.keys())
        schemes.append({
            "id": rxn_id,
            "name": f"REAL_{rxn_id}",
            "n_components": n_comp,
            "bb_keys": bb_keys,
            "n_blocks": [1] * n_comp,
            "n_blocks_total": n_comp,
            "smarts": "",
        })
    schemes.sort(key=lambda s: s["id"])
    return schemes


# ── Reaction Simulation ────────────────────────────────────────────────────────

def synthesize_from_bbs(bb_list: List[str], scheme: dict) -> Optional[str]:
    """Attempt synthesis from building blocks (simplified, returns joined SMILES)."""
    if any(b is None or not isinstance(b, str) or len(b) < 2 for b in bb_list):
        return None
    return ".".join(bb_list)


# ── Property Prediction (Scorer) ──────────────────────────────────────────────

class MolEScorer:
    """MolE ensemble for activity + Chemprop ensemble for solubility."""

    def __init__(self, cache_dir: Path = CACHE_DIR,
                 solubility_model_dir: Optional[Path] = None):
        self.cache_dir = cache_dir
        self.cache_file = cache_dir / "prediction_cache.json"
        self.cache = self._load_cache()
        self.ensemble = None
        self._load_ensemble()
        self.bb_mole_cache: Dict[str, np.ndarray] = {}
        self._load_bb_mole_cache()
        self.solubility_models: list = []
        self.solubility_featurizer = SimpleMoleculeMolGraphFeaturizer()
        self.solubility_rdkit_featurizer = MoleculeFeaturizerRegistry['v1_rdkit_2d_normalized']()
        self.solubility_model_dir = solubility_model_dir
        self.solubility_cache: Dict[str, float] = {}
        self.solubility_cache_dirty = False
        self._load_solubility_cache()
        if solubility_model_dir is not None:
            self._load_solubility_ensemble()

    def _load_solubility_ensemble(self):
        """Load 10-model Chemprop ensemble for solubility prediction."""
        model_dir = self.solubility_model_dir
        if not model_dir or not model_dir.exists():
            print(f"  WARNING: Solubility model dir not found: {model_dir}, using ESOL fallback")
            self.solubility_models = []
            return
        for i in range(10):
            ckpt = model_dir / f"model_{i}" / "best.pt"
            if ckpt.exists():
                try:
                    m = cp_load_model(ckpt)
                    m.eval()
                    m.to(DEVICE)
                    self.solubility_models.append(m)
                except Exception as e:
                    print(f"  Warning: solubility model {i} failed: {e}")
        print(f"  Loaded {len(self.solubility_models)} solubility models from {model_dir}")

    def _predict_solubility(self, smiles: str) -> float:
        """Predict logS using Chemprop ensemble (10-model mean), with cache."""
        cached = self.solubility_cache.get(smiles)
        if cached is not None:
            return cached
        if not self.solubility_models:
            val = self._esol_predict(smiles)
            self.solubility_cache[smiles] = val
            self.solubility_cache_dirty = True
            return val
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return -10.0
        mg = self.solubility_featurizer(mol)
        bmg = BatchMolGraph([mg])
        bmg.to(DEVICE)
        X_d = torch.tensor(self.solubility_rdkit_featurizer(mol), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        preds = []
        with torch.no_grad():
            for m in self.solubility_models:
                out = m(bmg, None, X_d)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                preds.append(out.item())
        val = float(np.mean(preds)) if preds else -10.0
        self.solubility_cache[smiles] = val
        self.solubility_cache_dirty = True
        return val

    def _predict_solubility_batched(self, smiles_list: List[str]) -> np.ndarray:
        """Batched logS prediction for many SMILES, with per-SMILES cache."""
        if not self.solubility_models:
            vals = np.array([self._esol_predict(s) for s in smiles_list], dtype=np.float32)
            for smi, v in zip(smiles_list, vals):
                self.solubility_cache[smi] = float(v)
                self.solubility_cache_dirty = True
            return vals

        n_total = len(smiles_list)
        result = np.full(n_total, -10.0, dtype=np.float32)
        uncached_idx = []
        uncached_smi = []
        for i, smi in enumerate(smiles_list):
            cached = self.solubility_cache.get(smi)
            if cached is not None:
                result[i] = cached
            else:
                uncached_idx.append(i)
                uncached_smi.append(smi)

        if not uncached_smi:
            return result

        mgs, X_d_list = [], []
        for smi in uncached_smi:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mgs.append(self.solubility_featurizer(mol))
                X_d_list.append(self.solubility_rdkit_featurizer(mol))
        if not mgs:
            return result

        bmg = BatchMolGraph(mgs)
        bmg.to(DEVICE)
        X_d = torch.tensor(np.array(X_d_list), dtype=torch.float32, device=DEVICE)
        all_preds = []
        with torch.no_grad():
            for m in self.solubility_models:
                out = m(bmg, None, X_d)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                all_preds.append(out.cpu().numpy().flatten())
        means = np.mean(all_preds, axis=0)
        for j, idx in enumerate(uncached_idx):
            val = float(means[j]) if j < len(means) else -10.0
            result[idx] = val
            self.solubility_cache[uncached_smi[j]] = val
            self.solubility_cache_dirty = True
        return result

    def _esol_predict(self, smiles: str) -> float:
        """Fallback ESOL prediction (Delaney 2004)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return -10.0
        try:
            logP = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            rb = Lipinski.NumRotatableBonds(mol)
            oh = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[OH]")))
            return 0.16 - 0.63 * logP - 0.0062 * mw + 0.066 * rb - 0.74 * oh
        except:
            return -10.0

    def _load_bb_mole_cache(self):
        """Load cached per-BB MolE embeddings if available."""
        bb_mole_path = SYNTHEMOL_DIR / "bb_mole_cache.npz"
        if bb_mole_path.exists():
            try:
                data = np.load(bb_mole_path, allow_pickle=True)
                self.bb_mole_cache = dict(zip(data["smiles"], data["embeddings"]))
                print(f"  Loaded {len(self.bb_mole_cache)} BB MolE embeddings")
            except Exception as e:
                print(f"  Warning: could not load BB MolE cache: {e}")
                self.bb_mole_cache = {}

    def _load_cache(self) -> Dict:
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError, EOFError):
                print(f"  Warning: corrupted cache file {self.cache_file}, starting fresh")
                self.cache_file.unlink(missing_ok=True)
        return {}

    def _save_cache(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)

    def _load_solubility_cache(self):
        """Load on-disk solubility cache (SMILES -> logS) from npz."""
        p = SOLUBILITY_CACHE_PATH
        if p.exists():
            try:
                data = np.load(p, allow_pickle=True)
                smiles_arr = data["smiles"]
                vals_arr = data["values"]
                self.solubility_cache = dict(zip(smiles_arr, vals_arr))
                print(f"  Loaded {len(self.solubility_cache)} cached solubility predictions")
            except Exception as e:
                print(f"  Warning: could not load solubility cache: {e}")
                self.solubility_cache = {}

    def _save_solubility_cache(self):
        """Save solubility cache to disk."""
        if not self.solubility_cache_dirty:
            return
        p = SOLUBILITY_CACHE_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p,
            smiles=list(self.solubility_cache.keys()),
            values=list(self.solubility_cache.values()))
        print(f"  Saved {len(self.solubility_cache)} solubility predictions to {p.name}")
        self.solubility_cache_dirty = False

    def _load_ensemble(self):
        """Load the 10-member multitask MolE ensemble."""
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "models"))
            from cheminfo_ensemble import MultitaskMolEEnsemble
            self.ensemble = MultitaskMolEEnsemble(
                model_dir=MODELS_DIR / "chemprop" / "multitask_mole",
                use_mole=True,
                n_ensemble=10,
                device=DEVICE,
            )
            print(f"  multitask_mole ensemble loaded (10 models)")
        except Exception as e:
            print(f"  WARNING: Could not load ensemble: {e}")
            self.ensemble = None

    def predict(self, smiles: str) -> Dict[str, float]:
        """Predict activity, per-species scores, and log solubility for a molecule."""
        if smiles in self.cache:
            return self.cache[smiles]

        result = {"activity": 0.0, "log_solubility": -10.0}

        # Log solubility via trained Chemprop ensemble (fallback ESOL)
        result["log_solubility"] = self._predict_solubility(smiles)

        # Activity from ensemble + per-species
        if self.ensemble is not None:
            try:
                logits, valid_mask = self.ensemble.predict_batched_logits([smiles])
                if valid_mask[0]:
                    logits_6 = logits[:, 0, :]  # (n_models, 6_species)
                    probs = 1.0 / (1.0 + np.exp(-logits_6))
                    mu_species = np.nanmean(probs, axis=0)
                    sigma_species = np.nanstd(probs, axis=0) + 1e-6
                    per_species = mu_species
                    # Geometric mean of per-species scores (C-MORAL ICLR 2026, MolRGen ICLR 2026):
                    # Provides higher dynamic range and naturally penalizes species bottlenecks,
                    # acting as a differentiable logical AND operator.
                    result["activity"] = float(np.exp(np.mean(np.log(np.maximum(per_species, GEOM_MEAN_EPS)))))
                    # Uncertainty-aware Z1 (NeurIPS 2025): geometric mean of P(s_i ≥ 0.5 | ensemble)
                    p_exceed = np.array([_normal_cdf(float((m - 0.5) / s))
                                         for m, s in zip(mu_species, sigma_species)])
                    result["z1_uncertainty"] = float(
                        np.exp(np.mean(np.log(np.maximum(p_exceed, GEOM_MEAN_EPS)))))
                    species_names = [
                        "A. baumannii", "E. coli", "H. pylori",
                        "K. pneumoniae", "P. aeruginosa", "S. aureus",
                    ]
                    for j, name in enumerate(species_names):
                        result[f"score_{name}"] = float(per_species[j])
                else:
                    pred = self.ensemble.predict(smiles)
                    result["activity"] = float(np.mean(pred))
            except:
                try:
                    pred = self.ensemble.predict(smiles)
                    result["activity"] = float(np.mean(pred))
                except:
                    pass

        self.cache[smiles] = result
        if len(self.cache) % 100 == 0:
            self._save_cache()
        return result

    def predict_approx(self, smiles: str) -> Dict[str, float]:
        """Fast approximate prediction using cached per-BB MolE embeddings.

        For joined SMILES (e.g. \"BB1.BB2\"), averages the per-BB MolE
        embeddings instead of running the MolE encoder on the joined molecule.
        This avoids the expensive DeBERTa forward pass (~5ms → ~0.01ms).

        The Chemprop models still receive the correct MolGraph (atom/bond
        features from the joined SMILES), so atom-level structure is exact.
        Only the MolE descriptor is approximated.

        Results are NOT cached — they're fast enough to recompute.
        The full `predict` (with real MolE encoder) is used for the
        selected molecule in _build_node and WILL cache its result.
        """
        if not self.bb_mole_cache:
            return self.predict(smiles)

        result = {"activity": 0.0, "log_solubility": -10.0}
        result["log_solubility"] = self._predict_solubility(smiles)

        if self.ensemble is not None:
            try:
                bbs = smiles.split(".")
                bb_embs = [self.bb_mole_cache.get(smi) for smi in bbs]
                bb_embs = [e for e in bb_embs if e is not None]
                if bb_embs:
                    avg_emb = np.mean(bb_embs, axis=0).astype(np.float32)
                    logits, valid_mask = self.ensemble.predict_batched_logits([smiles], mole_embs=[avg_emb])
                else:
                    logits, valid_mask = self.ensemble.predict_batched_logits([smiles])
                if valid_mask[0]:
                    logits_6 = logits[:, 0, :]
                    probs = 1.0 / (1.0 + np.exp(-logits_6))
                    mu_species = np.nanmean(probs, axis=0)
                    sigma_species = np.nanstd(probs, axis=0) + 1e-6
                    per_species = mu_species
                    result["activity"] = float(np.exp(np.mean(np.log(np.maximum(per_species, GEOM_MEAN_EPS)))))
                    p_exceed = np.array([_normal_cdf(float((m - 0.5) / s))
                                         for m, s in zip(mu_species, sigma_species)])
                    result["z1_uncertainty"] = float(
                        np.exp(np.mean(np.log(np.maximum(p_exceed, GEOM_MEAN_EPS)))))
                    species_names = [
                        "A. baumannii", "E. coli", "H. pylori",
                        "K. pneumoniae", "P. aeruginosa", "S. aureus",
                    ]
                    for j, name in enumerate(species_names):
                        result[f"score_{name}"] = float(per_species[j])
            except:
                pass

        return result

    def predict_approx_batched(self, smiles_list: List[str]) -> List[Dict[str, float]]:
        """Fast prediction for many SMILES: ESOL + batched MolE activity (scalar).

        Inline ESOL avoids the 10-model Chemprop GPU pass (~3s).
        Uses predict_batched for scalar mean activity (fast), not per-species.
        Per-species + full Chemprop solubility only in predict() for final molecules.
        """
        n = len(smiles_list)
        results = [{"activity": 0.0, "log_solubility": -10.0} for _ in range(n)]

        valid_smiles = []
        valid_indices = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    logP = Crippen.MolLogP(mol)
                    mw = Descriptors.MolWt(mol)
                    rb = Lipinski.NumRotatableBonds(mol)
                    oh = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[OH]")))
                    results[i]["log_solubility"] = (
                        0.16 - 0.63 * logP - 0.0062 * mw + 0.066 * rb - 0.74 * oh
                    )
                except Exception:
                    pass
                valid_smiles.append(smiles)
                valid_indices.append(i)

        if not valid_smiles or self.ensemble is None or not self.bb_mole_cache:
            return results

        # Build MolE embeddings from cache
        mole_embs = []
        ensemble_smiles = []
        ensemble_indices = []
        for i, smiles in zip(valid_indices, valid_smiles):
            bbs = smiles.split(".")
            bb_embs = [self.bb_mole_cache.get(smi) for smi in bbs]
            bb_embs = [e for e in bb_embs if e is not None]
            if bb_embs:
                avg_emb = np.mean(bb_embs, axis=0).astype(np.float32)
                mole_embs.append(avg_emb)
                ensemble_smiles.append(smiles)
                ensemble_indices.append(i)
            else:
                try:
                    pred = self.ensemble.predict(smiles)
                    results[i]["activity"] = float(np.mean(pred))
                except Exception:
                    pass

        # Batched ensemble prediction (scalar mean activity)
        if ensemble_smiles and mole_embs:
            try:
                batched_preds = self.ensemble.predict_batched(ensemble_smiles, mole_embs)
                for idx, pred in zip(ensemble_indices, batched_preds):
                    results[idx]["activity"] = float(pred)
            except Exception as e:
                print(f"    [debug] predict_batched failed: {e}, falling back to per-molecule")
                for idx, smiles in zip(ensemble_indices, ensemble_smiles):
                    try:
                        pred = self.ensemble.predict(smiles)
                        results[idx]["activity"] = float(np.mean(pred))
                    except Exception:
                        pass
        elif ensemble_smiles and not mole_embs:
            for idx, smiles in zip(ensemble_indices, ensemble_smiles):
                try:
                    pred = self.ensemble.predict(smiles)
                    results[idx]["activity"] = float(np.mean(pred))
                except Exception:
                    pass

        return results


# ── RL Models (Paper-Matching Architecture) ────────────────────────────────────

class ChempropMolEModel(nn.Module):
    """Chemprop + MolE RL value function model.

    Architecture matches MultitaskMolE scorer:
      - GNN: BondMessagePassing(300,3) + MeanAggregation -> 300-dim
      - MolE: 768 molecular embeddings (averaged across BBs)
      - Concat: 300 + 768 = 1068-dim
      - MLP: Linear(1068, 600) + ReLU + Linear(600, 600) + ReLU + Linear(600, n_tasks)
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, depth: int = DEPTH,
                 n_tasks: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mole_dim = MOLFEAT_DIM
        self.n_tasks = n_tasks
        self.featurizer = SimpleMoleculeMolGraphFeaturizer()

        self.message_passing = BondMessagePassing(
            d_h=hidden_dim,
            depth=depth,
        )
        self.aggregation = MeanAggregation()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + self.mole_dim, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, n_tasks),
        )
        self._init_weights()

        # External caches (set by RLGenerator after pre-computation)
        self.mole_cache: Dict[str, np.ndarray] = {}
        self.bb_gnn_cache: Dict[str, Tuple[np.ndarray, int]] = {}
        self.use_gnn_cache = False

    def _init_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def _get_mg(self, smiles: str):
        """Get MolGraph for SMILES, using global cache."""
        if smiles not in SMILES_TO_MOL_GRAPH:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                SMILES_TO_MOL_GRAPH[smiles] = self.featurizer(mol)
            else:
                SMILES_TO_MOL_GRAPH[smiles] = None
        return SMILES_TO_MOL_GRAPH[smiles]

    def precompute_gnn_encodings(self, all_bbs: List[str], batch_size: int = 1024):
        """Pre-compute GNN encodings for all individual BBs.

        Processes BBs in mini-batches to avoid OOM on GPU.
        Each mini-batch runs through message_passing + MeanAggregation,
        caches the (encoding, num_atoms) pair. During rollouts,
        joined-molecule encodings are reconstructed via weighted average
        of per-BB encodings (mathematically identical to MeanAggregation
        over disconnected components).

        This replaces O(N) MolGraph builds + GNN forward passes per rollout
        with a one-time O(N) cost + O(1) dict lookups at inference.
        """
        from tqdm import tqdm as _tqdm

        # Pre-build all MolGraphs (CPU, one-time)
        mg_data = []
        for smi in _tqdm(all_bbs, desc="  Building MolGraphs", ncols=80, leave=False):
            mg = self._get_mg(smi)
            if mg is not None:
                mg_data.append((smi, mg, mg.V.shape[0]))
            else:
                self.bb_gnn_cache[smi] = (np.zeros(self.hidden_dim, dtype=np.float32), 0)

        if not mg_data:
            return

        # Process in mini-batches
        for start in _tqdm(range(0, len(mg_data), batch_size), desc="  GNN forward", ncols=80, leave=False):
            batch = mg_data[start:start + batch_size]
            batch_mgs = [mg for _, mg, _ in batch]
            batch_mg = BatchMolGraph(batch_mgs)
            batch_mg.to(DEVICE)
            with torch.no_grad():
                encodings = self.message_passing(batch_mg)
                gnn_out = self.aggregation(encodings, batch_mg.batch).cpu().numpy()
            for j, (smi, _, n_atoms) in enumerate(batch):
                self.bb_gnn_cache[smi] = (gnn_out[j], n_atoms)

    def forward_batched(self, bb_lists: List[List[str]]) -> torch.Tensor:
        """Batched forward pass for N candidate BB combinations.

        In eval mode, uses cached GNN encodings (pre-computed per BB)
        to skip the MolGraph building + GNN forward pass entirely.
        In train mode, uses the original MolGraph pipeline for gradients.

        Args:
            bb_lists: List of BB tuples, each [smiles, ...].
        Returns:
            Tensor of shape (N, n_tasks) with predicted values.
        """
        if not bb_lists:
            return torch.tensor([], device=DEVICE)

        N = len(bb_lists)
        n_comps = [len(bbs) for bbs in bb_lists]
        same_n_comp = (max(n_comps) == min(n_comps))

        # MolE features: vectorized if uniform n_comp, else per-sample
        if same_n_comp:
            n_comp = n_comps[0]
            flat_bbs = [smi for bbs in bb_lists for smi in bbs]
            mole_feats = np.array(
                [self.mole_cache[smi] for smi in flat_bbs],
                dtype=np.float32,
            ).reshape(N, n_comp, self.mole_dim).mean(axis=1)
        else:
            mole_feats = np.zeros((N, self.mole_dim), dtype=np.float32)
            for i, bbs in enumerate(bb_lists):
                feats = [self.mole_cache.get(smi, np.zeros(self.mole_dim, dtype=np.float32)) for smi in bbs]
                mole_feats[i] = np.mean(feats, axis=0)
        mole_tensor = torch.from_numpy(mole_feats).to(DEVICE)

        if not self.training and self.use_gnn_cache and same_n_comp:
            n_comp = n_comps[0]
            flat_bbs = [smi for bbs in bb_lists for smi in bbs]
            enc_np = np.array(
                [self.bb_gnn_cache[smi][0] for smi in flat_bbs],
                dtype=np.float32,
            ).reshape(N, n_comp, self.hidden_dim)
            n_atoms_np = np.array(
                [self.bb_gnn_cache[smi][1] for smi in flat_bbs],
                dtype=np.float32,
            ).reshape(N, n_comp)
            total_atoms = np.maximum(n_atoms_np.sum(axis=1, keepdims=True), 1.0)
            gnn_np = (enc_np * (n_atoms_np / total_atoms)[:, :, None]).sum(axis=1)
            gnn_tensor = torch.from_numpy(gnn_np).to(DEVICE)
            combined = torch.cat([gnn_tensor, mole_tensor], dim=1)
            return self.mlp(combined)
        elif not self.training and self.use_gnn_cache and not same_n_comp:
            # Variable n_comp in eval: fall back to per-sample GNN cache
            gnn_np = np.zeros((N, self.hidden_dim), dtype=np.float32)
            for i, bbs in enumerate(bb_lists):
                gnn_sum = np.zeros(self.hidden_dim, dtype=np.float32)
                n_atoms_total = 0
                for smi in bbs:
                    enc, n_at = self.bb_gnn_cache[smi]
                    gnn_sum += enc * n_at
                    n_atoms_total += n_at
                gnn_np[i] = gnn_sum / max(n_atoms_total, 1)
            gnn_tensor = torch.from_numpy(gnn_np).to(DEVICE)
            combined = torch.cat([gnn_tensor, mole_tensor], dim=1)
            return self.mlp(combined)

        # Original MolGraph pipeline (train mode or no cache)
        smiles_list = [".".join(bbs) for bbs in bb_lists]
        mgs = []
        valid_mask = []
        for smi in smiles_list:
            mg = self._get_mg(smi)
            if mg is not None:
                mgs.append(mg)
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        if not mgs:
            return torch.zeros(N, self.n_tasks, device=DEVICE)

        batch_mg = BatchMolGraph(mgs)
        batch_mg.to(DEVICE)

        encodings = self.message_passing(batch_mg)
        gnn_out = self.aggregation(encodings, batch_mg.batch)

        valid_idx = torch.tensor([i for i, v in enumerate(valid_mask) if v],
                                  device=DEVICE, dtype=torch.long)
        combined = torch.cat([gnn_out, mole_tensor[valid_idx]], dim=1)
        preds = self.mlp(combined)

        result = torch.zeros(N, self.n_tasks, device=DEVICE)
        result[valid_idx] = preds
        return result


class ChempropRDKitModel(nn.Module):
    """Chemprop + RDKit descriptor value function model (Z2).

    Matches solubility pretrained model (aqsoildb_solubility_tdc_scaffold):
      - GNN: BondMessagePassing(300,3) + MeanAggregation -> 300-dim
      - RDKit: 200 molecular descriptors via chemprop v1_rdkit_2d_normalized
      - Concat: 300 + 200 = 500-dim
      - MLP: Linear(500, 300) + ReLU + Linear(300, n_tasks)
      - No LayerNorm (matches pretrained checkpoint exactly)
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, depth: int = DEPTH,
                 n_tasks: int = 1, rdkit_dim: int = RDKIT_FEAT_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rdkit_dim = rdkit_dim
        self.n_tasks = n_tasks
        self.featurizer = SimpleMoleculeMolGraphFeaturizer()
        self.rdkit_featurizer = MoleculeFeaturizerRegistry['v1_rdkit_2d_normalized']()

        self.message_passing = BondMessagePassing(d_h=hidden_dim, depth=depth)
        self.aggregation = MeanAggregation()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + rdkit_dim, 300),
            nn.ReLU(),
            nn.Linear(300, n_tasks),
        )
        self._init_weights()

        # Per-BB caches (eval mode speedup)
        self.bb_gnn_cache: Dict[str, Tuple[np.ndarray, int]] = {}
        self.use_gnn_cache = False
        self.bb_rdkit_cache: Dict[str, np.ndarray] = {}

    def _init_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def _get_mg(self, smiles: str):
        if smiles not in SMILES_TO_MOL_GRAPH:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                SMILES_TO_MOL_GRAPH[smiles] = self.featurizer(mol)
            else:
                SMILES_TO_MOL_GRAPH[smiles] = None
        return SMILES_TO_MOL_GRAPH[smiles]

    def precompute_rdkit_features(self, all_bbs: List[str]):
        """Pre-compute 200-dim RDKit descriptors for all unique BBs using chemprop featurizer."""
        from tqdm import tqdm as _tqdm
        for smi in _tqdm(all_bbs, desc="  BB RDKit descriptors (Z2)", ncols=80, leave=False):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                arr = self.rdkit_featurizer(mol)
                self.bb_rdkit_cache[smi] = np.array(arr, dtype=np.float32).flatten()
            else:
                self.bb_rdkit_cache[smi] = np.zeros(self.rdkit_dim, dtype=np.float32)

    def precompute_gnn_encodings(self, all_bbs: List[str], batch_size: int = 1024):
        from tqdm import tqdm as _tqdm
        mg_data = []
        for smi in _tqdm(all_bbs, desc="  Building MolGraphs (Z2)", ncols=80, leave=False):
            mg = self._get_mg(smi)
            if mg is not None:
                mg_data.append((smi, mg, mg.V.shape[0]))
            else:
                self.bb_gnn_cache[smi] = (np.zeros(self.hidden_dim, dtype=np.float32), 0)
        if not mg_data:
            return
        for start in _tqdm(range(0, len(mg_data), batch_size), desc="  GNN forward (Z2)", ncols=80, leave=False):
            batch = mg_data[start:start + batch_size]
            batch_mgs = [mg for _, mg, _ in batch]
            batch_mg = BatchMolGraph(batch_mgs)
            batch_mg.to(DEVICE)
            with torch.no_grad():
                encodings = self.message_passing(batch_mg)
                gnn_out = self.aggregation(encodings, batch_mg.batch).cpu().numpy()
            for j, (smi, _, n_atoms) in enumerate(batch):
                self.bb_gnn_cache[smi] = (gnn_out[j], n_atoms)

    def forward_batched(self, bb_lists: List[List[str]]) -> torch.Tensor:
        if not bb_lists:
            return torch.tensor([], device=DEVICE)

        N = len(bb_lists)
        n_comps = [len(bbs) for bbs in bb_lists]
        same_n_comp = (max(n_comps) == min(n_comps))

        # RDKit features: averaged from per-BB cache (fast, ~1us per lookup)
        if same_n_comp:
            n_comp = n_comps[0]
            flat_bbs = [smi for bbs in bb_lists for smi in bbs]
            rdkit_feats = np.array(
                [self.bb_rdkit_cache[smi] for smi in flat_bbs],
                dtype=np.float32,
            ).reshape(N, n_comp, self.rdkit_dim).mean(axis=1)
        else:
            rdkit_feats = np.zeros((N, self.rdkit_dim), dtype=np.float32)
            for i, bbs in enumerate(bb_lists):
                feats = [self.bb_rdkit_cache.get(smi, np.zeros(self.rdkit_dim, dtype=np.float32)) for smi in bbs]
                rdkit_feats[i] = np.mean(feats, axis=0)
        rdkit_tensor = torch.from_numpy(rdkit_feats).to(DEVICE)

        if not self.training and self.use_gnn_cache and same_n_comp:
            n_comp = n_comps[0]
            flat_bbs = [smi for bbs in bb_lists for smi in bbs]
            enc_np = np.array(
                [self.bb_gnn_cache[smi][0] for smi in flat_bbs],
                dtype=np.float32,
            ).reshape(N, n_comp, self.hidden_dim)
            n_atoms_np = np.array(
                [self.bb_gnn_cache[smi][1] for smi in flat_bbs],
                dtype=np.float32,
            ).reshape(N, n_comp)
            total_atoms = np.maximum(n_atoms_np.sum(axis=1, keepdims=True), 1.0)
            gnn_np = (enc_np * (n_atoms_np / total_atoms)[:, :, None]).sum(axis=1)
            gnn_tensor = torch.from_numpy(gnn_np).to(DEVICE)
            combined = torch.cat([gnn_tensor, rdkit_tensor], dim=1)
            return self.mlp(combined)
        elif not self.training and self.use_gnn_cache and not same_n_comp:
            gnn_np = np.zeros((N, self.hidden_dim), dtype=np.float32)
            for i, bbs in enumerate(bb_lists):
                gnn_sum = np.zeros(self.hidden_dim, dtype=np.float32)
                n_atoms_total = 0
                for smi in bbs:
                    enc, n_at = self.bb_gnn_cache[smi]
                    gnn_sum += enc * n_at
                    n_atoms_total += n_at
                gnn_np[i] = gnn_sum / max(n_atoms_total, 1)
            gnn_tensor = torch.from_numpy(gnn_np).to(DEVICE)
            combined = torch.cat([gnn_tensor, rdkit_tensor], dim=1)
            return self.mlp(combined)

        mgs = []
        valid_mask = []
        for bbs in bb_lists:
            smi = ".".join(bbs)
            mg = self._get_mg(smi)
            if mg is not None:
                mgs.append(mg)
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        if not mgs:
            return torch.zeros(N, self.n_tasks, device=DEVICE)

        batch_mg = BatchMolGraph(mgs)
        batch_mg.to(DEVICE)

        encodings = self.message_passing(batch_mg)
        gnn_out = self.aggregation(encodings, batch_mg.batch)

        valid_idx = torch.tensor([i for i, v in enumerate(valid_mask) if v],
                                  device=DEVICE, dtype=torch.long)
        combined = torch.cat([gnn_out, rdkit_tensor[valid_idx]], dim=1)
        preds = self.mlp(combined)

        result = torch.zeros(N, self.n_tasks, device=DEVICE)
        result[valid_idx] = preds
        return result


class RLModels:
    """Container for two independent RL models.

    Z1 (activity): scalar broad-spectrum activity.
    Z2 (log solubility): scalar log solubility.
    All prediction is batched — no per-molecule forward calls.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, depth: int = DEPTH):
        self.z1 = ChempropMolEModel(hidden_dim, depth, n_tasks=Z1_N_TASKS).to(DEVICE)
        self.z2 = ChempropRDKitModel(hidden_dim, depth, n_tasks=1).to(DEVICE)
        self.z1_opt = torch.optim.Adam(self.z1.parameters(), lr=LEARNING_RATE)
        self.z2_opt = torch.optim.Adam(self.z2.parameters(), lr=LEARNING_RATE)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.z1.parameters()) + \
               sum(p.numel() for p in self.z2.parameters())

    def set_caches(self, mole_cache: Dict[str, np.ndarray],
                   gnn_cache: Optional[Dict[str, Tuple[np.ndarray, int]]] = None,
                   rdkit_cache: Optional[Dict[str, np.ndarray]] = None):
        """Share pre-computed feature caches with both models."""
        self.z1.mole_cache = mole_cache
        if gnn_cache is not None:
            self.z1.bb_gnn_cache = gnn_cache
            self.z2.bb_gnn_cache = gnn_cache
            self.z1.use_gnn_cache = True
            self.z2.use_gnn_cache = True
        if rdkit_cache is not None:
            self.z2.bb_rdkit_cache = rdkit_cache

    def predict_values_batched(self, bb_lists: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched prediction for many candidate BB lists.

        Z1 returns scalar broad-spectrum activity (1-head sigmoid).
        Z2 returns scalar log solubility.

        Args:
            bb_lists: List of BB tuples, each [smiles, ...].
        Returns:
            (z1_vals, z2_vals) tensors of shape (N,).
        """
        with torch.no_grad():
            self.z1.eval()
            self.z2.eval()
            z1_vals = torch.sigmoid(self.z1.forward_batched(bb_lists).squeeze(-1))  # (N,)
            z2_vals = self.z2.forward_batched(bb_lists).squeeze(-1)  # (N,)
        return z1_vals, z2_vals

    def train_step_batched(self, bb_lists: List[List[str]],
                           z1_labels: torch.Tensor,
                           z2_labels: torch.Tensor) -> Dict[str, float]:
        """Single batched training step.

        Args:
            bb_lists: List of BB tuples for training.
            z1_labels: Scalar activity targets, tensor (N, 1).
            z2_labels: Solubility targets, tensor (N,).
        Returns:
            Dict of loss values.
        """
        self.z1.train()
        self.z2.train()

        z1_pred = torch.sigmoid(self.z1.forward_batched(bb_lists))  # (N, 1)
        z2_pred = self.z2.forward_batched(bb_lists).squeeze(-1)  # (N,)

        loss_z1 = F.mse_loss(z1_pred, z1_labels.unsqueeze(-1))
        loss_z2 = F.mse_loss(z2_pred, z2_labels)
        loss = loss_z1 + loss_z2

        self.z1_opt.zero_grad()
        self.z2_opt.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.z1.parameters(), CLIP_GRAD)
        torch.nn.utils.clip_grad_norm_(self.z2.parameters(), CLIP_GRAD)

        self.z1_opt.step()
        self.z2_opt.step()

        return {"loss": loss.item(), "loss_z1": loss_z1.item(), "loss_z2": loss_z2.item()}

    def save(self, path: Path):
        torch.save({
            "z1": self.z1.state_dict(),
            "z2": self.z2.state_dict(),
            "z1_opt": self.z1_opt.state_dict(),
            "z2_opt": self.z2_opt.state_dict(),
        }, path)

    def load(self, path: Path):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
        self.z1.load_state_dict(ckpt["z1"])
        self.z2.load_state_dict(ckpt["z2"])
        self.z1_opt.load_state_dict(ckpt["z1_opt"])
        self.z2_opt.load_state_dict(ckpt["z2_opt"])


# ── Trajectory Buffer ──────────────────────────────────────────────────────────

@dataclass
class Node:
    """A node in the search/generation trajectory.

    Matches paper's node definition: stores building blocks and RL values.
    z1_val / z2_val: RL model's own prediction (used for softmax selection).
    z1_target / z2_target: Fixed oracle training target (set after rollout completes,
                           same for all nodes in a trajectory = final molecule score).
    z1_target is a scalar broad-spectrum activity score.
    z2_target is a scalar log solubility.
    """
    bbs: List[str] = field(default_factory=list)
    z1_val: float = 0.0
    z2_val: float = 0.0
    z1_target: List[float] = field(default_factory=lambda: [0.0] * Z1_N_TASKS)
    z2_target: float = 0.0
    activity: float = 0.0
    log_solubility: float = -10.0
    z1_uncertainty: float = 0.0
    per_species: Dict[str, float] = field(default_factory=dict)
    smiles: str = ""
    rollout_id: int = -1
    complete: bool = False


class TrajectoryBuffer:
    """Stores trajectory nodes for RL training.

    Matches paper: stores both incomplete (partial) and complete (terminal) nodes.
    Each rollout generates multiple nodes: one per BB addition step.
    """

    def __init__(self, max_nodes: int = MAX_NODES):
        self.nodes: List[Node] = []
        self.max_nodes = max_nodes

    def add_node(self, node: Node):
        self.nodes.append(node)
        if len(self.nodes) > self.max_nodes:
            # Remove oldest nodes (FIFO)
            excess = len(self.nodes) - self.max_nodes
            self.nodes = self.nodes[excess:]

    def add_rollout(self, rollout_nodes: List[Node], rollout_id: int):
        """Add nodes from a rollout trajectory."""
        for node in rollout_nodes:
            node.rollout_id = rollout_id
            self.add_node(node)

    def sample_batch(self, batch_size: int = BATCH_SIZE) -> Tuple[List[Node], List[float], List[float]]:
        """Sample a random batch of nodes for training.

        Returns:
            (nodes, z1_targets, z2_targets)
        """
        if len(self.nodes) == 0:
            return [], [], []

        batch = random.sample(self.nodes, min(batch_size, len(self.nodes)))
        z1_targets = [n.z1_target for n in batch]
        z2_targets = [n.z2_target for n in batch]
        return batch, z1_targets, z2_targets

    @property
    def size(self) -> int:
        return len(self.nodes)


# ── Selection, Temperature, Weight Updates ─────────────────────────────────────

def softmax_selection(scores: Dict[str, float], tau: float) -> str:
    """Select BB via softmax over scores (GPU-accelerated).

    Matches paper: P(BB) = exp(Z(BB)/tau) / sum(exp(Z(BB')/tau)).

    Args:
        scores: Dict mapping BB SMILES to scores (combined Z value).
        tau: Temperature parameter.
    Returns:
        Selected BB SMILES string.
    """
    if not scores:
        return ""
    items = list(scores.items())
    if len(items) == 1:
        return items[0][0]

    values = torch.tensor([v for _, v in items], device=DEVICE, dtype=torch.float64)
    tau_t = max(tau, TAU_MIN)

    values = values - values.max()
    values = values.clamp(-500, 500)
    probs = torch.exp(values / tau_t)
    probs = probs / (probs.sum() + 1e-10)

    if torch.isnan(probs).any():
        probs = torch.ones_like(probs) / len(probs)

    idx = torch.multinomial(probs, 1).item()
    return items[idx][0]


def update_temperature(tau: float, lambda_avg: float,
                       target_lambda: float = TARGET_LAMBDA,
                       gamma: float = GAMMA) -> float:
    """Update temperature per paper Section 2.2.5.

    Paper formula:
      lambda_diff = (lambda_avg - target_lambda) / target_lambda
      tau_new = tau + lambda_diff * tau
      tau = gamma * tau + (1 - gamma) * tau_new
      tau = clip(tau, tau_min, tau_max)

    Args:
        tau: Current temperature.
        lambda_avg: EMA of max Tanimoto similarity.
        target_lambda: Target Tanimoto similarity (lambda=0.6).
        gamma: SMA decay (gamma=0.98).
    Returns:
        Updated temperature.
    """
    lambda_diff = (lambda_avg - target_lambda) / max(target_lambda, 1e-10)
    tau_new = tau + lambda_diff * tau
    tau = gamma * tau + (1.0 - gamma) * tau_new
    return float(np.clip(tau, TAU_MIN, TAU_MAX))


def update_weights(
    w_activity: float, w_solubility: float,
    sr_avg_activity: float, sr_avg_solubility: float,
    gamma: float = GAMMA,
) -> Tuple[float, float]:
    """Dynamic weight update per paper Section 2.2.5.

    Paper formula:
      s_avg = (1/L) * sum(s^k_avg)
      if s_avg > 0:
        s^k_diff = (s^k_avg - s_avg) / s_avg
        w_k = gamma * w_k + (1-gamma) * (w_k - s^k_diff * w_k)
      Normalize: w_k /= sum(w_k)
      Clip: w_k = max(w_k, w_min=0.001)
      Renormalize

    Where s^k_avg is the EMA success rate for objective k.

    Returns:
        (updated_w_activity, updated_w_solubility)
    """
    W_MIN = 0.001
    s_avgs = np.array([sr_avg_activity, sr_avg_solubility], dtype=np.float64)
    s_avg = np.mean(s_avgs)
    w = np.array([w_activity, w_solubility], dtype=np.float64)
    if s_avg > 0:
        s_diffs = (s_avgs - s_avg) / s_avg
        w = gamma * w + (1.0 - gamma) * (w - s_diffs * w)
    w = w / (np.sum(w) + 1e-10)
    w = np.maximum(w, W_MIN)
    w = w / (np.sum(w) + 1e-10)
    return float(w[0]), float(w[1])


# ── Tanimoto Similarity ────────────────────────────────────────────────────────

def compute_tanimoto(smiles1: str, smiles2: str) -> float:
    """Compute Tanimoto similarity between two molecules."""
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return 0.0
        fp1 = _morgan_fp(mol1)
        fp2 = _morgan_fp(mol2)
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except:
        return 0.0


def compute_lambda_from_buffer(buffer: TrajectoryBuffer, reference_fps: np.ndarray) -> float:
    """Compute average max Tanimoto similarity to reference molecules."""
    from rdkit import Chem, DataStructs
    if len(buffer.nodes) == 0 or reference_fps is None or len(reference_fps) == 0:
        return 0.0

    similarities = []
    for node in buffer.nodes[-100:]:  # Last 100 nodes
        if node.smiles:
            try:
                mol = Chem.MolFromSmiles(node.smiles)
                if mol:
                    fp = _morgan_fp(mol)
                    sims = [float(DataStructs.TanimotoSimilarity(fp, ref))
                            for ref in reference_fps]
                    similarities.append(max(sims) if sims else 0.0)
            except:
                pass

    return float(np.mean(similarities)) if similarities else 0.0


# ── RL Generator ───────────────────────────────────────────────────────────────

class RLGenerator:
    """Main RL generation loop matching paper architecture.

    Key aspects:
      - Dual RL models (Z1 for activity, Z2 for log solubility)
      - Multi-molecule node input via disconnected SMILES
      - Per-node trajectory storage
      - Dynamic temperature and weight updates
      - Built-in RL training loop with replay
      - 13 REAL reaction schemes selected at random per rollout
    """

    def __init__(
        self,
        building_blocks: Dict[int, Dict[str, List[str]]],
        reaction_schemes: List[Dict],
        scorer: Optional[MolEScorer] = None,
        n_rollout: int = N_ROLLOUT_DEFAULT,
        load_checkpoint: bool = False,
        w_solubility_init: float = W_SOLUBILITY_INIT,
    ):
        self.rxn_bbs = building_blocks  # {rxn_id: {pos_key: [smiles]}}
        self.reaction_schemes = reaction_schemes
        self.schemes_by_id = {s["id"]: s for s in reaction_schemes}
        self.scorer = scorer or MolEScorer()
        self.n_rollout = n_rollout

        # RL models
        self.models = RLModels()
        print(f"  RL models created: {self.models.n_params:,} params total")

        # Collect all unique BBs across all reaction schemes
        all_bbs = sorted(set(
            smi for bb_sets in building_blocks.values()
            for smis in bb_sets.values() for smi in smis
        ))

        # Pre-compute / load MolE 768-dim embeddings for ALL unique BBs
        mole_cache_path = SYNTHEMOL_DIR / "bb_mole_cache.npz"
        if mole_cache_path.exists():
            try:
                data = np.load(mole_cache_path, allow_pickle=True)
                self.scorer.bb_mole_cache = dict(zip(data["smiles"], data["embeddings"]))
                print(f"  Loaded {len(self.scorer.bb_mole_cache)} BB MolE embeddings")
            except Exception as e:
                print(f"  Warning: could not load BB MolE cache: {e}")
                self.scorer.bb_mole_cache = {}
        else:
            print("  Pre-computing BB MolE embeddings (one-time, saved for next run)...")
            if self.scorer.ensemble is not None and self.scorer.ensemble.mole_encoder is not None:
                mole_embs = self.scorer.ensemble.compute_mole_embeddings_batched(all_bbs, batch_size=256)
                self.scorer.bb_mole_cache = dict(zip(all_bbs, mole_embs))
                np.savez_compressed(mole_cache_path,
                    smiles=list(self.scorer.bb_mole_cache.keys()),
                    embeddings=list(self.scorer.bb_mole_cache.values()))
                print(f"  Saved {len(self.scorer.bb_mole_cache)} BB MolE embeddings to {mole_cache_path.name}")
            else:
                print("  WARNING: MolE encoder not available, skipping BB MolE cache")
                self.scorer.bb_mole_cache = {}

        # MolE cache is shared with RL models (replaces paper's RDKit 200-dim features)
        self.mole_cache: Dict[str, np.ndarray] = self.scorer.bb_mole_cache

        # Pre-compute / load GNN encodings for ALL unique BBs (one-time)
        gnn_cache_path = SYNTHEMOL_DIR / "bb_gnn_cache.npz"
        if gnn_cache_path.exists():
            print("  Loading cached BB GNN encodings...")
            data = np.load(gnn_cache_path, allow_pickle=True)
            smiles_arr = data["smiles"]
            encodings_arr = data["encodings"]
            atom_counts_arr = data["atom_counts"]
            self.gnn_cache = {}
            for i in range(len(smiles_arr)):
                self.gnn_cache[str(smiles_arr[i])] = (encodings_arr[i], int(atom_counts_arr[i]))
            missing = [smi for smi in all_bbs if smi not in self.gnn_cache]
            if missing:
                print(f"  Computing {len(missing)} missing BB GNN encodings...")
                self.models.z1.precompute_gnn_encodings(missing)
                self.models.z2.bb_gnn_cache = self.models.z1.bb_gnn_cache
                for smi in missing:
                    self.gnn_cache[smi] = self.models.z1.bb_gnn_cache[smi]
                np.savez_compressed(gnn_cache_path,
                    smiles=list(self.gnn_cache.keys()),
                    encodings=[v[0] for v in self.gnn_cache.values()],
                    atom_counts=[v[1] for v in self.gnn_cache.values()])
            print(f"  Loaded {len(self.gnn_cache)} BB GNN encodings")
        else:
            print("  Pre-computing BB GNN encodings (one-time, saved for next run)...")
            self.models.z1.precompute_gnn_encodings(all_bbs)
            self.models.z2.bb_gnn_cache = self.models.z1.bb_gnn_cache
            self.gnn_cache = self.models.z1.bb_gnn_cache
            np.savez_compressed(gnn_cache_path,
                smiles=list(self.gnn_cache.keys()),
                encodings=[v[0] for v in self.gnn_cache.values()],
                atom_counts=[v[1] for v in self.gnn_cache.values()])
            print(f"  Saved {len(self.gnn_cache)} BB GNN encodings to {gnn_cache_path.name}")

        # Ensure Z2 has its own GNN cache (RDKit model, separate from Z1's MolE model)
        if not hasattr(self.models.z2, 'bb_gnn_cache') or len(self.models.z2.bb_gnn_cache) == 0:
            self.models.z2.bb_gnn_cache = self.gnn_cache
            self.models.z2.use_gnn_cache = True

        # Pre-compute / load per-BB RDKit features for Z2 (chemprop v1_rdkit_2d_normalized)
        rdkit_cache_path = SYNTHEMOL_DIR / "bb_rdkit_cache.npz"
        if rdkit_cache_path.exists():
            data = np.load(rdkit_cache_path, allow_pickle=True)
            self.rdkit_cache = dict(zip(data["smiles"], data["features"]))
            print(f"  Loaded {len(self.rdkit_cache)} BB RDKit descriptor caches")
        else:
            print("  Pre-computing BB RDKit descriptors (one-time, saved for next run)...")
            self.models.z2.precompute_rdkit_features(all_bbs)
            self.rdkit_cache = self.models.z2.bb_rdkit_cache
            np.savez_compressed(rdkit_cache_path,
                smiles=list(self.rdkit_cache.keys()),
                features=list(self.rdkit_cache.values()))
            print(f"  Saved {len(self.rdkit_cache)} BB RDKit descriptor caches to {rdkit_cache_path.name}")

        self.models.set_caches(self.mole_cache, self.gnn_cache, self.rdkit_cache)

        # Initialize Z1/Z2 from pretrained Chemprop model weights (paper Section 2.2.5)
        # Loads only the GNN backbone (message_passing), FFN stays random for RL fine-tuning
        self._init_from_pretrained(self.models.z1, MODELS_DIR / "chemprop" / "multitask_mole")
        self._init_from_pretrained(self.models.z2, MODELS_DIR / "chemprop" / "aqsoildb_solubility_tdc_scaffold")

        # Trajectory buffer
        self.buffer = TrajectoryBuffer()

        # Generation state
        self.molecules: List[Dict] = []
        self.tau = TAU0
        self.w_activity = W_ACTIVITY_INIT
        self.w_solubility = w_solubility_init
        self.generation_step = 0
        # Rolling averages for paper temperature/weight updates
        self.lambda_avg = TARGET_LAMBDA
        self.sr_avg_activity = 0.0
        self.sr_avg_solubility = 0.0
        # HIR: BB pair visit counter for novelty bonus (Mol-AIR, JCIM 2025)
        self.bb_pair_counter: Dict[str, int] = {}

        # Statistics
        self.stats = {
            "steps": [], "tau": [], "w_activity": [], "w_solubility": [],
            "success_rate_activity": [], "success_rate_solubility": [],
            "loss": [], "lambda_avg": [],
            "avg_z1": [], "avg_z2": [],
        }

        # Timing
        self.t_start = time.time()

        # Load checkpoint
        self.checkpoint_path = CHECKPOINT_DIR / "rl_checkpoint.pt"
        if load_checkpoint and self.checkpoint_path.exists():
            self._load_checkpoint()

        # Load training active fingerprints for lambda computation
        self.training_fps = self._load_training_fps()

    def _load_training_fps(self) -> Optional[np.ndarray]:
        """Load fingerprints of training actives for lambda computation."""
        fp_cache = CACHE_DIR / "training_active_fps.npz"
        if fp_cache.exists():
            try:
                data = np.load(fp_cache)
                self.training_fps = data["fps"]
                print(f"  Loaded {len(self.training_fps)} training active fingerprints")
                return self.training_fps
            except:
                pass

        # Try to compute from scratch
        try:
            train_file = DATA_DIR / "processed" / "train_actives.smi"
            if train_file.exists():
                fps_list = []
                with open(train_file) as f:
                    for line in f:
                        smi = line.strip().split()[0]
                        mol = Chem.MolFromSmiles(smi)
                        if mol:
                            fp = _morgan_fp(mol)
                            fps_list.append(fp)
                arr = np.array(fps_list, dtype=object)
                np.savez(fp_cache, fps=arr)
                print(f"  Computed {len(arr)} training active fingerprints")
                return arr
        except Exception as e:
            print(f"  WARNING: Could not compute training fps: {e}")

        return None

    @staticmethod
    def _init_from_pretrained(model: nn.Module, model_dir: Path):
        """Initialize model weights from pretrained Chemprop checkpoint.

        Maps pretrained key names to RL model key names:
          predictor.ffn.0.0.* -> mlp.0.*  (first hidden layer)
          predictor.ffn.1.2.* -> mlp.2.*  (second hidden layer)
          predictor.ffn.2.2.* -> mlp.4.*  (output layer, averaged 6->1 for Z1)
          message_passing.*   -> same    (GNN backbone)

        Z2 (ChempropRDKitModel) has 2-layer MLP (no mlp.4), so the third
        layer is only mapped if it exists in both checkpoint and model.
        """
        ckpt_path = model_dir / "model_0" / "best.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: Pretrained checkpoint not found at {ckpt_path}, using random init")
            return
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            sd = ckpt.get("state_dict", ckpt)

            # Map pretrained keys to RL model keys
            has_mlp4 = "mlp.4.weight" in model.state_dict()
            key_map = {
                "predictor.ffn.0.0.weight": "mlp.0.weight",
                "predictor.ffn.0.0.bias": "mlp.0.bias",
                "predictor.ffn.1.2.weight": "mlp.2.weight",
                "predictor.ffn.1.2.bias": "mlp.2.bias",
            }
            if has_mlp4:
                key_map["predictor.ffn.2.2.weight"] = "mlp.4.weight"
                key_map["predictor.ffn.2.2.bias"] = "mlp.4.bias"

            # Build loadable dict: keep matched keys, skip shape-mismatched output
            loadable = {}
            mapped = []
            skipped = []
            skipped_keys = {"predictor.criterion.task_weights", "X_d_transform.mean",
                            "X_d_transform.scale", "predictor.output_transform.mean",
                            "predictor.output_transform.scale"}

            for k, v in sd.items():
                if any(k.startswith(s) for s in skipped_keys):
                    skipped.append(k)
                elif k in key_map:
                    new_k = key_map[k]
                    target_shape = tuple(model.state_dict()[new_k].shape)
                    source_shape = tuple(v.shape)
                    if source_shape != target_shape and has_mlp4 and "mlp.4" in new_k:
                        mapped.append((new_k, v))
                    elif source_shape == target_shape:
                        loadable[new_k] = v
                        mapped.append((new_k, v))
                    else:
                        skipped.append(f"{k} ({source_shape} -> {target_shape} mismatch)")
                elif k.startswith("message_passing"):
                    loadable[k] = v
                else:
                    skipped.append(k)

            missing, unexpected = model.load_state_dict(loadable, strict=False)

            # Average the 6 pretrained output heads into 1 for scalar Z1
            for k, v in mapped:
                if has_mlp4 and "mlp.4" in k:
                    if "weight" in k:
                        out_w = v.mean(dim=0, keepdim=True)  # (6,600)->(1,600)
                        model.mlp[4].weight.data.copy_(out_w.to(DEVICE))
                    elif "bias" in k:
                        out_b = v.mean(dim=0, keepdim=True)  # (6,)->(1,)
                        model.mlp[4].bias.data.copy_(out_b.to(DEVICE))

            n_gnn = len([k for k in loadable if k.startswith("message_passing")])
            n_ffn = len([k for k in loadable if k.startswith("mlp")])
            n_out = len([k for k, _ in mapped if "mlp.4" in k])
            print(f"  Loaded pretrained weights from {model_dir.name}:")
            print(f"    GNN: {n_gnn} layers  FFN: {n_ffn} layers  Output (averaged): {n_out} layers")
            if skipped:
                print(f"    Skipped (unmatched keys): {skipped}")
            if missing:
                print(f"    Missing in pretrained: {missing}")
        except Exception as e:
            print(f"  WARNING: Failed to load pretrained weights from {model_dir.name}: {e}")
            import traceback; traceback.print_exc()

    def _get_bb_pool(self, step: int, scheme_id: int) -> List[str]:
        """Get available building blocks for a given scheme and step."""
        scheme = self.schemes_by_id[scheme_id]
        if step >= len(scheme["bb_keys"]):
            return []
        key = scheme["bb_keys"][step]
        rxn_bbs = self.rxn_bbs.get(scheme_id, {})
        return list(rxn_bbs.get(key, []))

    def _score_bbs(self, bbs: List[str], step: int, scheme_id: int) -> Dict[str, float]:
        """Score candidate BBs: Z1+Z2 over all, ensemble re-rank top-100.

        Pass 1: Fast Z1+Z2 over ALL candidates (milliseconds, GPU-vectorized).
        Pass 2: Ensemble re-rank top 100 with property predictors (paper Section 2.2.5).
        """
        bb_pool = self._get_bb_pool(step, scheme_id)
        if not bb_pool:
            return {}
        bb_pool_list = list(bb_pool)
        w_a, w_q = self.w_activity, self.w_solubility

        # Pass 1: Z1 + Z2 over all candidates (milliseconds, GPU-vectorized)
        all_z1, all_z2 = [], []
        for start_idx in range(0, len(bb_pool_list), SCORE_BATCH_SIZE):
            batch = bb_pool_list[start_idx:start_idx + SCORE_BATCH_SIZE]
            z1_b, z2_b = self.models.predict_values_batched([bbs + [bb] for bb in batch])
            all_z1.append(z1_b); all_z2.append(z2_b)
        z1_vals = torch.cat(all_z1)
        z2_vals = torch.cat(all_z2)
        combined = w_a * z1_vals + w_q * z2_vals

        # Sort by Z1+Z2 and keep top-DIVERSITY_TOP_K for costly fingerprint operations
        paired = sorted(zip(bb_pool_list, combined.tolist()), key=lambda x: -x[1])

        # Pass 2: Ensemble re-rank on top-100 (paper: evaluate top candidates with full property predictor)
        top_k = paired[:min(100, len(paired))]
        preds = self.scorer.predict_approx_batched([".".join(bbs + [bb]) for bb, _ in top_k])

        ens_scores = [w_a * p["activity"] + w_q * p.get("log_solubility", -10.0) for p in preds]
        score_map = {bb: s for (bb, _), s in zip(top_k, ens_scores)}
        return {bb: score_map.get(bb, z) for bb, z in paired}

    def _build_node(self, bbs: List[str], rollout_id: int,
                    complete: bool = False) -> Node:
        """Build a node with RL values and optional scorer evaluation."""
        z1_t, z2_t = self.models.predict_values_batched([bbs])
        z1_val, z2_val = z1_t[0].item(), z2_t[0].item()

        node = Node(
            bbs=bbs,
            z1_val=z1_val,
            z2_val=z2_val,
            rollout_id=rollout_id,
            complete=complete,
        )

        if complete:
            mol_smiles = ".".join(bbs)
            node.smiles = mol_smiles
            try:
                pred = self.scorer.predict(mol_smiles)
                node.activity = pred["activity"]
                node.log_solubility = pred["log_solubility"]
                node.z1_uncertainty = pred.get("z1_uncertainty", 0.0)
                node.per_species = {k: v for k, v in pred.items()
                                    if k.startswith("score_")}
            except:
                pass

        return node

    def _single_rollout(self, rollout_id: int) -> Optional[Dict]:
        """Perform a single molecule generation rollout.

        Picks a random reaction scheme (paper: 13 REAL reactions),
        then adds BBs sequentially matching the scheme's component count.

        Returns:
            Dict with molecule info, or None if failed.
        """
        scheme = random.choice(self.reaction_schemes)
        scheme_id = scheme["id"]
        n_steps = scheme["n_components"]

        rollout_nodes = []
        bbs = []

        for step in range(n_steps):
            scores = self._score_bbs(bbs, step, scheme_id)
            if not scores:
                return None

            selected_bb = softmax_selection(scores, self.tau)
            if not selected_bb:
                return None

            bbs.append(selected_bb)
            complete = (step == n_steps - 1)
            node = self._build_node(bbs, rollout_id, complete=complete)
            rollout_nodes.append(node)

        # Store trajectory nodes
        self.buffer.add_rollout(rollout_nodes, rollout_id)

        # Final molecule
        mol_smiles = ".".join(bbs)
        final_node = rollout_nodes[-1]

        # Scalar Z1 target = broad-spectrum activity (geometric mean of per-species scores).
        # The RL model learns to predict this scalar directly.
        z1_target = final_node.activity
        for node in rollout_nodes:
            node.z1_target = z1_target
            node.z2_target = final_node.log_solubility

        # HIR: increment BB pair counter for novelty bonus next round
        if bbs:
            pair_key = ".".join(sorted(bbs))
            self.bb_pair_counter[pair_key] = self.bb_pair_counter.get(pair_key, 0) + 1

        # Update rolling averages per paper (EMA every rollout)
        if mol_smiles and len(self.molecules) > 0:
            try:
                mol_i = Chem.MolFromSmiles(mol_smiles)
                if mol_i is not None:
                    fp_i = _morgan_fp(mol_i)
                    sims = []
                    for m_prev in self.molecules[-100:]:
                        try:
                            mol_j = Chem.MolFromSmiles(m_prev.get('smiles', ''))
                            if mol_j is not None:
                                fp_j = _morgan_fp(mol_j)
                                sims.append(float(DataStructs.TanimotoSimilarity(fp_i, fp_j)))
                        except:
                            pass
                    max_sim = max(sims) if sims else 0.0
                    self.lambda_avg = GAMMA * self.lambda_avg + (1.0 - GAMMA) * max_sim
            except:
                pass
            sr_act_raw = 1.0 if final_node.activity >= THRESHOLD_ACTIVITY else 0.0
            sr_sol_raw = 1.0 if final_node.log_solubility >= THRESHOLD_SOLUBILITY else 0.0
            self.sr_avg_activity = GAMMA * self.sr_avg_activity + (1.0 - GAMMA) * sr_act_raw
            self.sr_avg_solubility = GAMMA * self.sr_avg_solubility + (1.0 - GAMMA) * sr_sol_raw

        return {
            "smiles": mol_smiles,
            "bbs": bbs,
            "activity": final_node.activity,
            "log_solubility": final_node.log_solubility,
            "z1_val": final_node.z1_val,
            "z2_val": final_node.z2_val,
            "rollout_id": rollout_id,
            **final_node.per_species,
        }

    def _compute_success_rates(self) -> Tuple[float, float]:
        """Compute success rates for activity and log solubility thresholds."""
        recent = self.molecules[-100:] if len(self.molecules) > 100 else self.molecules
        if not recent:
            return 0.0, 0.0

        n_act = sum(1 for m in recent if m["activity"] >= THRESHOLD_ACTIVITY)
        n_sol = sum(1 for m in recent if m["log_solubility"] >= THRESHOLD_SOLUBILITY)
        return n_act / len(recent), n_sol / len(recent)

    def _train_rl_models(self):
        """Train RL models on ALL buffer nodes (matches paper).

        Paper: every N_EPOCHS epochs over ALL nodes in the trajectory buffer.
        No sampling — the model sees every stored node each training cycle.
        """
        if self.buffer.size < 2:
            return {"loss": 0.0}

        all_nodes = self.buffer.nodes[:]
        random.shuffle(all_nodes)
        n_nodes = len(all_nodes)
        train_bs = min(BATCH_SIZE * 4, 256)  # GPU-effective batch size

        total_loss = 0.0
        n_batches = 0
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(N_EPOCHS):
            epoch_loss = 0.0
            n_epoch_batches = 0

            for start in range(0, n_nodes, train_bs):
                batch_nodes = all_nodes[start:start + train_bs]
                if len(batch_nodes) < 2:
                    continue

                bb_lists = [n.bbs for n in batch_nodes]
                z1_t = torch.tensor(
                    [n.z1_target for n in batch_nodes],
                    dtype=torch.float32, device=DEVICE)
                z2_t = torch.tensor(
                    [n.z2_target for n in batch_nodes],
                    dtype=torch.float32, device=DEVICE)

                loss_dict = self.models.train_step_batched(bb_lists, z1_t, z2_t)
                epoch_loss += loss_dict["loss"]
                n_epoch_batches += 1

            if n_epoch_batches > 0:
                avg_loss = epoch_loss / n_epoch_batches
                total_loss += avg_loss
                n_batches += 1

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        break

        self.models.z1.eval()
        self.models.z2.eval()

        return {"loss": total_loss / max(n_batches, 1)}

    def run(self) -> List[Dict]:
        """Main generation loop."""
        print(f"\n{'='*60}")
        print(f"RL Generation: {self.n_rollout} rollouts")
        print(f"{'='*60}")
        print(f"  Temperature: tau_0={TAU0}, gamma={GAMMA}")
        print(f"  Weights: w_act={self.w_activity:.2f}, w_sol={self.w_solubility:.2f}")
        print(f"  Thresholds: activity>={THRESHOLD_ACTIVITY}, logS>={THRESHOLD_SOLUBILITY}")
        print(f"  Total params: {self.models.n_params:,}")
        print()

        progress = tqdm(range(self.n_rollout), desc="RL Generation", ncols=80)

        for i in progress:
            self.generation_step = i
            self.diversity_progress = i / max(self.n_rollout - 1, 1)

            # Rollout
            mol = self._single_rollout(rollout_id=i)
            if mol is not None:
                self.molecules.append(mol)

            # Temperature update (every rollout — paper formula, Eq. 4-6)
            self.tau = update_temperature(self.tau, self.lambda_avg)

            # Weight update (every rollout — paper formula, Eq. 8-9)
            self.w_activity, self.w_solubility = update_weights(
                self.w_activity, self.w_solubility,
                self.sr_avg_activity, self.sr_avg_solubility,
            )

            # Periodic RL model training (every REPLAY_INTERVAL rollouts)
            if i > 0 and i % REPLAY_INTERVAL == 0:
                train_result = self._train_rl_models()
                loss = train_result.get("loss", 0.0)

                # Log stats
                recent = self.molecules[-100:] if len(self.molecules) >= 100 else self.molecules
                avg_z1 = np.mean([m.get("z1_val", 0) for m in recent]) if recent else 0
                avg_z2 = np.mean([m.get("z2_val", 0) for m in recent]) if recent else 0

                self.stats["steps"].append(i)
                self.stats["tau"].append(self.tau)
                self.stats["w_activity"].append(self.w_activity)
                self.stats["w_solubility"].append(self.w_solubility)
                self.stats["success_rate_activity"].append(self.sr_avg_activity)
                self.stats["success_rate_solubility"].append(self.sr_avg_solubility)
                self.stats["loss"].append(loss)
                self.stats["lambda_avg"].append(self.lambda_avg)
                self.stats["avg_z1"].append(avg_z1)
                self.stats["avg_z2"].append(avg_z2)

                progress.set_postfix({
                    "mol": len(self.molecules),
                    "tau": f"{self.tau:.3f}",
                    "w_a": f"{self.w_activity:.2f}",
                    "w_s": f"{self.w_solubility:.2f}",
                    "buf": self.buffer.size,
                })

            # Periodic checkpoint
            if i > 0 and i % (REPLAY_INTERVAL * 5) == 0:
                self.scorer._save_solubility_cache()
                self._save_checkpoint()
                self._save_results()

        # Final save
        self.scorer._save_solubility_cache()
        self._save_checkpoint()
        self._save_results()

        elapsed = time.time() - self.t_start
        print(f"\nGeneration complete: {len(self.molecules)} molecules in {elapsed/3600:.1f}h")
        print(f"  Final tau: {self.tau:.4f}")
        print(f"  Final weights: w_act={self.w_activity:.3f}, w_sol={self.w_solubility:.3f}")

        return self.molecules

    def _save_checkpoint(self):
        """Save full generation checkpoint."""
        self.models.save(self.checkpoint_path.with_suffix(".rl_models.pt"))
        ckpt = {
            "tau": self.tau,
            "w_activity": self.w_activity,
            "w_solubility": self.w_solubility,
            "generation_step": self.generation_step,
            "molecules": self.molecules[-1000:] if len(self.molecules) > 1000 else self.molecules,
            "stats": self.stats,
            "elapsed": time.time() - self.t_start,
        }
        torch.save(ckpt, self.checkpoint_path)
        print(f"  Checkpoint saved: {self.checkpoint_path}")

    def _load_checkpoint(self):
        """Load generation checkpoint."""
        try:
            models_path = self.checkpoint_path.with_suffix(".rl_models.pt")
            if models_path.exists():
                self.models.load(models_path)
            ckpt = torch.load(self.checkpoint_path, map_location=DEVICE, weights_only=False)
            self.tau = ckpt.get("tau", TAU0)
            self.w_activity = ckpt.get("w_activity", W_ACTIVITY_INIT)
            self.w_solubility = ckpt.get("w_solubility", W_SOLUBILITY_INIT)
            self.generation_step = ckpt.get("generation_step", 0)
            self.molecules = ckpt.get("molecules", [])
            self.stats = ckpt.get("stats", self.stats)
            print(f"  Loaded checkpoint: step={self.generation_step}, "
                  f"{len(self.molecules)} molecules, tau={self.tau:.4f}")
        except Exception as e:
            print(f"  WARNING: Could not load checkpoint: {e}")

    def _save_results(self):
        """Save generation results."""
        # Save molecules
        mol_file = SYNTHEMOL_DIR / "rl_generated_molecules.json"
        with open(mol_file, "w") as f:
            json.dump(self.molecules, f, indent=2)
        print(f"  Saved {len(self.molecules)} molecules to {mol_file}")

        # Save stats
        stats_file = SYNTHEMOL_DIR / "rl_generation_stats.json"
        with open(stats_file, "w") as f:
            json.dump(self.stats, f, indent=2)
        print(f"  Saved stats to {stats_file}")

        # Save summary CSV
        if self.molecules:
            df = pd.DataFrame(self.molecules)
            csv_file = SYNTHEMOL_DIR / "rl_generated_molecules.csv"
            df.to_csv(csv_file, index=False)
            print(f"  Saved CSV to {csv_file}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SyntheMol RL Generation (paper-matching)")
    parser.add_argument("--source", choices=["real", "commercial"], default="real",
                        help="Building block source")
    parser.add_argument("--n_rollout", type=int, default=N_ROLLOUT_DEFAULT,
                        help=f"Number of rollouts (default: {N_ROLLOUT_DEFAULT})")
    parser.add_argument("--load_checkpoint", action="store_true",
                        help="Load from checkpoint and resume")
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM,
                        help=f"MPNN hidden dim (default: {HIDDEN_DIM})")
    parser.add_argument("--depth", type=int, default=DEPTH,
                        help=f"MPNN depth (default: {DEPTH})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--tau0", type=float, default=TAU0,
                        help=f"Initial temperature (default: {TAU0})")
    parser.add_argument("--gamma", type=float, default=GAMMA,
                        help=f"Temperature decay (default: {GAMMA})")
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS,
                        help=f"Training epochs per replay (default: {N_EPOCHS})")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--solubility-model-dir", type=str, default=str(SOLUBILITY_MODEL_DIR),
                        help="Path to trained Chemprop solubility model directory")
    parser.add_argument("--w-solubility-init", type=float, default=W_SOLUBILITY_INIT,
                        help=f"Initial solubility weight (paper: 0.3, default: {W_SOLUBILITY_INIT})")
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"{'='*60}")
    print(f"SyntheMol RL Generation (Paper-Matching)")
    print(f"{'='*60}")
    print(f"  Source: {args.source}")
    print(f"  Rollouts: {args.n_rollout}")
    print(f"  Hidden dim: {args.hidden_dim}, Depth: {args.depth}")
    print(f"  LR: {args.lr}, tau_0: {args.tau0}, gamma: {args.gamma}")
    print(f"  Solubility weight init: {args.w_solubility_init}")
    print(f"  Device: {DEVICE}")

    # Load ALL 13 REAL reactions with their building block pools
    print("\nLoading building blocks...")
    reactions = load_building_blocks(source=args.source)

    # Generate scheme dicts from the loaded reactions
    schemes = get_reaction_schemes(reactions)
    print(f"  {len(schemes)} reaction scheme(s)")

    # Scorer (with optional solubility model)
    print("\nInitializing scorer...")
    sol_dir = Path(args.solubility_model_dir) if args.solubility_model_dir else None
    scorer = MolEScorer(solubility_model_dir=sol_dir)

    # Generator
    print("\nInitializing RL generator...")
    generator = RLGenerator(
        building_blocks=reactions,
        reaction_schemes=schemes,
        scorer=scorer,
        n_rollout=args.n_rollout,
        load_checkpoint=args.load_checkpoint,
        w_solubility_init=args.w_solubility_init,
    )

    # Override defaults with CLI args
    generator.tau = args.tau0

    # Run
    molecules = generator.run()

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    if molecules:
        activities = [m.get("activity", 0) for m in molecules]
        sols = [m.get("log_solubility", -10) for m in molecules]
        n_act = sum(1 for a in activities if a >= THRESHOLD_ACTIVITY)
        n_sol = sum(1 for s in sols if s >= THRESHOLD_SOLUBILITY)
        n_both = sum(1 for a, s in zip(activities, sols)
                      if a >= THRESHOLD_ACTIVITY and s >= THRESHOLD_SOLUBILITY)

        print(f"  Total molecules: {len(molecules)}")
        print(f"  Activity >= {THRESHOLD_ACTIVITY}: {n_act} ({100*n_act/len(molecules):.1f}%)")
        print(f"  Solubility >= {THRESHOLD_SOLUBILITY}: {n_sol} ({100*n_sol/len(molecules):.1f}%)")
        print(f"  Both criteria: {n_both} ({100*n_both/len(molecules):.1f}%)")
        print(f"  Mean activity: {np.mean(activities):.4f} +/- {np.std(activities):.4f}")
        print(f"  Mean solubility: {np.mean(sols):.4f} +/- {np.std(sols):.4f}")
    else:
        print("  No molecules generated!")

    print(f"\nResults saved to: {SYNTHEMOL_DIR}")
    print(f"Done in {time.time() - generator.t_start:.1f}s")


if __name__ == "__main__":
    main()
