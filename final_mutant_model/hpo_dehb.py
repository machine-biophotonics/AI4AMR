#!/usr/bin/env python3
"""
DEHB hyperparameter optimization for MIL training on metabolomics mutant data.

Proxy fold: train P2+P3, val P4  (test_plate P1 -> cyclic val P4, train P2,P3).
Objective: maximize validation accuracy -> fitness = -best_val_acc.
Budget (fidelity) = number of SC-MIL epochs (min 5, max 100). Early stopping patience 5.
Uses 2 L40 GPUs via single_node_with_gpus; Dask workers each run a training subprocess.

Checkpointing:
  - DEHB saves state/history/incumbent to output_path after every step (save_freq="step").
  - Resume with --resume (replays history, skips already-evaluated trials).
  - A human-readable trials.csv is appended after every trial.
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np

SCRIPT_DIR = os.path.expanduser("~/scripts")
RESULTS_ROOT = os.path.expanduser("/data/42-julia-hpc-bio-mbl/s529821/results/hpo_dehb")
DATA_ROOT = "/data/42-julia-hpc-bio-mbl/s529821/datasets/metabolomics/Mutants"
TRIALS_DIR = os.path.join(RESULTS_ROOT, "trials")
os.makedirs(TRIALS_DIR, exist_ok=True)

EARLY_STOPPING_PATIENCE = 5
BATCH_SIZE = 96
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
WALL_SECONDS = 22.5 * 3600  # leave ~30 min margin inside the 23h slurm limit

FIXED_ARGS = [
    "--data_mode", "metabolomics_mutant",
    "--data_root", DATA_ROOT,
    "--backbone", "efficientnet_b0",
    "--pretrained", "micronet",
    "--num_channels", "1",
    "--use_sc_mil",
    "--test_plate", "P1",  # proxy: train P2+P3, val P4
    "--batch_size", str(BATCH_SIZE),
    "--num_workers", str(NUM_WORKERS),
    "--prefetch_factor", str(PREFETCH_FACTOR),
    "--early_stopping_patience", str(EARLY_STOPPING_PATIENCE),
    "--checkpoint_every", "1000000",
    "--skip_test",
]


def build_configspace():
    cs = CS.ConfigurationSpace(seed=42)
    cs.add_hyperparameters([
        CSH.UniformFloatHyperparameter("lr", 1e-5, 1e-3, log=True, default_value=1e-4),
        CSH.UniformFloatHyperparameter("weight_decay", 1e-5, 1e-1, log=True, default_value=5e-2),
        CSH.UniformFloatHyperparameter("dropout", 0.1, 0.7, default_value=0.5),
        CSH.UniformFloatHyperparameter("label_smoothing", 0.0, 0.3, default_value=0.1),
        CSH.UniformFloatHyperparameter("entropy_loss_weight", 1e-4, 1e-1, log=True, default_value=1e-2),
        CSH.UniformFloatHyperparameter("sc_mil_weight", 0.1, 0.9, default_value=0.3),
        CSH.UniformFloatHyperparameter("sc_mil_temp", 2e-2, 0.5, log=True, default_value=7e-2),
        CSH.UniformFloatHyperparameter("instance_weight", 0.0, 1.0, default_value=0.5),
        CSH.CategoricalHyperparameter("contrastive_level", ["instance", "bag", "both"], default_value="both"),
        CSH.UniformFloatHyperparameter("attention_temp", 0.1, 1.0, default_value=0.5),
        CSH.UniformIntegerHyperparameter("num_heads", 1, 8, default_value=4),
        CSH.UniformIntegerHyperparameter("attn_hidden_dim", 128, 512, log=True, default_value=256),
        CSH.UniformIntegerHyperparameter("classifier_hidden_dim", 256, 2048, log=True, default_value=512),
        CSH.UniformIntegerHyperparameter("classifier_layers", 0, 3, default_value=0),
        CSH.CategoricalHyperparameter("pooling", ["attention", "simple_attention"], default_value="attention"),
    ])
    return cs


def config_to_args(config: dict) -> list:
    m = {
        "lr": "--lr", "weight_decay": "--weight_decay", "dropout": "--dropout",
        "label_smoothing": "--label_smoothing", "entropy_loss_weight": "--entropy_loss_weight",
        "sc_mil_weight": "--sc_mil_weight", "sc_mil_temp": "--sc_mil_temp",
        "instance_weight": "--instance_weight", "contrastive_level": "--contrastive_level",
        "attention_temp": "--attention_temp", "num_heads": "--num_heads",
        "attn_hidden_dim": "--attn_hidden_dim", "classifier_hidden_dim": "--classifier_hidden_dim",
        "classifier_layers": "--classifier_layers", "pooling": "--pooling",
    }
    args = []
    for key, flag in m.items():
        args.extend([flag, str(config[key])])
    return args


def parse_best_val_acc(trial_dir: str) -> float | None:
    csv_files = glob.glob(os.path.join(trial_dir, "**", "training_sc_mil_*.csv"), recursive=True)
    if not csv_files:
        return None
    best = None
    for f in csv_files:
        path = os.path.join(trial_dir, f)
        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header is None:
                continue
            try:
                idx = header.index("val_acc")
            except ValueError:
                continue
            for row in reader:
                if len(row) <= idx:
                    continue
                try:
                    v = float(row[idx])
                except ValueError:
                    continue
                best = v if best is None else max(best, v)
    return best


def append_trial_row(config_id: str, fidelity: int, config: dict, val_acc: float, fitness: float, cost: float, status: str):
    path = os.path.join(RESULTS_ROOT, "trials.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["config_id", "fidelity", "status", "val_acc", "fitness", "cost_s"] + sorted(config.keys()))
        writer.writerow([config_id, fidelity, status, val_acc, fitness, round(cost, 1)] + [config[k] for k in sorted(config.keys())])


def objective(config, fidelity=None, **kwargs):
    cfg = config.get_dictionary()
    # DEHB 0.1.2 does not pass config_id to f, so derive a deterministic unique id
    # from the config itself to avoid trial-dir collisions between workers.
    config_hash = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    trial_dir = os.path.join(TRIALS_DIR, f"{config_hash}_{int(fidelity)}")
    os.makedirs(trial_dir, exist_ok=True)

    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "train_mil.py"),
           "--output_dir", trial_dir,
           "--sc_mil_epochs", str(int(fidelity))]
    cmd += FIXED_ARGS
    cmd += config_to_args(cfg)

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    val_acc = parse_best_val_acc(trial_dir)
    if result.returncode != 0:
        status = "error"
        val_acc = None
    elif val_acc is None:
        status = "no_metrics"
    else:
        status = "ok"

    if val_acc is None:
        fitness = 1000.0  # worse than any achievable fitness
        val_acc_out = float("nan")
    else:
        fitness = float(-val_acc)
        val_acc_out = float(val_acc)

    append_trial_row(config_hash, int(fidelity), cfg, val_acc_out, fitness, elapsed, status)

    with open(os.path.join(trial_dir, "trial_result.json"), "w") as fh:
        json.dump({"config_id": config_hash, "fidelity": int(fidelity), "config": cfg,
                   "val_acc": val_acc_out, "fitness": fitness, "cost": elapsed,
                   "returncode": result.returncode, "status": status}, fh, indent=2)

    print(f"[HPO] {status.upper()} config_id={config_hash} fidelity={fidelity} "
          f"val_acc={val_acc_out:.2f} cost={elapsed:.0f}s cfg={json.dumps(cfg, sort_keys=True)}", flush=True)

    return {"fitness": fitness, "cost": elapsed, "info": {"val_acc": val_acc_out, "status": status, "trial_dir": trial_dir}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", default=False)
    ap.add_argument("--fevals", type=int, default=None)
    args = ap.parse_args()

    from dehb import DEHB

    cs = build_configspace()
    dehb = DEHB(
        f=objective, cs=cs, min_fidelity=5, max_fidelity=100, eta=3,
        n_workers=2, output_path=RESULTS_ROOT, save_freq="step",
        resume=args.resume, seed=42,
    )
    print(f"[HPO] DEHB started; workers=2, min_fidelity=5, max_fidelity=100, eta=3", flush=True)
    print(f"[HPO] Wall budget: {WALL_SECONDS}s; output: {RESULTS_ROOT}", flush=True)
    traj, runtime, history = dehb.run(fevals=args.fevals, total_cost=WALL_SECONDS,
                                      single_node_with_gpus=True)
    print(f"[HPO] Done. {len(traj)} evaluations. Final incumbent: {dehb.inc_score}", flush=True)


if __name__ == "__main__":
    main()
