#!/usr/bin/env python3
"""
Optuna search for U-Net shared loss weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import optuna
import torch
import torch.optim as optim

from dataset import get_dataloaders
from train import Config, evaluate, parse_features, set_seed, train_one_epoch
from unet import UNet


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = SCRIPT_DIR / "dataset_prepared"
DEFAULT_TUNING_ROOT = SCRIPT_DIR / "optuna_unet"


def parse_args():
    p = argparse.ArgumentParser(description="Optuna search for U-Net loss weights.")
    p.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--tuning_root", type=Path, default=DEFAULT_TUNING_ROOT)
    p.add_argument("--trials", type=int, default=24)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--features", type=str, default="16,32,64,128")
    p.add_argument("--print_every", type=int, default=100)
    p.add_argument("--prune_after", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_cfg(args, params, training_root):
    cfg = Config()
    cfg.dataset_root = args.dataset_root.resolve()
    cfg.training_root = training_root.resolve()
    cfg.checkpoint_dir = cfg.training_root / "checkpoints"
    cfg.results_dir = cfg.training_root / "results"
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.num_workers = args.num_workers
    cfg.features = parse_features(args.features)
    cfg.print_every = args.print_every
    cfg.save_every = args.epochs + 1
    cfg.seed = args.seed
    cfg.loss_l1_weight = params["loss_l1_weight"]
    cfg.loss_mse_weight = 1.0 - params["loss_l1_weight"]
    cfg.occ_bce_weight = params["occ_bce_weight"]
    cfg.occ_weight = params["occ_weight"]
    cfg.occ_pos_weight = params["occ_pos_weight"]
    cfg.occ_threshold = params["occ_threshold"]
    cfg.occ_logit_temp = params["occ_logit_temp"]
    return cfg


def write_trial_result(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def objective(args, trial):
    params = {
        "loss_l1_weight": trial.suggest_float("loss_l1_weight", 0.55, 0.85),
        "occ_bce_weight": trial.suggest_float("occ_bce_weight", 0.10, 0.60),
        "occ_weight": trial.suggest_float("occ_weight", 2.0, 6.0),
        "occ_pos_weight": trial.suggest_int("occ_pos_weight", 6, 20),
        "occ_threshold": trial.suggest_categorical("occ_threshold", [0.03, 0.05, 0.07]),
        "occ_logit_temp": trial.suggest_categorical("occ_logit_temp", [0.01, 0.02, 0.03]),
    }

    trial_root = args.tuning_root.resolve() / f"trial_{trial.number:03d}"
    cfg = build_cfg(args, params, trial_root)

    set_seed(cfg.seed)
    train_loader, val_loader, _ = get_dataloaders(
        cfg.dataset_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=cfg.in_channels, out_channels=cfg.out_channels, features=cfg.features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    scaler = torch.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    best_metrics = None
    best_epoch = None
    best_occ_iou = -1.0
    best_rmse = float("inf")
    history = []
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch)
        val_metrics = evaluate(model, val_loader, device, cfg)
        scheduler.step(val_metrics["masked_rmse"])

        row = {
            "trial": trial.number,
            "epoch": epoch,
            "seed": cfg.seed,
            "train_loss": train_loss,
            "val_masked_occ_iou": val_metrics["masked_occ_iou"],
            "val_masked_rmse": val_metrics["masked_rmse"],
            "val_fused_full_psnr": val_metrics["fused_full_psnr"],
        }
        history.append(row)

        is_better = (
            (val_metrics["masked_occ_iou"] > best_occ_iou + 1e-12)
            or (
                abs(val_metrics["masked_occ_iou"] - best_occ_iou) <= 1e-12
                and val_metrics["masked_rmse"] < best_rmse - 1e-12
            )
        )
        if is_better:
            best_occ_iou = val_metrics["masked_occ_iou"]
            best_rmse = val_metrics["masked_rmse"]
            best_metrics = val_metrics
            best_epoch = epoch

        trial.report(best_occ_iou, step=epoch)
        if epoch >= args.prune_after and trial.should_prune():
            payload = {
                "trial": trial.number,
                "state": "pruned",
                "seed": cfg.seed,
                "params": params,
                "best_epoch": best_epoch,
                "best_val_metrics": best_metrics,
                "time_minutes": (time.time() - t0) / 60.0,
            }
            write_trial_result(trial_root / "results" / "trial_result.json", payload)
            raise optuna.TrialPruned()

    csv_path = trial_root / "results" / "training_history.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    payload = {
        "trial": trial.number,
        "state": "complete",
        "seed": cfg.seed,
        "params": params,
        "best_epoch": best_epoch,
        "best_val_metrics": best_metrics,
        "time_minutes": (time.time() - t0) / 60.0,
    }
    write_trial_result(trial_root / "results" / "trial_result.json", payload)

    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("best_masked_rmse", best_metrics["masked_rmse"])
    trial.set_user_attr("best_fused_full_psnr", best_metrics["fused_full_psnr"])
    return best_occ_iou


def pick_best_trial(study):
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top = sorted(done, key=lambda t: t.value if t.value is not None else -math.inf, reverse=True)[:5]
    top = sorted(
        top,
        key=lambda t: (
            -(t.value if t.value is not None else -math.inf),
            t.user_attrs.get("best_masked_rmse", math.inf),
            -t.user_attrs.get("best_fused_full_psnr", -math.inf),
        ),
    )
    return top[0] if top else None


def main():
    args = parse_args()
    args.tuning_root = args.tuning_root.resolve()
    args.tuning_root.mkdir(parents=True, exist_ok=True)

    storage = f"sqlite:///{(args.tuning_root / 'optuna_study.db').as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=args.prune_after)
    study = optuna.create_study(
        study_name="unet_loss_search",
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    study.optimize(lambda trial: objective(args, trial), n_trials=args.trials)

    best_trial = pick_best_trial(study)
    if best_trial is None:
        raise SystemExit("No completed trials found.")

    best_params = dict(best_trial.params)
    best_params["loss_mse_weight"] = 1.0 - best_params["loss_l1_weight"]

    summary = {
        "study_name": study.study_name,
        "seed": args.seed,
        "n_trials_requested": args.trials,
        "n_trials_total": len(study.trials),
        "selected_trial": best_trial.number,
        "selected_value_masked_occ_iou": best_trial.value,
        "selected_best_epoch": best_trial.user_attrs.get("best_epoch"),
        "selected_best_masked_rmse": best_trial.user_attrs.get("best_masked_rmse"),
        "selected_best_fused_full_psnr": best_trial.user_attrs.get("best_fused_full_psnr"),
        "shared_loss_config": best_params,
    }

    with open(args.tuning_root / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    rows = []
    for t in study.trials:
        rows.append(
            {
                "trial": t.number,
                "state": t.state.name,
                "value_masked_occ_iou": t.value if t.value is not None else "",
                "best_epoch": t.user_attrs.get("best_epoch", ""),
                "best_masked_rmse": t.user_attrs.get("best_masked_rmse", ""),
                "best_fused_full_psnr": t.user_attrs.get("best_fused_full_psnr", ""),
                "loss_l1_weight": t.params.get("loss_l1_weight", ""),
                "loss_mse_weight": (1.0 - t.params["loss_l1_weight"]) if "loss_l1_weight" in t.params else "",
                "occ_bce_weight": t.params.get("occ_bce_weight", ""),
                "occ_weight": t.params.get("occ_weight", ""),
                "occ_pos_weight": t.params.get("occ_pos_weight", ""),
                "occ_threshold": t.params.get("occ_threshold", ""),
                "occ_logit_temp": t.params.get("occ_logit_temp", ""),
            }
        )

    with open(args.tuning_root / "study_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {args.tuning_root / 'best_params.json'}")
    print(f"Saved: {args.tuning_root / 'study_results.csv'}")


if __name__ == "__main__":
    main()
