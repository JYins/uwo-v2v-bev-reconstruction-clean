#!/usr/bin/env python3
"""
Optuna search for Pix2Pix adversarial weight with shared loss fixed.
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
from torch.amp import GradScaler, autocast

from dataset import get_dataloaders
from train import compute_shared_loss, evaluate, read_shared_loss_config, set_seed
from train_pix2pix import PatchDiscriminator
from unet import UNet


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = SCRIPT_DIR / "dataset_prepared"
DEFAULT_TUNING_ROOT = SCRIPT_DIR / "optuna_pix2pix_adv"
DEFAULT_SHARED_CONFIG = SCRIPT_DIR / "optuna_unet_run" / "best_params.json"


class EvalCfg:
    out_channels = 8

    def __init__(self, occ_threshold):
        self.occ_threshold = occ_threshold


def parse_args():
    p = argparse.ArgumentParser(description="Optuna search for Pix2Pix adversarial weight.")
    p.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--tuning_root", type=Path, default=DEFAULT_TUNING_ROOT)
    p.add_argument("--shared_config", type=Path, default=DEFAULT_SHARED_CONFIG)
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--g_lr", type=float, default=2e-4)
    p.add_argument("--d_lr", type=float, default=2e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--features", type=str, default="16,32,64,128")
    p.add_argument("--print_every", type=int, default=100)
    p.add_argument("--prune_after", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adv_min", type=float, default=0.1)
    p.add_argument("--adv_max", type=float, default=2.0)
    p.add_argument("--adv_step", type=float, default=0.1)
    p.add_argument("--enqueue_adv_values", type=str, default="0.1,0.5,1.0,2.0")
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def parse_features(text):
    return [int(part) for part in text.split(",") if part.strip()]


def parse_float_list(text):
    return [float(part) for part in text.split(",") if part.strip()]


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def objective(args, shared, trial):
    lambda_adv = trial.suggest_float("lambda_adv", args.adv_min, args.adv_max, step=args.adv_step)
    trial_root = args.tuning_root.resolve() / f"trial_{trial.number:03d}"
    results_dir = trial_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    train_loader, val_loader, _ = get_dataloaders(
        args.dataset_root.resolve(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    gen = UNet(in_channels=16, out_channels=8, features=parse_features(args.features)).to(device)
    disc = PatchDiscriminator(in_channels=24).to(device)

    g_opt = optim.Adam(gen.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    d_opt = optim.Adam(disc.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    g_scaler = GradScaler("cuda", enabled=use_amp)
    d_scaler = GradScaler("cuda", enabled=use_amp)

    best_occ_iou = -1.0
    best_rmse = float("inf")
    best_metrics = None
    best_epoch = None
    history = []
    t0 = time.time()
    eval_cfg = EvalCfg(shared["occ_threshold"])

    for epoch in range(1, args.epochs + 1):
        gen.train()
        disc.train()
        g_loss_sum = 0.0
        d_loss_sum = 0.0
        n = 0

        for step, (inp, tgt, mask) in enumerate(train_loader, start=1):
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            d_opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                fake = gen(inp)
                real_logit = disc(inp, tgt)
                fake_logit = disc(inp, fake.detach())
                d_loss = torch.relu(1.0 - real_logit).mean() + torch.relu(1.0 + fake_logit).mean()
            d_scaler.scale(d_loss).backward()
            d_scaler.step(d_opt)
            d_scaler.update()

            g_opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                fake = gen(inp)
                fake_logit = disc(inp, fake)
                g_adv = -fake_logit.mean()
                shared_loss, shared_parts = compute_shared_loss(
                    fake,
                    tgt,
                    mask,
                    shared["loss_l1_weight"],
                    shared["loss_mse_weight"],
                    shared["occ_weight"],
                    shared["occ_threshold"],
                    shared["occ_bce_weight"],
                    shared.get("occ_loss_type", "bce"),
                    shared.get("focal_gamma", 2.0),
                    shared.get("height_l1_weight", 0.0),
                    shared["occ_pos_weight"],
                    shared["occ_logit_temp"],
                )
                g_loss = lambda_adv * g_adv + shared_loss
            g_scaler.scale(g_loss).backward()
            g_scaler.step(g_opt)
            g_scaler.update()

            g_loss_sum += g_loss.item()
            d_loss_sum += d_loss.item()
            n += 1

            if step % args.print_every == 0:
                print(
                    f"trial {trial.number} epoch {epoch} step {step}/{len(train_loader)} "
                    f"g={g_loss_sum/n:.4f} d={d_loss_sum/n:.4f} "
                    f"adv={lambda_adv:.4f} "
                    f"shared_l1={shared_parts['l1']:.4f} "
                    f"shared_mse={shared_parts['mse']:.4f} "
                    f"shared_occ={shared_parts['occ']:.4f}"
                )

        val_metrics = evaluate(gen, val_loader, device, eval_cfg)
        row = {
            "trial": trial.number,
            "epoch": epoch,
            "seed": args.seed,
            "lambda_adv": lambda_adv,
            "g_loss": g_loss_sum / max(n, 1),
            "d_loss": d_loss_sum / max(n, 1),
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
            write_json(
                results_dir / "trial_result.json",
                {
                    "trial": trial.number,
                    "state": "pruned",
                    "seed": args.seed,
                    "lambda_adv": lambda_adv,
                    "best_epoch": best_epoch,
                    "best_val_metrics": best_metrics,
                    "time_minutes": (time.time() - t0) / 60.0,
                },
            )
            raise optuna.TrialPruned()

    with open(results_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    write_json(
        results_dir / "trial_result.json",
        {
            "trial": trial.number,
            "state": "complete",
            "seed": args.seed,
            "lambda_adv": lambda_adv,
            "best_epoch": best_epoch,
            "best_val_metrics": best_metrics,
            "time_minutes": (time.time() - t0) / 60.0,
        },
    )

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

    shared_blob = read_shared_loss_config(args.shared_config.resolve())
    shared = {
        "loss_l1_weight": shared_blob["loss_l1_weight"],
        "loss_mse_weight": shared_blob["loss_mse_weight"],
        "occ_bce_weight": shared_blob["occ_bce_weight"],
        "occ_weight": shared_blob["occ_weight"],
        "occ_pos_weight": shared_blob["occ_pos_weight"],
        "occ_threshold": shared_blob["occ_threshold"],
        "occ_logit_temp": shared_blob["occ_logit_temp"],
        "occ_loss_type": shared_blob.get("occ_loss_type", "bce"),
        "focal_gamma": shared_blob.get("focal_gamma", 2.0),
        "height_l1_weight": shared_blob.get("height_l1_weight", 0.0),
    }

    storage = f"sqlite:///{(args.tuning_root / 'optuna_study.db').as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=args.prune_after)
    study = optuna.create_study(
        study_name="pix2pix_adv_search",
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    for value in parse_float_list(args.enqueue_adv_values):
        study.enqueue_trial({"lambda_adv": value}, skip_if_exists=True)

    study.optimize(lambda trial: objective(args, shared, trial), n_trials=args.trials)

    best_trial = pick_best_trial(study)
    if best_trial is None:
        raise SystemExit("No completed Pix2Pix trials found.")

    summary = {
        "study_name": study.study_name,
        "seed": args.seed,
        "n_trials_requested": args.trials,
        "n_trials_total": len(study.trials),
        "shared_config": str(args.shared_config.resolve()),
        "selected_trial": best_trial.number,
        "selected_lambda_adv": best_trial.params["lambda_adv"],
        "selected_value_masked_occ_iou": best_trial.value,
        "selected_best_epoch": best_trial.user_attrs.get("best_epoch"),
        "selected_best_masked_rmse": best_trial.user_attrs.get("best_masked_rmse"),
        "selected_best_fused_full_psnr": best_trial.user_attrs.get("best_fused_full_psnr"),
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
                "lambda_adv": t.params.get("lambda_adv", ""),
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
