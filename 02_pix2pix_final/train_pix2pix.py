#!/usr/bin/env python3
"""
Pix2Pix training entry for BEV reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from dataset import get_dataloaders
from train import apply_shared_loss_overrides, compute_shared_loss, evaluate, read_shared_loss_config, set_seed
from unet import UNet


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset_prepared"
DEFAULT_TRAINING_ROOT = SCRIPT_DIR / "local_runs" / "training_pix2pix"


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 1),
        )

    def forward(self, cond, x):
        return self.net(torch.cat([cond, x], dim=1))


class EvalCfg:
    out_channels = 8
    occ_threshold = 0.05


def parse_args():
    p = argparse.ArgumentParser(description="Train Pix2Pix baseline for BEV reconstruction.")
    p.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--training_root", type=Path, default=DEFAULT_TRAINING_ROOT)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--g_lr", type=float, default=2e-4)
    p.add_argument("--d_lr", type=float, default=2e-4)
    p.add_argument("--num_workers", type=int, default=0 if os.name == "nt" else 4)
    p.add_argument("--lambda_adv", type=float, default=1.0)
    p.add_argument("--mask_variant", type=str, default="sector75", choices=["sector75", "front_rect", "front_blob"])
    p.add_argument("--preprocess_type", type=str, default="none", choices=["none", "register_layernorm"])
    p.add_argument("--registration_max_shift_px", type=int, default=24)
    p.add_argument("--shared_config", type=Path, default=None)
    p.add_argument("--loss_l1_weight", type=float, default=0.7)
    p.add_argument("--loss_mse_weight", type=float, default=0.3)
    p.add_argument("--occ_bce_weight", type=float, default=0.35)
    p.add_argument("--occ_loss_type", type=str, default="bce", choices=["bce", "focal", "bin_focal"])
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--height_l1_weight", type=float, default=0.0)
    p.add_argument("--occ_weight", type=float, default=3.0)
    p.add_argument("--occ_pos_weight", type=float, default=12.0)
    p.add_argument("--occ_logit_temp", type=float, default=0.02)
    p.add_argument("--occ_threshold", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.shared_config is not None:
        shared = read_shared_loss_config(args.shared_config)
        apply_shared_loss_overrides(args, shared, preserve_keys={"occ_loss_type", "focal_gamma", "height_l1_weight"})

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"

    train_root = args.training_root.resolve()
    ckpt_dir = train_root / "checkpoints"
    results_dir = train_root / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        args.dataset_root.resolve(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        mask_variant=args.mask_variant,
        preprocess_type=args.preprocess_type,
        registration_max_shift_px=args.registration_max_shift_px,
    )

    gen = UNet(in_channels=16, out_channels=8, features=[16, 32, 64, 128]).to(device)
    disc = PatchDiscriminator(in_channels=24).to(device)

    g_opt = optim.Adam(gen.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    d_opt = optim.Adam(disc.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    g_scaler = GradScaler("cuda", enabled=use_amp)
    d_scaler = GradScaler("cuda", enabled=use_amp)

    best_occ_iou = -1.0
    best_rmse = float("inf")
    best_epoch = None
    best_val_metrics = None
    history = []
    t_start = time.time()

    print("\nPix2Pix training")
    print(
        f"device={device} epochs={args.epochs} batch={args.batch_size} "
        f"seed={args.seed} amp={use_amp} "
        f"mask={args.mask_variant} preprocess={args.preprocess_type}"
    )

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

            # D step (hinge)
            d_opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                fake = gen(inp)
                real_logit = disc(inp, tgt)
                fake_logit = disc(inp, fake.detach())
                d_loss = torch.relu(1.0 - real_logit).mean() + torch.relu(1.0 + fake_logit).mean()
            d_scaler.scale(d_loss).backward()
            d_scaler.step(d_opt)
            d_scaler.update()

            # G step
            g_opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                fake = gen(inp)
                fake_logit = disc(inp, fake)
                g_adv = -fake_logit.mean()
                shared_loss, shared_parts = compute_shared_loss(
                    fake,
                    tgt,
                    mask,
                    args.loss_l1_weight,
                    args.loss_mse_weight,
                    args.occ_weight,
                    args.occ_threshold,
                    args.occ_bce_weight,
                    args.occ_loss_type,
                    args.focal_gamma,
                    args.height_l1_weight,
                    args.occ_pos_weight,
                    args.occ_logit_temp,
                )
                g_loss = args.lambda_adv * g_adv + shared_loss
            g_scaler.scale(g_loss).backward()
            g_scaler.step(g_opt)
            g_scaler.update()

            g_loss_sum += g_loss.item()
            d_loss_sum += d_loss.item()
            n += 1

            if step % args.print_every == 0:
                print(
                    f"  epoch {epoch} step {step}/{len(train_loader)} "
                    f"g={g_loss_sum/n:.4f} d={d_loss_sum/n:.4f} "
                    f"shared_l1={shared_parts['l1']:.4f} "
                    f"shared_mse={shared_parts['mse']:.4f} "
                    f"shared_occ={shared_parts['occ']:.4f} "
                    f"shared_height={shared_parts['height']:.4f}"
                )

        val_metrics = evaluate(gen, val_loader, device, EvalCfg()) if len(val_loader) > 0 else None
        row = {
            "epoch": epoch,
            "seed": args.seed,
            "g_loss": g_loss_sum / max(n, 1),
            "d_loss": d_loss_sum / max(n, 1),
            "val_masked_mae": val_metrics["masked_mae"] if val_metrics else "",
            "val_masked_rmse": val_metrics["masked_rmse"] if val_metrics else "",
            "val_masked_psnr": val_metrics["masked_psnr"] if val_metrics else "",
            "val_masked_occ_iou": val_metrics["masked_occ_iou"] if val_metrics else "",
            "val_fused_full_psnr": val_metrics["fused_full_psnr"] if val_metrics else "",
            "val_fused_full_occ_iou": val_metrics["fused_full_occ_iou"] if val_metrics else "",
            "val_full_psnr_raw": val_metrics["full_psnr"] if val_metrics else "",
        }
        history.append(row)

        if val_metrics:
            print(
                f"  epoch {epoch} done | "
                f"val IoU={val_metrics['masked_occ_iou']:.4f} "
                f"val RMSE={val_metrics['masked_rmse']:.6f} "
                f"fusedPSNR={val_metrics['fused_full_psnr']:.2f}"
            )
        is_better = (
            val_metrics
            and (
                (val_metrics["masked_occ_iou"] > best_occ_iou + 1e-12)
                or (
                    abs(val_metrics["masked_occ_iou"] - best_occ_iou) <= 1e-12
                    and val_metrics["masked_rmse"] < best_rmse - 1e-12
                )
            )
        )
        if is_better:
            best_occ_iou = val_metrics["masked_occ_iou"]
            best_rmse = val_metrics["masked_rmse"]
            best_epoch = epoch
            best_val_metrics = val_metrics
            torch.save(
                {
                    "epoch": epoch,
                    "generator": gen.state_dict(),
                    "discriminator": disc.state_dict(),
                    "val_metrics": val_metrics,
                },
                ckpt_dir / "best_pix2pix.pth",
            )
            print(f"  >>> best checkpoint epoch {epoch}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "generator": gen.state_dict(),
                    "discriminator": disc.state_dict(),
                },
                ckpt_dir / f"pix2pix_epoch_{epoch:03d}.pth",
            )

    total_min = (time.time() - t_start) / 60.0
    test_metrics = None

    best_path = ckpt_dir / "best_pix2pix.pth"
    if best_path.exists() and len(test_loader) > 0:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        gen.load_state_dict(ckpt["generator"])
        test_metrics = evaluate(gen, test_loader, device, EvalCfg())

    csv_path = results_dir / "training_history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "g_lr": args.g_lr,
            "d_lr": args.d_lr,
            "lambda_adv": args.lambda_adv,
            "mask_variant": args.mask_variant,
            "preprocess_type": args.preprocess_type,
            "registration_max_shift_px": args.registration_max_shift_px,
            "seed": args.seed,
            "shared_config": str(args.shared_config) if args.shared_config else None,
            "loss_l1_weight": args.loss_l1_weight,
            "loss_mse_weight": args.loss_mse_weight,
            "occ_bce_weight": args.occ_bce_weight,
            "occ_loss_type": args.occ_loss_type,
            "focal_gamma": args.focal_gamma,
            "height_l1_weight": args.height_l1_weight,
            "occ_weight": args.occ_weight,
            "occ_pos_weight": args.occ_pos_weight,
            "occ_logit_temp": args.occ_logit_temp,
            "occ_threshold": args.occ_threshold,
            "amp": use_amp,
        },
        "time_minutes": total_min,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }
    with open(results_dir / "pix2pix_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(results_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("done")
    print(f"saved {csv_path}")
    print(f"saved {results_dir / 'pix2pix_summary.json'}")


if __name__ == "__main__":
    main()
