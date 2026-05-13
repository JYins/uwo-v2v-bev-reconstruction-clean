#!/usr/bin/env python3
"""
Conditional diffusion-style training entry for BEV reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast

from dataset import get_dataloaders
from train import (
    apply_shared_loss_overrides,
    batch_metric_sums,
    compute_shared_loss,
    finalize_metrics,
    init_metric_acc,
    merge_metric_acc,
    read_shared_loss_config,
    set_seed,
)
from unet import UNet


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset_prepared"
DEFAULT_TRAINING_ROOT = SCRIPT_DIR / "local_runs" / "training_diffusion"


def parse_args():
    p = argparse.ArgumentParser(description="Train diffusion-style BEV recon baseline.")
    p.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--training_root", type=Path, default=DEFAULT_TRAINING_ROOT)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=5e-6)
    p.add_argument("--num_workers", type=int, default=0 if os.name == "nt" else 4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--max_train_steps", type=int, default=0)
    p.add_argument("--max_eval_batches", type=int, default=0)
    p.add_argument("--mask_variant", type=str, default="sector75", choices=["sector75", "front_rect", "front_blob"])
    p.add_argument(
        "--neighbor_preprocess",
        type=str,
        default="none",
        choices=["none", "register", "register_layernorm"],
    )
    p.add_argument("--registration_max_shift_px", type=int, default=24)
    p.add_argument("--shared_config", type=Path, default=None)
    p.add_argument("--noise_loss_weight", type=float, default=1.0)
    p.add_argument("--shared_loss_weight", type=float, default=1.0)
    p.add_argument("--empty_penalty_weight", type=float, default=0.0)
    p.add_argument("--loss_l1_weight", type=float, default=0.7)
    p.add_argument("--loss_mse_weight", type=float, default=0.3)
    p.add_argument("--occ_bce_weight", type=float, default=0.0)
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
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def alpha_bar_schedule(timesteps, device):
    beta = torch.linspace(1e-4, 2e-2, timesteps, device=device)
    alpha = 1.0 - beta
    return torch.cumprod(alpha, dim=0)


def q_sample(x0, t_idx, noise, alpha_bar):
    # x_t = sqrt(a_bar)*x0 + sqrt(1-a_bar)*noise
    a = alpha_bar[t_idx].view(-1, 1, 1, 1)
    return torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * noise


def estimate_x0(x_t, pred_noise, t_idx, alpha_bar):
    a = alpha_bar[t_idx].view(-1, 1, 1, 1)
    return (x_t - torch.sqrt(1.0 - a) * pred_noise) / torch.sqrt(a)


def masked_noise_l1(pred_noise, noise, mask):
    occluded = 1.0 - mask
    denom = (occluded.sum() * pred_noise.shape[1]).clamp(min=1.0)
    return (torch.abs(pred_noise - noise) * occluded).sum() / denom


def masked_empty_density_l1(pred, target, mask, occ_threshold):
    occluded = 1.0 - mask
    target_occ = (target[:, :4].sum(dim=1, keepdim=True) > occ_threshold).float()
    empty = occluded * (1.0 - target_occ)
    denom = (empty.sum() * pred.shape[1]).clamp(min=1.0)
    return (torch.abs(pred[:, :4]) * empty).sum() / denom


def make_condition(inp, mask):
    # Diffusion is an inpainting model: the denoiser needs to know which pixels
    # are fixed context and which pixels it is allowed to generate.
    return torch.cat([inp, mask], dim=1)


def make_sampling_schedule(timesteps, sample_steps):
    sample_steps = max(1, min(int(sample_steps), int(timesteps)))
    if sample_steps == timesteps:
        return list(range(timesteps - 1, -1, -1))
    step_size = timesteps / sample_steps
    schedule = []
    for idx in range(sample_steps):
        t = int(round(timesteps - 1 - idx * step_size))
        t = min(max(t, 0), timesteps - 1)
        if not schedule or t != schedule[-1]:
            schedule.append(t)
    if schedule[-1] != 0:
        schedule.append(0)
    return schedule


@torch.no_grad()
def ddim_sample(model, cond, mask, alpha_bar, timesteps, sample_steps):
    bsz = cond.shape[0]
    device = cond.device
    sample_schedule = make_sampling_schedule(timesteps, sample_steps)
    x = torch.randn(bsz, 8, cond.shape[2], cond.shape[3], device=device)
    visible = cond[:, :8]
    hidden = 1.0 - mask

    for idx, t_now in enumerate(sample_schedule):
        t_batch = torch.full((bsz,), t_now, device=device, dtype=torch.long)
        x = x * hidden + visible * mask
        pred_noise = model(torch.cat([x, cond], dim=1), timesteps=t_batch)
        x0_hat = estimate_x0(x, pred_noise, t_batch, alpha_bar).clamp(0.0, 1.0)
        x0_hat = x0_hat * hidden + visible * mask
        if idx == len(sample_schedule) - 1:
            x = x0_hat
            break

        t_prev = sample_schedule[idx + 1]
        a_prev = alpha_bar[t_prev].view(-1, 1, 1, 1)
        x = torch.sqrt(a_prev) * x0_hat + torch.sqrt(1.0 - a_prev) * pred_noise
        x = x * hidden + visible * mask

    return (x.clamp(0.0, 1.0) * hidden + visible * mask).clamp(0.0, 1.0)


def init_diffusion_diag_acc():
    return {
        "hidden_pixels": 0.0,
        "pred_occ_sum": 0.0,
        "target_occ_sum": 0.0,
        "empty_pixels": 0.0,
        "empty_false_positive_sum": 0.0,
        "pred_hidden_sum": 0.0,
        "target_hidden_sum": 0.0,
    }


@torch.no_grad()
def merge_diffusion_diag(acc, pred, target, mask, occ_threshold):
    hidden = 1.0 - mask
    hidden_bool = hidden > 0.5
    pred_occ = pred[:, :4].sum(dim=1, keepdim=True) > occ_threshold
    target_occ = target[:, :4].sum(dim=1, keepdim=True) > occ_threshold
    empty = (~target_occ) & hidden_bool

    hidden_pixels = hidden_bool.sum().item()
    empty_pixels = empty.sum().item()
    acc["hidden_pixels"] += hidden_pixels
    acc["pred_occ_sum"] += (pred_occ & hidden_bool).sum().item()
    acc["target_occ_sum"] += (target_occ & hidden_bool).sum().item()
    acc["empty_pixels"] += empty_pixels
    acc["empty_false_positive_sum"] += (pred_occ & empty).sum().item()
    acc["pred_hidden_sum"] += (pred * hidden).sum().item()
    acc["target_hidden_sum"] += (target * hidden).sum().item()


def finalize_diffusion_diag(acc, n_channels):
    hidden_pixels = max(acc["hidden_pixels"], 1.0)
    empty_pixels = max(acc["empty_pixels"], 1.0)
    hidden_values = max(acc["hidden_pixels"] * n_channels, 1.0)
    return {
        "pred_occ_rate": acc["pred_occ_sum"] / hidden_pixels,
        "target_occ_rate": acc["target_occ_sum"] / hidden_pixels,
        "empty_false_positive_rate": acc["empty_false_positive_sum"] / empty_pixels,
        "pred_hidden_mean": acc["pred_hidden_sum"] / hidden_values,
        "target_hidden_mean": acc["target_hidden_sum"] / hidden_values,
    }


@torch.no_grad()
def evaluate_sampled(
    model,
    loader,
    device,
    timesteps,
    sample_steps,
    alpha_bar,
    occ_threshold,
    max_batches=0,
):
    model.eval()
    n_channels = 8
    acc = init_metric_acc(n_channels)
    diag_acc = init_diffusion_diag_acc()

    for batch_idx, (inp, tgt, mask) in enumerate(loader, start=1):
        inp = inp.to(device, non_blocking=True)
        x0 = tgt.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        cond = make_condition(inp, mask)
        x0_hat = ddim_sample(model, cond, mask, alpha_bar, timesteps, sample_steps)

        batch = batch_metric_sums(
            x0_hat,
            x0,
            mask,
            occ_threshold,
            visible_bev=inp[:, :8],
        )
        merge_metric_acc(acc, batch)
        merge_diffusion_diag(diag_acc, x0_hat, x0, mask, occ_threshold)
        if max_batches > 0 and batch_idx >= max_batches:
            break

    metrics = finalize_metrics(acc, n_channels)
    metrics.update(finalize_diffusion_diag(diag_acc, n_channels))
    return metrics


def maybe_load_resume(path, model, opt, scaler, device):
    if path is None:
        return 1, -1.0, float("inf"), None, None, []

    resume_path = Path(path)
    if not resume_path.exists():
        print(f"resume checkpoint not found, starting fresh: {resume_path}")
        return 1, -1.0, float("inf"), None, None, []

    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    if "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_occ_iou = float(ckpt.get("best_occ_iou", -1.0))
    best_rmse = float(ckpt.get("best_rmse", float("inf")))
    best_epoch = ckpt.get("best_epoch")
    best_val_metrics = ckpt.get("best_val_metrics")
    history = ckpt.get("history", [])
    return start_epoch, best_occ_iou, best_rmse, best_epoch, best_val_metrics, history


def save_training_state(
    path,
    *,
    epoch,
    model,
    opt,
    scaler,
    best_occ_iou,
    best_rmse,
    best_epoch,
    best_val_metrics,
    history,
):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "best_occ_iou": best_occ_iou,
            "best_rmse": best_rmse,
            "best_epoch": best_epoch,
            "best_val_metrics": best_val_metrics,
            "history": history,
        },
        path,
    )


def compute_epoch_lr(epoch, total_epochs, base_lr, min_lr, warmup_epochs):
    total_epochs = max(int(total_epochs), 1)
    warmup_epochs = max(int(warmup_epochs), 0)
    base_lr = float(base_lr)
    min_lr = float(min_lr)

    if total_epochs == 1:
        return base_lr

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        warmup_progress = epoch / warmup_epochs
        return min_lr + (base_lr - min_lr) * warmup_progress

    if total_epochs <= warmup_epochs:
        return base_lr

    decay_progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    decay_progress = min(max(decay_progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr + (base_lr - min_lr) * cosine


def set_optimizer_lr(optimizer, lr_value):
    for group in optimizer.param_groups:
        group["lr"] = lr_value


def main():
    args = parse_args()
    if args.shared_config is not None:
        shared = read_shared_loss_config(args.shared_config)
        apply_shared_loss_overrides(
            args,
            shared,
            preserve_keys={"occ_bce_weight", "occ_loss_type", "focal_gamma", "height_l1_weight"},
        )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"

    train_root = args.training_root.resolve()
    ckpt_dir = train_root / "checkpoints"
    results_dir = train_root / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = ckpt_dir / "last_diffusion.pth"

    train_loader, val_loader, test_loader = get_dataloaders(
        args.dataset_root.resolve(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        mask_variant=args.mask_variant,
        neighbor_preprocess=args.neighbor_preprocess,
        registration_max_shift_px=args.registration_max_shift_px,
    )

    # input: x_t (8ch) + condition (masked ego 8ch + neighbor 8ch + mask 1ch) = 25 channels
    model = UNet(
        in_channels=25,
        out_channels=8,
        features=[16, 32, 64, 128],
        time_emb_dim=128,
        final_activation="identity",
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler("cuda", enabled=use_amp)

    alpha_bar = alpha_bar_schedule(args.timesteps, device)
    resume_path = args.resume if args.resume is not None else (last_ckpt_path if last_ckpt_path.exists() else None)
    start_epoch, best_occ_iou, best_rmse, best_epoch, best_val_metrics, history = maybe_load_resume(
        resume_path,
        model,
        opt,
        scaler,
        device,
    )
    start = time.time()

    print("\nDiffusion-style training")
    print(
        f"device={device} epochs={args.epochs} batch={args.batch_size} "
        f"seed={args.seed} amp={use_amp} sample_steps={args.sample_steps} "
        f"val_every={args.val_every} start_epoch={start_epoch} "
        f"lr={args.lr} min_lr={args.min_lr} warmup={args.warmup_epochs} "
        f"grad_clip={args.grad_clip} mask={args.mask_variant} "
        f"neighbor_preprocess={args.neighbor_preprocess}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        current_lr = compute_epoch_lr(
            epoch,
            args.epochs,
            args.lr,
            args.min_lr,
            args.warmup_epochs,
        )
        set_optimizer_lr(opt, current_lr)

        for step, (inp, tgt, mask) in enumerate(train_loader, start=1):
            inp = inp.to(device, non_blocking=True)    # 16ch
            x0 = tgt.to(device, non_blocking=True)     # 8ch
            mask = mask.to(device, non_blocking=True)
            cond = make_condition(inp, mask)           # 17ch

            t_idx = torch.randint(0, args.timesteps, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            x_t = q_sample(x0, t_idx, noise, alpha_bar)
            model_in = torch.cat([x_t, cond], dim=1)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                pred_noise = model(model_in, timesteps=t_idx)
                noise_loss = masked_noise_l1(pred_noise, noise, mask)
                x0_hat = estimate_x0(x_t, pred_noise, t_idx, alpha_bar).clamp(0.0, 1.0)
                shared_loss, shared_parts = compute_shared_loss(
                    x0_hat,
                    x0,
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
                empty_penalty = masked_empty_density_l1(
                    x0_hat,
                    x0,
                    mask,
                    args.occ_threshold,
                )
                loss = (
                    args.noise_loss_weight * noise_loss
                    + args.shared_loss_weight * shared_loss
                    + args.empty_penalty_weight * empty_penalty
                )
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            n += 1

            if step % args.print_every == 0:
                print(
                    f"  epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss={total_loss/n:.6f} noise={float(noise_loss.item()):.4f} "
                    f"lr={current_lr:.2e} "
                    f"shared_l1={shared_parts['l1']:.4f} "
                    f"shared_mse={shared_parts['mse']:.4f} "
                    f"shared_occ={shared_parts['occ']:.4f} "
                    f"shared_height={shared_parts['height']:.4f} "
                    f"empty={float(empty_penalty.item()):.4f}"
                )
            if args.max_train_steps > 0 and step >= args.max_train_steps:
                break

        should_validate = len(val_loader) > 0 and (epoch % args.val_every == 0 or epoch == args.epochs)
        val_metrics = evaluate_sampled(
            model,
            val_loader,
            device,
            args.timesteps,
            args.sample_steps,
            alpha_bar,
            occ_threshold=args.occ_threshold,
            max_batches=args.max_eval_batches,
        ) if should_validate else None

        row = {
            "epoch": epoch,
            "seed": args.seed,
            "lr": current_lr,
            "train_loss": total_loss / max(n, 1),
            "val_masked_mae": val_metrics["masked_mae"] if val_metrics else "",
            "val_masked_rmse": val_metrics["masked_rmse"] if val_metrics else "",
            "val_masked_psnr": val_metrics["masked_psnr"] if val_metrics else "",
            "val_masked_occ_iou": val_metrics["masked_occ_iou"] if val_metrics else "",
            "val_masked_occ_precision": val_metrics["masked_occ_precision"] if val_metrics else "",
            "val_masked_occ_recall": val_metrics["masked_occ_recall"] if val_metrics else "",
            "val_masked_occ_f1": val_metrics["masked_occ_f1"] if val_metrics else "",
            "val_fused_full_psnr": val_metrics["fused_full_psnr"] if val_metrics else "",
            "val_fused_full_occ_iou": val_metrics["fused_full_occ_iou"] if val_metrics else "",
            "val_full_psnr_raw": val_metrics["full_psnr"] if val_metrics else "",
            "val_pred_occ_rate": val_metrics["pred_occ_rate"] if val_metrics else "",
            "val_target_occ_rate": val_metrics["target_occ_rate"] if val_metrics else "",
            "val_empty_false_positive_rate": val_metrics["empty_false_positive_rate"] if val_metrics else "",
            "val_pred_hidden_mean": val_metrics["pred_hidden_mean"] if val_metrics else "",
            "val_target_hidden_mean": val_metrics["target_hidden_mean"] if val_metrics else "",
        }
        history.append(row)

        if val_metrics:
            print(
                f"  epoch {epoch} done | "
                f"val IoU={val_metrics['masked_occ_iou']:.4f} "
                f"prec={val_metrics['masked_occ_precision']:.4f} "
                f"rec={val_metrics['masked_occ_recall']:.4f} "
                f"val RMSE={val_metrics['masked_rmse']:.6f} "
                f"fusedPSNR={val_metrics['fused_full_psnr']:.2f} "
                f"predOcc={val_metrics['pred_occ_rate']:.3f} "
                f"emptyFP={val_metrics['empty_false_positive_rate']:.3f}"
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
            save_training_state(
                ckpt_dir / "best_diffusion.pth",
                epoch=epoch,
                model=model,
                opt=opt,
                scaler=scaler,
                best_occ_iou=best_occ_iou,
                best_rmse=best_rmse,
                best_epoch=best_epoch,
                best_val_metrics=best_val_metrics,
                history=history,
            )
            print(f"  >>> best checkpoint epoch {epoch}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_training_state(
                ckpt_dir / f"diffusion_epoch_{epoch:03d}.pth",
                epoch=epoch,
                model=model,
                opt=opt,
                scaler=scaler,
                best_occ_iou=best_occ_iou,
                best_rmse=best_rmse,
                best_epoch=best_epoch,
                best_val_metrics=best_val_metrics,
                history=history,
            )
        save_training_state(
            last_ckpt_path,
            epoch=epoch,
            model=model,
            opt=opt,
            scaler=scaler,
            best_occ_iou=best_occ_iou,
            best_rmse=best_rmse,
            best_epoch=best_epoch,
            best_val_metrics=best_val_metrics,
            history=history,
        )

    total_min = (time.time() - start) / 60.0
    test_metrics = None
    best_path = ckpt_dir / "best_diffusion.pth"

    if best_path.exists() and len(test_loader) > 0:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        test_metrics = evaluate_sampled(
            model,
            test_loader,
            device,
            args.timesteps,
            args.sample_steps,
            alpha_bar,
            occ_threshold=args.occ_threshold,
            max_batches=args.max_eval_batches,
        )

    csv_path = results_dir / "training_history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "timesteps": args.timesteps,
            "sample_steps": args.sample_steps,
            "val_every": args.val_every,
            "warmup_epochs": args.warmup_epochs,
            "max_train_steps": args.max_train_steps,
            "max_eval_batches": args.max_eval_batches,
            "mask_variant": args.mask_variant,
            "neighbor_preprocess": args.neighbor_preprocess,
            "registration_max_shift_px": args.registration_max_shift_px,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
            "resume": str(resume_path) if resume_path else None,
            "shared_config": str(args.shared_config) if args.shared_config else None,
            "noise_loss_weight": args.noise_loss_weight,
            "shared_loss_weight": args.shared_loss_weight,
            "empty_penalty_weight": args.empty_penalty_weight,
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
            "denoiser_final_activation": "identity",
            "condition_channels": 17,
            "model_input_channels": 25,
            "inpainting_visible_region": "preserved at every DDIM step",
        },
        "time_minutes": total_min,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }
    with open(results_dir / "diffusion_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(results_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("done")
    print(f"saved {csv_path}")
    print(f"saved {results_dir / 'diffusion_summary.json'}")


if __name__ == "__main__":
    main()
