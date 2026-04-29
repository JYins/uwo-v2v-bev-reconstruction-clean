#!/usr/bin/env python3
"""
Train U-Net for masked BEV reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from dataset import get_dataloaders
from unet import UNet, count_parameters


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset_prepared"
DEFAULT_TRAINING_ROOT = SCRIPT_DIR / "local_runs" / "training"


class Config:
    dataset_root = DEFAULT_DATASET_ROOT
    training_root = DEFAULT_TRAINING_ROOT

    epochs = 80
    batch_size = 12
    lr = 1e-4
    weight_decay = 1e-5
    num_workers = 0 if os.name == "nt" else 4

    in_channels = 16
    out_channels = 8
    features = [16, 32, 64, 128]
    mask_variant = "sector75"

    val_every = 1
    save_every = 10
    print_every = 10
    amp = True

    loss_l1_weight = 0.7
    loss_mse_weight = 0.3
    occ_weight = 3.0
    occ_threshold = 0.05
    occ_bce_weight = 0.0
    occ_loss_type = "bce"
    focal_gamma = 2.0
    height_l1_weight = 0.0
    occ_pos_weight = 8.0
    occ_logit_temp = 0.02
    seed = 42

    checkpoint_dir = training_root / "checkpoints"
    results_dir = training_root / "results"


def parse_features(text):
    return [int(x) for x in text.split(",") if x.strip()]


def psnr_from_mse(mse_value):
    if mse_value <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse_value)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def masked_pixel_count(mask):
    # mask shape: Bx1xHxW, 0=masked(occluded), 1=visible
    occluded = 1.0 - mask
    n = occluded.sum()
    return occluded, n


def masked_mse(pred, target, mask):
    occluded, n = masked_pixel_count(mask)
    if n.item() == 0:
        return torch.zeros((), device=pred.device)
    diff = (pred - target) ** 2
    return (diff * occluded).sum() / (n * pred.shape[1])


def masked_weighted_l1(pred, target, mask, occ_weight, occ_threshold):
    occluded, n = masked_pixel_count(mask)
    if n.item() == 0:
        return torch.zeros((), device=pred.device)

    target_occ = (target[:, :4].sum(dim=1, keepdim=True) > occ_threshold).float()
    weights = 1.0 + (occ_weight - 1.0) * target_occ * occluded

    abs_diff = torch.abs(pred - target)
    denom = weights.sum() * pred.shape[1]
    return (abs_diff * weights).sum() / denom


def masked_occ_bce(pred, target, mask, occ_threshold, occ_pos_weight, occ_logit_temp):
    occluded, n = masked_pixel_count(mask)
    if n.item() == 0:
        return torch.zeros((), device=pred.device)

    target_occ = (target[:, :4].sum(dim=1, keepdim=True) > occ_threshold).float()
    pred_occ = pred[:, :4].sum(dim=1, keepdim=True)
    temp = max(float(occ_logit_temp), 1e-6)
    logits = (pred_occ - occ_threshold) / temp

    bce = F.binary_cross_entropy_with_logits(logits, target_occ, reduction="none")
    weights = 1.0 + (occ_pos_weight - 1.0) * target_occ
    denom = (weights * occluded).sum().clamp(min=1.0)
    return ((bce * weights) * occluded).sum() / denom


def masked_occ_focal(
    pred,
    target,
    mask,
    occ_threshold,
    occ_pos_weight,
    occ_logit_temp,
    focal_gamma,
):
    occluded, n = masked_pixel_count(mask)
    if n.item() == 0:
        return torch.zeros((), device=pred.device)

    target_occ = (target[:, :4].sum(dim=1, keepdim=True) > occ_threshold).float()
    pred_occ = pred[:, :4].sum(dim=1, keepdim=True)
    temp = max(float(occ_logit_temp), 1e-6)
    gamma = max(float(focal_gamma), 0.0)
    logits = (pred_occ - occ_threshold) / temp

    bce = F.binary_cross_entropy_with_logits(logits, target_occ, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * target_occ + (1.0 - prob) * (1.0 - target_occ)
    focal = (1.0 - p_t).clamp(min=0.0).pow(gamma)
    weights = 1.0 + (occ_pos_weight - 1.0) * target_occ
    denom = (weights * occluded).sum().clamp(min=1.0)
    return ((bce * focal * weights) * occluded).sum() / denom


def masked_occ_bin_focal(
    pred,
    target,
    mask,
    occ_threshold,
    occ_pos_weight,
    occ_logit_temp,
    focal_gamma,
):
    occluded, n = masked_pixel_count(mask)
    if n.item() == 0:
        return torch.zeros((), device=pred.device)

    target_occ = (target[:, :4] > occ_threshold).float()
    pred_occ = pred[:, :4]
    temp = max(float(occ_logit_temp), 1e-6)
    gamma = max(float(focal_gamma), 0.0)
    logits = (pred_occ - occ_threshold) / temp

    bce = F.binary_cross_entropy_with_logits(logits, target_occ, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * target_occ + (1.0 - prob) * (1.0 - target_occ)
    focal = (1.0 - p_t).clamp(min=0.0).pow(gamma)
    weights = 1.0 + (occ_pos_weight - 1.0) * target_occ
    occ_mask = occluded.expand_as(target_occ)
    denom = (weights * occ_mask).sum().clamp(min=1.0)
    return ((bce * focal * weights) * occ_mask).sum() / denom


def masked_height_l1(pred, target, mask, occ_threshold):
    occluded, n = masked_pixel_count(mask)
    if n.item() == 0:
        return torch.zeros((), device=pred.device)

    target_occ = (target[:, :4] > occ_threshold).float()
    height_gate = occluded.expand_as(target_occ) * target_occ
    diff = torch.abs(pred[:, 4:8] - target[:, 4:8])
    denom = height_gate.sum().clamp(min=1.0)
    return (diff * height_gate).sum() / denom


def compute_shared_loss(
    pred,
    target,
    mask,
    loss_l1_weight,
    loss_mse_weight,
    occ_weight,
    occ_threshold,
    occ_bce_weight,
    occ_loss_type,
    focal_gamma,
    height_l1_weight,
    occ_pos_weight,
    occ_logit_temp,
):
    l1 = masked_weighted_l1(pred, target, mask, occ_weight, occ_threshold)
    mse = masked_mse(pred, target, mask)
    if occ_loss_type == "focal":
        occ_term = masked_occ_focal(
            pred,
            target,
            mask,
            occ_threshold,
            occ_pos_weight,
            occ_logit_temp,
            focal_gamma,
        )
    elif occ_loss_type == "bin_focal":
        occ_term = masked_occ_bin_focal(
            pred,
            target,
            mask,
            occ_threshold,
            occ_pos_weight,
            occ_logit_temp,
            focal_gamma,
        )
    else:
        occ_term = masked_occ_bce(
            pred,
            target,
            mask,
            occ_threshold,
            occ_pos_weight,
            occ_logit_temp,
        )
    height_term = masked_height_l1(pred, target, mask, occ_threshold)
    total = (
        loss_l1_weight * l1
        + loss_mse_weight * mse
        + occ_bce_weight * occ_term
        + height_l1_weight * height_term
    )
    return total, {
        "l1": float(l1.detach().item()),
        "mse": float(mse.detach().item()),
        "occ": float(occ_term.detach().item()),
        "height": float(height_term.detach().item()),
    }


def read_shared_loss_config(config_path):
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)

    cfg = blob.get("shared_loss_config", blob.get("config", blob))
    return {
        "loss_l1_weight": cfg["loss_l1_weight"],
        "loss_mse_weight": cfg["loss_mse_weight"],
        "occ_weight": cfg["occ_weight"],
        "occ_threshold": cfg["occ_threshold"],
        "occ_bce_weight": cfg.get("occ_bce_weight", 0.0),
        "occ_loss_type": cfg.get("occ_loss_type", "bce"),
        "focal_gamma": cfg.get("focal_gamma", 2.0),
        "height_l1_weight": cfg.get("height_l1_weight", 0.0),
        "occ_pos_weight": cfg.get("occ_pos_weight", 8.0),
        "occ_logit_temp": cfg.get("occ_logit_temp", 0.02),
    }


def apply_shared_loss_overrides(target, shared, preserve_keys=None):
    preserve = set(preserve_keys or [])
    for key, value in shared.items():
        if key in preserve:
            continue
        setattr(target, key, value)


def batch_metric_sums(pred, target, mask, occ_threshold, visible_bev=None):
    with torch.no_grad():
        b, c, _, _ = pred.shape
        abs_diff = torch.abs(pred - target)
        sq_diff = (pred - target) ** 2

        masked = (1.0 - mask)
        masked_count = masked.sum().item()
        full_count = float(b * pred.shape[2] * pred.shape[3])

        masked_mae_sum = (abs_diff * masked).sum().item()
        masked_mse_sum = (sq_diff * masked).sum().item()
        full_mae_sum = abs_diff.sum().item()
        full_mse_sum = sq_diff.sum().item()

        if visible_bev is None or visible_bev.shape[1] < c:
            fused = pred
        else:
            fused = pred * masked + visible_bev[:, :c] * mask

        fused_abs_diff = torch.abs(fused - target)
        fused_sq_diff = (fused - target) ** 2
        fused_full_mae_sum = fused_abs_diff.sum().item()
        fused_full_mse_sum = fused_sq_diff.sum().item()

        per_layer_masked_mae_sum = []
        per_layer_masked_mse_sum = []
        per_layer_full_mae_sum = []
        per_layer_full_mse_sum = []

        for ch in range(c):
            ch_abs = abs_diff[:, ch:ch + 1]
            ch_sq = sq_diff[:, ch:ch + 1]
            per_layer_masked_mae_sum.append((ch_abs * masked).sum().item())
            per_layer_masked_mse_sum.append((ch_sq * masked).sum().item())
            per_layer_full_mae_sum.append(ch_abs.sum().item())
            per_layer_full_mse_sum.append(ch_sq.sum().item())

        pred_occ = pred[:, :4].sum(dim=1, keepdim=True) > occ_threshold
        true_occ = target[:, :4].sum(dim=1, keepdim=True) > occ_threshold
        masked_bool = masked > 0.5

        masked_tp = ((pred_occ & true_occ) & masked_bool).sum().item()
        masked_pred_pos = (pred_occ & masked_bool).sum().item()
        masked_true_pos = (true_occ & masked_bool).sum().item()
        masked_union = ((pred_occ | true_occ) & masked_bool).sum().item()

        full_tp = (pred_occ & true_occ).sum().item()
        full_pred_pos = pred_occ.sum().item()
        full_true_pos = true_occ.sum().item()
        full_union = (pred_occ | true_occ).sum().item()

        fused_occ = fused[:, :4].sum(dim=1, keepdim=True) > occ_threshold
        fused_full_tp = (fused_occ & true_occ).sum().item()
        fused_full_pred_pos = fused_occ.sum().item()
        fused_full_true_pos = true_occ.sum().item()
        fused_full_union = (fused_occ | true_occ).sum().item()

        return {
            "masked_count": masked_count,
            "full_count": full_count,
            "masked_mae_sum": masked_mae_sum,
            "masked_mse_sum": masked_mse_sum,
            "full_mae_sum": full_mae_sum,
            "full_mse_sum": full_mse_sum,
            "fused_full_mae_sum": fused_full_mae_sum,
            "fused_full_mse_sum": fused_full_mse_sum,
            "masked_tp": masked_tp,
            "masked_pred_pos": masked_pred_pos,
            "masked_true_pos": masked_true_pos,
            "masked_union": masked_union,
            "full_tp": full_tp,
            "full_pred_pos": full_pred_pos,
            "full_true_pos": full_true_pos,
            "full_union": full_union,
            "fused_full_tp": fused_full_tp,
            "fused_full_pred_pos": fused_full_pred_pos,
            "fused_full_true_pos": fused_full_true_pos,
            "fused_full_union": fused_full_union,
            "per_layer_masked_mae_sum": per_layer_masked_mae_sum,
            "per_layer_masked_mse_sum": per_layer_masked_mse_sum,
            "per_layer_full_mae_sum": per_layer_full_mae_sum,
            "per_layer_full_mse_sum": per_layer_full_mse_sum,
        }


def init_metric_acc(n_channels):
    return {
        "masked_count": 0.0,
        "full_count": 0.0,
        "masked_mae_sum": 0.0,
        "masked_mse_sum": 0.0,
        "full_mae_sum": 0.0,
        "full_mse_sum": 0.0,
        "fused_full_mae_sum": 0.0,
        "fused_full_mse_sum": 0.0,
        "masked_tp": 0.0,
        "masked_pred_pos": 0.0,
        "masked_true_pos": 0.0,
        "masked_union": 0.0,
        "full_tp": 0.0,
        "full_pred_pos": 0.0,
        "full_true_pos": 0.0,
        "full_union": 0.0,
        "fused_full_tp": 0.0,
        "fused_full_pred_pos": 0.0,
        "fused_full_true_pos": 0.0,
        "fused_full_union": 0.0,
        "per_layer_masked_mae_sum": [0.0] * n_channels,
        "per_layer_masked_mse_sum": [0.0] * n_channels,
        "per_layer_full_mae_sum": [0.0] * n_channels,
        "per_layer_full_mse_sum": [0.0] * n_channels,
    }


def merge_metric_acc(acc, batch):
    for k in [
        "masked_count",
        "full_count",
        "masked_mae_sum",
        "masked_mse_sum",
        "full_mae_sum",
        "full_mse_sum",
        "fused_full_mae_sum",
        "fused_full_mse_sum",
        "masked_tp",
        "masked_pred_pos",
        "masked_true_pos",
        "masked_union",
        "full_tp",
        "full_pred_pos",
        "full_true_pos",
        "full_union",
        "fused_full_tp",
        "fused_full_pred_pos",
        "fused_full_true_pos",
        "fused_full_union",
    ]:
        acc[k] += batch[k]
    for i in range(len(acc["per_layer_masked_mae_sum"])):
        acc["per_layer_masked_mae_sum"][i] += batch["per_layer_masked_mae_sum"][i]
        acc["per_layer_masked_mse_sum"][i] += batch["per_layer_masked_mse_sum"][i]
        acc["per_layer_full_mae_sum"][i] += batch["per_layer_full_mae_sum"][i]
        acc["per_layer_full_mse_sum"][i] += batch["per_layer_full_mse_sum"][i]


def finalize_metrics(acc, n_channels):
    masked_den = max(acc["masked_count"] * n_channels, 1.0)
    full_den = max(acc["full_count"] * n_channels, 1.0)

    masked_mae = acc["masked_mae_sum"] / masked_den
    masked_mse_v = acc["masked_mse_sum"] / masked_den
    full_mae = acc["full_mae_sum"] / full_den
    full_mse_v = acc["full_mse_sum"] / full_den
    fused_full_mae = acc["fused_full_mae_sum"] / full_den
    fused_full_mse_v = acc["fused_full_mse_sum"] / full_den

    per_layer_masked_mae = []
    per_layer_masked_rmse = []
    per_layer_full_mae = []
    per_layer_full_rmse = []
    layer_masked_den = max(acc["masked_count"], 1.0)
    layer_full_den = max(acc["full_count"], 1.0)

    for i in range(n_channels):
        ch_masked_mae = acc["per_layer_masked_mae_sum"][i] / layer_masked_den
        ch_masked_mse = acc["per_layer_masked_mse_sum"][i] / layer_masked_den
        ch_full_mae = acc["per_layer_full_mae_sum"][i] / layer_full_den
        ch_full_mse = acc["per_layer_full_mse_sum"][i] / layer_full_den
        per_layer_masked_mae.append(ch_masked_mae)
        per_layer_masked_rmse.append(math.sqrt(max(ch_masked_mse, 0.0)))
        per_layer_full_mae.append(ch_full_mae)
        per_layer_full_rmse.append(math.sqrt(max(ch_full_mse, 0.0)))

    masked_iou = acc["masked_tp"] / acc["masked_union"] if acc["masked_union"] > 0 else 1.0
    full_iou = acc["full_tp"] / acc["full_union"] if acc["full_union"] > 0 else 1.0
    fused_full_iou = acc["fused_full_tp"] / acc["fused_full_union"] if acc["fused_full_union"] > 0 else 1.0

    masked_precision = acc["masked_tp"] / acc["masked_pred_pos"] if acc["masked_pred_pos"] > 0 else 1.0
    masked_recall = acc["masked_tp"] / acc["masked_true_pos"] if acc["masked_true_pos"] > 0 else 1.0
    masked_f1 = (
        2.0 * masked_precision * masked_recall / (masked_precision + masked_recall)
        if (masked_precision + masked_recall) > 0
        else 0.0
    )

    full_precision = acc["full_tp"] / acc["full_pred_pos"] if acc["full_pred_pos"] > 0 else 1.0
    full_recall = acc["full_tp"] / acc["full_true_pos"] if acc["full_true_pos"] > 0 else 1.0
    full_f1 = (
        2.0 * full_precision * full_recall / (full_precision + full_recall)
        if (full_precision + full_recall) > 0
        else 0.0
    )

    fused_full_precision = (
        acc["fused_full_tp"] / acc["fused_full_pred_pos"] if acc["fused_full_pred_pos"] > 0 else 1.0
    )
    fused_full_recall = (
        acc["fused_full_tp"] / acc["fused_full_true_pos"] if acc["fused_full_true_pos"] > 0 else 1.0
    )
    fused_full_f1 = (
        2.0 * fused_full_precision * fused_full_recall / (fused_full_precision + fused_full_recall)
        if (fused_full_precision + fused_full_recall) > 0
        else 0.0
    )

    metrics = {
        "masked_mae": masked_mae,
        "masked_mse": masked_mse_v,
        "masked_rmse": math.sqrt(max(masked_mse_v, 0.0)),
        "masked_psnr": psnr_from_mse(masked_mse_v),
        "masked_occ_iou": masked_iou,
        "masked_occ_precision": masked_precision,
        "masked_occ_recall": masked_recall,
        "masked_occ_f1": masked_f1,
        "full_mae": full_mae,
        "full_mse": full_mse_v,
        "full_rmse": math.sqrt(max(full_mse_v, 0.0)),
        "full_psnr": psnr_from_mse(full_mse_v),
        "full_occ_iou": full_iou,
        "full_occ_precision": full_precision,
        "full_occ_recall": full_recall,
        "full_occ_f1": full_f1,
        "fused_full_mae": fused_full_mae,
        "fused_full_mse": fused_full_mse_v,
        "fused_full_rmse": math.sqrt(max(fused_full_mse_v, 0.0)),
        "fused_full_psnr": psnr_from_mse(fused_full_mse_v),
        "fused_full_occ_iou": fused_full_iou,
        "fused_full_occ_precision": fused_full_precision,
        "fused_full_occ_recall": fused_full_recall,
        "fused_full_occ_f1": fused_full_f1,
        "per_layer_masked_mae": per_layer_masked_mae,
        "per_layer_masked_rmse": per_layer_masked_rmse,
        "per_layer_full_mae": per_layer_full_mae,
        "per_layer_full_rmse": per_layer_full_rmse,
    }
    return metrics


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, epoch):
    model.train()
    use_amp = cfg.amp and device.type == "cuda"
    total_loss = 0.0
    n_batches = 0

    for step, (inp, tgt, mask) in enumerate(loader, start=1):
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(inp)
            loss, parts = compute_shared_loss(
                pred,
                tgt,
                mask,
                cfg.loss_l1_weight,
                cfg.loss_mse_weight,
                cfg.occ_weight,
                cfg.occ_threshold,
                cfg.occ_bce_weight,
                cfg.occ_loss_type,
                cfg.focal_gamma,
                cfg.height_l1_weight,
                cfg.occ_pos_weight,
                cfg.occ_logit_temp,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

        if step % cfg.print_every == 0:
            print(
                f"    Epoch {epoch} | Batch {step}/{len(loader)} | "
                f"Loss: {total_loss / n_batches:.6f} | "
                f"L1: {parts['l1']:.4f} MSE: {parts['mse']:.4f} "
                f"Occ: {parts['occ']:.4f} Height: {parts['height']:.4f}"
            )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    n_channels = cfg.out_channels
    acc = init_metric_acc(n_channels)

    for inp, tgt, mask in loader:
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        pred = model(inp)
        visible_bev = inp[:, :cfg.out_channels] if inp.shape[1] >= cfg.out_channels else None
        batch = batch_metric_sums(pred, tgt, mask, cfg.occ_threshold, visible_bev=visible_bev)
        merge_metric_acc(acc, batch)

    return finalize_metrics(acc, n_channels)


def save_checkpoint(path, epoch, model, optimizer, extra):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            **extra,
        },
        path,
    )


def history_to_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_summary(path, cfg, n_params, best_epoch, best_metrics, test_metrics, total_minutes, amp_on):
    with open(path, "w", encoding="utf-8") as f:
        f.write("U-Net BEV Reconstruction - Training Summary\n")
        f.write("=" * 58 + "\n")
        f.write(f"Dataset root:    {cfg.dataset_root}\n")
        f.write(f"Training root:   {cfg.training_root}\n")
        f.write(f"Parameters:      {n_params:,}\n")
        f.write(f"Features:        {cfg.features}\n")
        f.write(f"Mask variant:    {cfg.mask_variant}\n")
        f.write(f"Epochs:          {cfg.epochs}\n")
        f.write(f"Batch size:      {cfg.batch_size}\n")
        f.write(f"Learning rate:   {cfg.lr}\n")
        f.write(f"Seed:            {cfg.seed}\n")
        f.write(f"AMP enabled:     {amp_on}\n")
        f.write(f"Training time:   {total_minutes:.1f} minutes\n")
        f.write("\nLoss setup\n")
        f.write(f"  loss_l1_weight:  {cfg.loss_l1_weight}\n")
        f.write(f"  loss_mse_weight: {cfg.loss_mse_weight}\n")
        f.write(f"  occ_weight:      {cfg.occ_weight}\n")
        f.write(f"  occ_threshold:   {cfg.occ_threshold}\n")
        f.write(f"  occ_bce_weight:  {cfg.occ_bce_weight}\n")
        f.write(f"  occ_loss_type:   {cfg.occ_loss_type}\n")
        f.write(f"  focal_gamma:     {cfg.focal_gamma}\n")
        f.write(f"  height_l1_weight:{cfg.height_l1_weight}\n")
        f.write(f"  occ_pos_weight:  {cfg.occ_pos_weight}\n")
        f.write(f"  occ_logit_temp:  {cfg.occ_logit_temp}\n")

        if best_epoch is not None and best_metrics is not None:
            f.write(f"\nBest epoch: {best_epoch}\n")
            f.write(f"  masked MAE:      {best_metrics['masked_mae']:.6f}\n")
            f.write(f"  masked RMSE:     {best_metrics['masked_rmse']:.6f}\n")
            f.write(f"  masked PSNR:     {best_metrics['masked_psnr']:.2f} dB\n")
            f.write(f"  masked Occ-IoU:  {best_metrics['masked_occ_iou']:.4f}\n")
            f.write(f"  masked Prec:     {best_metrics['masked_occ_precision']:.4f}\n")
            f.write(f"  masked Recall:   {best_metrics['masked_occ_recall']:.4f}\n")
            f.write(f"  masked F1:       {best_metrics['masked_occ_f1']:.4f}\n")
            f.write(f"  fused full MAE:  {best_metrics['fused_full_mae']:.6f}\n")
            f.write(f"  fused full RMSE: {best_metrics['fused_full_rmse']:.6f}\n")
            f.write(f"  fused full PSNR: {best_metrics['fused_full_psnr']:.2f} dB\n")
            f.write(f"  fused full IoU:  {best_metrics['fused_full_occ_iou']:.4f}\n")
            f.write(f"  fused full Prec: {best_metrics['fused_full_occ_precision']:.4f}\n")
            f.write(f"  fused full Rec:  {best_metrics['fused_full_occ_recall']:.4f}\n")
            f.write(f"  fused full F1:   {best_metrics['fused_full_occ_f1']:.4f}\n")
            f.write(f"  full MAE:        {best_metrics['full_mae']:.6f}\n")
            f.write(f"  full RMSE:       {best_metrics['full_rmse']:.6f}\n")
            f.write(f"  full PSNR:       {best_metrics['full_psnr']:.2f} dB\n")

        if test_metrics is not None:
            f.write("\nTest metrics\n")
            f.write(f"  masked MAE:      {test_metrics['masked_mae']:.6f}\n")
            f.write(f"  masked RMSE:     {test_metrics['masked_rmse']:.6f}\n")
            f.write(f"  masked PSNR:     {test_metrics['masked_psnr']:.2f} dB\n")
            f.write(f"  masked Occ-IoU:  {test_metrics['masked_occ_iou']:.4f}\n")
            f.write(f"  masked Prec:     {test_metrics['masked_occ_precision']:.4f}\n")
            f.write(f"  masked Recall:   {test_metrics['masked_occ_recall']:.4f}\n")
            f.write(f"  masked F1:       {test_metrics['masked_occ_f1']:.4f}\n")
            f.write(f"  fused full MAE:  {test_metrics['fused_full_mae']:.6f}\n")
            f.write(f"  fused full RMSE: {test_metrics['fused_full_rmse']:.6f}\n")
            f.write(f"  fused full PSNR: {test_metrics['fused_full_psnr']:.2f} dB\n")
            f.write(f"  fused full IoU:  {test_metrics['fused_full_occ_iou']:.4f}\n")
            f.write(f"  fused full Prec: {test_metrics['fused_full_occ_precision']:.4f}\n")
            f.write(f"  fused full Rec:  {test_metrics['fused_full_occ_recall']:.4f}\n")
            f.write(f"  fused full F1:   {test_metrics['fused_full_occ_f1']:.4f}\n")
            f.write(f"  full MAE:        {test_metrics['full_mae']:.6f}\n")
            f.write(f"  full RMSE:       {test_metrics['full_rmse']:.6f}\n")
            f.write(f"  full PSNR:       {test_metrics['full_psnr']:.2f} dB\n")


def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net for masked BEV reconstruction.")
    p.add_argument("--dataset_root", type=Path, default=Config.dataset_root)
    p.add_argument("--training_root", type=Path, default=Config.training_root)
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--num_workers", type=int, default=Config.num_workers)
    p.add_argument("--features", type=str, default="16,32,64,128")
    p.add_argument("--mask_variant", type=str, default=Config.mask_variant, choices=["sector75", "front_rect", "front_blob"])
    p.add_argument("--val_every", type=int, default=Config.val_every)
    p.add_argument("--save_every", type=int, default=Config.save_every)
    p.add_argument("--print_every", type=int, default=Config.print_every)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--shared_config", type=Path, default=None)
    p.add_argument("--loss_l1_weight", type=float, default=Config.loss_l1_weight)
    p.add_argument("--loss_mse_weight", type=float, default=Config.loss_mse_weight)
    p.add_argument("--occ_weight", type=float, default=Config.occ_weight)
    p.add_argument("--occ_threshold", type=float, default=Config.occ_threshold)
    p.add_argument("--occ_bce_weight", type=float, default=Config.occ_bce_weight)
    p.add_argument("--occ_loss_type", type=str, default=Config.occ_loss_type, choices=["bce", "focal", "bin_focal"])
    p.add_argument("--focal_gamma", type=float, default=Config.focal_gamma)
    p.add_argument("--height_l1_weight", type=float, default=Config.height_l1_weight)
    p.add_argument("--occ_pos_weight", type=float, default=Config.occ_pos_weight)
    p.add_argument("--occ_logit_temp", type=float, default=Config.occ_logit_temp)
    p.add_argument("--seed", type=int, default=Config.seed)
    return p.parse_args()


def main():
    args = parse_args()
    if args.shared_config is not None:
        shared = read_shared_loss_config(args.shared_config)
        apply_shared_loss_overrides(args, shared, preserve_keys={"occ_loss_type", "focal_gamma", "height_l1_weight"})

    cfg = Config()

    cfg.dataset_root = args.dataset_root.resolve()
    cfg.training_root = args.training_root.resolve()
    cfg.checkpoint_dir = cfg.training_root / "checkpoints"
    cfg.results_dir = cfg.training_root / "results"

    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.num_workers = args.num_workers
    cfg.features = parse_features(args.features)
    cfg.mask_variant = args.mask_variant
    cfg.val_every = args.val_every
    cfg.save_every = args.save_every
    cfg.print_every = args.print_every
    cfg.amp = not args.no_amp

    cfg.loss_l1_weight = args.loss_l1_weight
    cfg.loss_mse_weight = args.loss_mse_weight
    cfg.occ_weight = args.occ_weight
    cfg.occ_threshold = args.occ_threshold
    cfg.occ_bce_weight = args.occ_bce_weight
    cfg.occ_loss_type = args.occ_loss_type
    cfg.focal_gamma = args.focal_gamma
    cfg.height_l1_weight = args.height_l1_weight
    cfg.occ_pos_weight = args.occ_pos_weight
    cfg.occ_logit_temp = args.occ_logit_temp
    cfg.seed = args.seed

    set_seed(cfg.seed)

    print("\n" + "=" * 70)
    print("  U-Net Training - Masked BEV Reconstruction")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"\n  GPU: {gpu} ({mem:.1f} GB)")
    else:
        print("\n  WARNING: CUDA not found, using CPU.")

    print(f"\n  Dataset root:  {cfg.dataset_root}")
    print(f"  Training root: {cfg.training_root}")
    print(f"  Features:      {cfg.features}")
    print(f"  Mask variant:  {cfg.mask_variant}")
    print(f"  Seed:          {cfg.seed}")
    if args.shared_config is not None:
        print(f"  Shared config: {args.shared_config.resolve()}")

    train_loader, val_loader, test_loader = get_dataloaders(
        cfg.dataset_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        mask_variant=cfg.mask_variant,
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    if len(train_loader) == 0:
        raise SystemExit("\n  ERROR: no training data found.")

    model = UNet(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        features=cfg.features,
    ).to(device)
    n_params = count_parameters(model)
    print(f"\n  Model parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
    scaler = torch.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    best_occ_iou = -1.0
    best_rmse = float("inf")
    best_epoch = None
    best_metrics = None
    history = []

    print(f"\n  Training for {cfg.epochs} epochs")
    print(f"  Batch size: {cfg.batch_size} | LR: {cfg.lr}")
    print(
        f"  Loss: {cfg.loss_l1_weight}*weighted_L1 + "
        f"{cfg.loss_mse_weight}*masked_MSE + "
        f"{cfg.occ_bce_weight}*masked_occ_{cfg.occ_loss_type.upper()} + "
        f"{cfg.height_l1_weight}*masked_height_L1"
    )
    print(
        f"  occ_weight: {cfg.occ_weight} | occ_threshold: {cfg.occ_threshold} | "
        f"occ_loss_type: {cfg.occ_loss_type} | focal_gamma: {cfg.focal_gamma} | "
        f"height_l1_weight: {cfg.height_l1_weight} | "
        f"occ_pos_weight: {cfg.occ_pos_weight} | occ_logit_temp: {cfg.occ_logit_temp}"
    )
    print(f"  AMP: {'on' if scaler.is_enabled() else 'off'}")
    print("  " + "=" * 58)

    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch)

        val_metrics = None
        if len(val_loader) > 0 and epoch % cfg.val_every == 0:
            val_metrics = evaluate(model, val_loader, device, cfg)
            scheduler.step(val_metrics["masked_rmse"])

        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        if val_metrics is None:
            print(
                f"  Epoch {epoch:3d}/{cfg.epochs} | "
                f"Train Loss: {train_loss:.6f} | LR: {lr_now:.2e} | {dt:.1f}s"
            )
        else:
            print(
                f"  Epoch {epoch:3d}/{cfg.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val MAE: {val_metrics['masked_mae']:.6f} "
                f"RMSE: {val_metrics['masked_rmse']:.6f} "
                f"PSNR: {val_metrics['masked_psnr']:.2f} "
                f"IoU: {val_metrics['masked_occ_iou']:.4f} "
                f"Prec: {val_metrics['masked_occ_precision']:.4f} "
                f"Rec: {val_metrics['masked_occ_recall']:.4f} "
                f"FusedPSNR: {val_metrics['fused_full_psnr']:.2f} | "
                f"LR: {lr_now:.2e} | {dt:.1f}s"
            )

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
                best_epoch = epoch
                best_metrics = val_metrics
                save_checkpoint(
                    cfg.checkpoint_dir / "best_unet.pth",
                    epoch,
                    model,
                    optimizer,
                    {
                        "features": cfg.features,
                        "mask_variant": cfg.mask_variant,
                        "val_metrics": val_metrics,
                    },
                )
                print(f"  >>> New best checkpoint at epoch {epoch}")

        row = {
            "epoch": epoch,
            "seed": cfg.seed,
            "train_loss": train_loss,
            "lr": lr_now,
            "val_mse": val_metrics["masked_mse"] if val_metrics else "",
            "val_psnr": val_metrics["masked_psnr"] if val_metrics else "",
            "val_masked_mae": val_metrics["masked_mae"] if val_metrics else "",
            "val_masked_rmse": val_metrics["masked_rmse"] if val_metrics else "",
            "val_masked_occ_iou": val_metrics["masked_occ_iou"] if val_metrics else "",
            "val_masked_occ_precision": val_metrics["masked_occ_precision"] if val_metrics else "",
            "val_masked_occ_recall": val_metrics["masked_occ_recall"] if val_metrics else "",
            "val_masked_occ_f1": val_metrics["masked_occ_f1"] if val_metrics else "",
            "val_fused_full_mae": val_metrics["fused_full_mae"] if val_metrics else "",
            "val_fused_full_rmse": val_metrics["fused_full_rmse"] if val_metrics else "",
            "val_fused_full_psnr": val_metrics["fused_full_psnr"] if val_metrics else "",
            "val_fused_full_occ_iou": val_metrics["fused_full_occ_iou"] if val_metrics else "",
            "val_fused_full_occ_precision": val_metrics["fused_full_occ_precision"] if val_metrics else "",
            "val_fused_full_occ_recall": val_metrics["fused_full_occ_recall"] if val_metrics else "",
            "val_fused_full_occ_f1": val_metrics["fused_full_occ_f1"] if val_metrics else "",
            "val_full_mae": val_metrics["full_mae"] if val_metrics else "",
            "val_full_rmse": val_metrics["full_rmse"] if val_metrics else "",
            "val_full_psnr": val_metrics["full_psnr"] if val_metrics else "",
        }
        history.append(row)

        if epoch % cfg.save_every == 0:
            save_checkpoint(
                cfg.checkpoint_dir / f"unet_epoch_{epoch:03d}.pth",
                epoch,
                model,
                optimizer,
                {"features": cfg.features, "mask_variant": cfg.mask_variant},
            )

    total_minutes = (time.time() - start) / 60.0
    print("\n" + "=" * 58)
    print(f"  Training complete in {total_minutes:.1f} minutes")

    test_metrics = None
    best_path = cfg.checkpoint_dir / "best_unet.pth"
    if best_path.exists() and len(test_loader) > 0:
        print("\n  Running test with best checkpoint...")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics = evaluate(model, test_loader, device, cfg)
        print(f"    Best epoch: {ckpt['epoch']}")
        print(f"    Test masked MAE:      {test_metrics['masked_mae']:.6f}")
        print(f"    Test masked RMSE:     {test_metrics['masked_rmse']:.6f}")
        print(f"    Test masked PSNR:     {test_metrics['masked_psnr']:.2f} dB")
        print(f"    Test masked Occ-IoU:  {test_metrics['masked_occ_iou']:.4f}")
        print(f"    Test masked Prec/F1:  {test_metrics['masked_occ_precision']:.4f} / {test_metrics['masked_occ_f1']:.4f}")
        print(f"    Test fused full PSNR: {test_metrics['fused_full_psnr']:.2f} dB")

    history_npy = cfg.results_dir / "training_history.npy"
    history_csv = cfg.results_dir / "training_history.csv"
    summary_txt = cfg.results_dir / "unet_summary.txt"
    test_json = cfg.results_dir / "test_metrics.json"

    np.save(history_npy, history)
    history_to_csv(history, history_csv)
    write_summary(summary_txt, cfg, n_params, best_epoch, best_metrics, test_metrics, total_minutes, scaler.is_enabled())

    payload = {
        "config": {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "features": cfg.features,
            "mask_variant": cfg.mask_variant,
            "lr": cfg.lr,
            "seed": cfg.seed,
            "loss_l1_weight": cfg.loss_l1_weight,
            "loss_mse_weight": cfg.loss_mse_weight,
            "occ_weight": cfg.occ_weight,
            "occ_threshold": cfg.occ_threshold,
            "occ_bce_weight": cfg.occ_bce_weight,
            "occ_loss_type": cfg.occ_loss_type,
            "focal_gamma": cfg.focal_gamma,
            "height_l1_weight": cfg.height_l1_weight,
            "occ_pos_weight": cfg.occ_pos_weight,
            "occ_logit_temp": cfg.occ_logit_temp,
        },
        "best_epoch": best_epoch,
        "best_val_metrics": best_metrics,
        "test_metrics": test_metrics,
    }
    with open(test_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n  Saved: {history_npy}")
    print(f"  Saved: {history_csv}")
    print(f"  Saved: {summary_txt}")
    print(f"  Saved: {test_json}")
    print("  Done!")


if __name__ == "__main__":
    main()
