#!/usr/bin/env python3
"""
Build a colored final results report for the BEV reconstruction experiments.

Outputs:
  reports/final_results_2026-04-23/final_results_report_2026-04-23.md
  reports/final_results_2026-04-23/final_results_report_2026-04-23.html
  reports/final_results_2026-04-23/final_results_report_2026-04-23.pdf
"""

from __future__ import annotations

import csv
import html
import json
import math
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from dataset import BEVReconstructionDataset, discover_available_splits
from train_diffusion import alpha_bar_schedule, ddim_sample
from unet import UNet


ROOT = Path(__file__).resolve().parent
REPORT_ROOT = ROOT / "reports" / "final_results_2026-04-23"
ASSET_DIR = REPORT_ROOT / "assets"

DATASET_ROOT = ROOT / "dataset_prepared"
UNET_CKPT = ROOT / "training_unet_optuna_seed42" / "checkpoints" / "best_unet.pth"
PIX_CKPT = ROOT / "training_pix2pix_full_seed42" / "checkpoints" / "best_pix2pix.pth"
DIFF_CKPT = ROOT / "narval_runs" / "training_diffusion_full_seed42_v3" / "checkpoints" / "best_diffusion.pth"

UNET_SUMMARY = ROOT / "reports" / "unet_optuna_3seed_summary.json"
UNET_REFRESH = ROOT / "training_unet_optuna_seed42" / "results" / "test_metrics_refresh.json"
PIX_REFRESH = ROOT / "training_pix2pix_full_seed42" / "results" / "test_metrics_refresh.json"
PIX_SUMMARY = ROOT / "training_pix2pix_full_seed42" / "results" / "pix2pix_summary.json"
PIX_OPTUNA = ROOT / "optuna_pix2pix_adv_v2" / "best_params.json"
DIFF_SUMMARY = ROOT / "narval_runs" / "training_diffusion_full_seed42_v3" / "results" / "diffusion_summary.json"
DIFF_HISTORY = ROOT / "narval_runs" / "training_diffusion_full_seed42_v3" / "results" / "training_history.csv"

BIN_COLORS = np.array(
    [
        (255, 120, 40),
        (40, 230, 120),
        (40, 180, 255),
        (200, 60, 255),
    ],
    dtype=np.float64,
)
CHANNEL_LABELS = ["d0", "d1", "d2", "d3", "h0", "h1", "h2", "h3"]


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value, digits=4):
    if value is None or value == "":
        return "-"
    return f"{float(value):.{digits}f}"


def fmt_pm(metric_obj, digits=4):
    return f"{metric_obj['mean']:.{digits}f} +/- {metric_obj['std']:.{digits}f}"


def font(size=18, bold=False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf") if bold else Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf") if bold else Path("C:/Windows/Fonts/segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def bev_to_color(bev: np.ndarray) -> np.ndarray:
    """Render 8-channel BEV using density channels as color and height channels as brightness."""
    density = np.clip(bev[:, :, :4].astype(np.float64), 0.0, 1.0)
    height = np.clip(bev[:, :, 4:8].astype(np.float64), 0.0, 1.0)
    score = density * (0.7 + 0.3 * height)
    dominant = np.argmax(score, axis=2)
    dominant_score = np.max(score, axis=2)
    dominant_height = np.take_along_axis(height, dominant[:, :, None], axis=2)[:, :, 0]

    blend = np.zeros((bev.shape[0], bev.shape[1], 3), dtype=np.float64)
    blend_weight = score.sum(axis=2)
    for idx in range(4):
        blend += score[:, :, idx : idx + 1] * BIN_COLORS[idx]

    img = BIN_COLORS[dominant].copy()
    nonzero = blend_weight > 1e-8
    blend[nonzero] /= blend_weight[nonzero, None]
    img = 0.8 * img + 0.2 * blend

    brightness = np.power(np.clip(dominant_score, 0.0, 1.0), 0.45)
    brightness *= 0.55 + 0.45 * dominant_height
    img *= brightness[:, :, None]
    return np.clip(img, 0, 255).astype(np.uint8)


def resize_np(img: np.ndarray, size: int) -> Image.Image:
    return Image.fromarray(img).resize((size, size), Image.Resampling.NEAREST)


def captioned_tile(img: Image.Image, label: str, color: tuple[int, int, int], tile_size: int) -> Image.Image:
    header = 34
    canvas = Image.new("RGB", (tile_size, tile_size + header), (18, 24, 38))
    canvas.paste(img, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), label, fill=color, font=font(15, bold=True))
    return canvas


def load_unet(device):
    ckpt = torch.load(UNET_CKPT, map_location=device, weights_only=False)
    features = ckpt.get("features", [16, 32, 64, 128])
    model = UNet(in_channels=16, out_channels=8, features=features).to(device)
    model.load_state_dict(convert_legacy_unet_keys(ckpt["model_state_dict"]))
    model.eval()
    return model


def load_pix2pix(device):
    ckpt = torch.load(PIX_CKPT, map_location=device, weights_only=False)
    features = ckpt.get("features", [16, 32, 64, 128])
    model = UNet(in_channels=16, out_channels=8, features=features).to(device)
    model.load_state_dict(convert_legacy_unet_keys(ckpt["generator"]))
    model.eval()
    return model


def convert_legacy_unet_keys(state_dict):
    """Map pre-timestep ConvBlock sequential keys to the current explicit-layer keys."""
    converted = {}
    replacements = {
        ".block.0.": ".conv1.",
        ".block.1.": ".bn1.",
        ".block.3.": ".conv2.",
        ".block.4.": ".bn2.",
    }
    for key, value in state_dict.items():
        new_key = key
        for old, new in replacements.items():
            new_key = new_key.replace(old, new)
        converted[new_key] = value
    return converted


def load_diffusion(device):
    ckpt = torch.load(DIFF_CKPT, map_location=device, weights_only=False)
    model = UNet(in_channels=24, out_channels=8, features=[16, 32, 64, 128], time_emb_dim=128).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def sample_predictions(max_samples=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_splits = discover_available_splits(DATASET_ROOT)
    dataset = BEVReconstructionDataset(DATASET_ROOT, test_splits, augment=False)
    indices = np.linspace(0, len(dataset) - 1, num=max_samples, dtype=int).tolist()

    unet = load_unet(device)
    pix = load_pix2pix(device)
    diff = load_diffusion(device)
    alpha_bar = alpha_bar_schedule(1000, device)
    torch.manual_seed(42)

    samples = []
    with torch.no_grad():
        for idx in indices:
            inp, tgt, mask = dataset[idx]
            info = dataset.get_info(idx)
            inp_b = inp.unsqueeze(0).to(device)
            pred_unet = unet(inp_b).clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
            pred_pix = pix(inp_b).clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
            pred_diff = ddim_sample(diff, inp_b, alpha_bar, timesteps=1000, sample_steps=25)
            pred_diff = pred_diff.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            samples.append(
                {
                    "info": info,
                    "masked": inp[:8].numpy().transpose(1, 2, 0),
                    "neighbor": inp[8:].numpy().transpose(1, 2, 0),
                    "target": tgt.numpy().transpose(1, 2, 0),
                    "unet": pred_unet,
                    "pix2pix": pred_pix,
                    "diffusion": pred_diff,
                }
            )
    return samples


def render_model_comparison(sample, out_path: Path, tile_size=220):
    entries = [
        ("Masked Ego", sample["masked"], (80, 180, 255)),
        ("Ground Truth", sample["target"], (80, 255, 180)),
        ("U-Net", sample["unet"], (109, 160, 255)),
        ("Pix2Pix", sample["pix2pix"], (255, 190, 80)),
        ("Diffusion v3", sample["diffusion"], (255, 105, 120)),
    ]
    tiles = [captioned_tile(resize_np(bev_to_color(bev), tile_size), label, color, tile_size) for label, bev, color in entries]
    gap = 8
    footer = 42
    width = len(tiles) * tile_size + (len(tiles) - 1) * gap
    height = tiles[0].height + footer
    canvas = Image.new("RGB", (width, height), (12, 18, 30))
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 0))
        x += tile_size + gap
    draw = ImageDraw.Draw(canvas)
    info = sample["info"]
    title = f"{info['split']} / {info['scene']} / frame {info['frame']}   |   same test case, final checkpoints"
    draw.text((10, height - 30), title, fill=(195, 205, 220), font=font(15))
    canvas.save(out_path)


def channel_tile(channel, tile_size, cmap_name="turbo"):
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(np.clip(channel, 0.0, 1.0))
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((tile_size, tile_size), Image.Resampling.NEAREST)


def render_channel_split(sample, model_key: str, model_label: str, out_path: Path, tile_size=92):
    pred = sample[model_key]
    target = sample["target"]
    diff = np.abs(pred - target)
    rows = [("Prediction", pred), ("Ground Truth", target), ("Abs Diff", diff)]
    gap = 5
    left = 96
    top = 52
    row_gap = 30
    footer = 34
    width = left + 8 * tile_size + 7 * gap
    height = top + 3 * tile_size + 2 * row_gap + footer
    canvas = Image.new("RGB", (width, height), (14, 19, 31))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 14), f"{model_label}: 8-channel reconstruction split", fill=(245, 247, 250), font=font(20, True))
    for i, label in enumerate(CHANNEL_LABELS):
        x = left + i * (tile_size + gap) + 8
        draw.text((x, 34), label, fill=(210, 220, 235), font=font(13, True))
    for r, (row_label, bev) in enumerate(rows):
        y = top + r * (tile_size + row_gap)
        draw.text((12, y + tile_size // 2 - 8), row_label, fill=(190, 202, 220), font=font(14, True))
        for ch in range(8):
            tile = channel_tile(bev[:, :, ch], tile_size)
            x = left + ch * (tile_size + gap)
            canvas.paste(tile, (x, y))
    info = sample["info"]
    draw.text(
        (12, height - 24),
        f"{info['split']} / {info['scene']} / frame {info['frame']}   |   Abs Diff = |prediction - ground truth|",
        fill=(160, 175, 195),
        font=font(12),
    )
    canvas.save(out_path)


def build_metric_chart(out_path: Path, metrics):
    labels = ["U-Net", "Pix2Pix", "Diffusion"]
    colors = ["#4f8cff", "#f59f00", "#e8596f"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4), dpi=180)
    charts = [
        ("Masked Occ-IoU (higher better)", [metrics[m]["masked_occ_iou"] for m in labels], (0, 0.17)),
        ("Masked RMSE (lower better)", [metrics[m]["masked_rmse"] for m in labels], (0, 0.42)),
        ("Fused Full PSNR (higher better)", [metrics[m]["fused_full_psnr"] for m in labels], (0, 36)),
    ]
    for ax, (title, vals, ylim) in zip(axes, charts):
        bars = ax.bar(labels, vals, color=colors, width=0.62)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + (ylim[1] * 0.015), f"{val:.3f}", ha="center", fontsize=9)
        ax.tick_params(axis="x", labelrotation=18)
    fig.suptitle("Final Model Comparison", fontsize=14, fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_diffusion_curve(out_path: Path):
    rows = []
    with DIFF_HISTORY.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("val_masked_occ_iou"):
                rows.append(row)
    epochs = [int(r["epoch"]) for r in rows]
    iou = [float(r["val_masked_occ_iou"]) for r in rows]
    rmse = [float(r["val_masked_rmse"]) for r in rows]
    psnr = [float(r["val_fused_full_psnr"]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(9, 3.8), dpi=180)
    ax1.plot(epochs, iou, marker="o", color="#e8596f", label="val masked Occ-IoU")
    ax1.set_ylabel("Occ-IoU", color="#e8596f")
    ax1.set_ylim(0, 0.07)
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(epochs, rmse, marker="s", color="#4f8cff", label="val masked RMSE")
    ax2.plot(epochs, psnr, marker="^", color="#37b24d", label="val fused PSNR")
    ax2.set_ylabel("RMSE / PSNR")
    lines = ax1.lines + ax2.lines
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
    ax1.set_xlabel("Epoch")
    ax1.set_title("Diffusion v3 validation trend: stable but non-competitive", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_flowchart(out_path: Path):
    fig, ax = plt.subplots(figsize=(13, 7), dpi=180)
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)

    def box(x, y, w, h, text, color, edge="#1f2937"):
        patch = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=edge, linewidth=1.8, alpha=0.95)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, color="#111827", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.8, color="#334155"))

    box(0.3, 5.55, 2.2, 0.75, "8-channel BEV pipeline\nmasked ego + neighbor", "#dbeafe")
    box(3.0, 5.55, 2.2, 0.75, "U-Net baseline\n80 epochs, 3 seeds", "#bfdbfe")
    box(5.7, 5.55, 2.35, 0.75, "Optuna shared loss\noccupancy-aware loss", "#93c5fd")
    box(8.55, 5.55, 2.1, 0.75, "Diagnostics\nPrecision / Recall / F1", "#e0f2fe")
    box(11.0, 5.55, 1.65, 0.75, "Official U-Net\nbest baseline", "#bbf7d0")

    box(3.0, 3.55, 2.2, 0.75, "Pix2Pix smoke\nlambda_adv=1.0 too strong", "#fed7aa")
    box(5.7, 3.55, 2.35, 0.75, "Pix2Pix full\nlambda_adv=0.1", "#fdba74")
    box(8.55, 3.55, 2.1, 0.75, "Optuna adv search\n0.1 to 2.0", "#fb923c")
    box(11.0, 3.55, 1.65, 0.75, "Best adv\n0.1", "#ffedd5")

    box(3.0, 1.35, 2.2, 0.85, "Diffusion smoke\npipeline runs", "#fecdd3")
    box(5.7, 1.35, 2.35, 0.85, "Fix implementation\ntimestep + DDIM eval", "#fda4af")
    box(8.55, 1.35, 2.1, 0.85, "Narval v3 full\n120 epochs, A100", "#fb7185")
    box(11.0, 1.35, 1.65, 0.85, "Solid negative\nnon-competitive", "#ffe4e6")

    for y in [5.925, 3.925, 1.775]:
        arrow(2.5, y, 3.0, y)
        arrow(5.2, y, 5.7, y)
        arrow(8.05, y, 8.55, y)
        arrow(10.65, y, 11.0, y)

    ax.text(0.3, 6.65, "Experiment Evolution", fontsize=17, fontweight="bold", color="#0f172a")
    ax.text(0.3, 0.45, "Key decision: keep U-Net as official baseline; use Pix2Pix as valid GAN comparison; close Diffusion as a properly tested negative baseline.", fontsize=10.5, color="#334155")
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rel(path: Path) -> str:
    return path.relative_to(REPORT_ROOT).as_posix()


def make_tables():
    unet_summary = read_json(UNET_SUMMARY)
    unet_agg = unet_summary["aggregate"]
    unet_seed42 = read_json(UNET_REFRESH)["test_metrics"] if "test_metrics" in read_json(UNET_REFRESH) else read_json(UNET_REFRESH)
    pix = read_json(PIX_REFRESH)["test_metrics"] if "test_metrics" in read_json(PIX_REFRESH) else read_json(PIX_REFRESH)
    pix_summary = read_json(PIX_SUMMARY)
    pix_optuna = read_json(PIX_OPTUNA)
    diff = read_json(DIFF_SUMMARY)
    diff_test = diff["test_metrics"]

    final_metrics = {
        "U-Net": {
            "masked_occ_iou": unet_agg["masked_occ_iou"]["mean"],
            "masked_rmse": unet_agg["masked_rmse"]["mean"],
            "masked_mae": unet_agg["masked_mae"]["mean"],
            "masked_psnr": unet_agg["masked_psnr"]["mean"],
            "fused_full_psnr": unet_agg["fused_full_psnr"]["mean"],
            "fused_full_occ_iou": unet_agg["fused_full_occ_iou"]["mean"],
            "precision": unet_seed42.get("masked_occ_precision"),
            "recall": unet_seed42.get("masked_occ_recall"),
            "f1": unet_seed42.get("masked_occ_f1"),
        },
        "Pix2Pix": {
            "masked_occ_iou": pix["masked_occ_iou"],
            "masked_rmse": pix["masked_rmse"],
            "masked_mae": pix["masked_mae"],
            "masked_psnr": pix["masked_psnr"],
            "fused_full_psnr": pix["fused_full_psnr"],
            "fused_full_occ_iou": pix["fused_full_occ_iou"],
            "precision": pix.get("masked_occ_precision"),
            "recall": pix.get("masked_occ_recall"),
            "f1": pix.get("masked_occ_f1"),
        },
        "Diffusion": {
            "masked_occ_iou": diff_test["masked_occ_iou"],
            "masked_rmse": diff_test["masked_rmse"],
            "masked_mae": diff_test["masked_mae"],
            "masked_psnr": diff_test["masked_psnr"],
            "fused_full_psnr": diff_test["fused_full_psnr"],
            "fused_full_occ_iou": diff_test["fused_full_occ_iou"],
            "precision": diff_test.get("masked_occ_precision"),
            "recall": diff_test.get("masked_occ_recall"),
            "f1": diff_test.get("masked_occ_f1"),
        },
    }

    layer_metrics = {
        "U-Net": {
            "mae": unet_seed42.get("per_layer_masked_mae", []),
            "rmse": unet_seed42.get("per_layer_masked_rmse", []),
        },
        "Pix2Pix": {
            "mae": pix.get("per_layer_masked_mae", []),
            "rmse": pix.get("per_layer_masked_rmse", []),
        },
        "Diffusion v3": {
            "mae": diff_test.get("per_layer_masked_mae", []),
            "rmse": diff_test.get("per_layer_masked_rmse", []),
        },
    }

    context = {
        "unet_agg": unet_agg,
        "unet_seed42": unet_seed42,
        "pix": pix,
        "pix_summary": pix_summary,
        "pix_optuna": pix_optuna,
        "diff": diff,
        "diff_test": diff_test,
        "final_metrics": final_metrics,
        "layer_metrics": layer_metrics,
    }
    return context


def layer_table_rows(layer_data):
    rows = []
    for idx, label in enumerate(CHANNEL_LABELS):
        mae = layer_data["mae"][idx] if idx < len(layer_data["mae"]) else None
        rmse = layer_data["rmse"][idx] if idx < len(layer_data["rmse"]) else None
        if rmse is None:
            quality = "-"
        elif rmse < 0.04:
            quality = "Good"
        elif rmse < 0.08:
            quality = "Acceptable"
        elif rmse < 0.15:
            quality = "Weak"
        else:
            quality = "Poor"
        role = "density" if idx < 4 else "height"
        rows.append([label, role, fmt(mae, 5), fmt(rmse, 5), quality])
    return rows


def layer_judgement(model_name, layer_data):
    rmse = layer_data["rmse"]
    if not rmse:
        return "Judgement: per-layer metrics are not available."
    density = rmse[:4]
    height = rmse[4:8]
    density_mean = sum(density) / max(len(density), 1)
    height_mean = sum(height) / max(len(height), 1)
    worst_idx = max(range(len(rmse)), key=lambda i: rmse[i])
    worst = CHANNEL_LABELS[worst_idx]

    if model_name == "U-Net":
        return (
            f"Judgement: U-Net layer recovery is usable but not perfect. "
            f"Mean density RMSE = {density_mean:.5f}, mean height RMSE = {height_mean:.5f}, "
            f"and the weakest channel is {worst}. This supports the diagnosis that U-Net preserves the main occupancy structure, "
            f"but some layer-wise geometry, especially height-related structure, is weakened."
        )
    if model_name == "Pix2Pix":
        return (
            f"Judgement: Pix2Pix also loses layer detail. Mean density RMSE = {density_mean:.5f}, "
            f"mean height RMSE = {height_mean:.5f}, and the weakest channel is {worst}. "
            f"It is more conservative in occupancy, but that does not fully solve the layer-preservation issue."
        )
    return (
        f"Judgement: Diffusion v3 layer recovery is poor. Mean density RMSE = {density_mean:.5f}, "
        f"mean height RMSE = {height_mean:.5f}, and the weakest channel is {worst}. "
        f"The large per-layer errors confirm that Diffusion v3 does not preserve reliable 8-channel BEV structure."
    )


def build_report():
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    context = make_tables()

    samples = sample_predictions(max_samples=6)
    case_paths = []
    for idx, sample in enumerate(samples, start=1):
        case_path = ASSET_DIR / f"final_models_case_{idx:02d}.png"
        render_model_comparison(sample, case_path)
        case_paths.append(case_path)

    # Backward-compatible names used by earlier reports.
    render_model_comparison(samples[0], ASSET_DIR / "final_models_same_case.png")
    render_model_comparison(samples[-1], ASSET_DIR / "final_models_late_case.png")
    for key, label in [("unet", "U-Net"), ("pix2pix", "Pix2Pix"), ("diffusion", "Diffusion v3")]:
        render_channel_split(samples[0], key, label, ASSET_DIR / f"channel_split_{key}.png")

    build_metric_chart(ASSET_DIR / "final_metric_comparison.png", context["final_metrics"])
    build_diffusion_curve(ASSET_DIR / "diffusion_v3_curve.png")
    build_flowchart(ASSET_DIR / "experiment_flowchart.png")

    unet_agg = context["unet_agg"]
    fm = context["final_metrics"]
    pix_opt = context["pix_optuna"]
    diff = context["diff"]
    layer_metrics = context["layer_metrics"]

    model_rows = [
        ["U-Net", "42 / 43 / 44", "80", "Shared reconstruction + occupancy loss", fmt_pm(unet_agg["masked_occ_iou"]), fmt_pm(unet_agg["masked_rmse"], 5), fmt_pm(unet_agg["fused_full_psnr"], 2), "Best official baseline"],
        ["Pix2Pix", "42", "40", "0.1 * adversarial + shared loss", fmt(fm["Pix2Pix"]["masked_occ_iou"]), fmt(fm["Pix2Pix"]["masked_rmse"], 5), fmt(fm["Pix2Pix"]["fused_full_psnr"], 2), "Valid GAN comparison"],
        ["Diffusion v3", "42", "120", "noise prediction + shared loss", fmt(fm["Diffusion"]["masked_occ_iou"]), fmt(fm["Diffusion"]["masked_rmse"], 5), fmt(fm["Diffusion"]["fused_full_psnr"], 2), "Solid negative baseline"],
    ]

    diagnostics_rows = [
        ["U-Net seed42", fmt(context["unet_seed42"].get("masked_occ_precision")), fmt(context["unet_seed42"].get("masked_occ_recall")), fmt(context["unet_seed42"].get("masked_occ_f1")), "Higher recall, but visible false positives and layer weakening"],
        ["Pix2Pix seed42", fmt(context["pix"].get("masked_occ_precision")), fmt(context["pix"].get("masked_occ_recall")), fmt(context["pix"].get("masked_occ_f1")), "More conservative; precision higher, recall lower"],
        ["Diffusion v3 seed42", fmt(context["diff_test"].get("masked_occ_precision")), fmt(context["diff_test"].get("masked_occ_recall")), fmt(context["diff_test"].get("masked_occ_f1")), "Recall = 1.0, but precision collapses: predicts too much occupied area"],
    ]

    run_rows = [
        ["U-Net initial", "Baseline BEV reconstruction", "Established task pipeline and first baseline"],
        ["U-Net balanced / occupancy-aware", "Added weighted L1 + masked MSE + occupancy BCE", "Improved masked structure recovery"],
        ["U-Net Optuna", "Tuned shared loss weights", "Final shared loss selected"],
        ["U-Net 3 seeds", "Ran seed 42/43/44, 80 epochs", "Official baseline: masked Occ-IoU 0.1494 +/- 0.0023"],
        ["Threshold + diagnostics", "Added thresholded visualization, precision / recall / F1", "Found U-Net recall is high but false positives remain"],
        ["Focal U-Net ablation", "Replaced occ BCE with focal occupancy loss", "Did not beat baseline; rejected"],
        ["Layer-preserving U-Net probes", "Tried per-bin/height terms, then light height loss", "Stable but did not clearly improve layer preservation; baseline kept"],
        ["Pix2Pix smoke", "Initial lambda_adv=1.0", "Adversarial term too strong"],
        ["Pix2Pix full", "Reduced lambda_adv=0.1", "Best Pix2Pix single-seed run completed"],
        ["Pix2Pix Optuna", "Searched lambda_adv from 0.1 to 2.0, including integer-scale weights", "Confirmed lambda_adv=0.1; 1.0/2.0 underperformed"],
        ["Diffusion smoke", "Local low-memory smoke", "Pipeline ran, but early result was not meaningful"],
        ["Diffusion v1 cloud", "80 target but timed out around epoch 31, proxy eval and no timestep conditioning", "Invalid as final comparison"],
        ["Diffusion v2", "Added timestep conditioning and DDIM evaluation", "Correct pipeline, still weak"],
        ["Diffusion v3 final", "Fresh Narval A100 run, 120 epochs, LR schedule, grad clipping", "Completed; solid negative baseline"],
    ]

    def markdown_table(headers, rows):
        out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for row in rows:
            out.append("| " + " | ".join(str(x) for x in row) + " |")
        return "\n".join(out)

    md = f"""# Final BEV Reconstruction Results Report

<span style="color:#2563eb;font-weight:700;">Project:</span> Cooperative masked BEV reconstruction using ego masked BEV + neighbor BEV.  
<span style="color:#2563eb;font-weight:700;">Report date:</span> 2026-04-23  
<span style="color:#2563eb;font-weight:700;">Main conclusion:</span> U-Net remains the strongest final baseline. Pix2Pix is a valid but weaker GAN comparison. Diffusion v3 is now a properly tested negative baseline.

## 1. Executive Checklist

<div class="checklist">

- [DONE] Built the 8-channel BEV reconstruction pipeline and fixed train / val / test protocol.
- [DONE] Tuned and locked the shared U-Net reconstruction loss with Optuna.
- [DONE] Completed the final U-Net 3-seed run.
- [DONE] Added thresholded visualization and split Occ-IoU into precision / recall / F1.
- [DONE] Diagnosed the U-Net limitation: false positives plus layer-wise information weakening.
- [DONE] Tested focal occupancy loss and layer-preserving U-Net probes; neither beat the official baseline.
- [DONE] Completed Pix2Pix smoke, full seed42 run, and adversarial-weight Optuna search.
- [DONE] Confirmed Pix2Pix best `lambda_adv = 0.1`; integer-scale weights such as `1.0` and `2.0` were worse.
- [DONE] Fixed Diffusion implementation with timestep conditioning and DDIM-style sampled evaluation.
- [DONE] Completed a fresh Narval A100 Diffusion v3 full run for 120 epochs.

</div>

## 2. Experiment Evolution

![Experiment flowchart]({rel(ASSET_DIR / "experiment_flowchart.png")})

## 3. Training / Design Timeline

{markdown_table(["Stage", "What changed", "Outcome"], run_rows)}

## 4. Final Cross-Model Quantitative Comparison

![Metric comparison]({rel(ASSET_DIR / "final_metric_comparison.png")})

{markdown_table(["Model", "Seed(s)", "Epochs", "Final loss/objective", "Masked Occ-IoU (higher better)", "Masked RMSE (lower better)", "Fused Full PSNR (higher better)", "Status"], model_rows)}

## 5. Occupancy Diagnostics

These diagnostics explain why RMSE/PSNR alone are not enough. U-Net has the best main result, Pix2Pix is more conservative, and Diffusion predicts too much of the masked region as occupied.

{markdown_table(["Model", "Precision", "Recall", "F1", "Interpretation"], diagnostics_rows)}

## 6. Final Loss Definitions

Common shared loss:

```text
L_shared =
  0.8082 * masked_weighted_L1
+ 0.1918 * masked_MSE
+ 0.2784 * masked_occ_BCE
```

Occupancy-related internal hyperparameters:

```text
occ_weight = 2.7594
occ_pos_weight = 8
occ_threshold = 0.07
occ_logit_temp = 0.03
```

Model-specific objectives:

```text
U-Net      : L_unet      = L_shared
Pix2Pix    : L_pix2pix    = 0.1 * L_adv + L_shared
Diffusion  : L_diffusion  = L_noise + L_shared
```

## 7. Final Qualitative Outputs

The following panels show six different test cases using the same final checkpoints. Each panel uses the same layout:
`Masked Ego | Ground Truth | U-Net | Pix2Pix | Diffusion v3`.

### Case 1

![Final models case 1]({rel(case_paths[0])})

### Case 2

![Final models case 2]({rel(case_paths[1])})

### Case 3

![Final models case 3]({rel(case_paths[2])})

### Case 4

![Final models case 4]({rel(case_paths[3])})

### Case 5

![Final models case 5]({rel(case_paths[4])})

### Case 6

![Final models case 6]({rel(case_paths[5])})

## 8. Per-Layer Reconstruction Views

The split view directly checks whether the model preserves the 8 BEV channels. `Abs Diff` means `|prediction - ground truth|`; brighter areas are larger per-layer errors.

### U-Net

![U-Net channel split]({rel(ASSET_DIR / "channel_split_unet.png")})

{markdown_table(["Channel", "Role", "Masked MAE", "Masked RMSE", "Quality"], layer_table_rows(layer_metrics["U-Net"]))}

{layer_judgement("U-Net", layer_metrics["U-Net"])}

### Pix2Pix

![Pix2Pix channel split]({rel(ASSET_DIR / "channel_split_pix2pix.png")})

{markdown_table(["Channel", "Role", "Masked MAE", "Masked RMSE", "Quality"], layer_table_rows(layer_metrics["Pix2Pix"]))}

{layer_judgement("Pix2Pix", layer_metrics["Pix2Pix"])}

### Diffusion v3

![Diffusion channel split]({rel(ASSET_DIR / "channel_split_diffusion.png")})

{markdown_table(["Channel", "Role", "Masked MAE", "Masked RMSE", "Quality"], layer_table_rows(layer_metrics["Diffusion v3"]))}

{layer_judgement("Diffusion v3", layer_metrics["Diffusion v3"])}

## 9. Diffusion v3 Final Run

Diffusion v3 was the corrected full run:

- Narval A100 run.
- `seed = 42`.
- `epochs = 120`.
- `lr = 5e-5`, `min_lr = 5e-6`.
- `warmup_epochs = 2`.
- `grad_clip = 1.0`.
- `sample_steps = 25`.
- `val_every = 10`.
- timestep conditioning enabled.
- DDIM-style sampled validation/test enabled.

![Diffusion curve]({rel(ASSET_DIR / "diffusion_v3_curve.png")})

Final Diffusion v3 result:

```text
best epoch              = {diff["best_epoch"]}
test masked Occ-IoU     = {context["diff_test"]["masked_occ_iou"]:.4f}
test masked precision   = {context["diff_test"]["masked_occ_precision"]:.4f}
test masked recall      = {context["diff_test"]["masked_occ_recall"]:.4f}
test masked RMSE        = {context["diff_test"]["masked_rmse"]:.4f}
test fused full PSNR    = {context["diff_test"]["fused_full_psnr"]:.2f} dB
```

Interpretation: Diffusion v3 is no longer an invalid implementation run. It completed a full corrected 120-epoch training run, but the output occupancy collapses toward predicting too much occupied area. This gives recall `1.0`, but precision and Occ-IoU stay low. Therefore, it is a solid negative baseline rather than a competitive final model.

## 10. Final Conclusion

<div class="conclusion">

The final model ranking is clear:

1. **U-Net** is the official best baseline by masked Occ-IoU and RMSE.
2. **Pix2Pix** is a valid adversarial comparison, but it does not beat U-Net. Optuna confirmed that the small adversarial weight `0.1` is best.
3. **Diffusion v3** was properly corrected and fully trained, but it remains non-competitive. The main failure mode is over-predicting occupancy in the masked region.

</div>

## 11. Source Files

- U-Net 3-seed summary: `{UNET_SUMMARY}`
- U-Net seed42 diagnostics: `{UNET_REFRESH}`
- Pix2Pix summary: `{PIX_SUMMARY}`
- Pix2Pix Optuna best params: `{PIX_OPTUNA}`
- Diffusion v3 summary: `{DIFF_SUMMARY}`
- Diffusion v3 history: `{DIFF_HISTORY}`

"""

    md_path = REPORT_ROOT / "final_results_report_2026-04-23.md"
    html_path = REPORT_ROOT / "final_results_report_2026-04-23.html"
    pdf_path = REPORT_ROOT / "final_results_report_2026-04-23.pdf"
    md_path.write_text(md, encoding="utf-8")

    css = """
body { font-family: "Segoe UI", Arial, sans-serif; color:#172033; max-width: 1120px; margin: 36px auto; line-height:1.48; }
h1 { color:#0f3d77; font-size: 34px; border-bottom: 4px solid #60a5fa; padding-bottom: 10px; }
h2 { color:#155e75; margin-top: 34px; border-left: 7px solid #38bdf8; padding-left: 12px; }
h3 { color:#334155; }
table { border-collapse: collapse; width: 100%; margin: 14px 0 24px; font-size: 13px; }
th { background: #0f3d77; color: white; padding: 8px; text-align: left; }
td { border: 1px solid #d8e2ef; padding: 7px; vertical-align: top; }
tr:nth-child(even) td { background: #f8fafc; }
img { max-width: 100%; border-radius: 8px; border: 1px solid #dbe4f0; box-shadow: 0 4px 18px rgba(15,23,42,0.08); margin: 10px 0 20px; }
code, pre { background:#0f172a; color:#e2e8f0; border-radius:8px; }
pre { padding: 14px; overflow-x:auto; }
.checklist { background:#eff6ff; border:1px solid #bfdbfe; border-radius:12px; padding:12px 18px; }
.conclusion { background:#ecfdf5; border:1px solid #86efac; border-radius:12px; padding:12px 18px; }
"""

    body = md_to_html(md)
    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Final BEV Reconstruction Results Report</title>
<style>{css}</style>
</head>
<body>{body}</body>
</html>
"""
    html_path.write_text(html_doc, encoding="utf-8")

    chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
    if chrome.exists():
        subprocess.run(
            [
                str(chrome),
                "--headless=new",
                "--disable-gpu",
                f"--print-to-pdf={pdf_path}",
                "--print-to-pdf-no-header",
                html_path.as_uri(),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    print(f"Markdown: {md_path}")
    print(f"HTML:     {html_path}")
    print(f"PDF:      {pdf_path}")


def md_to_html(md: str) -> str:
    """Small markdown-to-html converter via pandoc if available."""
    tmp_md = REPORT_ROOT / "_tmp_report.md"
    tmp_html = REPORT_ROOT / "_tmp_report_body.html"
    tmp_md.write_text(md, encoding="utf-8")
    try:
        subprocess.run(["pandoc", str(tmp_md), "-f", "gfm", "-t", "html", "-o", str(tmp_html)], check=True)
        return tmp_html.read_text(encoding="utf-8")
    finally:
        if tmp_md.exists():
            tmp_md.unlink()
        if tmp_html.exists():
            tmp_html.unlink()


if __name__ == "__main__":
    build_report()
