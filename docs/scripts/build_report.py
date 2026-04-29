#!/usr/bin/env python3
"""
Build markdown report from training artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = SCRIPT_DIR / "dataset_prepared"
DEFAULT_TRAINING_ROOT = SCRIPT_DIR / "training"
DEFAULT_REPORTS_ROOT = SCRIPT_DIR / "reports"


def load_history(csv_path):
    if not csv_path.exists():
        return []
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_metrics(json_path):
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_plot(history, out_path):
    if not history:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    epochs = [int(r["epoch"]) for r in history]
    train_loss = [float(r["train_loss"]) for r in history]

    val_rows = [r for r in history if r.get("val_masked_mae", "") != ""]
    val_epochs = [int(r["epoch"]) for r in val_rows]
    val_mae = [float(r["val_masked_mae"]) for r in val_rows]
    val_rmse = [float(r["val_masked_rmse"]) for r in val_rows]
    val_psnr = [float(r["val_psnr"]) for r in val_rows]
    val_fused_psnr = [float(r["val_fused_full_psnr"]) for r in val_rows if r.get("val_fused_full_psnr", "") != ""]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, train_loss, color="#1f77b4")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")

    axes[1].plot(val_epochs, val_mae, label="MAE", color="#d62728")
    axes[1].plot(val_epochs, val_rmse, label="RMSE", color="#2ca02c")
    axes[1].set_title("Validation Masked")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(val_epochs, val_psnr, label="Masked PSNR", color="#9467bd")
    if len(val_fused_psnr) == len(val_epochs):
        axes[2].plot(val_epochs, val_fused_psnr, label="Fused full PSNR", color="#ff7f0e")
    axes[2].set_title("Validation PSNR")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def first_files(path, pattern, n):
    if not path.exists():
        return []
    return sorted(path.glob(pattern))[:n]


def fmt(x, digits=6):
    if x is None:
        return "N/A"
    return f"{float(x):.{digits}f}"


def fmt_psnr(x):
    if x is None:
        return "N/A"
    return f"{float(x):.2f}"


def per_layer_table(values, name):
    if not values:
        return [f"{name}: N/A"]
    lines = [f"### {name}", "", "| Channel | Value |", "|---|---|"]
    for i, v in enumerate(values):
        lines.append(f"| ch{i} | {float(v):.6f} |")
    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--training_root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--reports_root", type=Path, default=DEFAULT_REPORTS_ROOT)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    training_root = args.training_root.resolve()
    reports_root = args.reports_root.resolve()
    reports_root.mkdir(parents=True, exist_ok=True)

    dataset_stats = dataset_root / "dataset_stats.txt"
    summary_txt = training_root / "results" / "unet_summary.txt"
    history_csv = training_root / "results" / "training_history.csv"
    metrics_json = training_root / "results" / "test_metrics_refresh.json"
    if not metrics_json.exists():
        metrics_json = training_root / "results" / "test_metrics.json"
    prediction_dir = training_root / "results" / "predictions"
    vis_dir = dataset_root / "visualizations"

    history = load_history(history_csv)
    metrics_blob = load_metrics(metrics_json)
    curve_path = maybe_plot(history, reports_root / "training_curves.png")

    test_metrics = metrics_blob["test_metrics"] if metrics_blob and metrics_blob.get("test_metrics") else None
    best_metrics = None
    if metrics_blob:
        if metrics_blob.get("best_val_metrics"):
            best_metrics = metrics_blob["best_val_metrics"]
        elif metrics_blob.get("val_metrics"):
            best_metrics = metrics_blob["val_metrics"]

    dataset_preview = first_files(vis_dir, "*/*.png", 4)
    pred_preview = first_files(prediction_dir, "*.png", 6)

    lines = [
        "# V2V Cooperative Perception Report",
        "",
        "## 1) Main takeaway",
        "",
        "- Main table uses masked and fused-full metrics.",
        "- Raw full-image and per-layer metrics are in appendices.",
        "",
        "## 2) Dataset snapshot",
        "",
    ]

    if dataset_stats.exists():
        lines += ["```text", dataset_stats.read_text(encoding="utf-8"), "```", ""]
    else:
        lines += ["Dataset summary file is missing.", ""]

    lines += ["## 3) U-Net main results (masked + fused full)", ""]
    if test_metrics:
        lines += [
            "| Split | Masked MAE | Masked RMSE | Masked Occ-IoU | Masked Prec | Masked Recall | Masked F1 | Fused full PSNR | Fused full Occ-IoU |",
            "|---|---|---|---|---|---|---|---|---|",
            f"| Validation(best) | {fmt(best_metrics.get('masked_mae'))} | {fmt(best_metrics.get('masked_rmse'))} | {fmt(best_metrics.get('masked_occ_iou'), 4)} | {fmt(best_metrics.get('masked_occ_precision'), 4)} | {fmt(best_metrics.get('masked_occ_recall'), 4)} | {fmt(best_metrics.get('masked_occ_f1'), 4)} | {fmt_psnr(best_metrics.get('fused_full_psnr'))} | {fmt(best_metrics.get('fused_full_occ_iou'), 4)} |",
            f"| Test | {fmt(test_metrics.get('masked_mae'))} | {fmt(test_metrics.get('masked_rmse'))} | {fmt(test_metrics.get('masked_occ_iou'), 4)} | {fmt(test_metrics.get('masked_occ_precision'), 4)} | {fmt(test_metrics.get('masked_occ_recall'), 4)} | {fmt(test_metrics.get('masked_occ_f1'), 4)} | {fmt_psnr(test_metrics.get('fused_full_psnr'))} | {fmt(test_metrics.get('fused_full_occ_iou'), 4)} |",
            "",
        ]
    else:
        lines += ["No `test_metrics.json` yet.", ""]

    if curve_path:
        lines += ["## 4) Curves", "", f"![Training curves]({curve_path.as_posix()})", ""]

    lines += ["## 5) Qualitative samples", ""]
    for p in dataset_preview:
        lines.append(f"![Dataset sample]({p.as_posix()})")
    if dataset_preview:
        lines.append("")
    for p in pred_preview:
        lines.append(f"![Prediction sample]({p.as_posix()})")
    if pred_preview:
        lines.append("")

    lines += ["## Appendix A: Raw full-image metrics", ""]
    if test_metrics:
        lines += [
            "| Split | MAE | RMSE | PSNR | Occ-IoU |",
            "|---|---|---|---|---|",
            f"| Validation(best) | {fmt(best_metrics.get('full_mae'))} | {fmt(best_metrics.get('full_rmse'))} | {fmt_psnr(best_metrics.get('full_psnr'))} | {fmt(best_metrics.get('full_occ_iou'), 4)} |",
            f"| Test | {fmt(test_metrics.get('full_mae'))} | {fmt(test_metrics.get('full_rmse'))} | {fmt_psnr(test_metrics.get('full_psnr'))} | {fmt(test_metrics.get('full_occ_iou'), 4)} |",
            "",
        ]
    else:
        lines += ["N/A", ""]

    lines += ["## Appendix B: Per-layer metrics (test, masked)", ""]
    if test_metrics:
        lines += per_layer_table(test_metrics.get("per_layer_masked_mae"), "Per-layer MAE")
        lines += per_layer_table(test_metrics.get("per_layer_masked_rmse"), "Per-layer RMSE")
    else:
        lines += ["N/A", ""]

    lines += ["## Raw summary text", ""]
    if summary_txt.exists():
        lines += ["```text", summary_txt.read_text(encoding="utf-8"), "```", ""]

    report_path = reports_root / "project_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {report_path}")
    if curve_path:
        print(f"Saved: {curve_path}")


if __name__ == "__main__":
    main()
