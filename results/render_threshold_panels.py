#!/usr/bin/env python3
"""
Render raw and thresholded prediction panels for saved checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import BEVReconstructionDataset, discover_available_splits
from unet import UNet
from visualize_4columns import bev_to_color


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = SCRIPT_DIR / "dataset_prepared"


def parse_features(text):
    return [int(part) for part in text.split(",") if part.strip()]


def parse_thresholds(text):
    return [float(part) for part in text.split(",") if part.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Render thresholded occupancy panels.")
    p.add_argument("--model_kind", choices=["unet", "pix2pix"], required=True)
    p.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--training_root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--features", type=str, default="16,32,64,128")
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--occ_threshold", type=float, default=None)
    p.add_argument("--threshold_candidates", type=str, default="0.03,0.05,0.07,0.10,0.15,0.20")
    p.add_argument("--max_images", type=int, default=8)
    return p.parse_args()


def load_run_config(training_root: Path):
    for name in ["test_metrics.json", "pix2pix_summary.json", "diffusion_summary.json"]:
        path = training_root / "results" / name
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                blob = json.load(f)
            return blob.get("config", {})
    return {}


def load_checkpoint(model_kind, checkpoint_path: Path, device, features):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNet(in_channels=16, out_channels=8, features=features).to(device)
    state_key = "generator" if model_kind == "pix2pix" else "model_state_dict"
    model.load_state_dict(ckpt[state_key])
    model.eval()
    return model, ckpt


def predict_tensor(model, model_kind, inp_batch, device, timesteps):
    with torch.no_grad():
        return model(inp_batch.to(device))


def occ_counts(pred, target, mask, pred_tau, true_tau):
    pred_occ = pred[:, :4].sum(dim=1, keepdim=True) > pred_tau
    true_occ = target[:, :4].sum(dim=1, keepdim=True) > true_tau
    masked = (1.0 - mask) > 0.5
    tp = ((pred_occ & true_occ) & masked).sum().item()
    pred_pos = (pred_occ & masked).sum().item()
    true_pos = (true_occ & masked).sum().item()
    precision = tp / pred_pos if pred_pos > 0 else 1.0
    recall = tp / true_pos if true_pos > 0 else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def pick_threshold(model, model_kind, dataset, device, true_tau, candidates, timesteps):
    scores = []
    for tau in candidates:
        p_sum = 0.0
        r_sum = 0.0
        f_sum = 0.0
        n = 0
        for idx in range(len(dataset)):
            inp, tgt, mask = dataset[idx]
            pred = predict_tensor(model, model_kind, inp.unsqueeze(0), device, timesteps)
            precision, recall, f1 = occ_counts(
                pred.cpu(),
                tgt.unsqueeze(0),
                mask.unsqueeze(0),
                tau,
                true_tau,
            )
            p_sum += precision
            r_sum += recall
            f_sum += f1
            n += 1
        scores.append(
            {
                "threshold": tau,
                "precision": p_sum / max(n, 1),
                "recall": r_sum / max(n, 1),
                "f1": f_sum / max(n, 1),
            }
        )
    scores.sort(key=lambda row: (row["f1"], row["precision"]), reverse=True)
    return scores[0], scores


def threshold_bev(pred_bev, tau):
    occ = pred_bev[:, :, :4].sum(axis=2) > tau
    out = pred_bev.copy()
    out[~occ] = 0.0
    return out


def make_panel(masked_bev, neighbor_bev, pred_bev, thresh_bev, target_bev, title_text):
    import cv2

    tiles = [
        ("Masked Ego", (80, 180, 255), bev_to_color(masked_bev)),
        ("Neighbor", (255, 200, 80), bev_to_color(neighbor_bev)),
        ("Raw Output", (255, 255, 255), bev_to_color(pred_bev)),
        ("Thresholded", (255, 120, 120), bev_to_color(thresh_bev)),
        ("Ground Truth", (80, 255, 180), bev_to_color(target_bev)),
    ]

    height, width = tiles[0][2].shape[:2]
    gap = 4
    header = 50
    footer = 28
    canvas = np.ones((height + header + footer, len(tiles) * width + (len(tiles) - 1) * gap, 3), dtype=np.uint8) * 20

    font = cv2.FONT_HERSHEY_SIMPLEX
    for col_idx, (label, color, img) in enumerate(tiles):
        x = col_idx * (width + gap)
        canvas[header:header + height, x:x + width] = img
        cv2.putText(canvas, label, (x + 8, 30), font, 0.55, color, 1)

    cv2.putText(canvas, title_text, (10, header + height + 20), font, 0.45, (180, 180, 180), 1)
    return canvas


def choose_sample_indices(n_items, max_images):
    if n_items <= max_images:
        return list(range(n_items))
    return np.linspace(0, n_items - 1, num=max_images, dtype=int).tolist()


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    training_root = args.training_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = load_run_config(training_root)
    features = parse_features(args.features)
    if "features" in run_cfg and isinstance(run_cfg["features"], list):
        features = run_cfg["features"]
    occ_threshold = float(run_cfg.get("occ_threshold", args.occ_threshold if args.occ_threshold is not None else 0.07))
    timesteps = int(run_cfg.get("timesteps", args.timesteps))

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint.resolve()
    elif args.model_kind == "pix2pix":
        checkpoint_path = training_root / "checkpoints" / "best_pix2pix.pth"
    elif args.model_kind == "diffusion":
        checkpoint_path = training_root / "checkpoints" / "best_diffusion.pth"
    else:
        checkpoint_path = training_root / "checkpoints" / "best_unet.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.model_kind, checkpoint_path, device, features)

    _, val_splits, test_splits = discover_available_splits(dataset_root)
    val_dataset = BEVReconstructionDataset(dataset_root, val_splits, augment=False)
    test_dataset = BEVReconstructionDataset(dataset_root, test_splits, augment=False)

    candidates = parse_thresholds(args.threshold_candidates)
    best_row, sweep_rows = pick_threshold(
        model,
        args.model_kind,
        val_dataset,
        device,
        occ_threshold,
        candidates,
        timesteps,
    )
    render_tau = best_row["threshold"]

    sample_indices = choose_sample_indices(len(test_dataset), args.max_images)
    summary_lines = [
        f"{args.model_kind} thresholded prediction samples",
        f"checkpoint: {checkpoint_path}",
        f"epoch: {ckpt.get('epoch')}",
        f"occ_threshold(true): {occ_threshold}",
        f"render_threshold(pred): {render_tau}",
        "",
    ]

    import cv2

    for idx in sample_indices:
        inp, tgt, mask = test_dataset[idx]
        info = test_dataset.get_info(idx)
        pred = predict_tensor(model, args.model_kind, inp.unsqueeze(0), device, timesteps)

        masked_bev = inp[:8].numpy().transpose(1, 2, 0)
        neighbor_bev = inp[8:].numpy().transpose(1, 2, 0)
        pred_bev = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        thresh_bev = threshold_bev(pred_bev, render_tau)
        target_bev = tgt.numpy().transpose(1, 2, 0)

        precision, recall, f1 = occ_counts(
            pred.cpu(),
            tgt.unsqueeze(0),
            mask.unsqueeze(0),
            render_tau,
            occ_threshold,
        )
        title = (
            f"{info['split']}/{info['scene']} frame {info['frame']} | "
            f"thr={render_tau:.2f} p={precision:.3f} r={recall:.3f} f1={f1:.3f}"
        )
        panel = make_panel(masked_bev, neighbor_bev, pred_bev, thresh_bev, target_bev, title)
        filename = f"{info['split']}_{info['scene']}_{info['frame']}.png"
        save_path = output_dir / filename
        cv2.imwrite(str(save_path), panel)
        summary_lines.append(f"{filename}: p={precision:.4f} r={recall:.4f} f1={f1:.4f}")

    with open(output_dir / "prediction_samples.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    with open(output_dir / "threshold_sweep.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_kind": args.model_kind,
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": ckpt.get("epoch"),
                "true_occ_threshold": occ_threshold,
                "selected_render_threshold": render_tau,
                "sweep": sweep_rows,
            },
            f,
            indent=2,
        )

    print(f"saved {output_dir}")
    print(f"selected render threshold = {render_tau:.2f}")


if __name__ == "__main__":
    main()
