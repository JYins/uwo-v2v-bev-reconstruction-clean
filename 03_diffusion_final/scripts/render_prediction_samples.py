#!/usr/bin/env python3
"""
Render a few final diffusion v3 prediction panels.

This is intentionally just a small helper for the report figures.
It uses the same DDIM-style sampler as train_diffusion.py, not the older proxy
check that I used while debugging the first diffusion attempts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
REPO_ROOT = MODEL_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(RESULTS_DIR))

from dataset import BEVReconstructionDataset, discover_available_splits  # noqa: E402
from train_diffusion import alpha_bar_schedule, ddim_sample, make_condition  # noqa: E402
from unet import UNet  # noqa: E402
from visualize_4columns import bev_to_color  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Render final diffusion v3 prediction samples.")
    p.add_argument("--dataset_root", type=Path, default=REPO_ROOT / "dataset_prepared")
    p.add_argument(
        "--training_root",
        type=Path,
        default=MODEL_DIR / "local_runs" / "training_diffusion_full_seed42_v3",
    )
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--output_dir", type=Path, default=MODEL_DIR / "results" / "figures" / "prediction_samples")
    p.add_argument("--max_images", type=int, default=8)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--sample_steps", type=int, default=25)
    return p.parse_args()


def choose_sample_indices(n_items, max_images):
    if n_items <= max_images:
        return list(range(n_items))
    return np.linspace(0, n_items - 1, num=max_images, dtype=int).tolist()


def make_panel(masked_bev, neighbor_bev, pred_bev, target_bev, title_text):
    import cv2

    tiles = [
        ("Masked Ego", (80, 180, 255), bev_to_color(masked_bev)),
        ("Neighbor", (255, 200, 80), bev_to_color(neighbor_bev)),
        ("Diffusion v3", (255, 255, 255), bev_to_color(pred_bev)),
        ("Ground Truth", (80, 255, 180), bev_to_color(target_bev)),
    ]

    height, width = tiles[0][2].shape[:2]
    gap = 4
    header = 50
    footer = 28
    canvas = np.ones((height + header + footer, 4 * width + 3 * gap, 3), dtype=np.uint8) * 20

    font = cv2.FONT_HERSHEY_SIMPLEX
    for col_idx, (label, color, img) in enumerate(tiles):
        x = col_idx * (width + gap)
        canvas[header:header + height, x:x + width] = img
        cv2.putText(canvas, label, (x + 8, 30), font, 0.55, color, 1)

    cv2.putText(canvas, title_text, (10, header + height + 20), font, 0.45, (180, 180, 180), 1)
    return canvas


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    training_root = args.training_root.resolve()
    checkpoint_path = args.checkpoint or (training_root / "checkpoints" / "best_diffusion.pth")
    checkpoint_path = checkpoint_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_splits = discover_available_splits(dataset_root)
    dataset = BEVReconstructionDataset(dataset_root, test_splits, augment=False)
    if len(dataset) == 0:
        raise SystemExit(f"No test samples found under {dataset_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNet(
        in_channels=24,
        out_channels=8,
        features=[16, 32, 64, 128],
        time_emb_dim=128,
        final_activation="identity",
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    alpha_bar = alpha_bar_schedule(args.timesteps, device)

    import cv2

    lines = [
        "diffusion v3 DDIM prediction samples",
        f"checkpoint: {checkpoint_path}",
        f"epoch: {checkpoint.get('epoch')}",
        f"sample_steps: {args.sample_steps}",
        "",
    ]

    for idx in choose_sample_indices(len(dataset), args.max_images):
        inp, tgt, mask = dataset[idx]
        info = dataset.get_info(idx)

        inp_batch = inp.unsqueeze(0).to(device)
        mask_batch = mask.unsqueeze(0).to(device)
        cond = make_condition(inp_batch, mask_batch)

        with torch.no_grad():
            pred = ddim_sample(model, cond, mask_batch, alpha_bar, args.timesteps, args.sample_steps)

        masked_bev = inp[:8].numpy().transpose(1, 2, 0)
        neighbor_bev = inp[8:].numpy().transpose(1, 2, 0)
        pred_bev = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        target_bev = tgt.numpy().transpose(1, 2, 0)

        title = f"{info['split']}/{info['scene']} frame {info['frame']}"
        panel = make_panel(masked_bev, neighbor_bev, pred_bev, target_bev, title)
        filename = f"{info['split']}_{info['scene']}_{info['frame']}.png"
        save_path = output_dir / filename
        cv2.imwrite(str(save_path), panel)
        lines.append(filename)
        print(f"Saved: {save_path}")

    summary_path = output_dir / "prediction_samples.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
