#!/usr/bin/env python3
"""
Render per-channel prediction vs ground-truth grids.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import BEVReconstructionDataset, discover_available_splits
from unet import UNet


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = SCRIPT_DIR / "dataset_prepared"
CHANNEL_LABELS = ["d0", "d1", "d2", "d3", "h0", "h1", "h2", "h3"]


def parse_features(text):
    return [int(part) for part in text.split(",") if part.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Render per-channel prediction grids.")
    p.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--training_root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--features", type=str, default="16,32,64,128")
    p.add_argument("--model_kind", type=str, default="unet", choices=["unet", "pix2pix"])
    p.add_argument("--max_images", type=int, default=6)
    p.add_argument("--tile_size", type=int, default=140)
    return p.parse_args()


def load_model(checkpoint_path: Path, device, model_kind, features):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    features = checkpoint.get("features", features)
    model = UNet(in_channels=16, out_channels=8, features=features).to(device)
    state_key = "generator" if model_kind == "pix2pix" else "model_state_dict"
    model.load_state_dict(checkpoint[state_key])
    model.eval()
    return model, checkpoint


def choose_sample_indices(n_items, max_images):
    if n_items <= max_images:
        return list(range(n_items))
    return np.linspace(0, n_items - 1, num=max_images, dtype=int).tolist()


def channel_to_tile(channel, tile_size):
    channel_u8 = (np.clip(channel, 0.0, 1.0) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(channel_u8, cv2.COLORMAP_TURBO)
    return cv2.resize(colored, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)


def make_grid(pred_bev, target_bev, title, tile_size):
    rows = [("Prediction", pred_bev), ("Ground Truth", target_bev), ("Abs Diff", np.abs(pred_bev - target_bev))]
    gap = 4
    header = 36
    footer = 26
    row_gap = 24
    height = len(rows) * tile_size + (len(rows) - 1) * row_gap + header + footer
    width = len(CHANNEL_LABELS) * tile_size + (len(CHANNEL_LABELS) - 1) * gap
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 20

    font = cv2.FONT_HERSHEY_SIMPLEX
    for ch_idx, label in enumerate(CHANNEL_LABELS):
        x = ch_idx * (tile_size + gap)
        cv2.putText(canvas, label, (x + 8, 22), font, 0.55, (220, 220, 220), 1)

    for row_idx, (row_label, bev) in enumerate(rows):
        y = header + row_idx * (tile_size + row_gap)
        cv2.putText(canvas, row_label, (6, y - 6), font, 0.48, (180, 180, 180), 1)
        for ch_idx in range(8):
            x = ch_idx * (tile_size + gap)
            canvas[y:y + tile_size, x:x + tile_size] = channel_to_tile(bev[:, :, ch_idx], tile_size)

    cv2.putText(canvas, title, (10, height - 8), font, 0.42, (160, 160, 160), 1)
    return canvas


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        if args.model_kind == "pix2pix":
            checkpoint_path = args.training_root / "checkpoints" / "best_pix2pix.pth"
        else:
            checkpoint_path = args.training_root / "checkpoints" / "best_unet.pth"
    checkpoint_path = checkpoint_path.resolve()

    _, _, test_splits = discover_available_splits(args.dataset_root)
    dataset = BEVReconstructionDataset(args.dataset_root, test_splits, augment=False)
    model, checkpoint = load_model(checkpoint_path, device, args.model_kind, parse_features(args.features))

    sample_indices = choose_sample_indices(len(dataset), args.max_images)
    lines = [
        f"{args.model_kind} per-channel split view",
        f"checkpoint: {checkpoint_path}",
        f"epoch: {checkpoint.get('epoch')}",
        "",
    ]

    for idx in sample_indices:
        inp, tgt, _ = dataset[idx]
        info = dataset.get_info(idx)
        with torch.no_grad():
            pred = model(inp.unsqueeze(0).to(device))

        pred_bev = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        target_bev = tgt.numpy().transpose(1, 2, 0)
        title = f"{info['split']}/{info['scene']} frame {info['frame']}"
        grid = make_grid(pred_bev, target_bev, title, args.tile_size)
        filename = f"{info['split']}_{info['scene']}_{info['frame']}.png"
        save_path = output_dir / filename
        cv2.imwrite(str(save_path), grid)
        lines.append(filename)

    (output_dir / "channel_split_samples.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {output_dir}")


if __name__ == "__main__":
    main()
