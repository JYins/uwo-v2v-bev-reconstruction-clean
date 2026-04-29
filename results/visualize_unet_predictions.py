#!/usr/bin/env python3
"""
Save qualitative prediction panels for trained reconstruction models.

Panel layout:
  Masked Ego | Neighbor | Model Output | Ground Truth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset import BEVReconstructionDataset, discover_available_splits
from train import masked_mse
from unet import UNet
from visualize_4columns import bev_to_color


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = SCRIPT_DIR / "dataset_prepared"
DEFAULT_TRAINING_ROOT = SCRIPT_DIR / "training"
DEFAULT_OUTPUT_DIR = DEFAULT_TRAINING_ROOT / "results" / "predictions"


def parse_features(text):
    return [int(part) for part in text.split(",") if part.strip()]


def load_model(checkpoint_path: Path, device, features_override=None, model_kind="unet"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    features = checkpoint.get("features", features_override or [16, 32, 64, 128])
    model = UNet(in_channels=16, out_channels=8, features=features).to(device)
    if model_kind == "unet":
        state = checkpoint["model_state_dict"]
    elif model_kind == "pix2pix":
        state = checkpoint["generator"]
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint, features


def make_panel(masked_bev, neighbor_bev, pred_bev, target_bev, title_text, output_label):
    import cv2

    masked_img = bev_to_color(masked_bev)
    neighbor_img = bev_to_color(neighbor_bev)
    pred_img = bev_to_color(pred_bev)
    target_img = bev_to_color(target_bev)

    images = [masked_img, neighbor_img, pred_img, target_img]
    labels = [
        ("Masked Ego", (80, 180, 255)),
        ("Neighbor", (255, 200, 80)),
        (output_label, (255, 255, 255)),
        ("Ground Truth", (80, 255, 180)),
    ]

    height, width = masked_img.shape[:2]
    gap = 4
    header = 50
    footer = 28
    canvas = np.ones((height + header + footer, 4 * width + 3 * gap, 3), dtype=np.uint8) * 20

    font = cv2.FONT_HERSHEY_SIMPLEX
    for col_idx, (img, (label, color)) in enumerate(zip(images, labels)):
        x = col_idx * (width + gap)
        canvas[header:header + height, x:x + width] = img
        cv2.putText(canvas, label, (x + 8, 30), font, 0.55, color, 1)

    cv2.putText(canvas, title_text, (10, header + height + 20), font, 0.45, (180, 180, 180), 1)
    return canvas


def choose_sample_indices(n_items, max_images):
    if n_items <= max_images:
        return list(range(n_items))
    # Spread samples across the whole test set, not only the first few frames.
    return np.linspace(0, n_items - 1, num=max_images, dtype=int).tolist()


def main():
    parser = argparse.ArgumentParser(description="Generate prediction panels.")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--training_root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_images", type=int, default=16)
    parser.add_argument("--features", type=str, default="16,32,64,128")
    parser.add_argument("--model_kind", type=str, default="unet", choices=["unet", "pix2pix"])
    parser.add_argument("--output_label", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    elif args.model_kind == "pix2pix":
        checkpoint_path = args.training_root / "checkpoints" / "best_pix2pix.pth"
    else:
        checkpoint_path = args.training_root / "checkpoints" / "best_unet.pth"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    _, _, test_splits = discover_available_splits(args.dataset_root)
    if not test_splits:
        raise SystemExit(f"No test splits found in {args.dataset_root}")

    dataset = BEVReconstructionDataset(args.dataset_root, test_splits, augment=False)
    if len(dataset) == 0:
        raise SystemExit("Test dataset is empty.")

    model, checkpoint, features = load_model(
        checkpoint_path,
        device,
        parse_features(args.features),
        model_kind=args.model_kind,
    )
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch')}")
    print(f"Features: {features}")
    print(f"Device: {device}")
    output_label = args.output_label or ("Pix2Pix Output" if args.model_kind == "pix2pix" else "U-Net Output")

    sample_indices = choose_sample_indices(len(dataset), args.max_images)
    summary_lines = [
        f"{args.model_kind} prediction samples",
        f"Checkpoint: {checkpoint_path}",
        f"Epoch: {checkpoint.get('epoch')}",
        f"Num images: {len(sample_indices)}",
        "",
    ]

    for idx in sample_indices:
        inp, tgt, mask = dataset[idx]
        info = dataset.get_info(idx)

        inp_batch = inp.unsqueeze(0).to(device)
        tgt_batch = tgt.unsqueeze(0).to(device)
        mask_batch = mask.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(inp_batch)

        mse = masked_mse(pred, tgt_batch, mask_batch).item()

        masked_bev = inp[:8].numpy().transpose(1, 2, 0)
        neighbor_bev = inp[8:].numpy().transpose(1, 2, 0)
        pred_bev = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        target_bev = tgt.numpy().transpose(1, 2, 0)

        title = f"{info['split']}/{info['scene']} frame {info['frame']} | masked MSE {mse:.6f}"
        panel = make_panel(masked_bev, neighbor_bev, pred_bev, target_bev, title, output_label)

        filename = f"{info['split']}_{info['scene']}_{info['frame']}.png"
        save_path = output_dir / filename

        import cv2

        cv2.imwrite(str(save_path), panel)
        print(f"Saved: {save_path}")
        summary_lines.append(f"{filename}: masked_mse={mse:.6f}")

    summary_path = output_dir / "prediction_samples.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
