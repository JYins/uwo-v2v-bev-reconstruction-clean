#!/usr/bin/env python3
"""
Quick visualization script for dataset sanity check.

Panel order:
  LiDAR | Ego BEV | Masked Ego | Neighbor BEV
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR / "dataset_prepared"
OUTPUT_DIR = DATASET_ROOT / "visualizations"

X_RANGE = (-40.0, 40.0)
Y_RANGE = (-40.0, 40.0)
RESOLUTION = 0.16
BEV_W = 500
BEV_H = 500

HEIGHT_BINS = [
    (-3.0, -1.5),
    (-1.5, 0.0),
    (0.0, 1.0),
    (1.0, 2.0),
]
NUM_BINS = len(HEIGHT_BINS)
BIN_COLORS = [
    (255, 120, 40),
    (40, 230, 120),
    (40, 180, 255),
    (200, 60, 255),
]
VIS_DILATE_KERNEL = 2
FRAMES_PER_SCENE = 8
VIDEO_FPS = 10


def _bev_to_color_density_only(bev):
    import cv2

    height, width = bev.shape[:2]
    img = np.zeros((height, width, 3), dtype=np.float64)
    weights = np.zeros((height, width), dtype=np.float64)

    kernel = None
    if VIS_DILATE_KERNEL > 0:
        kernel_size = VIS_DILATE_KERNEL * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    for bin_idx in range(NUM_BINS):
        density = bev[:, :, bin_idx].astype(np.float64)
        if density.max() == 0:
            continue

        if kernel is not None:
            density_u8 = (np.clip(density, 0, 1) * 255).astype(np.uint8)
            density_u8 = cv2.dilate(density_u8, kernel, iterations=1)
            density = density_u8.astype(np.float64) / 255.0

        for channel_idx in range(3):
            img[:, :, channel_idx] += density * BIN_COLORS[bin_idx][channel_idx]
        weights += density

    nonzero = weights > 0
    for channel_idx in range(3):
        img[:, :, channel_idx][nonzero] /= weights[nonzero]

    brightness = np.clip(weights / max(weights.max(), 1e-8), 0, 1)
    brightness = np.power(brightness, 0.4)
    for channel_idx in range(3):
        img[:, :, channel_idx] *= brightness

    return np.clip(img, 0, 255).astype(np.uint8)


def _bev_to_color_multi_height(bev):
    density = np.clip(bev[:, :, :NUM_BINS].astype(np.float64), 0.0, 1.0)
    height_feat = np.clip(bev[:, :, NUM_BINS:NUM_BINS * 2].astype(np.float64), 0.0, 1.0)

    # Use density as the main occupancy signal, then let height modulate brightness.
    score = density * (0.7 + 0.3 * height_feat)
    dominant = np.argmax(score, axis=2)
    dominant_score = np.max(score, axis=2)
    dominant_height = np.take_along_axis(height_feat, dominant[:, :, None], axis=2)[:, :, 0]

    # A small weighted blend keeps overlapping bins from looking too harsh,
    # but the dominant bin still decides the main color so layers stay visible.
    blend = np.zeros((bev.shape[0], bev.shape[1], 3), dtype=np.float64)
    blend_weight = score.sum(axis=2)
    for bin_idx in range(NUM_BINS):
        for ch in range(3):
            blend[:, :, ch] += score[:, :, bin_idx] * BIN_COLORS[bin_idx][ch]

    img = np.zeros((bev.shape[0], bev.shape[1], 3), dtype=np.float64)
    for bin_idx in range(NUM_BINS):
        mask = dominant == bin_idx
        for ch in range(3):
            img[:, :, ch][mask] = BIN_COLORS[bin_idx][ch]

    nonzero = blend_weight > 1e-8
    for ch in range(3):
        blend[:, :, ch][nonzero] /= blend_weight[nonzero]
    img = 0.8 * img + 0.2 * blend

    brightness = np.power(np.clip(dominant_score, 0.0, 1.0), 0.45)
    brightness *= 0.55 + 0.45 * dominant_height
    for ch in range(3):
        img[:, :, ch] *= brightness

    return np.clip(img, 0, 255).astype(np.uint8)


def bev_to_color(bev):
    if bev.ndim != 3:
        raise ValueError(f"Expected HxWxC BEV, got shape {bev.shape}")

    channels = bev.shape[2]
    if channels >= NUM_BINS * 2:
        return _bev_to_color_multi_height(bev)
    if channels >= NUM_BINS:
        return _bev_to_color_density_only(bev[:, :, :NUM_BINS])
    raise ValueError(f"Expected at least {NUM_BINS} channels for visualization, got {channels}")


def render_lidar(filepath):
    import cv2
    import open3d as o3d

    img = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
    try:
        pcd = o3d.io.read_point_cloud(str(filepath))
        points = np.asarray(pcd.points, dtype=np.float32)
    except Exception:
        return img

    if len(points) == 0:
        return img

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    mask = (
        (x >= X_RANGE[0]) & (x < X_RANGE[1]) &
        (y >= Y_RANGE[0]) & (y < Y_RANGE[1])
    )
    x, y, z = x[mask], y[mask], z[mask]
    if len(x) == 0:
        return img

    col = np.clip(((x - X_RANGE[0]) / RESOLUTION).astype(np.int32), 0, BEV_W - 1)
    row = np.clip(((Y_RANGE[1] - y) / RESOLUTION).astype(np.int32), 0, BEV_H - 1)

    z_lo, z_hi = HEIGHT_BINS[0][0], HEIGHT_BINS[-1][1]
    z_norm = np.clip((z - z_lo) / (z_hi - z_lo), 0, 1)
    img[row, col, 0] = (255 * (1 - z_norm)).astype(np.uint8)
    img[row, col, 1] = (255 * z_norm).astype(np.uint8)

    if VIS_DILATE_KERNEL > 0:
        kernel_size = VIS_DILATE_KERNEL * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        img = cv2.dilate(img, kernel, iterations=1)

    return img


def make_4col(lidar_img, ego_img, masked_img, neighbor_img, mask_2d, frame_id, split_name, scene_name):
    import cv2

    height, width = ego_img.shape[:2]
    gap = 4
    header = 50
    canvas = np.ones((height + header, 4 * width + 3 * gap, 3), dtype=np.uint8) * 20

    masked_display = masked_img.copy()
    if mask_2d is not None:
        # Make the masked sector obvious, otherwise later easy to miss by eye.
        red = np.zeros_like(masked_display)
        red[:, :, 2] = 80
        masked_region = mask_2d < 0.5
        masked_display[masked_region] = cv2.addWeighted(
            masked_display[masked_region], 0.4, red[masked_region], 0.6, 0
        )

    columns = [lidar_img, ego_img, masked_display, neighbor_img]
    labels = [
        ("Original LiDAR", (170, 170, 170)),
        ("Ego BEV (Ground Truth)", (80, 255, 180)),
        ("Masked Ego (Sector)", (80, 180, 255)),
        ("Neighbor BEV", (255, 200, 80)),
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    for col_idx, (col_img, (label, color)) in enumerate(zip(columns, labels)):
        x = col_idx * (width + gap)
        canvas[header:header + height, x:x + width] = col_img
        cv2.putText(canvas, label, (x + 8, 32), font, 0.55, color, 1)

    cv2.putText(
        canvas,
        f"{split_name}/{scene_name} | Frame {frame_id}",
        (max(10, canvas.shape[1] - 560), 18),
        font,
        0.4,
        (100, 100, 100),
        1,
    )
    return canvas


def discover_processed(dataset_root: Path):
    entries = []
    if not dataset_root.is_dir():
        return entries

    for split_dir in sorted(dataset_root.iterdir()):
        if not split_dir.is_dir():
            continue
        if not any(split_dir.name.startswith(prefix) for prefix in ("train_", "test_", "val_")):
            continue

        for scene_dir in sorted(split_dir.iterdir()):
            if (scene_dir / "ego_bev").is_dir():
                entries.append({"split": split_dir.name, "scene": scene_dir.name, "path": scene_dir})
    return entries


def generate_images(dataset_root: Path, output_dir: Path):
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    entries = discover_processed(dataset_root)
    if not entries:
        print(f"  ERROR: no processed scenes found in {dataset_root}")
        sys.exit(1)

    print(f"  Found {len(entries)} processed scenes")

    for entry in entries:
        split_name = entry["split"]
        scene_name = entry["scene"]
        base = entry["path"]
        print(f"\n  [{split_name}/{scene_name}]")

        ego_dir = base / "ego_bev"
        masked_dir = base / "masked_ego"
        neighbor_dir = base / "neighbor_bev"
        lidar_dir = base / "original_lidar"
        mask_path = base / "sector_mask.npy"

        frames = sorted(path.stem for path in ego_dir.glob("*.npy"))
        if not frames:
            continue

        mask_2d = np.load(mask_path) if mask_path.exists() else None
        step = max(1, len(frames) // FRAMES_PER_SCENE)
        samples = frames[::step][:FRAMES_PER_SCENE]

        split_output_dir = output_dir / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)

        for frame_id in samples:
            ego_bev = np.load(ego_dir / f"{frame_id}.npy")
            masked_bev = np.load(masked_dir / f"{frame_id}.npy")
            neighbor_bev = np.load(neighbor_dir / f"{frame_id}.npy")

            ego_img = bev_to_color(ego_bev)
            masked_img = bev_to_color(masked_bev)
            neighbor_img = bev_to_color(neighbor_bev)

            pcd_path = lidar_dir / f"{frame_id}.pcd"
            lidar_img = render_lidar(pcd_path) if pcd_path.exists() else np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)

            canvas = make_4col(lidar_img, ego_img, masked_img, neighbor_img, mask_2d, frame_id, split_name, scene_name)
            save_path = split_output_dir / f"{scene_name}_{frame_id}.png"
            cv2.imwrite(str(save_path), canvas)
            print(f"    {save_path}")


def generate_videos(dataset_root: Path):
    import cv2

    entries = discover_processed(dataset_root)
    video_dir = dataset_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        split_name = entry["split"]
        scene_name = entry["scene"]
        base = entry["path"]

        ego_dir = base / "ego_bev"
        masked_dir = base / "masked_ego"
        neighbor_dir = base / "neighbor_bev"
        lidar_dir = base / "original_lidar"
        mask_path = base / "sector_mask.npy"

        frames = sorted(path.stem for path in ego_dir.glob("*.npy"))
        if not frames:
            continue

        mask_2d = np.load(mask_path) if mask_path.exists() else None

        gap = 4
        header = 50
        video_w = 4 * BEV_W + 3 * gap
        video_h = BEV_H + header
        video_path = video_dir / f"{split_name}_{scene_name}_4col.mp4"

        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            VIDEO_FPS,
            (video_w, video_h),
        )
        if not writer.isOpened():
            print(f"  WARNING: cannot create {video_path}")
            continue

        print(f"\n  Video: {video_path} ({len(frames)} frames)")
        for idx, frame_id in enumerate(frames, start=1):
            ego_bev = np.load(ego_dir / f"{frame_id}.npy")
            masked_bev = np.load(masked_dir / f"{frame_id}.npy")
            neighbor_bev = np.load(neighbor_dir / f"{frame_id}.npy")

            ego_img = bev_to_color(ego_bev)
            masked_img = bev_to_color(masked_bev)
            neighbor_img = bev_to_color(neighbor_bev)

            pcd_path = lidar_dir / f"{frame_id}.pcd"
            lidar_img = render_lidar(pcd_path) if pcd_path.exists() else np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
            writer.write(make_4col(lidar_img, ego_img, masked_img, neighbor_img, mask_2d, frame_id, split_name, scene_name))

            if idx % 30 == 0 or idx == len(frames):
                print(f"    [{idx}/{len(frames)}]")

        writer.release()
        print(f"    Done: {video_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 4-column visualizations for processed BEV data.")
    parser.add_argument("--dataset_root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip_videos", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print()
    print("=" * 70)
    print("  4-Column Visualization")
    print("=" * 70)
    print(f"\n  Dataset: {args.dataset_root}")
    print(f"  Output:  {args.output_dir}")

    try:
        import cv2  # noqa: F401
        import open3d  # noqa: F401
    except ImportError as exc:
        print(f"\n  ERROR: missing dependency: {exc}")
        sys.exit(1)

    print("\n  --- Static Images ---")
    generate_images(args.dataset_root.resolve(), args.output_dir.resolve())

    if not args.skip_videos:
        print("\n  --- Videos ---")
        generate_videos(args.dataset_root.resolve())

    print(f"\n{'=' * 70}")
    print("  Done")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
