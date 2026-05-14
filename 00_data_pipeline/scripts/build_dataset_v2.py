#!/usr/bin/env python3
"""
Build a separate dataset_v2 view for BEV reconstruction experiments.

This does not mutate dataset_prepared. It creates a teacher-facing layout with
clean images, noisy images, noise-type metadata, masks, and preprocessing
metadata. Raw clean/neighbor files are symlinked by default to save storage.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


RESOLUTION = 0.16
FRONT_R_MIN = 10.0
FRONT_R_MAX = 38.0
FRONT_RECT_HALF_WIDTH = 15.5
NOISE_TYPES = ("sector75", "front_rect", "front_blob")
PREPROCESS_TYPES = ("none", "register_layernorm")


def parse_args():
    p = argparse.ArgumentParser(description="Build dataset_v2 without modifying dataset_prepared.")
    p.add_argument("--src_root", type=Path, required=True)
    p.add_argument("--dst_root", type=Path, required=True)
    p.add_argument("--materialize_noisy", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Debug limit over base samples.")
    return p.parse_args()


def split_group(split_name):
    if split_name.startswith("train_"):
        return "train"
    if split_name.startswith("val_"):
        return "val"
    if split_name.startswith("test_"):
        return "test"
    return "other"


def generate_front_mask(shape, variant):
    if variant == "sector75":
        raise ValueError("sector75 mask should be read from source sector_mask.npy")

    height, width = shape
    center_row = height // 2
    center_col = width // 2
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    forward_m = (cols - center_col) * RESOLUTION
    lateral_m = (center_row - rows) * RESOLUTION
    in_forward = (forward_m >= FRONT_R_MIN) & (forward_m <= FRONT_R_MAX)

    mask = np.ones((height, width), dtype=np.float32)
    if variant == "front_rect":
        hidden = in_forward & (np.abs(lateral_m) <= FRONT_RECT_HALF_WIDTH)
    elif variant == "front_blob":
        phase = (forward_m - FRONT_R_MIN) / max(FRONT_R_MAX - FRONT_R_MIN, 1e-6)
        center_offset = 2.0 * np.sin(phase * np.pi * 1.3)
        half_width = (
            FRONT_RECT_HALF_WIDTH
            + 3.0 * np.sin(phase * np.pi * 2.0)
            + 1.2 * np.sin(phase * np.pi * 5.0)
        )
        half_width = np.clip(half_width, 10.0, 19.0)
        hidden = in_forward & (np.abs(lateral_m - center_offset) <= half_width)
    else:
        raise ValueError(f"unknown noise type: {variant}")

    mask[hidden] = 0.0
    return mask


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_symlink(src, dst, overwrite=False):
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    os.symlink(src, dst)


def save_npy(path, arr, overwrite=False):
    ensure_parent(path)
    if path.exists() and not overwrite:
        return
    np.save(path, arr)


def rel(path, root):
    return path.relative_to(root).as_posix()


def discover_samples(src_root):
    samples = []
    for split_path in sorted(path for path in src_root.iterdir() if path.is_dir()):
        group = split_group(split_path.name)
        if group == "other":
            continue
        for scene_path in sorted(path for path in split_path.iterdir() if path.is_dir()):
            ego_dir = scene_path / "ego_bev"
            masked_dir = scene_path / "masked_ego"
            neighbor_dir = scene_path / "neighbor_bev"
            mask_path = scene_path / "sector_mask.npy"
            if not (ego_dir.is_dir() and masked_dir.is_dir() and neighbor_dir.is_dir() and mask_path.exists()):
                continue
            for ego_path in sorted(ego_dir.glob("*.npy")):
                frame = ego_path.stem
                masked_path = masked_dir / ego_path.name
                neighbor_path = neighbor_dir / ego_path.name
                if masked_path.exists() and neighbor_path.exists():
                    samples.append(
                        {
                            "split": split_path.name,
                            "split_group": group,
                            "scene": scene_path.name,
                            "frame": frame,
                            "ego": ego_path,
                            "masked": masked_path,
                            "neighbor": neighbor_path,
                            "sector_mask": mask_path,
                        }
                    )
    return samples


def write_noise_type_metadata(dst_root):
    meta_dir = dst_root / "noisetype"
    meta_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "sector75": {
            "label": "sector75",
            "description": "Original front-sector occlusion used as the official main setting.",
            "geometry": {"type": "front_sector", "source": "dataset_prepared/sector_mask.npy"},
        },
        "front_rect": {
            "label": "front_rect",
            "description": "Regular front rectangular occlusion for robustness evaluation.",
            "geometry": {
                "type": "front_rectangle",
                "forward_m": [FRONT_R_MIN, FRONT_R_MAX],
                "lateral_m": [-FRONT_RECT_HALF_WIDTH, FRONT_RECT_HALF_WIDTH],
            },
        },
        "front_blob": {
            "label": "front_blob",
            "description": "Deterministic irregular front occlusion for robustness evaluation.",
            "geometry": {
                "type": "front_irregular_blob",
                "forward_m": [FRONT_R_MIN, FRONT_R_MAX],
                "area": "approximately matched to front_rect, with sinusoidal lateral boundary",
            },
        },
    }
    for name, payload in metadata.items():
        (meta_dir / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_readme(dst_root, materialize_noisy):
    text = f"""# dataset_v2

This folder is a separate dataset view. It does not modify `dataset_prepared`.

Layout:

- `img/clean`: symlinks to clean ego BEV ground truth.
- `neighborimg/raw`: symlinks to raw neighbor BEV.
- `noisyimg/<noise_type>`: masked/noisy ego BEV. `sector75` links the original masked ego files. `front_rect` and `front_blob` are {"materialized `.npy` files" if materialize_noisy else "represented in the manifest and generated by the loader"}.
- `masks/<noise_type>`: mask files for each scene/noise type.
- `noisetype`: JSON definitions for the occlusion/noise types.
- `preprocessedimg`: preprocessing method metadata. Heavy registered neighbor tensors are generated on the fly unless explicitly materialized later.
- `splits`: CSV manifests for train/val/test.
- `manifest.csv`: all rows across splits, noise types, and preprocessing types.

Preprocessing policy:

- `none`: raw neighbor BEV.
- `register_layernorm`: BEV occupancy-centroid registration plus per-layer visible-region calibration.

Scientific note:

The official baseline remains `sector75 + none`. Other noise and preprocessing settings are ablation/robustness experiments.
"""
    (dst_root / "README.md").write_text(text, encoding="utf-8")
    pre_dir = dst_root / "preprocessedimg" / "register_layernorm"
    pre_dir.mkdir(parents=True, exist_ok=True)
    (pre_dir / "README.md").write_text(
        "Registered/pre-layer neighbor BEV is generated by the dataset loader on the fly. "
        "This keeps dataset_v2 compact and reproducible.\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(f"source dataset not found: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)
    for folder in [
        "img/clean",
        "neighborimg/raw",
        "noisyimg",
        "masks",
        "noisetype",
        "preprocessedimg",
        "splits",
    ]:
        (dst_root / folder).mkdir(parents=True, exist_ok=True)

    samples = discover_samples(src_root)
    if args.limit > 0:
        samples = samples[: args.limit]

    write_noise_type_metadata(dst_root)
    write_readme(dst_root, args.materialize_noisy)

    manifest_rows = []
    split_rows = {"train": [], "val": [], "test": []}
    mask_cache = {}

    for idx, sample in enumerate(samples, start=1):
        base_rel = Path(sample["split"]) / sample["scene"] / f"{sample['frame']}.npy"
        clean_dst = dst_root / "img" / "clean" / base_rel
        neighbor_dst = dst_root / "neighborimg" / "raw" / base_rel
        safe_symlink(sample["ego"], clean_dst, overwrite=args.overwrite)
        safe_symlink(sample["neighbor"], neighbor_dst, overwrite=args.overwrite)

        if idx % 500 == 0:
            print(f"processed base samples: {idx}/{len(samples)}", flush=True)

        for noise_type in NOISE_TYPES:
            scene_mask_dst = dst_root / "masks" / noise_type / sample["split"] / sample["scene"] / "mask.npy"
            if noise_type == "sector75":
                safe_symlink(sample["sector_mask"], scene_mask_dst, overwrite=args.overwrite)
            else:
                key = (sample["split"], sample["scene"], noise_type)
                if key not in mask_cache:
                    source_mask = np.load(sample["sector_mask"]).astype(np.float32)
                    mask_cache[key] = generate_front_mask(source_mask.shape, noise_type)
                save_npy(scene_mask_dst, mask_cache[key], overwrite=args.overwrite)

            noisy_dst = dst_root / "noisyimg" / noise_type / base_rel
            if noise_type == "sector75":
                safe_symlink(sample["masked"], noisy_dst, overwrite=args.overwrite)
            elif args.materialize_noisy:
                ego = np.load(sample["ego"]).astype(np.float32)
                mask = mask_cache[(sample["split"], sample["scene"], noise_type)]
                save_npy(noisy_dst, ego * mask[:, :, np.newaxis], overwrite=args.overwrite)

            noisy_ref = rel(noisy_dst, dst_root) if noisy_dst.exists() or noisy_dst.is_symlink() else f"generated://{noise_type}"
            for preprocess_type in PREPROCESS_TYPES:
                pre_ref = (
                    rel(neighbor_dst, dst_root)
                    if preprocess_type == "none"
                    else f"on_the_fly://{preprocess_type}"
                )
                row = {
                    "sample_id": f"{sample['split']}__{sample['scene']}__{sample['frame']}__{noise_type}__{preprocess_type}",
                    "split_group": sample["split_group"],
                    "split": sample["split"],
                    "scene": sample["scene"],
                    "frame": sample["frame"],
                    "noise_type": noise_type,
                    "preprocess_type": preprocess_type,
                    "clean_img": rel(clean_dst, dst_root),
                    "noisy_img": noisy_ref,
                    "neighbor_raw": rel(neighbor_dst, dst_root),
                    "preprocessed_img": pre_ref,
                    "mask": rel(scene_mask_dst, dst_root),
                    "source_clean": str(sample["ego"]),
                    "source_neighbor": str(sample["neighbor"]),
                }
                manifest_rows.append(row)
                split_rows[sample["split_group"]].append(row)

    fieldnames = list(manifest_rows[0].keys()) if manifest_rows else []
    with (dst_root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    for group, rows in split_rows.items():
        with (dst_root / "splits" / f"{group}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    stats = {
        "source_root": str(src_root),
        "dataset_v2_root": str(dst_root),
        "base_samples": len(samples),
        "noise_types": list(NOISE_TYPES),
        "preprocess_types": list(PREPROCESS_TYPES),
        "manifest_rows": len(manifest_rows),
        "materialize_noisy": bool(args.materialize_noisy),
        "split_rows": {key: len(value) for key, value in split_rows.items()},
    }
    (dst_root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
