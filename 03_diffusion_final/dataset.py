#!/usr/bin/env python3
"""
Dataset loader for my V2V4Real BEV reconstruction experiments.

Small reminder to myself:
  - input  = masked ego + neighbor
  - target = clean ego
  - loss only cares the masked area, so the mask needs to come out too
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset_prepared"

RESOLUTION = 0.16
FRONT_R_MIN = 10.0
FRONT_R_MAX = 38.0
FRONT_RECT_HALF_WIDTH = 15.5
MASK_VARIANTS = {"sector75", "front_rect", "front_blob"}
NEIGHBOR_PREPROCESS_MODES = {"none", "register", "register_layernorm"}


def generate_front_mask(shape, variant):
    if variant not in MASK_VARIANTS:
        raise ValueError(f"Unknown mask_variant={variant!r}; expected one of {sorted(MASK_VARIANTS)}")
    if variant == "sector75":
        return None

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
    else:
        phase = (forward_m - FRONT_R_MIN) / max(FRONT_R_MAX - FRONT_R_MIN, 1e-6)
        center_offset = 2.0 * np.sin(phase * np.pi * 1.3)
        half_width = (
            FRONT_RECT_HALF_WIDTH
            + 3.0 * np.sin(phase * np.pi * 2.0)
            + 1.2 * np.sin(phase * np.pi * 5.0)
        )
        half_width = np.clip(half_width, 10.0, 19.0)
        hidden = in_forward & (np.abs(lateral_m - center_offset) <= half_width)

    mask[hidden] = 0.0
    return mask


def _weighted_center_of_mass(weight_map):
    total = float(weight_map.sum())
    if total <= 1e-6:
        return None
    rows, cols = np.indices(weight_map.shape)
    row = float((rows * weight_map).sum() / total)
    col = float((cols * weight_map).sum() / total)
    return row, col


def _shift_bev_zero(bev, row_shift, col_shift):
    if row_shift == 0 and col_shift == 0:
        return bev

    shifted = np.zeros_like(bev)
    h, w, _ = bev.shape

    src_r0 = max(0, -row_shift)
    src_r1 = min(h, h - row_shift)
    dst_r0 = max(0, row_shift)
    dst_r1 = min(h, h + row_shift)

    src_c0 = max(0, -col_shift)
    src_c1 = min(w, w - col_shift)
    dst_c0 = max(0, col_shift)
    dst_c1 = min(w, w + col_shift)

    if src_r1 > src_r0 and src_c1 > src_c0:
        shifted[dst_r0:dst_r1, dst_c0:dst_c1, :] = bev[src_r0:src_r1, src_c0:src_c1, :]
    return shifted


def register_neighbor_to_visible_ego(masked_ego, neighbor, mask, max_shift_px=24):
    # Estimate one BEV translation from density occupancy, then apply it to all
    # density/height layers together so the multi-layer structure stays aligned.
    visible = mask > 0.5
    ego_occ = masked_ego[:, :, :4].sum(axis=2) * visible
    neighbor_occ = neighbor[:, :, :4].sum(axis=2) * visible

    ego_center = _weighted_center_of_mass(ego_occ)
    neighbor_center = _weighted_center_of_mass(neighbor_occ)
    if ego_center is None or neighbor_center is None:
        return neighbor

    row_shift = int(np.clip(round(ego_center[0] - neighbor_center[0]), -max_shift_px, max_shift_px))
    col_shift = int(np.clip(round(ego_center[1] - neighbor_center[1]), -max_shift_px, max_shift_px))
    return _shift_bev_zero(neighbor, row_shift, col_shift)


def layerwise_match_neighbor_to_ego(masked_ego, neighbor, mask):
    # Per-layer calibration: match neighbor channel statistics to the visible ego
    # region. This avoids treating the 8-channel BEV like a 3-channel RGB image.
    visible = mask > 0.5
    if visible.sum() <= 4:
        return neighbor

    out = neighbor.copy()
    for ch in range(out.shape[2]):
        ego_vals = masked_ego[:, :, ch][visible]
        neighbor_vals = out[:, :, ch][visible]
        ego_std = float(ego_vals.std())
        neighbor_std = float(neighbor_vals.std())
        if neighbor_std <= 1e-6:
            continue
        out[:, :, ch] = (out[:, :, ch] - float(neighbor_vals.mean())) / neighbor_std
        out[:, :, ch] = out[:, :, ch] * max(ego_std, 1e-6) + float(ego_vals.mean())
    return np.clip(out, 0.0, 1.0)


def preprocess_neighbor(masked_ego, neighbor, mask, mode, registration_max_shift_px):
    if mode not in NEIGHBOR_PREPROCESS_MODES:
        raise ValueError(
            f"Unknown neighbor_preprocess={mode!r}; expected one of {sorted(NEIGHBOR_PREPROCESS_MODES)}"
        )
    if mode == "none":
        return neighbor

    neighbor = register_neighbor_to_visible_ego(
        masked_ego,
        neighbor,
        mask,
        max_shift_px=registration_max_shift_px,
    )
    if mode == "register_layernorm":
        neighbor = layerwise_match_neighbor_to_ego(masked_ego, neighbor, mask)
    return neighbor


class BEVReconstructionDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        splits,
        augment=False,
        mask_variant="sector75",
        neighbor_preprocess="none",
        registration_max_shift_px=24,
    ):
        self.dataset_root = Path(dataset_root)
        self.augment = augment
        self.mask_variant = mask_variant
        self.neighbor_preprocess = neighbor_preprocess
        self.registration_max_shift_px = int(registration_max_shift_px)
        self.samples = []
        if self.mask_variant not in MASK_VARIANTS:
            raise ValueError(
                f"Unknown mask_variant={self.mask_variant!r}; expected one of {sorted(MASK_VARIANTS)}"
            )
        if self.neighbor_preprocess not in NEIGHBOR_PREPROCESS_MODES:
            raise ValueError(
                "Unknown neighbor_preprocess="
                f"{self.neighbor_preprocess!r}; expected one of {sorted(NEIGHBOR_PREPROCESS_MODES)}"
            )

        for split in splits:
            split_path = self.dataset_root / split
            if not split_path.is_dir():
                print(f"  WARNING: split not found: {split_path}")
                continue

            for scene_path in sorted(split_path.iterdir()):
                if not scene_path.is_dir():
                    continue

                ego_dir = scene_path / "ego_bev"
                masked_dir = scene_path / "masked_ego"
                neighbor_dir = scene_path / "neighbor_bev"
                mask_file = scene_path / "sector_mask.npy"

                if not (ego_dir.is_dir() and masked_dir.is_dir() and neighbor_dir.is_dir()):
                    continue
                if not mask_file.exists():
                    continue

                frame_ids = sorted(path.stem for path in ego_dir.glob("*.npy"))
                for frame_id in frame_ids:
                    ego_path = ego_dir / f"{frame_id}.npy"
                    masked_path = masked_dir / f"{frame_id}.npy"
                    neighbor_path = neighbor_dir / f"{frame_id}.npy"

                    if ego_path.exists() and masked_path.exists() and neighbor_path.exists():
                        self.samples.append(
                            {
                                "ego": ego_path,
                                "masked": masked_path,
                                "neighbor": neighbor_path,
                                "mask": mask_file,
                                "split": split,
                                "scene": scene_path.name,
                                "frame": frame_id,
                            }
                        )

        print(f"  Dataset: {len(self.samples)} samples from {len(splits)} splits")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        ego_bev = np.load(sample["ego"]).astype(np.float32)
        masked_ego = np.load(sample["masked"]).astype(np.float32)
        neighbor = np.load(sample["neighbor"]).astype(np.float32)
        mask = np.load(sample["mask"]).astype(np.float32)

        variant_mask = generate_front_mask(mask.shape, self.mask_variant)
        if variant_mask is not None:
            mask = variant_mask
            masked_ego = ego_bev * mask[:, :, np.newaxis]

        neighbor = preprocess_neighbor(
            masked_ego,
            neighbor,
            mask,
            self.neighbor_preprocess,
            self.registration_max_shift_px,
        )

        if self.augment and np.random.random() > 0.5:
            # Horizontal flip is enough for now. Simple but works ok.
            ego_bev = np.flip(ego_bev, axis=1).copy()
            masked_ego = np.flip(masked_ego, axis=1).copy()
            neighbor = np.flip(neighbor, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # Keep channel order in the last dim here, transpose only once before torch.
        input_bev = np.concatenate([masked_ego, neighbor], axis=2)

        input_tensor = torch.from_numpy(input_bev.transpose(2, 0, 1))
        target_tensor = torch.from_numpy(ego_bev.transpose(2, 0, 1))
        mask_tensor = torch.from_numpy(mask[np.newaxis, :, :])

        return input_tensor, target_tensor, mask_tensor

    def get_info(self, idx):
        return self.samples[idx]


def discover_available_splits(dataset_root):
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        return [], [], []

    all_dirs = [path.name for path in dataset_root.iterdir() if path.is_dir()]
    train_splits = sorted(name for name in all_dirs if name.startswith("train_"))
    val_splits = sorted(name for name in all_dirs if name.startswith("val_"))
    test_splits = sorted(name for name in all_dirs if name.startswith("test_"))
    return train_splits, val_splits, test_splits


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def get_dataloaders(
    dataset_root,
    batch_size=8,
    num_workers=None,
    seed=None,
    mask_variant="sector75",
    neighbor_preprocess="none",
    registration_max_shift_px=24,
):
    train_splits, val_splits, test_splits = discover_available_splits(dataset_root)

    print(f"\n  Train splits: {train_splits}")
    print(f"  Val splits:   {val_splits}")
    print(f"  Test splits:  {test_splits}")

    print(f"  Mask variant: {mask_variant}")
    print(f"  Neighbor preprocess: {neighbor_preprocess}")

    train_dataset = BEVReconstructionDataset(
        dataset_root,
        train_splits,
        augment=True,
        mask_variant=mask_variant,
        neighbor_preprocess=neighbor_preprocess,
        registration_max_shift_px=registration_max_shift_px,
    )
    val_dataset = BEVReconstructionDataset(
        dataset_root,
        val_splits,
        augment=False,
        mask_variant=mask_variant,
        neighbor_preprocess=neighbor_preprocess,
        registration_max_shift_px=registration_max_shift_px,
    )
    test_dataset = BEVReconstructionDataset(
        dataset_root,
        test_splits,
        augment=False,
        mask_variant=mask_variant,
        neighbor_preprocess=neighbor_preprocess,
        registration_max_shift_px=registration_max_shift_px,
    )

    if num_workers is None:
        # Windows dataloader can be annoying sometimes, so keep it safe there.
        num_workers = 0 if os.name == "nt" else 4

    pin_memory = torch.cuda.is_available()
    drop_last = len(train_dataset) >= batch_size
    loader_gen = None
    worker_init = None

    if seed is not None:
        loader_gen = torch.Generator()
        loader_gen.manual_seed(seed)
        worker_init = seed_worker

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init,
        generator=loader_gen,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init,
        generator=loader_gen,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init,
        generator=loader_gen,
    )
    return train_loader, val_loader, test_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Quick dataset loader sanity check.")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--mask_variant", type=str, default="sector75", choices=sorted(MASK_VARIANTS))
    parser.add_argument("--neighbor_preprocess", type=str, default="none", choices=sorted(NEIGHBOR_PREPROCESS_MODES))
    parser.add_argument("--registration_max_shift_px", type=int, default=24)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Testing dataset from: {args.dataset_root}")

    train_loader, val_loader, test_loader = get_dataloaders(
        args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=42,
        mask_variant=args.mask_variant,
        neighbor_preprocess=args.neighbor_preprocess,
        registration_max_shift_px=args.registration_max_shift_px,
    )

    print(f"\n  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    for batch in train_loader:
        inp, tgt, mask = batch
        print(f"\n  Input shape:  {inp.shape}")
        print(f"  Target shape: {tgt.shape}")
        print(f"  Mask shape:   {mask.shape}")
        print(f"  Input range:  [{inp.min():.4f}, {inp.max():.4f}]")
        print(f"  Target range: [{tgt.min():.4f}, {tgt.max():.4f}]")
        print(f"  Mask values:  {mask.unique().tolist()}")
        break

    print("\n  Dataset OK!")
