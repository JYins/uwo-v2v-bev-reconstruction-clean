#!/usr/bin/env python3
"""
Prepare V2V4Real into the dataset format I use for reconstruction training.

Main jobs:
1. unzip raw splits if needed
2. find synchronized ego / neighbor frames
3. convert both sides into the same 8-channel BEV
4. mask the ego forward sector
5. save everything in a layout that training code can read directly
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import time
import warnings
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")


# ===================================================================
#  DEFAULT PATHS
# ===================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "dataset_prepared"

SPLIT_PREFIXES = ("train_", "test_", "val_")
ZIP_SPLITS = [
    "train_01",
    "train_02",
    "train_03",
    "train_04",
    "train_05",
    "train_06",
    "train_07",
    "train_08",
    "test_01",
    "test_02",
    "test_03",
    "val_01",
]

ZIP_NAME_MAP = {
    "val_01": "val.zip",
}

EGO_ID = "0"
NEIGHBOR_ID = "1"


# ===================================================================
#  BEV PARAMETERS
# ===================================================================

X_RANGE = (-40.0, 40.0)
Y_RANGE = (-40.0, 40.0)
RESOLUTION = 0.16

BEV_W = int((X_RANGE[1] - X_RANGE[0]) / RESOLUTION)
BEV_H = int((Y_RANGE[1] - Y_RANGE[0]) / RESOLUTION)

HEIGHT_BINS = [
    (-3.0, -1.5),
    (-1.5, 0.0),
    (0.0, 1.0),
    (1.0, 2.0),
]
NUM_BINS = len(HEIGHT_BINS)
NUM_CHANNELS = NUM_BINS * 2
DENSITY_CAP = 20


# ===================================================================
#  SECTOR MASK PARAMETERS
# ===================================================================

SECTOR_ANGLE_DEG = 75
SECTOR_START_DEG = -37.5
SECTOR_R_MIN = 10.0
SECTOR_R_MAX = 38.0


# ===================================================================
#  RAW DATA EXTRACTION
# ===================================================================

def archive_name_for_split(split_name: str) -> str:
    return ZIP_NAME_MAP.get(split_name, f"{split_name}.zip")


def infer_prefix_to_strip(split_name: str) -> Optional[str]:
    if split_name == "val_01":
        return "val/"
    return None


def split_has_scene_data(split_dir: Path) -> bool:
    if not split_dir.is_dir():
        return False
    for child in split_dir.iterdir():
        if child.is_dir():
            ego_dir = child / EGO_ID
            nbr_dir = child / NEIGHBOR_ID
            if ego_dir.is_dir() and nbr_dir.is_dir():
                return True
    return False


def extract_archive(archive_path: Path, target_dir: Path, strip_prefix: Optional[str] = None) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        total = len(members)
        print(f"  Extracting {archive_path.name} -> {target_dir} ({total} files)")

        for idx, member in enumerate(members, start=1):
            member_name = member.filename.replace("\\", "/")
            if strip_prefix and member_name.startswith(strip_prefix):
                rel_name = member_name[len(strip_prefix):]
            else:
                rel_name = member_name

            if not rel_name:
                continue

            out_path = target_dir / rel_name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(member, "r") as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            if idx == 1 or idx % 1000 == 0 or idx == total:
                print(f"    [{idx:5d}/{total}] {rel_name}")


def ensure_raw_splits(project_root: Path) -> dict[str, str]:
    """
    Ensure train/test/val split folders exist on disk.

    Returns a dict split_name -> status string.
    """
    statuses: dict[str, str] = {}

    for split_name in ZIP_SPLITS:
        split_dir = project_root / split_name
        archive_path = project_root / archive_name_for_split(split_name)

        if split_has_scene_data(split_dir):
            statuses[split_name] = "ready"
            continue

        if not archive_path.exists():
            statuses[split_name] = "missing_archive"
            continue

        extract_archive(
            archive_path=archive_path,
            target_dir=split_dir,
            strip_prefix=infer_prefix_to_strip(split_name),
        )
        statuses[split_name] = "extracted"

    return statuses


# ===================================================================
#  POINT CLOUD LOADING
# ===================================================================

def load_pcd(path: str):
    import open3d as o3d

    try:
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points)
        return pts if len(pts) > 0 else None
    except Exception:
        return None


# ===================================================================
#  8-CHANNEL BEV
# ===================================================================

def points_to_bev(points):
    bev = np.zeros((BEV_H, BEV_W, NUM_CHANNELS), dtype=np.float32)

    if points is None or len(points) == 0:
        return bev

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    mask = (
        (x >= X_RANGE[0]) & (x < X_RANGE[1]) &
        (y >= Y_RANGE[0]) & (y < Y_RANGE[1])
    )
    x, y, z = x[mask], y[mask], z[mask]

    if len(x) == 0:
        return bev

    col = np.clip(((x - X_RANGE[0]) / RESOLUTION).astype(np.int32), 0, BEV_W - 1)
    row = np.clip(((Y_RANGE[1] - y) / RESOLUTION).astype(np.int32), 0, BEV_H - 1)

    log_cap = np.log1p(DENSITY_CAP)

    for b, (z_lo, z_hi) in enumerate(HEIGHT_BINS):
        mz = (z >= z_lo) & (z < z_hi)
        if mz.sum() == 0:
            continue

        br, bc, bz = row[mz], col[mz], z[mz]

        cnt = np.zeros((BEV_H, BEV_W), dtype=np.float32)
        np.add.at(cnt, (br, bc), 1.0)
        bev[:, :, b] = np.clip(np.log1p(cnt) / log_cap, 0, 1)

        dz = z_hi - z_lo
        if dz > 0:
            z_n = (bz - z_lo) / dz
            mh = np.full((BEV_H, BEV_W), -1.0, dtype=np.float32)
            np.maximum.at(mh, (br, bc), z_n)
            mh[mh < 0] = 0
            bev[:, :, NUM_BINS + b] = mh

    return bev


# ===================================================================
#  SECTOR MASK
# ===================================================================

def generate_sector_mask():
    mask = np.ones((BEV_H, BEV_W), dtype=np.float32)

    center_row = BEV_H // 2
    center_col = BEV_W // 2

    r_min_px = SECTOR_R_MIN / RESOLUTION
    r_max_px = SECTOR_R_MAX / RESOLUTION

    angle_start = np.radians(SECTOR_START_DEG)
    angle_end = np.radians(SECTOR_START_DEG + SECTOR_ANGLE_DEG)

    rows, cols = np.meshgrid(np.arange(BEV_H), np.arange(BEV_W), indexing="ij")
    d_forward = cols - center_col
    d_lateral = center_row - rows

    dist = np.sqrt(d_forward ** 2 + d_lateral ** 2)
    angle = np.arctan2(d_lateral, d_forward)

    in_sector = (
        (dist >= r_min_px) & (dist <= r_max_px) &
        (angle >= angle_start) & (angle <= angle_end)
    )
    mask[in_sector] = 0.0
    return mask


def apply_mask(bev_8ch, mask_2d):
    return bev_8ch * mask_2d[:, :, np.newaxis]


# ===================================================================
#  DISCOVERY
# ===================================================================

def discover_all_splits(project_root: Path):
    entries = []

    for item in sorted(project_root.iterdir()):
        if not item.is_dir():
            continue
        if not any(item.name.startswith(prefix) for prefix in SPLIT_PREFIXES):
            continue

        for scene_dir in sorted(item.iterdir()):
            if not scene_dir.is_dir():
                continue

            ego_path = scene_dir / EGO_ID
            nbr_path = scene_dir / NEIGHBOR_ID
            if not (ego_path.is_dir() and nbr_path.is_dir()):
                continue

            ego_pcds = list(ego_path.glob("*.pcd"))
            nbr_pcds = list(nbr_path.glob("*.pcd"))
            if ego_pcds and nbr_pcds:
                entries.append(
                    {
                        "split": item.name,
                        "scene": scene_dir.name,
                        "scene_path": str(scene_dir),
                        "ego_path": str(ego_path),
                        "nbr_path": str(nbr_path),
                    }
                )

    return entries


def get_common_frames(ego_path, nbr_path):
    def ids(folder):
        return {
            os.path.splitext(f)[0]
            for f in os.listdir(folder)
            if f.endswith(".pcd")
        }

    return sorted(ids(ego_path) & ids(nbr_path))


def cast_bev_for_storage(array: np.ndarray, save_dtype: str) -> np.ndarray:
    if save_dtype == "float16":
        return array.astype(np.float16)
    return array.astype(np.float32)


def load_or_build_outputs(
    frame_id: str,
    output_dirs: dict[str, Path],
    entry: dict[str, str],
    sector_mask: np.ndarray,
    force_reprocess: bool,
    save_dtype: str,
):
    ego_out = output_dirs["ego"] / f"{frame_id}.npy"
    masked_out = output_dirs["masked"] / f"{frame_id}.npy"
    neighbor_out = output_dirs["neighbor"] / f"{frame_id}.npy"
    lidar_out = output_dirs["lidar"] / f"{frame_id}.pcd"

    if (
        not force_reprocess and
        ego_out.exists() and
        masked_out.exists() and
        neighbor_out.exists() and
        lidar_out.exists()
    ):
        # If files are already there, just reuse them. No need to suffer twice.
        ego_bev = np.load(ego_out).astype(np.float32)
        nbr_bev = np.load(neighbor_out).astype(np.float32)
        return ego_bev, nbr_bev, True

    src_pcd = Path(entry["ego_path"]) / f"{frame_id}.pcd"
    if force_reprocess or not lidar_out.exists():
        shutil.copy2(src_pcd, lidar_out)

    ego_pts = load_pcd(str(src_pcd))
    nbr_pts = load_pcd(str(Path(entry["nbr_path"]) / f"{frame_id}.pcd"))

    ego_bev = points_to_bev(ego_pts)
    nbr_bev = points_to_bev(nbr_pts)
    masked_ego = apply_mask(ego_bev, sector_mask)

    np.save(ego_out, cast_bev_for_storage(ego_bev, save_dtype))
    np.save(masked_out, cast_bev_for_storage(masked_ego, save_dtype))
    np.save(neighbor_out, cast_bev_for_storage(nbr_bev, save_dtype))

    return ego_bev, nbr_bev, False


# ===================================================================
#  PROCESSING
# ===================================================================

def process_scene(
    entry,
    scene_idx,
    total,
    sector_mask,
    output_root: Path,
    force_reprocess: bool,
    save_dtype: str,
):
    split = entry["split"]
    scene = entry["scene"]

    print(f"\n{'=' * 70}")
    print(f"  [{scene_idx + 1}/{total}] {split} / {scene}")
    print(f"{'=' * 70}")

    frames = get_common_frames(entry["ego_path"], entry["nbr_path"])
    print(f"  Sync frames: {len(frames)}")

    if not frames:
        print("  SKIP: no synchronized frames")
        return {"split": split, "scene": scene, "frames": 0, "ego_fill": [], "nbr_fill": []}

    base = output_root / split / scene
    output_dirs = {
        "lidar": base / "original_lidar",
        "ego": base / "ego_bev",
        "masked": base / "masked_ego",
        "neighbor": base / "neighbor_bev",
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    np.save(base / "sector_mask.npy", sector_mask)

    stats = {"split": split, "scene": scene, "frames": len(frames), "ego_fill": [], "nbr_fill": []}
    reused_frames = 0
    built_frames = 0
    start_time = time.time()

    for i, frame_id in enumerate(frames):
        if i == 0 or (i + 1) % 20 == 0 or i == len(frames) - 1:
            elapsed = time.time() - start_time
            fps = (i + 1) / max(elapsed, 0.01)
            eta = (len(frames) - i - 1) / max(fps, 0.01)
            print(f"    [{i + 1:4d}/{len(frames)}] {frame_id} ({fps:.1f} f/s, ETA {eta:.0f}s)")

        ego_bev, nbr_bev, reused = load_or_build_outputs(
            frame_id=frame_id,
            output_dirs=output_dirs,
            entry=entry,
            sector_mask=sector_mask,
            force_reprocess=force_reprocess,
            save_dtype=save_dtype,
        )

        if reused:
            reused_frames += 1
        else:
            built_frames += 1

        ego_fill = (ego_bev[:, :, :NUM_BINS].sum(axis=2) > 0).sum()
        nbr_fill = (nbr_bev[:, :, :NUM_BINS].sum(axis=2) > 0).sum()
        stats["ego_fill"].append(ego_fill)
        stats["nbr_fill"].append(nbr_fill)

        if i == 0:
            total_px = BEV_H * BEV_W
            sect_pct = (1 - sector_mask.mean()) * 100
            print("    --- First frame check ---")
            print(f"    BEV shape:      {ego_bev.shape}")
            print(f"    Ego occupancy:  {ego_fill:,}/{total_px:,} ({100 * ego_fill / total_px:.1f}%)")
            print(f"    Nbr occupancy:  {nbr_fill:,}/{total_px:,} ({100 * nbr_fill / total_px:.1f}%)")
            print(f"    Sector masked:  {sect_pct:.1f}% of BEV")

    elapsed = time.time() - start_time
    print(f"  Done: {len(frames)} frames in {elapsed:.1f}s (built {built_frames}, reused {reused_frames})")
    return stats


# ===================================================================
#  SUMMARY
# ===================================================================

def write_summary(all_stats, output_root: Path, save_dtype: str):
    total_frames = sum(s["frames"] for s in all_stats)
    total_px = BEV_H * BEV_W

    splits = {}
    for stat in all_stats:
        splits.setdefault(stat["split"], []).append(stat)

    lines = [
        "=" * 65,
        "  V2V4Real Dataset - Prepared for Reconstruction",
        "=" * 65,
        "",
        f"  Output:        {output_root}",
        f"  Total scenes:  {len(all_stats)}",
        f"  Total frames:  {total_frames}",
        "",
        "  BEV:",
        f"    {BEV_W}x{BEV_H} @ {RESOLUTION}m/px, +/-{X_RANGE[1]}m, 8 channels",
        f"    Shape: (H={BEV_H}, W={BEV_W}, C=8) float32",
        "",
        f"  Sector mask: {SECTOR_ANGLE_DEG} deg, {SECTOR_R_MIN}-{SECTOR_R_MAX}m",
        f"  Saved dtype:  {save_dtype} on disk (loaded as float32 for training)",
        "",
        "  Per split:",
    ]

    for split_name in sorted(splits.keys()):
        split_stats = splits[split_name]
        split_frames = sum(s["frames"] for s in split_stats)
        lines.append(f"\n    [{split_name}] - {len(split_stats)} scenes, {split_frames} frames")
        for stat in split_stats:
            if stat["frames"] == 0:
                continue
            avg_ego = float(np.mean(stat["ego_fill"]))
            avg_nbr = float(np.mean(stat["nbr_fill"]))
            lines.append(
                f"      {stat['scene'][:45]}: {stat['frames']} fr, "
                f"ego {avg_ego / total_px * 100:.1f}%, "
                f"nbr {avg_nbr / total_px * 100:.1f}%"
            )

    lines.extend(
        [
            "",
            "  Output per scene:",
            "    {split}/{scene}/",
            "      original_lidar/",
            "      ego_bev/",
            "      masked_ego/",
            "      neighbor_bev/",
            "      sector_mask.npy",
            "",
            "  Model training:",
            "    Input  = concat(masked_ego, neighbor_bev) -> (H,W,16)",
            "    Target = ego_bev                          -> (H,W,8)",
            "    Loss on masked region (sector_mask == 0)",
        ]
    )

    summary_text = "\n".join(lines)
    summary_path = output_root / "dataset_stats.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"\n{summary_text}")
    print(f"\n  Saved: {summary_path}")


def print_split_statuses(statuses: dict[str, str]) -> None:
    print("\n  Raw split status:")
    for split_name in ZIP_SPLITS:
        status = statuses.get(split_name, "unknown")
        print(f"    {split_name:<8} {status}")


# ===================================================================
#  MAIN
# ===================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare V2V4Real dataset for BEV reconstruction.")
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output_root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--skip_extract", action="store_true", help="Do not auto-extract zip archives.")
    parser.add_argument("--extract_only", action="store_true", help="Only extract archives, do not process frames.")
    parser.add_argument("--force_reprocess", action="store_true", help="Rebuild outputs even if .npy files already exist.")
    parser.add_argument("--save_dtype", choices=["float16", "float32"], default="float16")
    return parser.parse_args()


def main():
    args = parse_args()

    print()
    print("=" * 70)
    print("  Dataset Preparation - 8-Channel BEV + Sector Mask")
    print("  Auto-extracts archives and resumes existing outputs")
    print("=" * 70)

    try:
        import open3d  # noqa: F401
    except ImportError:
        print("\n  ERROR: open3d is not installed in the current Python environment.")
        print("  Use the v2v environment first, otherwise this script cannot move.")
        sys.exit(1)

    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()

    print(f"\n  Project root: {project_root}")
    print(f"  Output root:  {output_root}")
    print(f"  BEV:          {BEV_W}x{BEV_H} @ {RESOLUTION}m/px")
    print(f"  Save dtype:   {args.save_dtype}")

    statuses = {split_name: "not_checked" for split_name in ZIP_SPLITS}
    if not args.skip_extract:
        statuses = ensure_raw_splits(project_root)
        print_split_statuses(statuses)
    else:
        print("\n  Skipping archive extraction by request.")

    if args.extract_only:
        print("\n  Extraction finished. Exiting because --extract_only was set.")
        return

    entries = discover_all_splits(project_root)
    if not entries:
        print(f"\n  ERROR: no valid train/test/val scenes found under {project_root}")
        sys.exit(1)

    splits_found = {}
    for entry in entries:
        splits_found.setdefault(entry["split"], 0)
        splits_found[entry["split"]] += 1

    print(f"\n  Found {len(entries)} scenes across {len(splits_found)} splits:")
    for split_name in sorted(splits_found.keys()):
        print(f"    {split_name}: {splits_found[split_name]} scenes")

    sector_mask = generate_sector_mask()
    sect_pct = (1 - sector_mask.mean()) * 100
    print(f"\n  Sector mask: {sect_pct:.1f}% of BEV masked")

    output_root.mkdir(parents=True, exist_ok=True)

    all_stats = []
    start_time = time.time()
    for idx, entry in enumerate(entries):
        stats = process_scene(
            entry=entry,
            scene_idx=idx,
            total=len(entries),
            sector_mask=sector_mask,
            output_root=output_root,
            force_reprocess=args.force_reprocess,
            save_dtype=args.save_dtype,
        )
        all_stats.append(stats)

    total_time = time.time() - start_time
    total_frames = sum(s["frames"] for s in all_stats)

    print(f"\n{'=' * 70}")
    print("  ALL DONE")
    print(f"  {total_frames} frames, {len(entries)} scenes, {total_time / 60:.1f} minutes")
    print(f"{'=' * 70}")

    write_summary(all_stats, output_root, args.save_dtype)

    for entry in entries:
        sample_dir = output_root / entry["split"] / entry["scene"] / "ego_bev"
        npy_files = sorted(sample_dir.glob("*.npy"))
        if npy_files:
            arr = np.load(npy_files[0])
            print(f"\n  Verification ({entry['split']}/{entry['scene']}):")
            print(f"    Shape: {arr.shape}  dtype: {arr.dtype}")
            print(f"    Range: [{arr.min():.4f}, {arr.max():.4f}]")
            break

    print("\n  Next: python visualize_4columns.py")


if __name__ == "__main__":
    main()
