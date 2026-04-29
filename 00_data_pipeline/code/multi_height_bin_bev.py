#!/usr/bin/env python3
"""
=============================================================================
Multi-Height-Bin BEV for V2V4Real  —  v3 FINAL (Literature-Optimized)
=============================================================================

MEng Thesis: V2V Cooperative Perception via Generative Reconstruction

Parameters chosen from literature:
  - Range  ±40m : tighter than v2's ±50m. The outer ring is extremely sparse
    and adds noise without information. CORE uses ±51.2m but at 0.4m voxels
    giving 256×256. We use ±40m at 0.16m → 500×500 which is a better
    resolution for our image-based generative models (U-Net/GAN/Diffusion).
  - Resolution 0.16m : matches PointPillars standard (CVPR2019).
    V2V4Real official uses 0.4m for detection, but we need finer resolution
    for pixel-level reconstruction. 0.16m is the finest standard in literature.
  - Height [-3, 2]m, 4 bins : CORE uses [-3, 1]. We extend to 2m to capture
    truck tops and tree canopy while cutting empty sky above 2m.

Encoding (8 channels per vehicle):
  Ch 0-3 : Log-normalized point density per height bin
  Ch 4-7 : Max height within each bin (normalized to [0,1])

Two output types:
  1. RAW numpy (.npy) — clean 8-channel data for model training
  2. ENHANCED visualization (.png/.mp4) — morphological dilation + glow
     effect to make sparse LiDAR BEV visually impressive for presentations.
     This does NOT alter the training data.

Multi-scene: processes ALL scenes found in BASE_DIR, or a single scene.

Author: Shi Yin (MEng research project, Western University)
Supervisor: Dr. Fadi AlMahamid
=============================================================================
"""

import os
import sys
import glob
import time
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ===================================================================
#  USER CONFIGURATION
# ===================================================================

# Option A: Process ALL scenes under this directory.
# This script is older than prepare_dataset.py, but I keep it because it
# records the exact BEV design choices from the early pipeline work.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BASE_DIR = str(REPO_ROOT / "train_01")

# Option B: Process only ONE specific scene (set to None to use BASE_DIR)
# SINGLE_SCENE = str(REPO_ROOT / "train_01" / "testoutput_CAV_data_2022-03-15-09-54-40_0")
SINGLE_SCENE = None

# Vehicle sub-folder names inside each scene
EGO_VEHICLE = "0"
NEIGHBOR_VEHICLE = "1"

# Output root
OUTPUT_DIR = str(REPO_ROOT / "output_v3")


# ===================================================================
#  BEV PARAMETERS  (literature-optimized)
# ===================================================================

# Spatial range (meters, centered on vehicle)
# Tighter than ±50m — outer ring is almost empty, wastes pixels
X_RANGE = (-40.0, 40.0)
Y_RANGE = (-40.0, 40.0)

# Resolution: 0.16 m/px matches PointPillars (CVPR2019)
# At ±40m → 80m / 0.16 = 500 pixels per side → 500×500 BEV
RESOLUTION = 0.16

# Height bins (meters, in LiDAR frame)
# Based on CORE (ICCV2023): [-3, 1]m range, we extend to 2m
HEIGHT_BINS = [
    (-3.0, -1.5),   # Bin 0: Road surface, deep ground, dips
    (-1.5,  0.0),   # Bin 1: Curbs, car undersides, low barriers
    ( 0.0,  1.0),   # Bin 2: Car bodies, pedestrians, cyclists
    ( 1.0,  2.0),   # Bin 3: Car/truck tops, trees, signs
]

# Density normalization cap
# log(1 + count) / log(1 + cap). At 0.16m cells, ~15-20 pts max
DENSITY_CAP = 20

# Visualization colors (BGR)
BIN_COLORS = [
    (255, 120, 40),    # Bin 0: Blue    (ground)
    (40,  230, 120),   # Bin 1: Green   (low obstacles)
    (40,  180, 255),   # Bin 2: Orange  (car bodies)  — most important
    (200, 60,  255),   # Bin 3: Magenta (tall objects)
]

BIN_NAMES = ["Ground", "Low Obj", "Car Body", "Tall Obj"]

# Processing
MAX_FRAMES = 200
DETAIL_FRAMES = 5
VIDEO_FPS = 10

# Visualization enhancement (for DISPLAY only, raw data unaffected)
VIS_DILATE_KERNEL = 2      # morphological dilation radius in pixels
VIS_GLOW_SIGMA = 1.0       # gaussian blur sigma for glow effect
VIS_GLOW_BLEND = 0.4       # blend ratio: 0=no glow, 1=full glow


# ===================================================================
#  DEPENDENCY CHECK
# ===================================================================

def check_deps():
    missing = []
    for mod, pkg in [("open3d","open3d"),("cv2","opencv-python"),("matplotlib","matplotlib")]:
        try: __import__(mod)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"\nMissing: {', '.join(missing)}")
        print(f"Run:  pip install {' '.join(missing)}")
        sys.exit(1)


# ===================================================================
#  SCENE DISCOVERY
# ===================================================================

def find_all_scenes(base_dir):
    """Find all scene folders that contain vehicle sub-folders with PCD files."""
    scenes = []
    if not os.path.exists(base_dir):
        print(f"\nERROR: Base directory not found: {base_dir}")
        sys.exit(1)

    for name in sorted(os.listdir(base_dir)):
        scene_path = os.path.join(base_dir, name)
        if not os.path.isdir(scene_path):
            continue
        ego_path = os.path.join(scene_path, EGO_VEHICLE)
        nbr_path = os.path.join(scene_path, NEIGHBOR_VEHICLE)
        if os.path.isdir(ego_path) and os.path.isdir(nbr_path):
            ego_pcds = glob.glob(os.path.join(ego_path, "*.pcd"))
            nbr_pcds = glob.glob(os.path.join(nbr_path, "*.pcd"))
            if ego_pcds and nbr_pcds:
                scenes.append((name, scene_path))
    return scenes


def get_common_frames(scene_path):
    """Get sorted list of frame IDs common to both vehicles."""
    ego_path = os.path.join(scene_path, EGO_VEHICLE)
    nbr_path = os.path.join(scene_path, NEIGHBOR_VEHICLE)

    def ids(folder):
        return {os.path.splitext(os.path.basename(f))[0]
                for f in glob.glob(os.path.join(folder, "*.pcd"))}

    common = sorted(ids(ego_path) & ids(nbr_path))
    return common, ego_path, nbr_path


# ===================================================================
#  POINT CLOUD LOADING
# ===================================================================

def load_pcd(path):
    """Load PCD → numpy (N,3+). Returns None on failure."""
    import open3d as o3d
    try:
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points)
        return pts if len(pts) > 0 else None
    except:
        return None


# ===================================================================
#  CORE BEV CONVERSION
# ===================================================================

def points_to_bev(points, x_range=X_RANGE, y_range=Y_RANGE,
                  resolution=RESOLUTION, height_bins=HEIGHT_BINS,
                  density_cap=DENSITY_CAP):
    """
    3D point cloud → 8-channel dense BEV.

    Channels 0-3: log-normalized density per bin
    Channels 4-7: max-height within bin (normalized [0,1])

    Returns: (H, W, 8) float32 array, values in [0, 1]
    """
    n_bins = len(height_bins)
    W = int((x_range[1] - x_range[0]) / resolution)
    H = int((y_range[1] - y_range[0]) / resolution)
    bev = np.zeros((H, W, n_bins * 2), dtype=np.float32)

    if points is None or len(points) == 0:
        return bev

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Spatial crop
    mask = ((x >= x_range[0]) & (x < x_range[1]) &
            (y >= y_range[0]) & (y < y_range[1]))
    x, y, z = x[mask], y[mask], z[mask]

    if len(x) == 0:
        return bev

    # Pixel coords (Y flipped: forward = top)
    col = np.clip(((x - x_range[0]) / resolution).astype(np.int32), 0, W-1)
    row = np.clip(((y_range[1] - y) / resolution).astype(np.int32), 0, H-1)

    log_cap = np.log1p(density_cap)

    for b, (z_lo, z_hi) in enumerate(height_bins):
        mz = (z >= z_lo) & (z < z_hi)
        if mz.sum() == 0:
            continue
        br, bc, bz = row[mz], col[mz], z[mz]

        # Density channel
        cnt = np.zeros((H, W), dtype=np.float32)
        np.add.at(cnt, (br, bc), 1.0)
        bev[:, :, b] = np.clip(np.log1p(cnt) / log_cap, 0, 1)

        # Max-height channel
        dz = z_hi - z_lo
        if dz > 0:
            z_n = (bz - z_lo) / dz
            mh = np.full((H, W), -1.0, dtype=np.float32)
            np.maximum.at(mh, (br, bc), z_n)
            mh[mh < 0] = 0
            bev[:, :, n_bins + b] = mh

    return bev


def points_to_simple_bev(points, x_range=X_RANGE, y_range=Y_RANGE,
                          resolution=RESOLUTION):
    """Old single-channel BEV for comparison."""
    W = int((x_range[1] - x_range[0]) / resolution)
    H = int((y_range[1] - y_range[0]) / resolution)
    bev = np.zeros((H, W), dtype=np.float32)

    if points is None or len(points) == 0:
        return bev

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    mask = ((x >= x_range[0]) & (x < x_range[1]) &
            (y >= y_range[0]) & (y < y_range[1]))
    x, y, z = x[mask], y[mask], z[mask]

    col = np.clip(((x - x_range[0]) / resolution).astype(np.int32), 0, W-1)
    row = np.clip(((y_range[1] - y) / resolution).astype(np.int32), 0, H-1)

    z_lo, z_hi = HEIGHT_BINS[0][0], HEIGHT_BINS[-1][1]
    z_n = np.clip((z - z_lo) / (z_hi - z_lo), 0, 1)
    np.maximum.at(bev, (row, col), z_n)
    return bev


# ===================================================================
#  VISUALIZATION — RAW (for data accuracy comparison)
# ===================================================================

def bev_to_color_raw(bev):
    """8ch BEV → color image. No enhancement. True to data."""
    n_bins = len(HEIGHT_BINS)
    H, W = bev.shape[:2]
    img = np.zeros((H, W, 3), dtype=np.float64)
    wt = np.zeros((H, W), dtype=np.float64)

    for b in range(n_bins):
        d = bev[:, :, b].astype(np.float64)
        if d.max() == 0: continue
        for c in range(3):
            img[:, :, c] += d * BIN_COLORS[b][c]
        wt += d

    m = wt > 0
    for c in range(3):
        img[:, :, c][m] /= wt[m]

    # Brightness from density with gamma lift
    br = np.clip(wt / max(wt.max(), 1e-8), 0, 1)
    br = np.power(br, 0.45)  # gamma lift: makes sparse points visible
    for c in range(3):
        img[:, :, c] *= br

    return np.clip(img, 0, 255).astype(np.uint8)


# ===================================================================
#  VISUALIZATION — ENHANCED (for presentations, does NOT touch data)
# ===================================================================

def bev_to_color_enhanced(bev):
    """
    8ch BEV → ENHANCED color image for presentations.

    Enhancement pipeline:
      1. Morphological dilation — thickens each occupied pixel by a few px
         so sparse LiDAR points become visible structural lines
      2. Gaussian glow — soft bloom effect around bright regions
      3. Blend glow with original for depth/atmosphere

    This is purely cosmetic. The .npy training data is unaffected.
    Papers do similar post-processing for their figures.
    """
    import cv2

    n_bins = len(HEIGHT_BINS)
    H, W = bev.shape[:2]
    img = np.zeros((H, W, 3), dtype=np.float64)
    wt = np.zeros((H, W), dtype=np.float64)

    # Dilate each bin's density before rendering
    kernel = None
    if VIS_DILATE_KERNEL > 0:
        ks = VIS_DILATE_KERNEL * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))

    for b in range(n_bins):
        d = bev[:, :, b].astype(np.float64)
        if d.max() == 0:
            continue

        # Dilate: expand occupied pixels
        if kernel is not None:
            d_u8 = (np.clip(d, 0, 1) * 255).astype(np.uint8)
            d_u8 = cv2.dilate(d_u8, kernel, iterations=1)
            d = d_u8.astype(np.float64) / 255.0

        for c in range(3):
            img[:, :, c] += d * BIN_COLORS[b][c]
        wt += d

    m = wt > 0
    for c in range(3):
        img[:, :, c][m] /= wt[m]

    # Brightness with gamma
    br = np.clip(wt / max(wt.max(), 1e-8), 0, 1)
    br = np.power(br, 0.4)
    for c in range(3):
        img[:, :, c] *= br

    base = np.clip(img, 0, 255).astype(np.uint8)

    # Glow: gaussian blur + additive blend
    if VIS_GLOW_SIGMA > 0 and VIS_GLOW_BLEND > 0:
        glow_size = int(VIS_GLOW_SIGMA * 6) | 1  # must be odd
        glow = cv2.GaussianBlur(base, (glow_size, glow_size), VIS_GLOW_SIGMA)
        base = cv2.addWeighted(base, 1.0, glow, VIS_GLOW_BLEND, 0)

    return base


def single_bin_enhanced(bev, b):
    """Render single bin with dilation for display."""
    import cv2
    H, W = bev.shape[:2]
    d = bev[:, :, b].copy()

    if VIS_DILATE_KERNEL > 0:
        ks = VIS_DILATE_KERNEL * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        d_u8 = (np.clip(d, 0, 1) * 255).astype(np.uint8)
        d_u8 = cv2.dilate(d_u8, kernel, iterations=1)
        d = d_u8.astype(np.float64) / 255.0

    img = np.zeros((H, W, 3), dtype=np.float64)
    br = np.power(np.clip(d, 0, 1), 0.4)
    for c in range(3):
        img[:, :, c] = br * BIN_COLORS[b][c]
    return np.clip(img, 0, 255).astype(np.uint8)


def maxh_heatmap(bev, b):
    """Max-height heatmap for one bin with dilation."""
    import cv2
    n_bins = len(HEIGHT_BINS)
    H, W = bev.shape[:2]
    d = bev[:, :, b]
    mh = bev[:, :, n_bins + b]

    vis = np.zeros((H, W), dtype=np.uint8)
    occupied = d > 0
    vis[occupied] = (mh[occupied] * 255).astype(np.uint8)

    if VIS_DILATE_KERNEL > 0:
        ks = VIS_DILATE_KERNEL * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        vis = cv2.dilate(vis, kernel, iterations=1)

    hmap = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    hmap[vis == 0] = 0
    return hmap


# ===================================================================
#  COMPOSITE IMAGES
# ===================================================================

def make_comparison(simple_bev, multi_bev, frame_id):
    """Old gray vs new enhanced color — side by side."""
    import cv2

    gray = (simple_bev * 255).astype(np.uint8)
    gray_bgr = np.stack([gray]*3, axis=-1)
    color = bev_to_color_enhanced(multi_bev)

    h, w = gray_bgr.shape[:2]
    gap = 8
    lh = 50
    canvas = np.zeros((h + lh, w*2 + gap, 3), dtype=np.uint8)

    canvas[lh:lh+h, :w] = gray_bgr
    canvas[lh:lh+h, w+gap:] = color
    canvas[lh:, w:w+gap] = 60

    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Old: 1-Ch Max-Height BEV", (10,35), f, 0.8, (170,170,170), 2)
    cv2.putText(canvas, "New: 8-Ch Dense Multi-Bin BEV", (w+gap+10,35), f, 0.8, (80,255,180), 2)
    cv2.putText(canvas, f"Frame {frame_id}", (w*2+gap-200,35), f, 0.55, (120,120,120), 1)
    return canvas


def make_detail(multi_bev, frame_id):
    """Per-bin breakdown: composite + density row + height row."""
    import cv2

    n_bins = len(HEIGHT_BINS)
    H, W = multi_bev.shape[:2]
    comp = bev_to_color_enhanced(multi_bev)

    pw, ph = W//2, H//2
    f = cv2.FONT_HERSHEY_SIMPLEX
    sp = 6
    rl = 35
    th = 55

    tpw = pw * n_bins + sp * (n_bins - 1)
    cw = max(W, tpw)
    ch = th + H + sp + rl + ph + sp + rl + ph + 35
    canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
    canvas[:] = 12

    cv2.putText(canvas, f"8-Channel Dense BEV — Frame {frame_id}",
                (10,38), f, 1.0, (255,255,255), 2)

    y0 = th
    xo = (cw - W) // 2
    canvas[y0:y0+H, xo:xo+W] = comp
    cv2.putText(canvas, "Composite (density-weighted, enhanced)", (xo, y0-8), f, 0.5, (190,190,190), 1)

    # Density row
    yr2 = y0 + H + sp + rl
    xs = (cw - tpw) // 2
    cv2.putText(canvas, "DENSITY per height bin (log-normalized):", (xs, yr2-12), f, 0.5, (210,210,210), 1)
    for b in range(n_bins):
        panel = single_bin_enhanced(multi_bev, b)
        small = cv2.resize(panel, (pw, ph), interpolation=cv2.INTER_AREA)
        x = xs + b * (pw + sp)
        canvas[yr2:yr2+ph, x:x+pw] = small

        lo, hi = HEIGHT_BINS[b]
        occ = (multi_bev[:,:,b] > 0).sum()
        cv2.putText(canvas, f"{BIN_NAMES[b]} [{lo},{hi}m]", (x, yr2+ph+14), f, 0.4, BIN_COLORS[b], 1)
        cv2.putText(canvas, f"{occ:,} px", (x, yr2+ph+27), f, 0.35, (130,130,130), 1)

    # Height heatmap row
    yr3 = yr2 + ph + sp + rl + 15
    cv2.putText(canvas, "MAX HEIGHT within bin (heatmap: dark=low, bright=high):", (xs, yr3-12), f, 0.5, (210,210,210), 1)
    for b in range(n_bins):
        hm = maxh_heatmap(multi_bev, b)
        small = cv2.resize(hm, (pw, ph), interpolation=cv2.INTER_AREA)
        x = xs + b * (pw + sp)
        canvas[yr3:yr3+ph, x:x+pw] = small
        lo, hi = HEIGHT_BINS[b]
        cv2.putText(canvas, f"{BIN_NAMES[b]} [{lo},{hi}m]", (x, yr3+ph+14), f, 0.4, BIN_COLORS[b], 1)

    return canvas


def make_dual(ego_bev, nbr_bev, frame_id):
    """Ego + Neighbor enhanced BEV side by side."""
    import cv2

    ec = bev_to_color_enhanced(ego_bev)
    nc = bev_to_color_enhanced(nbr_bev)
    h, w = ec.shape[:2]

    lh, gap = 60, 8
    canvas = np.zeros((h + lh, w*2 + gap, 3), dtype=np.uint8)
    canvas[:] = 10

    canvas[lh:lh+h, :w] = ec
    canvas[lh:lh+h, w+gap:] = nc
    canvas[lh:, w:w+gap] = 40

    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Ego Vehicle (CAV 0)", (10,28), f, 0.85, (100,200,255), 2)
    cv2.putText(canvas, "Neighbor Vehicle (CAV 1)", (w+gap+10,28), f, 0.85, (100,255,150), 2)
    cv2.putText(canvas, f"Frame {frame_id}", (w*2+gap-200,28), f, 0.55, (120,120,120), 1)
    cv2.putText(canvas, f"8ch dense | {RESOLUTION}m/px | range {X_RANGE[0]}~{X_RANGE[1]}m",
                (10,50), f, 0.4, (90,90,90), 1)

    # Legend
    ly = lh + h - 28
    for i in range(n_bins := len(HEIGHT_BINS)):
        x = 10 + i * 200
        cv2.rectangle(canvas, (x,ly), (x+12,ly+12), BIN_COLORS[i], -1)
        lo, hi = HEIGHT_BINS[i]
        cv2.putText(canvas, f"{BIN_NAMES[i]} [{lo},{hi}m]", (x+18,ly+11), f, 0.38, (160,160,160), 1)

    return canvas


# ===================================================================
#  VIDEO
# ===================================================================

def make_video(img_dir, out_path, pattern="*.png", fps=VIDEO_FPS):
    import cv2
    files = sorted(glob.glob(os.path.join(img_dir, pattern)))
    if not files: return None

    first = cv2.imread(files[0])
    if first is None: return None
    h, w = first.shape[:2]

    for codec, ext in [('mp4v','.mp4'),('XVID','.avi')]:
        try:
            p = os.path.splitext(out_path)[0] + ext
            wr = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*codec), fps, (w,h))
            if wr.isOpened():
                for f in files:
                    im = cv2.imread(f)
                    if im is not None:
                        if im.shape[:2] != (h,w): im = cv2.resize(im,(w,h))
                        wr.write(im)
                wr.release()
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    return p
        except: continue
    return None


# ===================================================================
#  STATISTICS
# ===================================================================

def analyze_bev(bev, label=""):
    """Print detailed statistics for a BEV array."""
    n_bins = len(HEIGHT_BINS)
    H, W = bev.shape[:2]
    total_px = H * W

    print(f"  {label} BEV stats ({H}x{W}, {bev.shape[2]}ch):")
    any_occ = (bev[:,:,:n_bins].sum(axis=2) > 0).sum()
    print(f"    Overall occupancy: {any_occ:,}/{total_px:,} ({100*any_occ/total_px:.1f}%)")

    for b in range(n_bins):
        d = bev[:,:,b]
        occ = (d > 0).sum()
        if occ > 0:
            avg_d = d[d > 0].mean()
            max_d = d.max()
            mh = bev[:,:,n_bins+b]
            avg_h = mh[mh > 0].mean() if (mh > 0).sum() > 0 else 0
            lo, hi = HEIGHT_BINS[b]
            print(f"    Bin {b} [{lo:+.1f},{hi:+.1f}m] {BIN_NAMES[b]:>8s}: "
                  f"{occ:>6,}px ({100*occ/total_px:>4.1f}%) "
                  f"density avg={avg_d:.3f} max={max_d:.3f} "
                  f"height avg={avg_h:.2f}")
        else:
            print(f"    Bin {b}: empty")


# ===================================================================
#  MAIN
# ===================================================================

def process_scene(scene_name, scene_path, output_root):
    """Process one scene: generate all BEV data and visualizations."""
    import cv2

    frames, ego_path, nbr_path = get_common_frames(scene_path)
    if not frames:
        print(f"  SKIP {scene_name}: no common frames")
        return 0

    n = min(len(frames), MAX_FRAMES)
    frames = frames[:n]

    # Output dirs for this scene
    sd = os.path.join(output_root, scene_name)
    dirs = {
        'comp':    os.path.join(sd, 'comparisons'),
        'detail':  os.path.join(sd, 'multi_bin_bev'),
        'dual':    os.path.join(sd, 'dual_vehicle'),
        'npy_ego': os.path.join(sd, 'numpy_data', 'ego'),
        'npy_nbr': os.path.join(sd, 'numpy_data', 'neighbor'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f"\n  Scene: {scene_name}")
    print(f"  Frames: {n} | Ego: {ego_path} | Nbr: {nbr_path}")
    print(f"  {'-'*50}")

    t0 = time.time()
    for i, fid in enumerate(frames):
        ft = time.time()
        print(f"    [{i+1:4d}/{n}] {fid}", end="", flush=True)

        ego_pts = load_pcd(os.path.join(ego_path, f"{fid}.pcd"))
        nbr_pts = load_pcd(os.path.join(nbr_path, f"{fid}.pcd"))

        if ego_pts is None and nbr_pts is None:
            print(" SKIP"); continue

        # Core conversion
        ego_bev = points_to_bev(ego_pts)
        nbr_bev = points_to_bev(nbr_pts)
        ego_simple = points_to_simple_bev(ego_pts)

        # Save raw numpy (CLEAN, no enhancement)
        np.save(os.path.join(dirs['npy_ego'], f"{fid}.npy"), ego_bev)
        np.save(os.path.join(dirs['npy_nbr'], f"{fid}.npy"), nbr_bev)

        # Visualizations (ENHANCED for display)
        comp = make_comparison(ego_simple, ego_bev, fid)
        cv2.imwrite(os.path.join(dirs['comp'], f"comp_{fid}.png"), comp)

        dual = make_dual(ego_bev, nbr_bev, fid)
        cv2.imwrite(os.path.join(dirs['dual'], f"dual_{fid}.png"), dual)

        if i < DETAIL_FRAMES:
            det = make_detail(ego_bev, fid)
            cv2.imwrite(os.path.join(dirs['detail'], f"detail_{fid}.png"), det)

        dt = time.time() - ft
        print(f" {dt:.1f}s", end="")

        # First frame: detailed stats
        if i == 0 and ego_pts is not None:
            print()
            analyze_bev(ego_bev, "Ego")
            analyze_bev(nbr_bev, "Neighbor")
        else:
            print()

    total = time.time() - t0
    print(f"  {'-'*50}")
    print(f"  {n} frames in {total:.1f}s ({total/n:.2f}s/frame)")

    # Videos
    v1 = make_video(dirs['comp'], os.path.join(sd, "video_comparison.mp4"), "comp_*.png")
    v2 = make_video(dirs['dual'], os.path.join(sd, "video_dual_vehicle.mp4"), "dual_*.png")
    if v1: print(f"  Video: {v1}")
    if v2: print(f"  Video: {v2}")

    return n


def main():
    print()
    print("=" * 65)
    print("  Multi-Height-Bin BEV v3 FINAL — Literature-Optimized")
    print("  MEng Thesis: V2V Cooperative Perception")
    print("=" * 65)
    print()

    check_deps()

    bw = int((X_RANGE[1]-X_RANGE[0]) / RESOLUTION)
    bh = int((Y_RANGE[1]-Y_RANGE[0]) / RESOLUTION)
    n_bins = len(HEIGHT_BINS)

    print("Parameters (from CORE/PointPillars/V2V4Real literature):")
    print(f"  BEV size:     {bw}x{bh} px")
    print(f"  Range:        [{X_RANGE[0]}, {X_RANGE[1]}] x [{Y_RANGE[0]}, {Y_RANGE[1]}] m")
    print(f"  Resolution:   {RESOLUTION} m/px  (PointPillars standard: 0.16m)")
    print(f"  Channels:     {n_bins*2}  ({n_bins} bins x [density + max_height])")
    print(f"  Density cap:  {DENSITY_CAP}  (log normalization)")
    print(f"  Height bins:")
    for i, (lo, hi) in enumerate(HEIGHT_BINS):
        print(f"    Bin {i}: [{lo:+.1f}, {hi:+.1f}m]  {BIN_NAMES[i]}")
    print(f"  Visualization: dilation={VIS_DILATE_KERNEL}px, glow={VIS_GLOW_SIGMA}/{VIS_GLOW_BLEND}")
    print()

    # Find scenes
    if SINGLE_SCENE:
        name = os.path.basename(SINGLE_SCENE)
        scenes = [(name, SINGLE_SCENE)]
    else:
        scenes = find_all_scenes(BASE_DIR)

    if not scenes:
        print("ERROR: No valid scenes found!")
        print(f"  Looked in: {BASE_DIR if not SINGLE_SCENE else SINGLE_SCENE}")
        print(f"  Expected sub-folders: {EGO_VEHICLE}/ and {NEIGHBOR_VEHICLE}/ with .pcd files")
        sys.exit(1)

    print(f"Found {len(scenes)} scene(s):")
    for name, path in scenes:
        print(f"  {name}")
    print()

    # Process all scenes
    total_frames = 0
    t_all = time.time()
    for name, path in scenes:
        total_frames += process_scene(name, path, OUTPUT_DIR)

    elapsed = time.time() - t_all

    # Final summary
    print()
    print("=" * 65)
    print("  ALL DONE!")
    print("=" * 65)
    print(f"  Scenes:       {len(scenes)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Total time:   {elapsed:.1f}s")
    print(f"  Output:       {os.path.abspath(OUTPUT_DIR)}/")
    print()

    # Print numpy spec
    first_scene = scenes[0][0]
    first_frames, _, _ = get_common_frames(scenes[0][1])
    if first_frames:
        sample_path = os.path.join(OUTPUT_DIR, first_scene, 'numpy_data', 'ego', f"{first_frames[0]}.npy")
        if os.path.exists(sample_path):
            s = np.load(sample_path)
            print("  NumPy array spec (for model training):")
            print(f"    Shape:    {s.shape}  = ({bh}, {bw}, {n_bins*2})")
            print(f"    Ch 0-{n_bins-1}:   density per bin   (float32, [0,1])")
            print(f"    Ch {n_bins}-{n_bins*2-1}:   max-height per bin (float32, [0,1])")
            print(f"    Non-zero: {(s>0).sum():,} / {s.size:,} ({100*(s>0).sum()/s.size:.1f}%)")
            print(f"    Load:     np.load('{sample_path}')")
    print()

    # Advisor meeting guidance
    print("  FOR YOUR ADVISOR:")
    for name, _ in scenes:
        sd = os.path.join(OUTPUT_DIR, name)
        details = sorted(glob.glob(os.path.join(sd, 'multi_bin_bev', '*.png')))
        vid_c = os.path.join(sd, 'video_comparison.mp4')
        vid_d = os.path.join(sd, 'video_dual_vehicle.mp4')
        print(f"\n  [{name}]")
        if details:
            print(f"    1. {details[0]}")
        if os.path.exists(vid_c):
            print(f"    2. {vid_c}")
        if os.path.exists(vid_d):
            print(f"    3. {vid_d}")

    print()
    print("=" * 65)
    print("  Next step: mask ego BEV + train U-Net/GAN/Diffusion")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
