#!/usr/bin/env python3
"""
Compute LPIPS and FID on rendered 3-channel BEV outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lpips
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from dataset import BEVReconstructionDataset, discover_available_splits
from train_diffusion import alpha_bar_schedule, estimate_x0, q_sample
from unet import UNet
from visualize_4columns import bev_to_color


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = SCRIPT_DIR / "dataset_prepared"


def parse_features(text):
    return [int(part) for part in text.split(",") if part.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Compute LPIPS/FID on rendered BEV panels.")
    p.add_argument("--model_kind", choices=["unet", "pix2pix", "diffusion"], required=True)
    p.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--training_root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--output_json", type=Path, required=True)
    p.add_argument("--features", type=str, default="16,32,64,128")
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_run_config(training_root: Path):
    for name in ["test_metrics.json", "pix2pix_summary.json", "diffusion_summary.json"]:
        path = training_root / "results" / name
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                blob = json.load(f)
            return blob.get("config", {})
    return {}


def resolve_checkpoint(args):
    if args.checkpoint is not None:
        return args.checkpoint.resolve()
    if args.model_kind == "pix2pix":
        return args.training_root.resolve() / "checkpoints" / "best_pix2pix.pth"
    if args.model_kind == "diffusion":
        return args.training_root.resolve() / "checkpoints" / "best_diffusion.pth"
    return args.training_root.resolve() / "checkpoints" / "best_unet.pth"


def load_model(model_kind, checkpoint_path: Path, device, features):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if model_kind == "diffusion":
        model = UNet(in_channels=24, out_channels=8, features=features).to(device)
        model.load_state_dict(ckpt["model"])
    else:
        model = UNet(in_channels=16, out_channels=8, features=features).to(device)
        state = ckpt["generator"] if model_kind == "pix2pix" else ckpt["model_state_dict"]
        model.load_state_dict(state)
    model.eval()
    return model, ckpt


def predict_tensor(model, model_kind, inp_batch, device, timesteps):
    with torch.no_grad():
        if model_kind == "diffusion":
            cond = inp_batch.to(device)
            x0_shape = (cond.shape[0], 8, cond.shape[2], cond.shape[3])
            x0_dummy = torch.zeros(x0_shape, device=device)
            t_idx = torch.full((cond.shape[0],), timesteps - 1, device=device, dtype=torch.long)
            alpha_bar = alpha_bar_schedule(timesteps, device)
            noise = torch.randn_like(x0_dummy)
            x_t = q_sample(x0_dummy, t_idx, noise, alpha_bar)
            pred_noise = model(torch.cat([x_t, cond], dim=1))
            return estimate_x0(x_t, pred_noise, t_idx, alpha_bar).clamp(0.0, 1.0)
        return model(inp_batch.to(device))


def bev_to_rgb_tensor(bev_tensor):
    bev_np = bev_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    bgr = bev_to_color(bev_np)
    rgb = bgr[:, :, ::-1].copy()
    uint8_img = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    float_img = uint8_img.float().div(127.5).sub(1.0)
    return uint8_img, float_img


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    training_root = args.training_root.resolve()
    output_json = args.output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    run_cfg = load_run_config(training_root)
    features = parse_features(args.features)
    if isinstance(run_cfg.get("features"), list):
        features = run_cfg["features"]
    timesteps = int(run_cfg.get("timesteps", args.timesteps))

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    checkpoint_path = resolve_checkpoint(args)
    model, ckpt = load_model(args.model_kind, checkpoint_path, device, features)

    _, _, test_splits = discover_available_splits(dataset_root)
    dataset = BEVReconstructionDataset(dataset_root, test_splits, augment=False)
    n_items = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    lpips_metric = lpips.LPIPS(net="alex").to(device)
    lpips_metric.eval()

    lpips_sum = 0.0
    for idx in tqdm(range(n_items), desc=f"{args.model_kind} perceptual"):
        inp, tgt, _ = dataset[idx]
        pred = predict_tensor(model, args.model_kind, inp.unsqueeze(0), device, timesteps).squeeze(0)

        pred_uint8, pred_lpips = bev_to_rgb_tensor(pred)
        tgt_uint8, tgt_lpips = bev_to_rgb_tensor(tgt)

        fid.update(tgt_uint8.unsqueeze(0).to(device), real=True)
        fid.update(pred_uint8.unsqueeze(0).to(device), real=False)

        with torch.no_grad():
            lp = lpips_metric(pred_lpips.unsqueeze(0).to(device), tgt_lpips.unsqueeze(0).to(device))
        lpips_sum += float(lp.item())

    payload = {
        "model_kind": args.model_kind,
        "training_root": str(training_root),
        "checkpoint": str(checkpoint_path),
        "epoch": ckpt.get("epoch"),
        "samples": n_items,
        "lpips_mean": lpips_sum / max(n_items, 1),
        "fid": float(fid.compute().item()),
        "notes": "Computed on consistently rendered 3-channel BEV images for supplementary realism comparison.",
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {output_json}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
