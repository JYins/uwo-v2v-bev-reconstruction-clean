#!/usr/bin/env python3
"""
Re-evaluate a saved model checkpoint with the current metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dataset import get_dataloaders
from train import Config as TrainCfg
from train import evaluate
from train_diffusion import alpha_bar_schedule, evaluate_proxy
from unet import UNet


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = SCRIPT_DIR / "dataset_prepared"


def parse_features(text):
    return [int(x) for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Re-evaluate a saved checkpoint with current metrics.")
    p.add_argument("--model_kind", choices=["unet", "pix2pix", "diffusion"], required=True)
    p.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--training_root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--features", type=str, default="16,32,64,128")
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--occ_threshold", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def load_run_config(training_root: Path):
    for name in ["test_metrics.json", "pix2pix_summary.json", "diffusion_summary.json"]:
        path = training_root / "results" / name
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                blob = json.load(f)
            return blob.get("config", {})
    return {}


def load_unet_like(checkpoint_path: Path, device, state_key: str, features):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNet(in_channels=16, out_channels=8, features=features).to(device)
    model.load_state_dict(ckpt[state_key])
    model.eval()
    return model, ckpt


def load_diffusion_model(checkpoint_path: Path, device, features):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNet(in_channels=24, out_channels=8, features=features).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    training_root = args.training_root.resolve()
    run_cfg = load_run_config(training_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = parse_features(args.features)
    if "features" in run_cfg and isinstance(run_cfg["features"], list):
        features = run_cfg["features"]

    occ_threshold = args.occ_threshold
    if occ_threshold is None:
        occ_threshold = float(run_cfg.get("occ_threshold", 0.07))

    timesteps = int(run_cfg.get("timesteps", args.timesteps))

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint.resolve()
    elif args.model_kind == "pix2pix":
        checkpoint_path = training_root / "checkpoints" / "best_pix2pix.pth"
    elif args.model_kind == "diffusion":
        checkpoint_path = training_root / "checkpoints" / "best_diffusion.pth"
    else:
        checkpoint_path = training_root / "checkpoints" / "best_unet.pth"

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    del train_loader

    if args.model_kind == "pix2pix":
        model, ckpt = load_unet_like(checkpoint_path, device, "generator", features)
        cfg = TrainCfg()
        cfg.out_channels = 8
        cfg.occ_threshold = occ_threshold
        val_metrics = evaluate(model, val_loader, device, cfg) if len(val_loader) > 0 else None
        test_metrics = evaluate(model, test_loader, device, cfg) if len(test_loader) > 0 else None
    elif args.model_kind == "diffusion":
        model, ckpt = load_diffusion_model(checkpoint_path, device, features)
        alpha_bar = alpha_bar_schedule(timesteps, device)
        val_metrics = (
            evaluate_proxy(model, val_loader, device, timesteps, alpha_bar, occ_threshold)
            if len(val_loader) > 0
            else None
        )
        test_metrics = (
            evaluate_proxy(model, test_loader, device, timesteps, alpha_bar, occ_threshold)
            if len(test_loader) > 0
            else None
        )
    else:
        model, ckpt = load_unet_like(checkpoint_path, device, "model_state_dict", features)
        cfg = TrainCfg()
        cfg.out_channels = 8
        cfg.occ_threshold = occ_threshold
        val_metrics = evaluate(model, val_loader, device, cfg) if len(val_loader) > 0 else None
        test_metrics = evaluate(model, test_loader, device, cfg) if len(test_loader) > 0 else None

    payload = {
        "model_kind": args.model_kind,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "config": {
            **run_cfg,
            "features": features,
            "occ_threshold": occ_threshold,
            "timesteps": timesteps if args.model_kind == "diffusion" else None,
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    output_path = args.output.resolve() if args.output else training_root / "results" / "test_metrics_refresh.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"saved {output_path}")
    if test_metrics:
        print(
            "test "
            f"iou={test_metrics['masked_occ_iou']:.4f} "
            f"prec={test_metrics['masked_occ_precision']:.4f} "
            f"rec={test_metrics['masked_occ_recall']:.4f} "
            f"f1={test_metrics['masked_occ_f1']:.4f} "
            f"rmse={test_metrics['masked_rmse']:.6f}"
        )


if __name__ == "__main__":
    main()
