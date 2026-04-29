#!/usr/bin/env python3
"""
Re-evaluate a saved final checkpoint with the current metric code.

This script loads the model code from the cleaned final folders. It is mainly
for sanity checks after moving files around; normal review can use the copied
JSON summaries already in each model folder.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset_prepared"
MODEL_DIRS = {
    "unet": REPO_ROOT / "01_unet_final",
    "pix2pix": REPO_ROOT / "02_pix2pix_final",
    "diffusion": REPO_ROOT / "03_diffusion_final",
}


def parse_features(text):
    return [int(x) for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Re-evaluate a saved final checkpoint.")
    p.add_argument("--model_kind", choices=["unet", "pix2pix", "diffusion"], required=True)
    p.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--training_root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--features", type=str, default="16,32,64,128")
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--sample_steps", type=int, default=25)
    p.add_argument("--occ_threshold", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def use_model_folder(model_kind):
    model_dir = MODEL_DIRS[model_kind]
    sys.path.insert(0, str(model_dir))
    return model_dir


def load_run_config(training_root: Path):
    for name in ["test_metrics_refresh.json", "test_metrics.json", "pix2pix_summary.json", "diffusion_summary.json"]:
        path = training_root / "results" / name
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                blob = json.load(f)
            return blob.get("config", {})
    return {}


def default_checkpoint(model_kind, training_root):
    if model_kind == "pix2pix":
        return training_root / "checkpoints" / "best_pix2pix.pth"
    if model_kind == "diffusion":
        return training_root / "checkpoints" / "best_diffusion.pth"
    return training_root / "checkpoints" / "best_unet.pth"


def main():
    args = parse_args()
    use_model_folder(args.model_kind)

    dataset_mod = importlib.import_module("dataset")
    train_mod = importlib.import_module("train")
    unet_mod = importlib.import_module("unet")

    dataset_root = args.dataset_root.resolve()
    training_root = args.training_root.resolve()
    run_cfg = load_run_config(training_root)
    features = run_cfg.get("features", parse_features(args.features))
    occ_threshold = args.occ_threshold
    if occ_threshold is None:
        occ_threshold = float(run_cfg.get("occ_threshold", 0.07))

    checkpoint_path = (args.checkpoint.resolve() if args.checkpoint else default_checkpoint(args.model_kind, training_root))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, test_loader = dataset_mod.get_dataloaders(
        dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if args.model_kind == "diffusion":
        diffusion_mod = importlib.import_module("train_diffusion")
        model = unet_mod.UNet(
            in_channels=24,
            out_channels=8,
            features=features,
            time_emb_dim=128,
            final_activation="identity",
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        timesteps = int(run_cfg.get("timesteps", args.timesteps))
        sample_steps = int(run_cfg.get("sample_steps", args.sample_steps))
        alpha_bar = diffusion_mod.alpha_bar_schedule(timesteps, device)
        val_metrics = diffusion_mod.evaluate_sampled(
            model, val_loader, device, timesteps, sample_steps, alpha_bar, occ_threshold
        ) if len(val_loader) else None
        test_metrics = diffusion_mod.evaluate_sampled(
            model, test_loader, device, timesteps, sample_steps, alpha_bar, occ_threshold
        ) if len(test_loader) else None
    else:
        state_key = "generator" if args.model_kind == "pix2pix" else "model_state_dict"
        model = unet_mod.UNet(in_channels=16, out_channels=8, features=features).to(device)
        model.load_state_dict(ckpt[state_key])
        model.eval()
        cfg = train_mod.Config()
        cfg.out_channels = 8
        cfg.occ_threshold = occ_threshold
        val_metrics = train_mod.evaluate(model, val_loader, device, cfg) if len(val_loader) else None
        test_metrics = train_mod.evaluate(model, test_loader, device, cfg) if len(test_loader) else None

    payload = {
        "model_kind": args.model_kind,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "config": {
            **run_cfg,
            "features": features,
            "occ_threshold": occ_threshold,
            "timesteps": int(run_cfg.get("timesteps", args.timesteps)) if args.model_kind == "diffusion" else None,
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    output_path = args.output.resolve() if args.output else training_root / "results" / "test_metrics_refresh.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

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
