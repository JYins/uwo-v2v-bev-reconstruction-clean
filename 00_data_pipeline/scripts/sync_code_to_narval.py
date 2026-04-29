#!/usr/bin/env python3
"""
Sync only the minimal training/evaluation code bundle to Narval.
"""

from __future__ import annotations

import argparse
import socket
from pathlib import Path
from stat import S_ISDIR

import paramiko


DEFAULT_FILES = [
    "dataset.py",
    "unet.py",
    "train.py",
    "train_pix2pix.py",
    "train_diffusion.py",
    "tune_unet_optuna.py",
    "tune_pix2pix_adv.py",
    "compute_perceptual_metrics.py",
    "reevaluate_saved_model.py",
    "render_threshold_panels.py",
    "visualize_4columns.py",
    "visualize_channel_splits.py",
    "visualize_unet_predictions.py",
    "optuna_unet_run/best_params.json",
]


def parse_args():
    p = argparse.ArgumentParser(description="Sync minimal code bundle to Narval.")
    p.add_argument("--local_root", type=Path, required=True)
    p.add_argument("--remote_root", type=str, required=True)
    p.add_argument("--host", type=str, default="narval.alliancecan.ca")
    p.add_argument("--user", type=str, required=True)
    p.add_argument("--password", type=str, required=True)
    p.add_argument("--duo_code", type=str, required=True)
    p.add_argument("--key_path", type=Path, required=True)
    p.add_argument("--extra", type=Path, nargs="*", default=[])
    return p.parse_args()


def keyboard_handler(password, duo_code):
    def _handler(title, instructions, prompts):
        answers = []
        for prompt, _show_input in prompts:
            low = prompt.lower()
            if "password" in low:
                answers.append(password)
            elif "duo" in low or "passcode" in low or "verification code" in low:
                answers.append(duo_code)
            else:
                answers.append("")
        return answers

    return _handler


def ensure_remote_dir(sftp, path):
    cur = ""
    for piece in path.strip("/").split("/"):
        cur += "/" + piece
        try:
            st = sftp.stat(cur)
            if not S_ISDIR(st.st_mode):
                raise RuntimeError(f"Remote path exists but is not a directory: {cur}")
        except IOError:
            sftp.mkdir(cur)


def sync_one(sftp, src: Path, dst: str):
    ensure_remote_dir(sftp, str(Path(dst).parent).replace("\\", "/"))
    sftp.put(str(src), dst)


def main():
    args = parse_args()
    key = paramiko.Ed25519Key.from_private_key_file(str(args.key_path))

    sock = socket.create_connection((args.host, 22), timeout=30)
    transport = paramiko.Transport(sock)
    transport.start_client(timeout=30)
    transport.auth_publickey(args.user, key)
    if not transport.is_authenticated():
        transport.auth_interactive(args.user, keyboard_handler(args.password, args.duo_code))

    sftp = paramiko.SFTPClient.from_transport(transport)
    ensure_remote_dir(sftp, args.remote_root)

    manifest = [Path(item) for item in DEFAULT_FILES] + list(args.extra)
    synced = 0

    print(f"Local root:  {args.local_root}", flush=True)
    print(f"Remote root: {args.remote_root}", flush=True)
    print("Syncing manifest:", flush=True)

    for rel in manifest:
        src = args.local_root / rel
        if not src.exists():
            print(f"  MISSING {rel}", flush=True)
            continue
        dst = f"{args.remote_root.rstrip('/')}/{rel.as_posix()}"
        sync_one(sftp, src, dst)
        synced += 1
        print(f"  OK {rel.as_posix()}", flush=True)

    print(f"Done. Synced files: {synced}", flush=True)
    sftp.close()
    transport.close()


if __name__ == "__main__":
    main()
