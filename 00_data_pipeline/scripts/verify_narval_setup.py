#!/usr/bin/env python3
"""
Quick remote verification for the Narval minimal upload layout.
"""

from __future__ import annotations

import argparse
import socket

import paramiko


def parse_args():
    p = argparse.ArgumentParser(description="Verify Narval layout and Python env.")
    p.add_argument("--host", type=str, default="narval.alliancecan.ca")
    p.add_argument("--user", type=str, required=True)
    p.add_argument("--password", type=str, required=True)
    p.add_argument("--duo_code", type=str, required=True)
    p.add_argument("--key_path", type=str, required=True)
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


def run(transport, cmd: str):
    chan = transport.open_session(timeout=30)
    chan.exec_command(f'bash -lc "{cmd}"')
    stdout = chan.makefile("r", -1).read().strip()
    stderr = chan.makefile_stderr("r", -1).read().strip()
    status = chan.recv_exit_status()
    return status, stdout, stderr


def main():
    args = parse_args()
    key = paramiko.Ed25519Key.from_private_key_file(args.key_path)

    sock = socket.create_connection((args.host, 22), timeout=30)
    transport = paramiko.Transport(sock)
    transport.start_client(timeout=30)
    transport.auth_publickey(args.user, key)
    if not transport.is_authenticated():
        transport.auth_interactive(args.user, keyboard_handler(args.password, args.duo_code))

    checks = [
        ("layout", "find ~/scratch/MEng_Project -maxdepth 1 -mindepth 1 -type d | sed 's|.*/MEng_Project/||' | sort"),
        ("data_count", "find ~/scratch/MEng_Project/data/dataset_prepared -type f | wc -l"),
        ("code_count", "find ~/scratch/MEng_Project/code -type f | wc -l"),
        ("python_env", "source ~/scratch/MEng_Project/venv/bin/activate && python - <<'PY'\nimport torch, optuna, lpips, torchmetrics\nprint('OK')\nPY"),
        ("train_help", "source ~/scratch/MEng_Project/venv/bin/activate && cd ~/scratch/MEng_Project/code && python train.py --help | head -n 3"),
    ]

    for name, cmd in checks:
        status, stdout, stderr = run(transport, cmd)
        print(f"[{name}] status={status}")
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        print("---")

    transport.close()


if __name__ == "__main__":
    main()
