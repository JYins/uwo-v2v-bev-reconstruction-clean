#!/usr/bin/env python3
"""
Sync local dataset_prepared to Narval scratch with resume support.
"""

from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path
from stat import S_ISDIR

import paramiko


def parse_args():
    p = argparse.ArgumentParser(description="Sync dataset_prepared to Narval.")
    p.add_argument("--local_root", type=Path, required=True)
    p.add_argument("--remote_root", type=str, required=True)
    p.add_argument("--host", type=str, default="narval.alliancecan.ca")
    p.add_argument("--user", type=str, required=True)
    p.add_argument("--password", type=str, required=True)
    p.add_argument("--duo_code", type=str, required=True)
    p.add_argument("--key_path", type=Path, required=True)
    p.add_argument("--log_every", type=int, default=100)
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

    local_files = [p for p in args.local_root.rglob("*") if p.is_file()]
    uploaded = 0
    skipped = 0
    transferred_bytes = 0

    print(f"Local files: {len(local_files)}", flush=True)
    print(f"Local root:  {args.local_root}", flush=True)
    print(f"Remote root: {args.remote_root}", flush=True)

    for idx, src in enumerate(local_files, start=1):
        rel = src.relative_to(args.local_root).as_posix()
        dst = f"{args.remote_root.rstrip('/')}/{rel}"
        ensure_remote_dir(sftp, str(Path(dst).parent).replace("\\", "/"))

        try:
            st = sftp.stat(dst)
            if st.st_size == src.stat().st_size:
                skipped += 1
                if idx % args.log_every == 0:
                    print(
                        f"[{idx}/{len(local_files)}] skipped={skipped} uploaded={uploaded} bytes={transferred_bytes}",
                        flush=True,
                    )
                continue
        except IOError:
            pass

        sftp.put(str(src), dst)
        uploaded += 1
        transferred_bytes += src.stat().st_size

        if idx % args.log_every == 0:
            print(
                f"[{idx}/{len(local_files)}] skipped={skipped} uploaded={uploaded} bytes={transferred_bytes}",
                flush=True,
            )

    print("Sync complete.", flush=True)
    print(f"Uploaded files: {uploaded}", flush=True)
    print(f"Skipped files:  {skipped}", flush=True)
    print(f"Uploaded GB:    {transferred_bytes / (1024**3):.2f}", flush=True)

    sftp.close()
    transport.close()


if __name__ == "__main__":
    main()
