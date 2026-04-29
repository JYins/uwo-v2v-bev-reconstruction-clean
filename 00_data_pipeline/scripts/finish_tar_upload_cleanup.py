#!/usr/bin/env python3
"""
Wait for a local tar archive to finish, upload it to Narval, extract it remotely,
then clean up the local tar file.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path
from stat import S_ISDIR

import paramiko


def parse_args():
    p = argparse.ArgumentParser(description="Finish tar -> upload -> extract -> cleanup flow.")
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--remote_tar", type=str, required=True)
    p.add_argument("--remote_extract_dir", type=str, required=True)
    p.add_argument("--host", type=str, default="narval.alliancecan.ca")
    p.add_argument("--user", type=str, required=True)
    p.add_argument("--password", type=str, required=True)
    p.add_argument("--duo_code", type=str, required=True)
    p.add_argument("--key_path", type=Path, required=True)
    p.add_argument("--poll_seconds", type=int, default=20)
    p.add_argument("--stable_polls", type=int, default=3)
    p.add_argument("--retry_wait_seconds", type=int, default=20)
    p.add_argument("--delete_remote_tar", action="store_true")
    return p.parse_args()


def log(msg: str):
    print(msg, flush=True)


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


def wait_for_archive_to_stabilize(path: Path, poll_seconds: int, stable_polls: int):
    stable = 0
    last_size = -1
    while True:
        if not path.exists():
            log(f"Waiting for archive to appear: {path}")
            time.sleep(poll_seconds)
            continue

        size = path.stat().st_size
        log(f"Archive size: {size / (1024**3):.2f} GB")
        if size > 0 and size == last_size:
            stable += 1
            log(f"Archive stable check {stable}/{stable_polls}")
        else:
            stable = 0
        last_size = size

        if stable >= stable_polls:
            log("Archive size has stabilized. Proceeding to upload.")
            return

        time.sleep(poll_seconds)


def connect(host: str, user: str, password: str, duo_code: str, key_path: Path):
    key = paramiko.Ed25519Key.from_private_key_file(str(key_path))
    sock = socket.create_connection((host, 22), timeout=30)
    transport = paramiko.Transport(sock)
    transport.start_client(timeout=30)
    transport.auth_publickey(user, key)
    if not transport.is_authenticated():
        transport.auth_interactive(user, keyboard_handler(password, duo_code))
    sftp = paramiko.SFTPClient.from_transport(transport)
    return transport, sftp


def upload_with_progress(sftp, local_path: Path, remote_path: str):
    ensure_remote_dir(sftp, str(Path(remote_path).parent).replace("\\", "/"))
    total = local_path.stat().st_size
    state = {"last_report": 0.0}

    def callback(done, all_bytes):
        now = time.time()
        if now - state["last_report"] < 5 and done != all_bytes:
            return
        state["last_report"] = now
        pct = (done / all_bytes * 100.0) if all_bytes else 100.0
        log(f"Upload progress: {done / (1024**3):.2f}/{all_bytes / (1024**3):.2f} GB ({pct:.1f}%)")

    remote_tmp = remote_path + ".part"
    offset = 0
    try:
        offset = sftp.stat(remote_tmp).st_size
    except IOError:
        offset = 0

    mode = "ab" if offset > 0 else "wb"
    log(
        f"Uploading {local_path} -> {remote_path} "
        f"(resume offset {offset / (1024**3):.2f} GB)"
    )

    with local_path.open("rb") as src, sftp.open(remote_tmp, mode) as dst:
        if offset > 0:
            src.seek(offset)
        transferred = offset
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            transferred += len(chunk)
            callback(transferred, total)
        dst.flush()

    try:
        sftp.remove(remote_path)
    except IOError:
        pass
    sftp.rename(remote_tmp, remote_path)
    log(f"Upload complete: {total / (1024**3):.2f} GB")


def run_remote(transport, cmd: str):
    chan = transport.open_session(timeout=30)
    chan.exec_command(f'bash -lc "{cmd}"')
    stdout = chan.makefile("r", -1).read().strip()
    stderr = chan.makefile_stderr("r", -1).read().strip()
    status = chan.recv_exit_status()
    return status, stdout, stderr


def main():
    args = parse_args()

    wait_for_archive_to_stabilize(args.archive, args.poll_seconds, args.stable_polls)

    while True:
        try:
            transport, sftp = connect(args.host, args.user, args.password, args.duo_code, args.key_path)
            try:
                upload_with_progress(sftp, args.archive, args.remote_tar)
                break
            finally:
                sftp.close()
                transport.close()
        except Exception as exc:
            log(f"Upload interrupted, will retry in {args.retry_wait_seconds}s: {exc}")
            time.sleep(args.retry_wait_seconds)

    transport, sftp = connect(args.host, args.user, args.password, args.duo_code, args.key_path)
    try:
        extract_cmd = (
            f"mkdir -p {args.remote_extract_dir} && "
            f"tar -xf {args.remote_tar} -C {args.remote_extract_dir}"
        )
        log("Starting remote extract...")
        status, stdout, stderr = run_remote(transport, extract_cmd)
        log(f"Remote extract status: {status}")
        if stdout:
            log(stdout)
        if stderr:
            log(stderr)
        if status != 0:
            raise RuntimeError("Remote extract failed.")

        count_cmd = f"find {args.remote_extract_dir}/dataset_prepared -type f | wc -l"
        status, stdout, stderr = run_remote(transport, count_cmd)
        log(f"Remote file count status: {status}")
        if stdout:
            log(f"Remote extracted file count: {stdout}")
        if stderr:
            log(stderr)

        if args.delete_remote_tar:
            run_remote(transport, f"rm -f {args.remote_tar}")
            log("Deleted remote tar archive.")

    finally:
        sftp.close()
        transport.close()

    try:
        args.archive.unlink()
        log(f"Deleted local archive: {args.archive}")
    except Exception as exc:
        log(f"WARNING: failed to delete local archive: {exc}")

    log("Finish-tar-upload-cleanup flow completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FATAL: {exc}")
        sys.exit(1)
