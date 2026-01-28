#!/usr/bin/env python3
"""
Resumable Hugging Face file downloader for very large artifacts.

Why:
- `hf download ...` can abort on flaky networks (SSL EOF / timeouts).
- For huge single files (e.g. 20GB parts), a simple HTTP resume loop is often
  more robust than the Hub client stack.

This script uses `curl -L -C -` (HTTP Range) + retries to resume downloads.

Example:
  python AgentGym2/AgentGym/scripts/resumable_hf_download.py \\
    --repo-id HelloKKMe/grounding_dataset \\
    --repo-type dataset \\
    --filename image.part.aa \\
    --out /data/pengchen/AgentGym_qwen/data/hf_grounding_dataset/image.part.aa
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def _repo_resolve_url(repo_id: str, repo_type: str, filename: str, revision: str) -> str:
    if repo_type == "dataset":
        return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{filename}"
    if repo_type == "model":
        return f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
    if repo_type == "space":
        return f"https://huggingface.co/spaces/{repo_id}/resolve/{revision}/{filename}"
    raise ValueError(f"Unsupported repo_type={repo_type!r}")


def _curl_download(url: str, out_path: Path, token: Optional[str], extra_env: dict) -> int:
    """
    Use curl with:
    -L follow redirects (Hub resolves to signed URLs)
    -C - resume
    --retry ... retry on transient errors
    --speed-* abort if stuck (lets retry kick in)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "curl",
        "--fail",
        "--location",
        "--continue-at",
        "-",
        "--output",
        str(out_path),
        "--retry",
        "999999",
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "--retry-max-time",
        "0",
        "--speed-time",
        "60",
        "--speed-limit",
        "10240",  # 10KB/s for 60s -> treat as stalled
        url,
    ]

    # Optional auth (needed for gated/private repos).
    env = os.environ.copy()
    env.update(extra_env or {})
    if token:
        cmd.insert(1, "--header")
        cmd.insert(2, f"Authorization: Bearer {token}")

    return subprocess.call(cmd, env=env)


def _remote_size_via_head(url: str, token: Optional[str]) -> Optional[int]:
    """
    Best-effort: ask final redirect target for Content-Length.
    If not available, return None (we can still resume blindly).
    """
    try:
        import requests  # type: ignore
    except Exception:
        return None

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.head(url, headers=headers, allow_redirects=True, timeout=30)
        r.raise_for_status()
        cl = r.headers.get("Content-Length")
        if cl:
            return int(cl)
    except Exception:
        return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--filename", required=True)
    ap.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"])
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", required=True, help="Output filepath (download target).")
    ap.add_argument("--sleep", type=float, default=10.0, help="Sleep seconds between retries.")
    ap.add_argument("--max-attempts", type=int, default=0, help="0 = infinite retry.")
    ap.add_argument(
        "--disable-xet",
        action="store_true",
        default=False,
        help="Set HF_HUB_DISABLE_XET=1 (kept for consistency; curl doesn't use Xet anyway).",
    )
    args = ap.parse_args()

    if shutil.which("curl") is None:
        print("ERROR: curl not found in PATH; install curl or use `hf download`.", file=sys.stderr)
        return 2

    url = _repo_resolve_url(args.repo_id, args.repo_type, args.filename, args.revision)
    out_path = Path(args.out)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    extra_env = {}
    if args.disable_xet:
        extra_env["HF_HUB_DISABLE_XET"] = "1"

    expected = _remote_size_via_head(url, token)
    if expected:
        print(f"[info] remote size: {expected} bytes")
    else:
        print("[info] remote size: unknown (no Content-Length); will download until curl succeeds")

    attempt = 0
    while True:
        attempt += 1
        if args.max_attempts and attempt > args.max_attempts:
            print(f"[error] reached max attempts={args.max_attempts}", file=sys.stderr)
            return 1

        local_size = out_path.stat().st_size if out_path.exists() else 0
        if expected and local_size >= expected:
            print(f"[ok] already downloaded: {out_path} ({local_size} bytes)")
            return 0

        print(f"[info] attempt {attempt} | local={local_size} bytes | url={url}")
        rc = _curl_download(url, out_path, token, extra_env)

        # If curl succeeded, verify size if we can.
        if rc == 0:
            local_size = out_path.stat().st_size if out_path.exists() else 0
            if expected and local_size < expected:
                print(
                    f"[warn] curl succeeded but file smaller than expected "
                    f"({local_size} < {expected}); retrying...",
                    file=sys.stderr,
                )
            else:
                print(f"[ok] downloaded: {out_path} ({local_size} bytes)")
                return 0
        else:
            print(f"[warn] curl exit code={rc}; retrying after {args.sleep}s", file=sys.stderr)

        time.sleep(args.sleep)


if __name__ == "__main__":
    raise SystemExit(main())

