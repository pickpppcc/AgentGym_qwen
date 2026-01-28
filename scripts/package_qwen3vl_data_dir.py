#!/usr/bin/env python3
"""
Prepare a portable data directory for Qwen3VL tool-call training.

Why:
- Existing JSONL often contains absolute image paths like:
    /dfs/data/AgentGym_qwen/data/images_vl/...
  which are platform-specific.
- AgentGym2/AgentGym/agentenv/agentenv/trainer/bc_trainer.py can resolve *relative*
  image paths against the JSONL directory, so we can make the dataset portable.

This script:
1) Rewrites image paths inside JSONL (both obj["images"] and the prompt tail fields
   ScreenshotRelPath/ScreenshotAbsPath) to be relative to a new root.
2) Copies any image files that exist locally into the output root, preserving the
   relative path layout (e.g. images_vl/...).
3) Optionally copies the grounding parquet into the output root.
4) Writes a missing image list for paths that can't be found locally.

Typical usage:
  python AgentGym2/AgentGym/scripts/package_qwen3vl_data_dir.py \\
    --input AgentGym2/AgentGym/data/vl_all_prefix.dfs.qwen3vl_toolcall.fixed.grounding30.shuf.jsonl \\
    --output-root /data/pengchen/AgentGym_qwen/data \\
    --local-base AgentGym2/AgentGym/data \\
    --copy-parquet AgentGym2/AgentGym/data/0000.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


DFS_PREFIX = "/dfs/data/AgentGym_qwen/data/"


def _rewrite_one_path(p: str) -> str:
    """Rewrite known absolute prefixes to a portable relative path."""
    if not isinstance(p, str):
        return p
    if p.startswith(DFS_PREFIX):
        return p[len(DFS_PREFIX) :].lstrip("/")
    return p


def _rewrite_prompt_paths(text: str) -> str:
    """Rewrite embedded ScreenshotRelPath/ScreenshotAbsPath values if they use DFS_PREFIX."""
    if not isinstance(text, str) or DFS_PREFIX not in text:
        return text
    return text.replace(DFS_PREFIX, "")


def _iter_image_relpaths(obj: dict) -> Iterable[str]:
    images = obj.get("images")
    if not isinstance(images, list):
        return []
    for round_imgs in images:
        if not isinstance(round_imgs, list):
            continue
        for p in round_imgs:
            if isinstance(p, str) and p:
                yield p


def _candidate_local_sources(
    rel_path: str, *, local_base: Path, input_jsonl_dir: Path
) -> List[Path]:
    """
    Try to find where the file exists locally.
    - If rel_path is absolute and exists, use it as-is.
    - Else try: input_jsonl_dir/rel_path
    - Else try: local_base/rel_path
    """
    p = Path(rel_path)
    out: List[Path] = []
    if p.is_absolute() and p.exists():
        out.append(p)
        return out
    cand1 = (input_jsonl_dir / rel_path).resolve()
    out.append(cand1)
    out.append((local_base / rel_path).resolve())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL (toolcall dataset).")
    ap.add_argument("--output-root", required=True, help="Target data dir root.")
    ap.add_argument(
        "--local-base",
        default="AgentGym2/AgentGym/data",
        help="Local base dir that contains images_vl/ etc (used to locate files to copy).",
    )
    ap.add_argument(
        "--copy-parquet",
        default="",
        help="Optional parquet file path to copy into output-root/parquet/.",
    )
    ap.add_argument(
        "--output-jsonl-name",
        default="train.jsonl",
        help="Output JSONL filename under output-root/ (default: train.jsonl).",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_root = Path(args.output_root)
    local_base = Path(args.local_base)
    out_jsonl = out_root / args.output_jsonl_name
    missing_path = out_root / "missing_images.txt"

    out_root.mkdir(parents=True, exist_ok=True)

    input_dir = in_path.parent.resolve()
    local_base = local_base.resolve()

    missing: List[str] = []
    copied = 0
    total_imgs = 0

    # Stream rewrite JSONL and collect/copy images.
    with in_path.open("r", encoding="utf-8") as fin, out_jsonl.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            obj = json.loads(line)

            # Rewrite images field.
            images = obj.get("images")
            if isinstance(images, list):
                new_images = []
                for round_imgs in images:
                    if not isinstance(round_imgs, list):
                        new_images.append(round_imgs)
                        continue
                    new_round = []
                    for p in round_imgs:
                        if isinstance(p, str):
                            new_round.append(_rewrite_one_path(p))
                        else:
                            new_round.append(p)
                    new_images.append(new_round)
                obj["images"] = new_images

            # Rewrite embedded prompt path hints.
            conv = obj.get("conversations")
            if isinstance(conv, list):
                for m in conv:
                    if isinstance(m, dict) and isinstance(m.get("value"), str):
                        m["value"] = _rewrite_prompt_paths(m["value"])

            # Copy images that exist locally into the output root.
            for rel in _iter_image_relpaths(obj):
                rel = _rewrite_one_path(rel)
                if not rel:
                    continue
                total_imgs += 1
                src = None
                for cand in _candidate_local_sources(rel, local_base=local_base, input_jsonl_dir=input_dir):
                    if cand.exists() and cand.is_file():
                        src = cand
                        break
                if src is None:
                    missing.append(rel)
                    continue
                dst = out_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src, dst)
                    copied += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Copy parquet if requested.
    if args.copy_parquet:
        pq_src = Path(args.copy_parquet)
        if pq_src.exists():
            pq_dst_dir = out_root / "parquet"
            pq_dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pq_src, pq_dst_dir / pq_src.name)

    # Write missing list (unique + stable).
    if missing:
        uniq = sorted(set(missing))
        missing_path.write_text("\n".join(uniq) + "\n", encoding="utf-8")
    else:
        if missing_path.exists():
            missing_path.unlink()

    print(
        json.dumps(
            {
                "input": str(in_path),
                "output_root": str(out_root),
                "output_jsonl": str(out_jsonl),
                "local_base": str(local_base),
                "images_seen": total_imgs,
                "images_copied": copied,
                "missing_images": len(set(missing)),
                "missing_list": str(missing_path) if missing else None,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

