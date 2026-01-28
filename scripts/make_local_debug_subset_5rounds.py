#!/usr/bin/env python3
"""
Extract one 5-round (round_idx=5) multi-turn sample from the training JSONL and
rewrite image paths from /dfs/... to local /data/pengchen/OSWorld/Osworld/... paths.

Also performs a lightweight local debug:
  - checks image paths exist
  - opens images with PIL and prints their sizes
  - prints a short message outline
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


DFS_VL_PREFIX = "/dfs/data/AgentGym_qwen/data/images_vl"
LOCAL_VL_PREFIX = "/data/pengchen/OSWorld/Osworld/AgentGym2/AgentGym/data/images_vl"

DFS_LINUX_PREFIX = "/dfs/data/AgentGym_qwen/data/linux_images_1920x1080"
LOCAL_LINUX_PREFIX = (
    "/data/pengchen/OSWorld/Osworld/AgentGym2/AgentGym/data/linux_images_1920x1080"
)


def _map_image_path(p: str) -> str:
    if p.startswith(DFS_VL_PREFIX):
        return LOCAL_VL_PREFIX + p[len(DFS_VL_PREFIX) :]
    if p.startswith(DFS_LINUX_PREFIX):
        return LOCAL_LINUX_PREFIX + p[len(DFS_LINUX_PREFIX) :]
    return p


def _rewrite_paths_in_item(obj: Dict[str, Any]) -> Dict[str, Any]:
    o = copy.deepcopy(obj)
    for m in o.get("messages", []) or []:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image":
                continue
            p = part.get("image")
            if isinstance(p, str) and p:
                part["image"] = _map_image_path(p)
    return o


def _collect_user_images(obj: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for m in obj.get("messages", []) or []:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        for part in m.get("content", []) or []:
            if isinstance(part, dict) and part.get("type") == "image":
                p = part.get("image")
                if isinstance(p, str) and p:
                    out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL dataset (messages format).")
    ap.add_argument("--output", required=True, help="Output JSONL debug subset.")
    ap.add_argument("--round-idx", type=int, default=5, help="Pick sample with this round_idx (default: 5).")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    picked = None
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            oid = obj.get("orig_item_id") or ""
            if isinstance(oid, str) and oid.startswith("linux_grounding_"):
                continue
            if int(obj.get("round_idx") or 0) != int(args.round_idx):
                continue
            picked = obj
            break

    if picked is None:
        raise SystemExit(f"Failed to find a non-grounding item with round_idx={args.round_idx}")

    out_obj = _rewrite_paths_in_item(picked)

    # Local debug: check/open images.
    imgs = _collect_user_images(out_obj)
    missing = [p for p in imgs if not Path(p).exists()]

    sizes: List[Tuple[str, Tuple[int, int]]] = []
    try:
        from PIL import Image

        for p in imgs:
            try:
                with Image.open(p) as im:
                    sizes.append((p, im.size))
            except Exception:
                sizes.append((p, (-1, -1)))
    except Exception:
        pass

    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    # Print a compact report.
    msg_len = len(out_obj.get("messages", []) or [])
    print(
        json.dumps(
            {
                "picked_item_id": out_obj.get("item_id"),
                "orig_item_id": out_obj.get("orig_item_id"),
                "round_idx": out_obj.get("round_idx"),
                "num_rounds": out_obj.get("num_rounds"),
                "messages_len": msg_len,
                "user_images": len(imgs),
                "missing_images": len(missing),
                "output": str(out_path),
                "example_image_size": sizes[0] if sizes else None,
            },
            ensure_ascii=False,
        )
    )
    if missing:
        print("missing_paths:")
        for p in missing[:20]:
            print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

