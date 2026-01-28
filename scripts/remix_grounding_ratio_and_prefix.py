#!/usr/bin/env python3
"""
1) Rewrite linux-grounding image path prefixes in a Qwen3-VL JSONL dataset.
2) Downsample linux-grounding items to reach a base:grounding ratio of ~N:1.
3) Mix items by interleaving N base items then 1 grounding item (after shuffling within each group).

This operates on the per-item schema produced by convert_qwen3vl_toolcall_jsonl_to_messages.py:
  {
    "item_id": ...,
    "orig_item_id": ...,
    "round_idx": ...,
    "num_rounds": ...,
    "messages": [...]
  }
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def _is_linux_grounding(obj: Dict[str, Any]) -> bool:
    oid = obj.get("orig_item_id") or obj.get("item_id") or ""
    return isinstance(oid, str) and oid.startswith("linux_grounding_")


def _rewrite_prefix_in_messages(
    obj: Dict[str, Any], *, old_prefix: str, new_prefix: str
) -> None:
    msgs = obj.get("messages")
    if not isinstance(msgs, list):
        return
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "image":
                continue
            p = part.get("image")
            if not isinstance(p, str):
                continue
            if old_prefix and p.startswith(old_prefix):
                part["image"] = new_prefix + p[len(old_prefix) :]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL path.")
    ap.add_argument("--output", required=True, help="Output JSONL path.")
    ap.add_argument(
        "--base-per-grounding",
        type=int,
        default=5,
        help="Target ratio: base:grounding ~= N:1 (default: 5).",
    )
    ap.add_argument("--seed", type=int, default=42, help="Shuffle/sample seed.")
    ap.add_argument(
        "--old-prefix",
        required=True,
        help="Old image path prefix to replace (exact prefix match).",
    )
    ap.add_argument(
        "--new-prefix",
        required=True,
        help="New image path prefix to use.",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = int(args.base_per_grounding)
    if n <= 0:
        raise SystemExit("--base-per-grounding must be > 0")

    base_items: List[Dict[str, Any]] = []
    grounding_items: List[Dict[str, Any]] = []

    # Load + rewrite prefix on grounding items.
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            if _is_linux_grounding(obj):
                _rewrite_prefix_in_messages(
                    obj, old_prefix=str(args.old_prefix), new_prefix=str(args.new_prefix)
                )
                grounding_items.append(obj)
            else:
                base_items.append(obj)

    base_count = len(base_items)
    if base_count <= 0:
        raise RuntimeError("No base items found (non-linux_grounding_).")

    target_grounding = base_count // n
    target_grounding = min(target_grounding, len(grounding_items))

    rng = random.Random(int(args.seed))
    rng.shuffle(base_items)
    rng.shuffle(grounding_items)
    grounding_keep = (
        rng.sample(grounding_items, k=target_grounding) if target_grounding else []
    )
    rng.shuffle(grounding_keep)

    # Interleave: N base, then 1 grounding.
    out_items: List[Dict[str, Any]] = []
    bi = 0
    for g in grounding_keep:
        out_items.extend(base_items[bi : bi + n])
        bi += n
        out_items.append(g)
    out_items.extend(base_items[bi:])

    with out_path.open("w", encoding="utf-8") as f:
        for obj in out_items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input": str(in_path),
                "output": str(out_path),
                "base_items": base_count,
                "grounding_items_in": len(grounding_items),
                "grounding_items_out": len(grounding_keep),
                "target_ratio_base_per_grounding": n,
                "actual_ratio_base_per_grounding": (
                    round(base_count / len(grounding_keep), 4) if grounding_keep else None
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

