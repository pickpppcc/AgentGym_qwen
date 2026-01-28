#!/usr/bin/python3
"""
Augment a Qwen3VL tool-call JSONL training set with grounding samples stored in a parquet file.

We cap the number of added grounding items to <= ratio * (num_base_items).

Parquet schema expected (as in AgentGym2/AgentGym/data/0000.parquet):
  - image: str (path)
  - bbox: list[int] (optional; not used)
  - conversations: list[{"from": "human"|"gpt", "value": str}]
    with gpt value like "(x, y)".

Output items follow the existing training set schema:
  {"item_id","orig_item_id","round_idx","num_rounds","conversations","images"}

Notes:
- This script uses system Python (/usr/bin/python3) with pyarrow installed.
- Image paths from parquet are kept as-is; if they are relative, training will resolve them
  relative to the train_file directory (see QwenVLBCTrainer).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow.parquet as pq


COORD_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")


def _load_human_prefix(base_jsonl: Path) -> str:
    """Reuse the tool prompt prefix from the base dataset (everything before 'Instruction:')."""
    with base_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            conv = obj.get("conversations") or []
            if not conv:
                continue
            human = conv[0].get("value", "")
            if not isinstance(human, str) or "Instruction:" not in human:
                continue
            prefix = human.split("Instruction:", 1)[0]
            return prefix
    raise RuntimeError(f"Failed to find a usable human prompt prefix in {base_jsonl}")


def _parse_xy(text: str) -> Optional[Tuple[int, int]]:
    if not isinstance(text, str):
        return None
    nums = COORD_RE.findall(text)
    if len(nums) < 2:
        return None
    try:
        x = int(round(float(nums[0])))
        y = int(round(float(nums[1])))
    except Exception:
        return None
    # Qwen3VL relative coordinate convention is 0..999.
    x = max(0, min(999, x))
    y = max(0, min(999, y))
    return x, y


def _clean_instruction(text: str) -> str:
    t = (text or "").strip()
    if t.lower().startswith("<image>"):
        t = t[len("<image>") :].strip()
    return t


def _build_grounding_item(
    *,
    row_idx: int,
    human_prefix: str,
    instruction: str,
    image_path: str,
    x: int,
    y: int,
) -> Dict:
    human_val = (
        human_prefix
        + "Instruction: [SEP] "
        + instruction
        + " [SEP] Step: 1/1 [SEP] ScreenshotRelPath: "
        + image_path
        + " [SEP] ScreenshotAbsPath: "
        + image_path
    )

    gpt_val = (
        "Action: Left click.\n"
        "<tool_call>\n"
        + json.dumps(
            {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [x, y]}},
            ensure_ascii=False,
        )
        + "\n</tool_call>"
    )

    return {
        "item_id": f"grounding_0000_{row_idx:07d}",
        "orig_item_id": f"grounding_0000_{row_idx:07d}",
        "round_idx": 1,
        "num_rounds": 1,
        "conversations": [
            {"from": "human", "value": human_val, "loss": None},
            {"from": "gpt", "value": gpt_val, "loss": True},
        ],
        # QwenVLBCTrainer expects per-round image path arrays.
        "images": [[image_path]],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        required=True,
        help="Base Qwen3VL toolcall JSONL (recommended: *.fixed.jsonl).",
    )
    ap.add_argument("--parquet", required=True, help="Grounding parquet file path.")
    ap.add_argument("--output", required=True, help="Output augmented JSONL.")
    ap.add_argument(
        "--ratio",
        type=float,
        default=0.30,
        help="Max grounding items to add = floor(ratio * num_base_items).",
    )
    ap.add_argument(
        "--max-grounding",
        type=int,
        default=0,
        help="Optional hard cap on added grounding items (0 = no extra cap).",
    )
    ap.add_argument(
        "--image-prefix",
        type=str,
        default="",
        help="If set and parquet image path is relative, prefix it (e.g. /dfs/data/AgentGym_qwen/data).",
    )
    args = ap.parse_args()

    base_path = Path(args.base)
    parquet_path = Path(args.parquet)
    out_path = Path(args.output)

    # Count base items
    base_count = 0
    with base_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                base_count += 1

    if base_count <= 0:
        raise RuntimeError(f"Base dataset appears empty: {base_path}")

    ratio = float(args.ratio)
    if ratio < 0:
        ratio = 0.0

    max_add = int(math.floor(base_count * ratio))
    if args.max_grounding and args.max_grounding > 0:
        max_add = min(max_add, int(args.max_grounding))

    human_prefix = _load_human_prefix(base_path)

    # Stream parquet and pick first max_add valid rows (fast and deterministic).
    pf = pq.ParquetFile(str(parquet_path))
    added = 0
    total_seen = 0

    # Prepare output dir
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with base_path.open("r", encoding="utf-8") as fbase, out_path.open("w", encoding="utf-8") as fout:
        # 1) Copy base dataset verbatim
        for line in fbase:
            fout.write(line.rstrip("\n") + "\n")

        # 2) Append grounding items
        if max_add > 0:
            for batch in pf.iter_batches(batch_size=4096, columns=["image", "conversations"]):
                for row in batch.to_pylist():
                    total_seen += 1
                    if added >= max_add:
                        break

                    image_path = row.get("image")
                    conv = row.get("conversations") or []
                    if not isinstance(image_path, str) or not isinstance(conv, list) or len(conv) < 2:
                        continue

                    # Find first human instruction and gpt coordinate
                    human_msg = None
                    gpt_msg = None
                    for m in conv:
                        if not isinstance(m, dict):
                            continue
                        if human_msg is None and m.get("from") in {"human", "user"}:
                            human_msg = m.get("value")
                        if gpt_msg is None and m.get("from") in {"gpt", "assistant"}:
                            gpt_msg = m.get("value")
                    if not isinstance(human_msg, str) or not isinstance(gpt_msg, str):
                        continue

                    xy = _parse_xy(gpt_msg)
                    if xy is None:
                        continue
                    x, y = xy

                    inst = _clean_instruction(human_msg)
                    if not inst:
                        continue

                    # Prefix relative image paths if requested.
                    if args.image_prefix and not os.path.isabs(image_path):
                        pref = str(args.image_prefix).rstrip("/")
                        image_path_use = pref + "/" + image_path.lstrip("/")
                    else:
                        image_path_use = image_path

                    item = _build_grounding_item(
                        row_idx=total_seen,
                        human_prefix=human_prefix,
                        instruction=inst,
                        image_path=image_path_use,
                        x=x,
                        y=y,
                    )
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    added += 1

                if added >= max_add:
                    break

    print(
        json.dumps(
            {
                "base_items": base_count,
                "ratio": ratio,
                "max_add": max_add,
                "grounding_added": added,
                "parquet_rows_scanned": total_seen,
                "output": str(out_path),
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

