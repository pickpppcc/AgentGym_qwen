#!/usr/bin/env python3
"""
Augment a Qwen3VL tool-call JSONL training set with Linux grounding samples from
`linux_splited.json` (image + elements with normalized bbox).

This script:
- Filters out elements whose normalized bbox has any coord outside [0, 1].
- Converts bbox -> a click coordinate in Qwen3-VL "screen space" (0..999) by taking
  the bbox center and scaling by 1000, then clipping to [0, 999].
- Appends the grounding items to the base dataset so that grounding is ~target_ratio
  of the *final* dataset size.

Output schema matches the existing tool-call training set:
  {"item_id","orig_item_id","round_idx","num_rounds","conversations","images"}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _count_jsonl(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _load_human_prefix(base_jsonl: Path) -> str:
    """Reuse the tool prompt prefix from the base dataset (everything before 'Instruction:')."""
    for obj in _iter_jsonl(base_jsonl):
        conv = obj.get("conversations") or []
        if not conv:
            continue
        human = conv[0].get("value", "")
        if not isinstance(human, str) or "Instruction:" not in human:
            continue
        return human.split("Instruction:", 1)[0]
    raise RuntimeError(f"Failed to find a usable human prompt prefix in {base_jsonl}")


def _bbox_is_normalized(b: List[float]) -> bool:
    if not isinstance(b, list) or len(b) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in b]
    except Exception:
        return False
    if any(math.isnan(v) or math.isinf(v) for v in (x1, y1, x2, y2)):
        return False
    if any(v < 0.0 or v > 1.0 for v in (x1, y1, x2, y2)):
        return False
    # Keep degenerate-ish boxes out.
    if x2 <= x1 or y2 <= y1:
        return False
    return True


def _norm_to_qwen_coord(v: float) -> int:
    # Qwen3VL relative coordinate convention in this repo is 0..999.
    # Scale by 1000 (so 1.0 becomes 1000), then clip to 999.
    x = int(round(float(v) * 1000.0))
    return max(0, min(999, x))


def _bbox_center_xy_999(b: List[float]) -> Tuple[int, int]:
    x1, y1, x2, y2 = [float(v) for v in b]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return _norm_to_qwen_coord(cx), _norm_to_qwen_coord(cy)


def _build_grounding_item(
    *,
    item_id: str,
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
        "<think>\n"
        "I should click the target UI element.\n"
        "</think>\n\n"
        "Action: Left click.\n"
        "<tool_call>\n"
        + json.dumps(
            {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [x, y]}},
            ensure_ascii=False,
        )
        + "\n</tool_call>"
    )

    return {
        "item_id": f"{item_id}__r001",
        "orig_item_id": item_id,
        "round_idx": 1,
        "num_rounds": 1,
        "conversations": [
            {"from": "human", "value": human_val, "loss": None},
            {"from": "gpt", "value": gpt_val, "loss": True},
        ],
        "images": [[image_path]],
    }


def _maybe_join_image_path(image_root: str, img_filename: str) -> str:
    if not image_root:
        return img_filename
    return str(Path(image_root) / img_filename)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base Qwen3VL toolcall JSONL (*.fixed.jsonl).")
    ap.add_argument("--linux-grounding", required=True, help="Path to linux_splited.json.")
    ap.add_argument("--output", required=True, help="Output augmented JSONL.")
    ap.add_argument(
        "--target-ratio",
        type=float,
        default=0.20,
        help="Target grounding fraction in the final dataset (default: 0.20).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling grounding elements.",
    )
    ap.add_argument(
        "--linux-image-root",
        type=str,
        default="",
        help="Root dir containing the screenshots referenced by img_filename (e.g. .../linux_images_1920x1080).",
    )
    ap.add_argument(
        "--out-image-root",
        type=str,
        default="",
        help="If set, write image paths under this root and copy used images there.",
    )
    ap.add_argument(
        "--copy-images",
        action="store_true",
        help="If set, copy sampled images to --out-image-root (requires --out-image-root).",
    )
    args = ap.parse_args()

    base_path = Path(args.base)
    linux_path = Path(args.linux_grounding)
    out_path = Path(args.output)

    if args.copy_images and not args.out_image_root:
        raise SystemExit("--copy-images requires --out-image-root")

    base_count = _count_jsonl(base_path)
    if base_count <= 0:
        raise RuntimeError(f"Base dataset appears empty: {base_path}")

    target_ratio = float(args.target_ratio)
    if not (0.0 <= target_ratio < 1.0):
        raise SystemExit("--target-ratio must be in [0, 1)")

    # Want: add / (base + add) ~= target_ratio  =>  add ~= base * target_ratio / (1-target_ratio)
    add_count = int(round(base_count * target_ratio / (1.0 - target_ratio)))

    human_prefix = _load_human_prefix(base_path)

    with linux_path.open("r", encoding="utf-8") as f:
        linux = json.load(f)
    if not isinstance(linux, list):
        raise RuntimeError(f"Unexpected linux grounding JSON shape: expected list, got {type(linux)}")

    # Flatten eligible elements.
    eligible: List[Tuple[str, str, List[float], str]] = []
    # (img_filename, instruction, bbox, data_type)
    filtered_bbox = 0
    filtered_missing = 0
    for ex in linux:
        img_fn = ex.get("img_filename")
        if not isinstance(img_fn, str) or not img_fn:
            continue
        for el in ex.get("elements", []) or []:
            if not isinstance(el, dict):
                continue
            bbox = el.get("bbox")
            if not _bbox_is_normalized(bbox):
                filtered_bbox += 1
                continue
            instr = (el.get("instruction") or "").strip()
            if not instr:
                continue
            data_type = (el.get("data_type") or "").strip()

            # Check image exists if a root is given; otherwise just keep.
            if args.linux_image_root:
                src = Path(args.linux_image_root) / img_fn
                if not src.exists():
                    filtered_missing += 1
                    continue

            eligible.append((img_fn, instr, bbox, data_type))

    if not eligible:
        raise RuntimeError("No eligible grounding elements after filtering.")

    if add_count <= 0:
        add_count = 0

    add_count = min(add_count, len(eligible))

    rng = random.Random(int(args.seed))
    sampled = rng.sample(eligible, k=add_count) if add_count else []

    # Ensure output dir exists.
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy images if requested.
    if args.copy_images and sampled:
        dst_root = Path(args.out_image_root)
        dst_root.mkdir(parents=True, exist_ok=True)
        src_root = Path(args.linux_image_root) if args.linux_image_root else None
        if src_root is None:
            raise SystemExit("--copy-images requires --linux-image-root")
        seen = set()
        for img_fn, _, _, _ in sampled:
            if img_fn in seen:
                continue
            seen.add(img_fn)
            src = src_root / img_fn
            dst = dst_root / img_fn
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)

    # Write: base dataset verbatim, then append grounding items.
    added = 0
    with base_path.open("r", encoding="utf-8") as fbase, out_path.open("w", encoding="utf-8") as fout:
        for line in fbase:
            fout.write(line.rstrip("\n") + "\n")

        for i, (img_fn, instr, bbox, data_type) in enumerate(sampled):
            x, y = _bbox_center_xy_999(bbox)
            # Keep the original instruction text (closest to the source grounding labels).
            final_instr = instr

            if args.out_image_root:
                image_path = _maybe_join_image_path(args.out_image_root, img_fn)
            else:
                image_path = _maybe_join_image_path(args.linux_image_root, img_fn)

            item_id = f"linux_grounding_{i:07d}"
            obj = _build_grounding_item(
                item_id=item_id,
                human_prefix=human_prefix,
                instruction=final_instr,
                image_path=image_path,
                x=x,
                y=y,
            )
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            added += 1

    print(
        json.dumps(
            {
                "base_items": base_count,
                "eligible_elements": len(eligible),
                "filtered_bbox_or_invalid": filtered_bbox,
                "filtered_missing_image": filtered_missing,
                "added_grounding_items": added,
                "target_ratio": target_ratio,
                "output": str(out_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
