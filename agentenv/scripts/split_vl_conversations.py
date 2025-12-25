#!/usr/bin/env python3
"""
Split multi-round VL conversations into per-round prefix samples and separate images from text.

Input (JSON): list[{"item_id": str, "conversations": list[{"from": "human"|"gpt", "value": str, ...}], ...}]

Output:
  - merged JSONL: each line is a sample containing the first k rounds (k=1..N)
    plus `images` (per-round image path arrays)
  - image files: decoded from screenshot_base64, optionally downsampled

Rules (per prefix sample k):
  - Keep text for rounds 1..k
  - Images are saved only once per original item per round (no duplication).
    The JSONL stores references (paths) so downstream code can decide how many
    previous-round images to include and how to downsample them.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Pillow is required for image decoding. Install with `pip install pillow`."
    ) from e


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _loads_maybe_json(s: Any) -> Optional[dict]:
    if not isinstance(s, str):
        return None
    t = s.strip()
    if not t.startswith("{"):
        return None
    try:
        obj = json.loads(t)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _strip_data_url_prefix(b64_str: str) -> str:
    t = b64_str.strip()
    if t.startswith("data:") and "," in t:
        return t.split(",", 1)[-1]
    return t


def _decode_image_from_base64(b64_str: str) -> Image.Image:
    raw = base64.b64decode(_strip_data_url_prefix(b64_str))
    img = Image.open(BytesIO(raw))
    return img.convert("RGB")


def _merge_text_from_inner(inner: dict) -> str:
    instr = inner.get("instruction") or ""
    step_text = inner.get("step_text") or ""
    merged = (str(instr).strip() + "\n" + str(step_text).strip()).strip()
    return merged


@dataclass
class ParsedMessage:
    role_from: str
    text: str
    screenshot_b64: Optional[str]
    extra: Dict[str, Any]


def _parse_message(msg: Dict[str, Any]) -> ParsedMessage:
    role_from = str(msg.get("from", "")).strip()
    raw_val = msg.get("value", "")
    inner = _loads_maybe_json(raw_val)

    screenshot_b64 = None
    text = raw_val if isinstance(raw_val, str) else str(raw_val)
    if inner is not None:
        screenshot_b64 = inner.get("screenshot_base64")
        merged = _merge_text_from_inner(inner)
        if merged:
            text = merged

    extra = {k: v for k, v in msg.items() if k not in ("from", "value")}
    return ParsedMessage(role_from=role_from, text=text, screenshot_b64=screenshot_b64, extra=extra)


def _pair_rounds(conversations: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    if len(conversations) % 2 != 0:
        raise ValueError(f"Expected even number of messages (human/gpt pairs), got {len(conversations)}")
    pairs = []
    for i in range(0, len(conversations), 2):
        pairs.append((conversations[i], conversations[i + 1]))
    return pairs


def _write_jsonl(path: str, rows: List[dict]) -> None:
    _safe_makedirs(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to processed_conversations_vl_train.json (list JSON)")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL path (merged: text + images paths)")
    ap.add_argument("--out_image_dir", required=True, help="Directory to write decoded image files")
    ap.add_argument("--max_rounds", type=int, default=15, help="Max rounds (pairs) to keep per conversation")
    ap.add_argument("--image_ext", default="png", choices=["png", "jpg", "jpeg"], help="Image file extension")
    ap.add_argument("--limit_items", type=int, default=0, help="For debugging: only process first N items (0=all)")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of items")
    if args.limit_items and args.limit_items > 0:
        data = data[: args.limit_items]

    _safe_makedirs(args.out_image_dir)

    merged_rows: List[dict] = []

    for item_idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        orig_item_id = str(item.get("item_id", f"item_{item_idx}"))
        conversations = item.get("conversations", [])
        if not isinstance(conversations, list) or not conversations:
            continue

        pairs = _pair_rounds(conversations)
        total_rounds = len(pairs)
        keep_rounds = min(total_rounds, int(args.max_rounds))

        # Pre-parse all rounds once
        parsed_rounds: List[Tuple[ParsedMessage, ParsedMessage]] = []
        for (m1, m2) in pairs[:keep_rounds]:
            parsed_rounds.append((_parse_message(m1), _parse_message(m2)))

        # Save images once per original item, per round (no duplication across prefix samples)
        saved_paths_by_round: List[List[str]] = [[] for _ in range(keep_rounds)]
        orig_dir = os.path.join(args.out_image_dir, orig_item_id)
        _safe_makedirs(orig_dir)
        for r1 in range(1, keep_rounds + 1):  # 1-based round index
            round_idx0 = r1 - 1
            pm1, pm2 = parsed_rounds[round_idx0]
            paths: List[str] = []
            for msg_i, pm in enumerate((pm1, pm2)):
                if not pm.screenshot_b64:
                    continue
                try:
                    img = _decode_image_from_base64(str(pm.screenshot_b64))
                except Exception:
                    continue
                fname = f"round{r1:03d}_msg{msg_i}.{args.image_ext}"
                fpath = os.path.join(orig_dir, fname)
                try:
                    if args.image_ext in ("jpg", "jpeg"):
                        img.save(fpath, format="JPEG", quality=95)
                    else:
                        img.save(fpath, format="PNG")
                    paths.append(
                        os.path.relpath(
                            fpath,
                            start=os.path.dirname(args.out_jsonl) or ".",
                        )
                    )
                except Exception:
                    continue
            saved_paths_by_round[round_idx0] = paths

        for k in range(1, keep_rounds + 1):
            new_item_id = f"{orig_item_id}__r{k:03d}"

            # Text: keep first k rounds, strip screenshot_base64 from values
            out_convs: List[dict] = []
            for r in range(k):
                pm1, pm2 = parsed_rounds[r]
                out_convs.append({"from": pm1.role_from, "value": pm1.text, **pm1.extra})
                out_convs.append({"from": pm2.role_from, "value": pm2.text, **pm2.extra})

            # Images: arrays per round in prefix; references to per-round images saved once above.
            images_per_round: List[List[str]] = []
            for r in range(k):
                images_per_round.append(saved_paths_by_round[r])

            merged_rows.append(
                {
                    "item_id": new_item_id,
                    "orig_item_id": orig_item_id,
                    "round_idx": k,
                    "num_rounds": keep_rounds,
                    "conversations": out_convs,
                    "images": images_per_round,
                }
            )

    _write_jsonl(args.out_jsonl, merged_rows)
    print(
        f"wrote rows={len(merged_rows)} to {args.out_jsonl}\n"
        f"images in {args.out_image_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
