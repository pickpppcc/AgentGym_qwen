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
import re
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

def _save_img(img: Image.Image, fpath: str, ext: str) -> bool:
    try:
        if ext in ("jpg", "jpeg"):
            img.save(fpath, format="JPEG", quality=95)
        else:
            img.save(fpath, format="PNG")
        return True
    except Exception:
        return False


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

_ACTION_FUNCS = ("click", "left_double", "right_single", "drag", "hotkey", "type", "scroll", "wait", "finished", "call_user")


def _normalize_whitespace(s: str) -> str:
    s = (s or "").strip()
    # collapse all whitespace (including newlines) to single spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _extract_first_action_call(text: str) -> Optional[str]:
    """
    Extract a single action call from messy model text.
    Accepts optional 'pyautogui.' prefix and ignores trailing explanations.
    """
    if not text:
        return None
    s = text.strip()

    # Prefer content after the last "Action:" marker if present.
    if "Action:" in s:
        s = s.split("Action:")[-1].strip()

    # Find first known call.
    m = re.search(
        r"(?:pyautogui\.)?(?:"
        + "|".join(re.escape(fn) for fn in _ACTION_FUNCS)
        + r")\s*\(",
        s,
    )
    if not m:
        # Handle bare terminal actions like "finished" / "wait" without parentheses (rare)
        m2 = re.search(r"(?:pyautogui\.)?(finished|wait|call_user)\b", s)
        if m2:
            return f"{m2.group(1)}()"
        return None

    start = m.start()
    open_idx = s.find("(", start)
    if open_idx < 0:
        return s[start:].strip()

    depth = 0
    in_single = False
    in_double = False
    escape = False
    for i, ch in enumerate(s[open_idx:], start=open_idx):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if in_single:
            if ch == "'":
                in_single = False
            continue
        if in_double:
            if ch == '"':
                in_double = False
            continue
        if ch == "'":
            in_single = True
            continue
        if ch == '"':
            in_double = True
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return s[start : i + 1].strip()
    return s[start:].strip()


def _escape_single_quotes(s: str) -> str:
    # Escape unescaped single quotes for single-quoted python strings.
    return re.sub(r"(?<!\\)'", r"\\'", s)


def _normalize_action_call(call: str) -> Optional[str]:
    """
    Normalize to OSWorld-friendly UITARS-style calls:
      - no 'pyautogui.' prefix
      - hotkey uses keyword 'key' with space-separated keys: hotkey(key='ctrl z')
      - finished/wait/call_user have no args
      - type uses keyword 'content' and single quotes (escaping inner quotes)
    """
    if not call:
        return None
    s = call.strip()
    if s.startswith("pyautogui."):
        s = s[len("pyautogui.") :].lstrip()

    # Normalize terminal actions that sometimes include extra args
    for name in ("finished", "wait", "call_user"):
        if s.startswith(name):
            return f"{name}()"

    m = re.match(r"^([a-zA-Z_]+)\s*\((.*)\)\s*$", s, flags=re.DOTALL)
    if not m:
        return None
    func = m.group(1).strip()
    args = (m.group(2) or "").strip()
    if func not in _ACTION_FUNCS:
        return None

    if func == "hotkey":
        # Prefer existing key=... if present.
        km = re.search(r"key\s*=\s*(['\"])(.*?)\1", args, flags=re.DOTALL)
        if km:
            key = km.group(2).strip()
            key = _normalize_whitespace(key)
            key = key.replace("+", " ")
            return f"hotkey(key='{_escape_single_quotes(key)}')"

        # Positional: hotkey('ctrl','z') / hotkey(\"ctrl\", \"z\")
        keys = re.findall(r"(['\"])(.*?)\1", args, flags=re.DOTALL)
        key_parts = [_normalize_whitespace(v) for _, v in keys if _normalize_whitespace(v)]
        if not key_parts:
            return "hotkey(key='')"
        key = " ".join(key_parts).replace("+", " ")
        return f"hotkey(key='{_escape_single_quotes(key)}')"

    if func == "type":
        cm = re.search(r"content\s*=\s*(['\"])(.*?)\1", args, flags=re.DOTALL)
        if cm:
            content = cm.group(2)
        else:
            # positional: type('hello')
            m1 = re.match(r"^(['\"])(.*?)\1\s*$", args, flags=re.DOTALL)
            content = m1.group(2) if m1 else args
        return f"type(content='{_escape_single_quotes(content)}')"

    # For other calls, keep only known keywords to reduce noise.
    def _kw(name: str) -> Optional[str]:
        km = re.search(rf"\b{name}\s*=\s*(['\"])(.*?)\1", args, flags=re.DOTALL)
        return km.group(2) if km else None

    if func in ("click", "left_double", "right_single", "scroll"):
        start_box = _kw("start_box")
        if start_box is None:
            # Fallback: pyautogui.click(x, y, button='right') / click(850, 298, ...)
            xm = re.search(r"\bx\s*=\s*(-?\d+(?:\.\d+)?)", args)
            ym = re.search(r"\by\s*=\s*(-?\d+(?:\.\d+)?)", args)
            if xm and ym:
                x = xm.group(1)
                y = ym.group(1)
            else:
                mxy = re.match(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*(?:,|$)", args)
                if not mxy:
                    return None
                x, y = mxy.group(1), mxy.group(2)

            # button/clicks mapping (pyautogui-compatible -> UITARS)
            button = "left"
            bm = re.search(r"\bbutton\s*=\s*(['\"])(.*?)\1", args, flags=re.DOTALL)
            if bm:
                button = _normalize_whitespace(bm.group(2)).lower()
            clicks = 1
            cm = re.search(r"\bclicks\s*=\s*(\d+)", args)
            if cm:
                try:
                    clicks = int(cm.group(1))
                except Exception:
                    clicks = 1

            mapped_func = func
            if button == "right":
                mapped_func = "right_single"
            elif clicks >= 2:
                mapped_func = "left_double"
            elif func == "right_single":
                mapped_func = "right_single"
            elif func == "left_double":
                mapped_func = "left_double"
            else:
                mapped_func = "click"

            start_box = f"({x},{y})"
            if func == "scroll":
                direction = _kw("direction") or ""
                direction = _normalize_whitespace(direction)
                return f"scroll(start_box='{_escape_single_quotes(start_box)}', direction='{_escape_single_quotes(direction)}')"
            return f"{mapped_func}(start_box='{_escape_single_quotes(start_box)}')"

        start_box = _normalize_whitespace(start_box)
        if func == "scroll":
            direction = _kw("direction") or ""
            direction = _normalize_whitespace(direction)
            return f"scroll(start_box='{_escape_single_quotes(start_box)}', direction='{_escape_single_quotes(direction)}')"
        return f"{func}(start_box='{_escape_single_quotes(start_box)}')"

    if func == "drag":
        start_box = _kw("start_box")
        end_box = _kw("end_box")
        if start_box is None or end_box is None:
            return None
        start_box = _normalize_whitespace(start_box)
        end_box = _normalize_whitespace(end_box)
        return f"drag(start_box='{_escape_single_quotes(start_box)}', end_box='{_escape_single_quotes(end_box)}')"

    return None


def _normalize_assistant_value(text: str) -> Optional[str]:
    """
    Convert arbitrary assistant text into exactly 2 lines:
      Thought: ...
      Action: <call>
    """
    if not text:
        return None
    t = text.strip()

    # Thought: capture between Thought: and Action: if present.
    thought = ""
    tm = re.search(r"Thought:\s*(.+?)(?=\n\s*Action:|$)", t, flags=re.DOTALL)
    if tm:
        thought = _normalize_whitespace(tm.group(1))
    else:
        # Fallback: use first line as thought if it doesn't look like an action
        first = t.splitlines()[0].strip()
        if first and "Action:" not in first:
            thought = _normalize_whitespace(first)

    raw_call = _extract_first_action_call(t)
    call = _normalize_action_call(raw_call or "")
    if not call:
        return None

    if not thought:
        thought = "..."
    return f"Thought: {thought}\nAction: {call}"


def _maybe_extract_preamble(conversations: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Detect the common preamble pattern:
      human: (format + action space)
      gpt:   Ok.   (loss=false)
    Return (preamble_text, remaining_conversations).
    """
    if len(conversations) < 4:
        return None, conversations
    h0 = conversations[0]
    a0 = conversations[1]
    if str(h0.get("from", "")).strip() != "human":
        return None, conversations
    if str(a0.get("from", "")).strip() != "gpt":
        return None, conversations

    h0_text = str(h0.get("value", "") or "")
    a0_text = str(a0.get("value", "") or "").strip().lower()
    a0_loss = a0.get("loss", None)

    looks_like_prefix = ("Output Format" in h0_text) or ("Action Space" in h0_text) or ("You are a GUI agent" in h0_text)
    looks_like_ok = a0_text in {"ok.", "ok", "okay"} and (a0_loss is False)
    if not (looks_like_prefix and looks_like_ok):
        return None, conversations

    return h0_text, conversations[2:]


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
    # Defaults are "on" so users don't need to pass extra flags for strict UITARS format.
    ap.set_defaults(normalize_actions=True, drop_unparseable=True)
    ap.add_argument(
        "--normalize_actions",
        action="store_true",
        help="Normalize assistant outputs to strict Thought/Action + UITARS action calls (default: on)",
    )
    ap.add_argument(
        "--no_normalize_actions",
        action="store_false",
        dest="normalize_actions",
        help="Disable assistant action normalization",
    )
    ap.add_argument(
        "--drop_unparseable",
        action="store_true",
        help="Drop prefix samples that contain unparseable assistant actions (default: on)",
    )
    ap.add_argument(
        "--no_drop_unparseable",
        action="store_false",
        dest="drop_unparseable",
        help="Keep prefix samples even if assistant actions are unparseable",
    )
    ap.add_argument(
        "--save_assistant_images",
        action="store_true",
        default=False,
        help="Also save assistant (msg1) screenshots (default: only msg0)",
    )
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of items")
    if args.limit_items and args.limit_items > 0:
        data = data[: args.limit_items]

    _safe_makedirs(args.out_image_dir)

    merged_rows: List[dict] = []
    stats = {
        "items_total": 0,
        "items_used": 0,
        "prefix_rows_written": 0,
        "prefix_rows_dropped": 0,
        "assistant_msgs_normalized": 0,
        "assistant_msgs_failed": 0,
        "preamble_stripped": 0,
    }

    for item_idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        stats["items_total"] += 1
        orig_item_id = str(item.get("item_id", f"item_{item_idx}"))
        conversations = item.get("conversations", [])
        if not isinstance(conversations, list) or not conversations:
            continue

        preamble, conversations_wo_preamble = _maybe_extract_preamble(conversations)
        if preamble:
            stats["preamble_stripped"] += 1
            conversations = conversations_wo_preamble

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
                if msg_i == 1 and not args.save_assistant_images:
                    continue
                if not pm.screenshot_b64:
                    continue
                try:
                    img = _decode_image_from_base64(str(pm.screenshot_b64))
                except Exception:
                    continue
                fname = f"round{r1:03d}_msg{msg_i}.{args.image_ext}"
                fpath = os.path.join(orig_dir, fname)
                try:
                    if _save_img(img, fpath, args.image_ext):
                        rel = os.path.relpath(
                            fpath,
                            start=os.path.dirname(args.out_jsonl) or ".",
                        )
                        paths.append(rel)
                except Exception:
                    continue
            saved_paths_by_round[round_idx0] = paths

        for k in range(1, keep_rounds + 1):
            new_item_id = f"{orig_item_id}__r{k:03d}"

            # Text: keep first k rounds, strip screenshot_base64 from values
            out_convs: List[dict] = []
            bad_prefix = False
            for r in range(k):
                pm1, pm2 = parsed_rounds[r]

                human_text = pm1.text
                # Merge preamble into the first real user message to keep alternation without training on the "Ok." turn.
                if preamble and r == 0:
                    human_text = (str(preamble).rstrip() + "\n\n" + str(human_text).lstrip()).strip()

                out_convs.append({"from": pm1.role_from, "value": human_text, **pm1.extra})

                assistant_text = pm2.text
                if args.normalize_actions and pm2.role_from == "gpt":
                    normalized = _normalize_assistant_value(assistant_text)
                    if normalized is None:
                        stats["assistant_msgs_failed"] += 1
                        if args.drop_unparseable:
                            bad_prefix = True
                            break
                    else:
                        stats["assistant_msgs_normalized"] += 1
                        assistant_text = normalized
                out_convs.append({"from": pm2.role_from, "value": assistant_text, **pm2.extra})

            if bad_prefix:
                stats["prefix_rows_dropped"] += 1
                continue

            # Images: arrays per round in prefix; references to per-round images saved once above.
            images_per_round: List[List[str]] = []
            for r in range(k):
                images_per_round.append(saved_paths_by_round[r])

            out_row = {
                "item_id": new_item_id,
                "orig_item_id": orig_item_id,
                "round_idx": k,
                "num_rounds": keep_rounds,
                "conversations": out_convs,
                "images": images_per_round,
            }

            merged_rows.append(out_row)
            stats["prefix_rows_written"] += 1

        stats["items_used"] += 1

    _write_jsonl(args.out_jsonl, merged_rows)
    print(
        f"wrote rows={len(merged_rows)} to {args.out_jsonl}\n"
        f"images in {args.out_image_dir}"
    )
    print("stats:", json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
