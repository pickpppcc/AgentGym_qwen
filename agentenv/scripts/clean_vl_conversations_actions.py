#!/usr/bin/env python3
"""
Clean/validate Action lines in processed_conversations_vl_*.json (list JSON).

Why:
  - AgentGym/OSWorld VL datasets often store assistant targets as:
      Thought: ...
      Action: pyautogui.click(start_box='(x,y)')
  - A tiny fraction of samples may contain corrupted/partial Action lines, which can
    hurt supervised training (the model learns to emit unparsable actions).

This script streams a large JSON array and either:
  - marks bad assistant turns as loss=False (default, safest), or
  - attempts a best-effort repair for simple broken start_box patterns.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple


def iter_json_array(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> Generator[dict, None, None]:
    dec = json.JSONDecoder()
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        buf = ""
        # Seek '['
        while True:
            ch = f.read(1)
            if not ch:
                return
            if ch.isspace():
                continue
            if ch == "[":
                break
            raise ValueError(f"{path} is not a JSON array")

        while True:
            # Skip whitespace/commas/end bracket
            while True:
                if not buf:
                    more = f.read(chunk_size)
                    if not more:
                        return
                    buf += more
                i = 0
                n = len(buf)
                while i < n and buf[i].isspace():
                    i += 1
                if i < n and buf[i] == ",":
                    buf = buf[i + 1 :]
                    continue
                if i < n and buf[i] == "]":
                    return
                buf = buf[i:]
                break

            while True:
                try:
                    obj, idx = dec.raw_decode(buf)
                except json.JSONDecodeError:
                    more = f.read(chunk_size)
                    if not more:
                        raise
                    buf += more
                    continue
                buf = buf[idx:]
                if isinstance(obj, dict):
                    yield obj
                else:
                    yield {"_non_dict_item": obj}
                break


def extract_action_line(text: Any) -> Optional[str]:
    if not isinstance(text, str):
        return None
    if "Action:" not in text:
        return None
    tail = text.split("Action:")[-1].strip()
    if not tail:
        return None
    # Strip code fences if present
    if tail.startswith("```"):
        tail = tail.strip("`").strip()
    for line in tail.splitlines():
        s = line.strip()
        if s:
            return s
    return None


def is_parseable_action(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.upper() in {"WAIT", "DONE", "FAIL"}:
        return True
    try:
        node = ast.parse(s, mode="eval")
        return isinstance(node, ast.Expression) and isinstance(node.body, ast.Call)
    except Exception:
        return False


_RE_START_BOX_INTS = re.compile(r"start_box\s*=\s*'?\(\s*(\d+)\s*,\s*(\d+)\s*", re.IGNORECASE)
_RE_FUNC_PREFIX = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_\.]*)\s*\(")


def try_repair_action(line: str) -> Optional[str]:
    """
    Best-effort repair for common corruptions like:
      pyautogui.click(start_box='(935,41出错了
    """
    s = line.strip()
    if is_parseable_action(s):
        return s

    # If it looks like it started a click with start_box ints, reconstruct.
    m_xy = _RE_START_BOX_INTS.search(s)
    if m_xy:
        x, y = m_xy.group(1), m_xy.group(2)
        m_fn = _RE_FUNC_PREFIX.search(s)
        fn = (m_fn.group(1) if m_fn else "pyautogui.click").strip()
        repaired = f"{fn}(start_box='({x},{y})')"
        if is_parseable_action(repaired):
            return repaired

    return None


def write_json_array(path: Path, items: Generator[dict, None, None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first = True
    with path.open("w", encoding="utf-8") as f:
        f.write("[\n")
        for item in items:
            if not first:
                f.write(",\n")
            first = False
            f.write(json.dumps(item, ensure_ascii=False))
        f.write("\n]\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--mode",
        choices=["mark_loss_false", "repair_or_mark_loss_false"],
        default="mark_loss_false",
        help="How to handle bad Action lines for assistant turns with loss=true.",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    total_items = 0
    total_loss_true_msgs = 0
    bad_msgs = 0
    repaired_msgs = 0

    def cleaned_items() -> Generator[dict, None, None]:
        nonlocal total_items, total_loss_true_msgs, bad_msgs, repaired_msgs
        for item in iter_json_array(in_path):
            total_items += 1
            convs = item.get("conversations")
            if isinstance(convs, list):
                for m in convs:
                    role = m.get("from") or m.get("role")
                    if role not in {"gpt", "assistant"}:
                        continue
                    if m.get("loss") is not True:
                        continue
                    total_loss_true_msgs += 1
                    text = m.get("value") or m.get("content") or ""
                    action = extract_action_line(text)
                    if action is None or not is_parseable_action(action):
                        bad_msgs += 1
                        if args.mode == "repair_or_mark_loss_false" and action:
                            repaired = try_repair_action(action)
                            if repaired and isinstance(text, str):
                                # Replace the last Action line (best-effort) while keeping Thought.
                                before, _sep, _after = text.rpartition("Action:")
                                m["value"] = (before + "Action:\n" + repaired).strip()
                                repaired_msgs += 1
                                continue
                        # Safest: exclude from loss.
                        m["loss"] = False
            yield item

    write_json_array(out_path, cleaned_items())

    print(
        json.dumps(
            {
                "input": str(in_path),
                "output": str(out_path),
                "items": total_items,
                "assistant_loss_true_msgs": total_loss_true_msgs,
                "bad_msgs": bad_msgs,
                "repaired_msgs": repaired_msgs,
                "mode": args.mode,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
