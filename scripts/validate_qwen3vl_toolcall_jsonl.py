#!/usr/bin/env python3
"""
Validate that a JSONL dataset matches the Qwen3VLAgent tool-call output schema:

Assistant message should contain:
  ... </think>

  Action: ...
  <tool_call>
  {"name":"computer_use","arguments":{...}}
  </tool_call>

This is a format/consistency check (not model quality).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

ALLOWED_ACTIONS = {
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "scroll",
    "wait",
    "terminate",
}


@dataclass
class Stats:
    items: int = 0
    gpt_msgs: int = 0
    ok_msgs: int = 0
    bad_msgs: int = 0


def _err(prefix: str, item_id: str, msg_idx: int, detail: str) -> str:
    return f"{prefix} item_id={item_id} msg_idx={msg_idx}: {detail}"


def _validate_tool_call(tc: Dict[str, Any]) -> Optional[str]:
    if tc.get("name") != "computer_use":
        return f"tool name != computer_use: {tc.get('name')!r}"
    args = tc.get("arguments")
    if not isinstance(args, dict):
        return "arguments is not an object"
    action = args.get("action")
    if action not in ALLOWED_ACTIONS:
        return f"invalid arguments.action: {action!r}"

    if action == "key":
        keys = args.get("keys")
        if not isinstance(keys, list) or not all(isinstance(k, str) and k for k in keys):
            return "action=key requires non-empty list[str] 'keys'"

    if action == "type":
        if not isinstance(args.get("text"), str):
            return "action=type requires string 'text'"

    if action in {
        "mouse_move",
        "left_click",
        "left_click_drag",
        "right_click",
        "middle_click",
        "double_click",
    }:
        coord = args.get("coordinate")
        if (
            not isinstance(coord, list)
            or len(coord) != 2
            or not all(isinstance(v, (int, float)) for v in coord)
        ):
            return f"action={action} requires coordinate=[x,y]"
        x, y = int(coord[0]), int(coord[1])
        if not (0 <= x <= 999 and 0 <= y <= 999):
            return f"coordinate out of range 0..999: {coord!r}"

    if action == "scroll":
        if not isinstance(args.get("pixels"), (int, float)):
            return "action=scroll requires numeric 'pixels'"

    if action == "wait":
        if not isinstance(args.get("time"), (int, float)):
            return "action=wait requires numeric 'time'"

    if action == "terminate":
        status = args.get("status")
        if status not in {"success", "failure"}:
            return "action=terminate requires status in {success,failure}"

    return None


def _validate_gpt_message(text: str) -> Tuple[bool, str]:
    if not isinstance(text, str):
        return False, "assistant value is not a string"
    # Accept both:
    # - thinking style: (thought)\n</think>\n\nAction: ...\n<tool_call>...\n</tool_call>
    # - non-thinking style: Action: ...\n<tool_call>...\n</tool_call>
    after = text.split("</think>", 1)[1] if "</think>" in text else text
    if "<tool_call>" not in after:
        return False, "missing <tool_call>"
    if "Action:" not in after:
        return False, "missing Action:"

    blocks = TOOL_CALL_BLOCK_RE.findall(after)
    if not blocks:
        return False, "no parseable <tool_call> JSON blocks"

    for b in blocks:
        try:
            tc = json.loads(b)
        except Exception as e:
            return False, f"tool_call JSON decode error: {e}"
        if not isinstance(tc, dict):
            return False, "tool_call JSON is not an object"
        msg = _validate_tool_call(tc)
        if msg:
            return False, msg
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL.")
    ap.add_argument("--limit", type=int, default=0, help="Only validate first N items (0=all).")
    ap.add_argument("--max-errors", type=int, default=20, help="Print up to N errors.")
    args = ap.parse_args()

    stats = Stats()
    errors: List[str] = []

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if args.limit and stats.items >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            stats.items += 1
            item_id = str(obj.get("item_id") or obj.get("orig_item_id") or stats.items)
            convs = obj.get("conversations", [])
            if not isinstance(convs, list):
                continue
            for mi, m in enumerate(convs):
                if not isinstance(m, dict) or m.get("from") != "gpt":
                    continue
                stats.gpt_msgs += 1
                ok, detail = _validate_gpt_message(m.get("value", ""))
                if ok:
                    stats.ok_msgs += 1
                else:
                    stats.bad_msgs += 1
                    if len(errors) < args.max_errors:
                        errors.append(_err("BAD", item_id, mi, detail))

    out = {
        "input": args.input,
        "items": stats.items,
        "gpt_msgs": stats.gpt_msgs,
        "ok_msgs": stats.ok_msgs,
        "bad_msgs": stats.bad_msgs,
    }
    print(json.dumps(out, ensure_ascii=False), file=sys.stderr)
    if errors:
        print("\n".join(errors), file=sys.stderr)
    return 0 if stats.bad_msgs == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
