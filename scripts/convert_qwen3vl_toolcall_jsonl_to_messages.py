#!/usr/bin/env python3
"""
Convert the legacy Qwen3-VL tool-call JSONL format used in this repo:
  - conversations: [{"from":"human"|"gpt","value": "...<think>...</think>...<tool_call>{...}</tool_call>"}]
  - images: [[img_path], [img_path], ...] (per-round)

into a structured chat format:
  - messages: [{"role":"user"|"assistant", ...}]
  - assistant: {"reasoning_content": "...", "content": "Action: ...", "tool_calls": [...]}
  - user: {"content": [{"type":"image","image": "...path..."}, {"type":"text","text":"...pure instruction..."}]}

This script is intended to *reduce dataset redundancy* and make training rely on:
  - a global system prompt injected in trainer, and
  - tools passed via tokenizer.apply_chat_template(tools=...).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)


def _extract_pure_instruction(text: Any) -> str:
    """
    Input typically looks like:
      Instruction: [SEP] <instruction> [SEP] Step: ... [SEP] ScreenshotRelPath: ... [SEP] ...
    We keep only <instruction>.
    """
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if not t:
        return ""
    if "Instruction:" in t:
        t = t.split("Instruction:", 1)[1].strip()
    # Remove leading [SEP]
    if t.startswith("[SEP]"):
        t = t[len("[SEP]") :].strip()
    # Take only the first segment before next [SEP]
    if "[SEP]" in t:
        t = t.split("[SEP]", 1)[0].strip()
    return t


def _parse_assistant(value: Any) -> Tuple[Optional[str], str, List[Dict[str, Any]]]:
    if not isinstance(value, str):
        return None, "", []
    s = value

    reasoning = None
    m = THINK_RE.search(s)
    if m:
        reasoning = m.group(1).strip()

    # Extract the first "Action:" line (keep it with the prefix).
    action_line = ""
    action_idx = s.find("Action:")
    if action_idx >= 0:
        tail = s[action_idx:].splitlines()
        if tail:
            action_line = tail[0].strip()

    tool_calls: List[Dict[str, Any]] = []
    for block in TOOL_CALL_BLOCK_RE.findall(s):
        try:
            obj = json.loads(block)
        except Exception:
            continue
        if isinstance(obj, dict):
            # Normalize to {"name":..., "arguments":...}
            if "function" in obj and isinstance(obj.get("function"), dict):
                obj = obj["function"]
            if "name" in obj and "arguments" in obj:
                tool_calls.append({"name": obj["name"], "arguments": obj["arguments"]})

    return reasoning, action_line, tool_calls


def _convert_item(obj: Dict[str, Any]) -> Dict[str, Any]:
    conversations = obj.get("conversations") or []
    images_by_round = obj.get("images") or []

    messages: List[Dict[str, Any]] = []

    for mi, m in enumerate(conversations):
        if not isinstance(m, dict):
            continue
        frm = m.get("from")
        val = m.get("value")

        if frm == "human":
            round_r = (mi // 2) + 1
            round_paths: List[str] = []
            if isinstance(images_by_round, list) and (round_r - 1) < len(images_by_round):
                rp = images_by_round[round_r - 1]
                if isinstance(rp, list):
                    round_paths = [p for p in rp if isinstance(p, str) and p]

            instruction = _extract_pure_instruction(val)
            content: List[Dict[str, Any]] = [{"type": "image", "image": p} for p in round_paths]
            content.append({"type": "text", "text": instruction})
            messages.append({"role": "user", "content": content})

        elif frm == "gpt":
            reasoning, action_line, tool_calls = _parse_assistant(val)
            out: Dict[str, Any] = {
                "role": "assistant",
                "content": action_line,
            }
            if isinstance(reasoning, str) and reasoning:
                out["reasoning_content"] = reasoning
            if tool_calls:
                out["tool_calls"] = tool_calls
            messages.append(out)

    # Remove legacy fields and attach new messages.
    obj = dict(obj)
    obj.pop("conversations", None)
    obj.pop("images", None)
    obj["messages"] = messages
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input legacy JSONL.")
    ap.add_argument("--output", required=True, help="Output JSONL with messages.")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = 0
    user_msgs = 0
    asst_msgs = 0
    tool_calls_total = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            new_obj = _convert_item(obj)
            items += 1
            for msg in new_obj.get("messages", []) or []:
                if msg.get("role") == "user":
                    user_msgs += 1
                elif msg.get("role") == "assistant":
                    asst_msgs += 1
                    tc = msg.get("tool_calls") or []
                    if isinstance(tc, list):
                        tool_calls_total += len(tc)
            fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input": str(in_path),
                "output": str(out_path),
                "items": items,
                "user_msgs": user_msgs,
                "assistant_msgs": asst_msgs,
                "tool_calls_total": tool_calls_total,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

