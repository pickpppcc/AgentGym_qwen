#!/usr/bin/env python3
"""
Strip the repeated "# Tools ... </tools> ... Rules ... Instruction:" prompt prefix from
Qwen3-VL tool-call JSONL datasets.

For each conversation message with from=="human", keep only the substring starting at
the first occurrence of "Instruction:" (inclusive), but only when the preceding text
looks like the tool/system prompt (e.g. contains "# Tools" or "<tools>").

This is useful to shrink datasets and to centralize the system prompt in training code
(e.g. QwenVLBCTrainer adds system_prompt once via role="system").
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple


def _strip_one(text: Any) -> Tuple[Any, bool]:
    if not isinstance(text, str):
        return text, False
    idx = text.find("Instruction:")
    if idx < 0:
        return text, False
    head = text[:idx]
    # Only strip when the prefix matches the known tool prompt shape to avoid
    # accidentally trimming other datasets that happen to contain "Instruction:".
    if ("# Tools" in head) or ("<tools>" in head) or ("You may call one or more functions" in head):
        return text[idx:].lstrip(), True
    return text, False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL.")
    ap.add_argument("--output", required=True, help="Output JSONL.")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = 0
    human_msgs = 0
    stripped_msgs = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            conv = obj.get("conversations")
            if isinstance(conv, list):
                for m in conv:
                    if not isinstance(m, dict):
                        continue
                    if m.get("from") != "human":
                        continue
                    human_msgs += 1
                    new_v, did = _strip_one(m.get("value"))
                    if did:
                        stripped_msgs += 1
                        m["value"] = new_v
            items += 1
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input": str(in_path),
                "output": str(out_path),
                "items": items,
                "human_msgs": human_msgs,
                "stripped_human_msgs": stripped_msgs,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

