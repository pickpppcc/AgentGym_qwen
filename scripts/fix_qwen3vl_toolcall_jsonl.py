#!/usr/bin/env python3
"""
Fix Qwen3VL tool-call JSONL datasets that contain multiple top-level JSON objects
inside a single <tool_call>...</tool_call> block, e.g.:

  <tool_call>
  {...}
  {...}
  </tool_call>

Those blocks are not valid JSON and break strict parsers (json.loads -> "Extra data").

This script rewrites such blocks into multiple <tool_call> blocks (one JSON object
per block), while preserving the original JSON object text as much as possible:

  <tool_call>
  {...}
  </tool_call>
  <tool_call>
  {...}
  </tool_call>

Other content is left unchanged.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class Stats:
    lines_in: int = 0
    lines_out: int = 0
    items: int = 0
    msgs_seen: int = 0
    toolcall_blocks_seen: int = 0
    toolcall_blocks_fixed: int = 0
    msgs_fixed: int = 0
    items_fixed: int = 0


_DECODER = json.JSONDecoder()


def _split_concatenated_json_objects(raw: str) -> Tuple[List[str], str]:
    """
    Parse a string that may contain concatenated JSON objects and return:
      - list of raw JSON substrings (each a complete JSON value; we only keep dicts)
      - remainder (non-whitespace) after parsing stops

    We intentionally return the original substrings (not json.dumps(obj)) to
    minimize drift in formatting/tokens for training.
    """
    cleaned = (raw or "").strip()
    if not cleaned:
        return [], ""

    idx = 0
    parts: List[str] = []
    while idx < len(cleaned):
        while idx < len(cleaned) and cleaned[idx] in " \t\r\n,":
            idx += 1
        if idx >= len(cleaned):
            break
        try:
            obj, end = _DECODER.raw_decode(cleaned, idx)
        except json.JSONDecodeError:
            break
        # Only keep dicts; ignore other JSON values.
        if isinstance(obj, dict):
            parts.append(cleaned[idx:end])
        idx = end

    rest = cleaned[idx:].strip()
    return parts, rest


def _fix_tool_calls_in_text(text: str, stats: Stats) -> Tuple[str, bool]:
    if not isinstance(text, str) or "<tool_call>" not in text:
        return text, False

    out: List[str] = []
    pos = 0
    changed = False

    while True:
        start = text.find("<tool_call>", pos)
        if start == -1:
            out.append(text[pos:])
            break

        end = text.find("</tool_call>", start)
        if end == -1:
            # malformed; leave the rest untouched
            out.append(text[pos:])
            break

        out.append(text[pos:start])
        inner = text[start + len("<tool_call>") : end]
        stats.toolcall_blocks_seen += 1

        inner_stripped = inner.strip()
        if not inner_stripped:
            # empty block; keep as-is
            out.append(text[start : end + len("</tool_call>")])
            pos = end + len("</tool_call>")
            continue

        # If it's valid JSON already, keep it.
        try:
            json.loads(inner_stripped)
            out.append(text[start : end + len("</tool_call>")])
            pos = end + len("</tool_call>")
            continue
        except json.JSONDecodeError:
            pass

        parts, rest = _split_concatenated_json_objects(inner_stripped)
        if len(parts) >= 2 and not rest:
            # Rewrite into multiple tool_call blocks (one object per block).
            blocks = []
            for p in parts:
                blocks.append("<tool_call>\n" + p.strip() + "\n</tool_call>")
            out.append("\n".join(blocks))
            stats.toolcall_blocks_fixed += 1
            changed = True
        else:
            # Unknown/unhandled; keep original to avoid corrupting data.
            out.append(text[start : end + len("</tool_call>")])

        pos = end + len("</tool_call>")

    return "".join(out), changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL path.")
    ap.add_argument("--output", required=True, help="Output JSONL path.")
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Write to a temporary file then replace --input (also writes --output as backup).",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    tmp_path = out_path
    if args.inplace:
        # When in-place, we treat --output as a backup location.
        tmp_path = in_path.with_suffix(in_path.suffix + ".tmp")

    stats = Stats()

    with in_path.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            stats.lines_in += 1
            line = line.rstrip("\n")
            if not line.strip():
                fout.write(line + "\n")
                stats.lines_out += 1
                continue

            try:
                obj = json.loads(line)
            except Exception:
                # Keep unparseable lines unchanged.
                fout.write(line + "\n")
                stats.lines_out += 1
                continue

            if not isinstance(obj, dict):
                fout.write(line + "\n")
                stats.lines_out += 1
                continue

            stats.items += 1
            item_changed = False

            convs = obj.get("conversations")
            if isinstance(convs, list):
                for m in convs:
                    if not isinstance(m, dict):
                        continue
                    if m.get("from") not in {"gpt", "assistant"}:
                        continue
                    stats.msgs_seen += 1
                    v = m.get("value")
                    if not isinstance(v, str):
                        continue
                    fixed, changed = _fix_tool_calls_in_text(v, stats)
                    if changed:
                        m["value"] = fixed
                        stats.msgs_fixed += 1
                        item_changed = True

            if item_changed:
                stats.items_fixed += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            stats.lines_out += 1

    if args.inplace:
        # Move original to backup (--output), then replace it with tmp.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        in_path.replace(out_path)
        tmp_path.replace(in_path)

    # Print stats to stderr so stdout can be piped if needed.
    import sys

    print(json.dumps(stats.__dict__, ensure_ascii=False), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

