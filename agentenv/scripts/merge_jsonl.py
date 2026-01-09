#!/usr/bin/env python3
"""
Merge multiple JSONL files into a single JSONL file (streaming, low-memory).

Examples:
  python merge_jsonl.py \
    --output /dfs/data/AgentGym_qwen/data/vl_all_prefix_merged.strict.jsonl \
    /dfs/data/AgentGym_qwen/data/vl_train_prefix_merged.strict.jsonl \
    /dfs/data/AgentGym_qwen/data/vl_validation_prefix_merged.strict.jsonl \
    /dfs/data/AgentGym_qwen/data/vl_test_prefix_merged.strict.jsonl

Optional de-duplication (keeps first occurrence):
  python merge_jsonl.py --dedupe_key item_id --output merged.jsonl a.jsonl b.jsonl c.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Set


def _merge(
    inputs: list[str],
    output: str,
    *,
    validate_json: bool,
    dedupe_key: Optional[str],
) -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)

    seen: Optional[Set[str]] = set() if dedupe_key else None
    stats = {
        "inputs": inputs,
        "output": output,
        "lines_in": 0,
        "lines_out": 0,
        "lines_skipped_blank": 0,
        "lines_skipped_invalid_json": 0,
        "lines_skipped_deduped": 0,
    }

    with open(output, "w", encoding="utf-8") as w:
        for path in inputs:
            with open(path, "r", encoding="utf-8") as r:
                for line in r:
                    stats["lines_in"] += 1
                    if not line.strip():
                        stats["lines_skipped_blank"] += 1
                        continue

                    if validate_json or seen is not None:
                        try:
                            obj = json.loads(line)
                        except Exception:
                            stats["lines_skipped_invalid_json"] += 1
                            continue

                        if seen is not None:
                            key_val = obj.get(dedupe_key)
                            if key_val is None:
                                # If missing key, keep it (can't dedupe)
                                pass
                            else:
                                key_str = str(key_val)
                                if key_str in seen:
                                    stats["lines_skipped_deduped"] += 1
                                    continue
                                seen.add(key_str)

                        # Write normalized JSON if we parsed it
                        w.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        stats["lines_out"] += 1
                    else:
                        # Fast-path: raw concatenate
                        w.write(line if line.endswith("\n") else line + "\n")
                        stats["lines_out"] += 1

    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge JSONL files into one JSONL.")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument(
        "--no_validate_json",
        action="store_true",
        help="Do not parse JSON; just concatenate lines (faster, no checks).",
    )
    ap.add_argument(
        "--dedupe_key",
        default=None,
        help="Optional key to dedupe by (e.g., item_id). Keeps first occurrence.",
    )
    ap.add_argument("inputs", nargs="+", help="Input JSONL files (in order)")
    args = ap.parse_args()

    stats = _merge(
        args.inputs,
        args.output,
        validate_json=not args.no_validate_json,
        dedupe_key=args.dedupe_key,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

