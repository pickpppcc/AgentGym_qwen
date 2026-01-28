#!/usr/bin/env python3
import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Tuple


ACTION_SPLIT_RE = re.compile(r"\n\s*Action\s*[:：]\s*", flags=re.IGNORECASE)
THOUGHT_PREFIX_RE = re.compile(r"^\s*Thought\s*[:：]\s*", flags=re.IGNORECASE)


@dataclass
class Stats:
    total_lines: int = 0
    changed_messages: int = 0
    skipped_already_think: int = 0
    unparsed_messages: int = 0
    wrapped_unparsed: int = 0


def split_thought_action(text: str) -> Tuple[str, str] | None:
    """
    Parse strings like:
      Thought: ...
      Action: ...
    Return (thought, action) without labels, or None if not parseable.
    """
    if "</think>" in text:
        return None

    stripped = text.strip("\ufeff")
    if not THOUGHT_PREFIX_RE.match(stripped):
        return None

    body = THOUGHT_PREFIX_RE.sub("", stripped, count=1)
    parts = ACTION_SPLIT_RE.split(body, maxsplit=1)
    if len(parts) != 2:
        return None

    thought, action = parts[0].strip(), parts[1].strip()
    if not thought or not action:
        return None
    return thought, action


def _escape_single_quotes(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def convert_assistant_value(
    text: str,
    stats: Stats,
    on_unparsed: str,
    fallback_action: str,
) -> str:
    if "</think>" in text:
        stats.skipped_already_think += 1
        return text

    parsed = split_thought_action(text)
    if parsed is None:
        stats.unparsed_messages += 1
        if on_unparsed == "keep":
            return text
        if on_unparsed == "wrap":
            stats.wrapped_unparsed += 1
            thought = f"原始输出无法解析：{text.strip()}\n采取保守策略执行默认动作。".strip()
            return f"{thought}\n</think>\n\n{fallback_action.strip()}"
        if on_unparsed == "wrap_finished":
            stats.wrapped_unparsed += 1
            content = _escape_single_quotes(text.strip())
            return (
                "原始输出无法解析，直接结束本轮。"
                "\n</think>\n\n"
                f"finished(content='{content}')"
            )
        raise ValueError(f"Unknown on_unparsed mode: {on_unparsed}")

    stats.changed_messages += 1
    thought, action = parsed
    return f"{thought}\n</think>\n\n{action}"


def convert_obj(
    obj: Dict[str, Any],
    stats: Stats,
    on_unparsed: str,
    fallback_action: str,
) -> Dict[str, Any]:
    conversations = obj.get("conversations")
    if isinstance(conversations, list):
        for message in conversations:
            if not isinstance(message, dict):
                continue
            if message.get("from") != "gpt":
                continue
            value = message.get("value")
            if isinstance(value, str):
                message["value"] = convert_assistant_value(
                    value,
                    stats,
                    on_unparsed=on_unparsed,
                    fallback_action=fallback_action,
                )
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert AgentGym VL JSONL assistant messages from "
            "'Thought: ...\\nAction: ...' to '...\\n</think>\\n\\n...'."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL path (e.g., data/vl_all_prefix.dfs.jsonl).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--on-unparsed",
        choices=["keep", "wrap", "wrap_finished"],
        default="wrap",
        help=(
            "What to do if a GPT message cannot be parsed. "
            "'keep' leaves it unchanged; 'wrap' converts it to think+fallback action; "
            "'wrap_finished' converts it to think+finished(content=...). (default: wrap)"
        ),
    )
    parser.add_argument(
        "--fallback-action",
        default="wait()",
        help="Fallback action used with --on-unparsed=wrap (default: wait()).",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding (default: utf-8).",
    )
    args = parser.parse_args()

    stats = Stats()
    with open(args.input, "r", encoding=args.encoding) as fin, open(
        args.output, "w", encoding=args.encoding
    ) as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            stats.total_lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"JSON parse error at line {line_no}: {e}") from e

            obj = convert_obj(
                obj,
                stats,
                on_unparsed=args.on_unparsed,
                fallback_action=args.fallback_action,
            )
            fout.write(json.dumps(obj, ensure_ascii=False))
            fout.write("\n")

    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "total_lines": stats.total_lines,
                "changed_messages": stats.changed_messages,
                "skipped_already_think": stats.skipped_already_think,
                "unparsed_messages": stats.unparsed_messages,
                "wrapped_unparsed": stats.wrapped_unparsed,
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
