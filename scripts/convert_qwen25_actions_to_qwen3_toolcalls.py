#!/usr/bin/env python3
"""
Convert AgentGym prefix JSONL (UITARS-style actions like click/drag/...) into
Qwen3VLAgent-style tool calls:

  Action: ...
  <tool_call>
  {"name":"computer_use","arguments":{...}}
  </tool_call>

Input is typically the output of `convert_vl_jsonl_to_qwen3vl_think.py`, where
assistant messages already look like:

  {thought}
  </think>

  click(start_box='(x,y)')
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


COORD_RE = re.compile(
    r"\(?\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)?"
)


QWEN3_ACTION_SPACE_BLOCK = """## Output Format

...
</think>

Action: ...
<tool_call>
{"name": "computer_use", "arguments": {...}}
</tool_call>

## Action Space

You must call tool `computer_use` and output JSON inside `<tool_call>...</tool_call>`.

- `action`: one of `key`, `type`, `mouse_move`, `left_click`, `left_click_drag`, `right_click`, `middle_click`, `double_click`, `scroll`, `wait`, `terminate`
- `keys`: array of strings (required only by `action=key`)
- `text`: string (required only by `action=type`)
- `coordinate`: `[x, y]` (required only by mouse actions; coordinates are in a 0..999 grid)
- `pixels`: number (required only by `action=scroll`; positive=up, negative=down)
- `time`: number (required only by `action=wait`; seconds)
- `status`: `success` or `failure` (required only by `action=terminate`)
"""


@dataclass
class Stats:
    total_lines: int = 0
    total_gpt_msgs: int = 0
    converted_gpt_msgs: int = 0
    unparsed_gpt_msgs: int = 0
    action_counts: Dict[str, int] = field(default_factory=dict)
    multi_tool_calls: int = 0
    human_prompts_rewritten: int = 0


def _first_nonempty_line(text: str) -> str:
    for ln in text.splitlines():
        s = ln.strip()
        if s:
            return s
    return ""


def _parse_coord(s: str) -> Optional[Tuple[int, int]]:
    m = COORD_RE.search(s)
    if not m:
        return None
    x = int(float(m.group(1)))
    y = int(float(m.group(2)))
    return x, y


def _parse_call(call_str: str) -> Optional[Dict[str, Any]]:
    s = call_str.strip()
    if not s:
        return None
    try:
        node = ast.parse(s, mode="eval")
        if not isinstance(node, ast.Expression) or not isinstance(node.body, ast.Call):
            return None
        call = node.body
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            return None
        kwargs: Dict[str, Any] = {}
        for kw in call.keywords:
            if not kw.arg:
                continue
            if isinstance(kw.value, ast.Constant):
                kwargs[kw.arg] = kw.value.value
            elif isinstance(kw.value, ast.Str):
                kwargs[kw.arg] = kw.value.s
            else:
                kwargs[kw.arg] = None
        return {"function": func_name, "args": kwargs}
    except Exception:
        return None


def _tool_call_block(arguments: Dict[str, Any]) -> str:
    payload = {"name": "computer_use", "arguments": arguments}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"


def _hotkey_to_keys(key_str: str) -> List[str]:
    s = (key_str or "").strip().lower()
    if not s:
        return []
    # Common variants
    s = s.replace("control", "ctrl")
    s = s.replace("command", "cmd")
    s = s.replace("print screen", "printscreen")
    # Split on '+' or whitespace
    parts = [p for p in re.split(r"[+\\s]+", s) if p]
    return parts


def _convert_action_call_to_qwen3(
    call_str: str,
    *,
    scroll_pixels: int,
    scroll_drag_delta: int,
    wait_time: float,
) -> Optional[str]:
    parsed = _parse_call(call_str)
    if parsed is None:
        # allow plain WAIT/DONE style tokens as a last resort
        up = call_str.strip().upper()
        if up == "WAIT":
            return "Action: Wait.\n" + _tool_call_block({"action": "wait", "time": wait_time})
        if up == "DONE":
            return "Action: Terminate the task.\n" + _tool_call_block({"action": "terminate", "status": "success"})
        if up == "FAIL":
            return "Action: Terminate the task.\n" + _tool_call_block({"action": "terminate", "status": "failure"})
        return None

    fn = parsed["function"]
    args = parsed["args"] or {}

    if fn == "click":
        xy = _parse_coord(str(args.get("start_box", "")))
        if not xy:
            return None
        x, y = xy
        return "Action: Left click.\n" + _tool_call_block({"action": "left_click", "coordinate": [x, y]})

    if fn == "left_double":
        xy = _parse_coord(str(args.get("start_box", "")))
        if not xy:
            return None
        x, y = xy
        return "Action: Double click.\n" + _tool_call_block({"action": "double_click", "coordinate": [x, y]})

    if fn == "right_single":
        xy = _parse_coord(str(args.get("start_box", "")))
        if not xy:
            return None
        x, y = xy
        return "Action: Right click.\n" + _tool_call_block({"action": "right_click", "coordinate": [x, y]})

    if fn == "drag":
        start_xy = _parse_coord(str(args.get("start_box", "")))
        end_xy = _parse_coord(str(args.get("end_box", "")))
        if not start_xy or not end_xy:
            return None
        sx, sy = start_xy
        ex, ey = end_xy
        blocks = [
            _tool_call_block({"action": "mouse_move", "coordinate": [sx, sy]}),
            _tool_call_block({"action": "left_click_drag", "coordinate": [ex, ey]}),
        ]
        return (
            f"Action: Drag from ({sx},{sy}) to ({ex},{ey}).\n" + "\n".join(blocks)
        )

    if fn == "scroll":
        direction = str(args.get("direction", "")).strip().lower()
        start_xy = _parse_coord(str(args.get("start_box", "")))
        if not start_xy:
            return None
        x, y = start_xy
        if direction in {"down", "up"}:
            pixels = -abs(scroll_pixels) if direction == "down" else abs(scroll_pixels)
            blocks = [
                _tool_call_block({"action": "mouse_move", "coordinate": [x, y]}),
                _tool_call_block({"action": "scroll", "pixels": pixels}),
            ]
            return f"Action: Scroll {direction}.\n" + "\n".join(blocks)

        # Qwen3VLAgent tool schema does not provide hscroll; best-effort: drag horizontally.
        if direction in {"right", "left"}:
            dx = abs(scroll_drag_delta) * (1 if direction == "right" else -1)
            tx = max(0, min(999, x + dx))
            blocks = [
                _tool_call_block({"action": "mouse_move", "coordinate": [x, y]}),
                _tool_call_block({"action": "left_click_drag", "coordinate": [tx, y]}),
            ]
            return f"Action: Scroll {direction} (approx. by dragging).\n" + "\n".join(blocks)

        return None

    if fn == "hotkey":
        keys = _hotkey_to_keys(str(args.get("key", "")))
        if not keys:
            return None
        return "Action: Press keys.\n" + _tool_call_block({"action": "key", "keys": keys})

    if fn == "type":
        text = args.get("content", "")
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        return "Action: Type text.\n" + _tool_call_block({"action": "type", "text": text})

    if fn == "wait":
        return "Action: Wait.\n" + _tool_call_block({"action": "wait", "time": wait_time})

    if fn == "finished":
        return (
            "Action: Terminate the task.\n"
            + _tool_call_block({"action": "terminate", "status": "success"})
        )

    return None


def _rewrite_human_prompt(text: str) -> Tuple[str, bool]:
    if not isinstance(text, str) or "## Output Format" not in text:
        return text, False
    # Replace the entire Output Format + Action Space section; keep Note and everything after.
    pattern = r"## Output Format\s*\n.*?\n## Note\s*\n"
    repl = QWEN3_ACTION_SPACE_BLOCK + "\n## Note\n"
    new_text, n = re.subn(pattern, repl, text, flags=re.DOTALL)
    if n > 0:
        # Rewrite Note lines to avoid the literal "Thought" label while keeping intent.
        new_text = re.sub(
            r"^-\s*Use\s+Chinese\s+in\s+`Thought`\s+part\.\s*$",
            "- Use Chinese in the reasoning part (before `</think>`).",
            new_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        new_text = re.sub(
            r"^-\s*Write\s+a\s+small\s+plan.*?in\s+`Thought`\s+part\s*part\.\s*$",
            "- Write a small plan and finally summarize your next action (with its target element) in the reasoning part (before `</think>`).",
            new_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        # If variants exist, do a softer fallback rewrite.
        new_text = re.sub(
            r"`Thought`\s*part",
            "the reasoning part (before `</think>`)",
            new_text,
            flags=re.IGNORECASE,
        )
    return new_text, n > 0


def _convert_assistant_value(
    text: str,
    stats: Stats,
    *,
    scroll_pixels: int,
    scroll_drag_delta: int,
    wait_time: float,
) -> str:
    if not isinstance(text, str) or "</think>" not in text:
        return text
    before, _sep, after = text.partition("</think>")

    # Some rare corrupted samples may contain extra lines after </think>, e.g.:
    #   click(start_box='(935,41出错了
    #   ...
    #   Action: click(start_box='(491,385)')
    # Prefer the first parseable action call we can find.
    action_line = ""
    for ln in after.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.lower().startswith("action:"):
            s = s.split(":", 1)[1].strip()
        if _convert_action_call_to_qwen3(
            s,
            scroll_pixels=scroll_pixels,
            scroll_drag_delta=scroll_drag_delta,
            wait_time=wait_time,
        ):
            action_line = s
            break
        if not action_line:
            action_line = s  # fallback to first non-empty (may be unparseable)

    stats.total_gpt_msgs += 1
    if not action_line:
        stats.unparsed_gpt_msgs += 1
        return text

    converted = _convert_action_call_to_qwen3(
        action_line,
        scroll_pixels=scroll_pixels,
        scroll_drag_delta=scroll_drag_delta,
        wait_time=wait_time,
    )
    if converted is None:
        stats.unparsed_gpt_msgs += 1
        return text

    fn = action_line.split("(", 1)[0].strip()
    stats.action_counts[fn] = stats.action_counts.get(fn, 0) + 1
    if converted.count("<tool_call>") > 1:
        stats.multi_tool_calls += 1

    stats.converted_gpt_msgs += 1
    # Keep reasoning content, but drop the literal "Thought:" label if present.
    thought_text = before.strip()
    thought_text = re.sub(r"^\s*Thought\s*[:：]\s*", "", thought_text, flags=re.IGNORECASE)
    return thought_text.rstrip() + "\n</think>\n\n" + converted.strip()


def convert_obj(
    obj: Dict[str, Any],
    stats: Stats,
    *,
    scroll_pixels: int,
    scroll_drag_delta: int,
    wait_time: float,
    rewrite_human_prompts: bool,
) -> Dict[str, Any]:
    conversations = obj.get("conversations")
    if isinstance(conversations, list):
        for msg in conversations:
            if not isinstance(msg, dict):
                continue
            role = msg.get("from")
            if role == "human" and rewrite_human_prompts:
                v = msg.get("value")
                if isinstance(v, str):
                    nv, changed = _rewrite_human_prompt(v)
                    if changed:
                        msg["value"] = nv
                        stats.human_prompts_rewritten += 1
            if role != "gpt":
                continue
            v = msg.get("value")
            if isinstance(v, str):
                msg["value"] = _convert_assistant_value(
                    v,
                    stats,
                    scroll_pixels=scroll_pixels,
                    scroll_drag_delta=scroll_drag_delta,
                    wait_time=wait_time,
                )
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL path.")
    ap.add_argument("--output", required=True, help="Output JSONL path.")
    ap.add_argument(
        "--rewrite-human-prompts",
        action="store_true",
        default=True,
        help="Rewrite human prompt Output Format/Action Space to Qwen3 tool-call style (default: on).",
    )
    ap.add_argument(
        "--no-rewrite-human-prompts",
        action="store_false",
        dest="rewrite_human_prompts",
        help="Keep human prompts unchanged.",
    )
    ap.add_argument(
        "--scroll-pixels",
        type=int,
        default=500,
        help="Pixels for vertical scroll (default: 500).",
    )
    ap.add_argument(
        "--scroll-drag-delta",
        type=int,
        default=200,
        help="Delta x (0..999 grid) for approximating horizontal scroll via dragging (default: 200).",
    )
    ap.add_argument(
        "--wait-time",
        type=float,
        default=5.0,
        help="Seconds for wait() (default: 5.0).",
    )
    ap.add_argument("--encoding", default="utf-8")
    args = ap.parse_args()

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
            if not isinstance(obj, dict):
                continue
            obj = convert_obj(
                obj,
                stats,
                scroll_pixels=args.scroll_pixels,
                scroll_drag_delta=args.scroll_drag_delta,
                wait_time=args.wait_time,
                rewrite_human_prompts=args.rewrite_human_prompts,
            )
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "total_lines": stats.total_lines,
                "total_gpt_msgs": stats.total_gpt_msgs,
                "converted_gpt_msgs": stats.converted_gpt_msgs,
                "unparsed_gpt_msgs": stats.unparsed_gpt_msgs,
                "multi_tool_calls": stats.multi_tool_calls,
                "human_prompts_rewritten": stats.human_prompts_rewritten,
                "action_counts_top20": sorted(
                    stats.action_counts.items(), key=lambda kv: kv[1], reverse=True
                )[:20],
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
