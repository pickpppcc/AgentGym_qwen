from __future__ import annotations

import re
from typing import Any, Mapping
import os
import time
import json
import base64
import struct

import requests

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput
import sys
sys.path.append('/data/pengchen/OSWorld/Osworld/OSWorld')
from mm_agents.uitars15_v1 import (
    parse_action,
    UITARS_ACTION_SPACE,
    UITARS_USR_PROMPT_THOUGHT,
    UITARS_SYS_PROMPT_THOUGHT_STRICT,
)


class OSWorldEnvClient(BaseEnvClient):

    def __init__(self, env_server_base: str, data_len: int = 1, timeout: int = 300, create_cfg: dict[str, Any] | None = None, observe_format: str = "text", examples_dir: str | None = None, post_reset_delay_seconds: float | int = 3, step_pause_seconds: float | int = 3, save_dir: str | None = None):
        super().__init__()
        self.env_server_base = env_server_base.rstrip("/")
        self.timeout = timeout
        self.data_len = data_len
        self.observe_format = observe_format
        self.example_files: list[str] = []
        self._post_reset_delay_seconds: float = float(post_reset_delay_seconds)
        self._step_pause_seconds: float = float(step_pause_seconds)
        self._save_dir: str | None = save_dir
        self._run_dir: str | None = None
        self._current_task_id: str | None = None
        self._step_index: int = 0
        self._recording_active: bool = False
        # UI-TARS models often emit normalized coordinates in [0, 1000] (not raw pixels).
        # Scale them back to the current screenshot resolution before sending to OSWorld server.
        try:
            self._uitars_coord_factor: int = int(os.environ.get("AGENTGYM_UITARS_COORD_FACTOR", "1000"))
        except Exception:
            self._uitars_coord_factor = 1000
        self._last_screen_size: tuple[int, int] | None = None

        # 创建 OSWorld 环境实例；若后端返回 500 等错误，尽量把服务端返回的报错信息打出来，方便排查。
        try:
            ok = requests.post(
                f"{self.env_server_base}/create",
                json=create_cfg or {},
                timeout=self.timeout,
            )
            ok.raise_for_status()
        except requests.HTTPError as e:
            # 把状态码、返回文本和当前 create_cfg 打印出来，便于用户定位是 server 端哪个参数不对
            body = None
            status = None
            try:
                status = ok.status_code  # type: ignore[name-defined]
                body = ok.text           # type: ignore[name-defined]
            except Exception:
                body = None
            debug_msg = (
                f"[OSWorldEnvClient] /create failed with HTTP {status}: {body}\n"
                f"  env_server_base={self.env_server_base}\n"
                f"  create_cfg={json.dumps(create_cfg or {}, ensure_ascii=False)}"
            )
            # 直接打印到控制台；同时抛出更有信息量的异常
            try:
                print(debug_msg)
            except Exception:
                pass
            raise RuntimeError(debug_msg) from e

        j = ok.json()
        self.env_id = j["id"]
        self.info: dict[str, Any] | None = None

        # Collect OSWorld evaluation examples if provided
        if examples_dir:
            collected: list[str] = []
            try:
                for root, _dirs, files in os.walk(examples_dir):
                    for fn in files:
                        if fn.endswith('.json'):
                            collected.append(os.path.join(root, fn))
                collected.sort()
            except Exception:
                collected = []
            self.example_files = collected
            if self.example_files:
                self.data_len = len(self.example_files)

    @property
    def conversation_start(self) -> tuple[ConversationMessage, ...]:
        """
        构造与 UI-TARS 风格一致的首轮对话 prompt。
        - 使用 UITARS_USR_PROMPT_THOUGHT 作为模板；
        - {instruction} 来自当前 OSWorld 任务的 instruction 字段；
        - {action_space} 使用 UITARS_ACTION_SPACE；
        - {language} 固定为英文，方便大多数模型对齐（如需中文可自行改成 'Chinese'）。
        """
        try:
            instr = self.get_instruction() or ""
        except Exception:
            instr = ""

        user_prompt = UITARS_USR_PROMPT_THOUGHT.format(
            instruction=instr,
            action_space=UITARS_ACTION_SPACE,
            language="English",
        )

        # 在 UITARS 基础模板后追加一个 few-shot 示例，帮助 Qwen 学会正确的 Thought / Action 格式。
        # 这里示例一个简单的“打开浏览器并访问 Amazon 页面”的步骤。
        example_block = (
            "\n\n"
            "## Example (format demonstration)\n"
            "Below is an example of the expected output format. "
            "Follow this style in your own responses.\n\n"
            "User Instruction: Open the browser and go to https://www.amazon.com, then click the 'Sign in' button.\n"
            "Observation: The desktop is visible and a browser icon is on the taskbar.\n\n"
            "Thought: I should first click the browser icon on the taskbar to open the browser, "
            "then I can type the Amazon URL in the address bar.\n"
            "Action: click(start_box='(120,1040)')\n\n"
            "Now, for the current task, think step by step and respond in the same Thought and Action format."
        )

        full_prompt = user_prompt + example_block

        # NOTE: keep the strict system prompt as a first "human" message so that:
        # - Text-only chat templates (that may not support system role) won't break;
        # - API/VL paths can promote it to a real system role and drop it from user content.
        return (
            # ConversationMessage(
            #     {
            #         "from": "human",
            #         "loss": None,
            #         "value": UITARS_SYS_PROMPT_THOUGHT_STRICT,
            #     }
            # ),
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": full_prompt,
                }
            ),
            ConversationMessage(
                {
                    "from": "gpt",
                    "loss": False,
                    "value": "OK. I will act carefully and finish the task.",
                }
            ),
        )

    def __len__(self) -> int:
        return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(f"{self.env_server_base}/{path}", json=data, timeout=self.timeout)
        res.raise_for_status()
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(f"{self.env_server_base}/{path}?id={self.env_id}", timeout=self.timeout)
        res.raise_for_status()
        return res.json()

    def _maybe_end_recording(self) -> None:
        try:
            if not self._recording_active:
                return
            if not self._run_dir:
                return
            dest = os.path.join(self._run_dir, "recording.mp4")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            self._post("end_recording", {"dest": dest})
        except Exception:
            pass
        finally:
            self._recording_active = False

    def _maybe_start_recording(self) -> None:
        try:
            if self._recording_active:
                return
            if not self._run_dir:
                return
            self._post("start_recording", {})
            self._recording_active = True
        except Exception:
            self._recording_active = False

    def observe(self) -> str:
        if self.info:
            ob = self.info["observation"]  # type: ignore[index]
        else:
            r = self._get("observation")
            ob = r["observation"]  # type: ignore[index]

        if self.observe_format == "text":
            return self._format_observation(ob)
        elif self.observe_format == "screenshot_base64":
            """
            为视觉语言模型返回「文本 + 截图」的组合：
            - instruction: 任务指令文本
            - step_text:   当前环境状态的简短说明（这里先留空，可后续扩展）
            - screenshot_base64: 当前帧截图（data:image/png;base64,...）

            统一打包成 JSON 字符串，方便下游像离线数据一样解析：
            {"instruction": ..., "step_text": ..., "screenshot_base64": ...}
            """
            try:
                instr: str | None = None
                step_text: str = ""
                screenshot_b64: str = ""
                if isinstance(ob, dict):
                    raw_instr = ob.get("instruction")
                    if isinstance(raw_instr, str) and raw_instr.strip():
                        instr = raw_instr.strip()
                    raw_ss = ob.get("screenshot_base64", "")
                    if isinstance(raw_ss, str):
                        screenshot_b64 = raw_ss
                payload = {
                    "instruction": instr or "",
                    "step_text": step_text,
                    "screenshot_base64": screenshot_b64,
                }
                return json.dumps(payload, ensure_ascii=False)
            except Exception:
                # 兜底：退化为仅返回截图，保持旧行为尽量不崩
                try:
                    return ob.get("screenshot_base64", "")  # type: ignore[union-attr]
                except Exception:
                    return ""
        elif self.observe_format == "raw":
            return str(ob)
        else:
            return self._format_observation(ob)

    def get_instruction(self) -> str | None:
        try:
            ob: Any
            if self.info and isinstance(self.info, dict):
                ob = self.info.get("observation")
            else:
                r = self._get("observation")
                ob = r.get("observation")
            if isinstance(ob, dict):
                instr = ob.get("instruction")
                if isinstance(instr, str) and instr.strip():
                    return instr.strip()
            return None
        except Exception:
            return None

    def step(self, action: str) -> StepOutput:
        action_matches = re.findall(r"Action:\s*(.*?)(?=\n|$)", action, re.DOTALL)
        if len(action_matches) > 1:
            return StepOutput(
                state=("Error: Only one 'Action' is allowed per response. Please adjust your response."),
                reward=0,
                done=False,
            )
        act = action_matches[-1] if action_matches else action
        if isinstance(act, str):
            act = act.strip()

        def _maybe_refresh_screen_size() -> None:
            # Best-effort: infer current screen size from last observation screenshot_base64.
            # Falls back to (1920,1080) if anything fails.
            try:
                if not isinstance(self.info, dict):
                    return
                obs = self.info.get("observation")
                if not isinstance(obs, dict):
                    return
                b64 = obs.get("screenshot_base64")
                if not isinstance(b64, str) or not b64:
                    return
                b64_str = b64
                if "," in b64_str and b64_str.strip().startswith("data:"):
                    b64_str = b64_str.split(",", 1)[-1]
                img_bytes = base64.b64decode(b64_str)
                # PNG IHDR: width/height stored at bytes 16..24 (big-endian)
                if len(img_bytes) >= 24 and img_bytes[:8] == b"\x89PNG\r\n\x1a\n":
                    w = int(struct.unpack(">I", img_bytes[16:20])[0])
                    h = int(struct.unpack(">I", img_bytes[20:24])[0])
                    if w > 0 and h > 0:
                        self._last_screen_size = (w, h)
            except Exception:
                return

        def _scale_point(x: float, y: float) -> tuple[float, float]:
            factor = int(self._uitars_coord_factor or 0)
            if factor <= 0:
                return x, y
            # Heuristic: if coords are within [0,factor], treat as normalized and scale to screen.
            if not (0.0 <= float(x) <= float(factor) and 0.0 <= float(y) <= float(factor)):
                return x, y
            w, h = self._last_screen_size or (1920, 1080)
            try:
                return float(x) / float(factor) * float(w), float(y) / float(factor) * float(h)
            except Exception:
                return x, y

        def _parse_point(s: str) -> tuple[float, float] | None:
            try:
                inner = s.strip().strip("'").strip('"').strip()
                inner = (
                    inner.replace("<|box_start|>", "")
                    .replace("<|box_end|>", "")
                    .strip()
                )
                if inner.startswith("(") and inner.endswith(")"):
                    x_str, y_str = inner[1:-1].split(",")
                    _maybe_refresh_screen_size()
                    x0, y0 = float(x_str.strip()), float(y_str.strip())
                    return _scale_point(x0, y0)
                return None
            except Exception:
                return None

        def _to_computer13(a: str):
            s = a.strip()
            # WAIT / DONE / FAIL 直接透传
            if s.upper() == "WAIT":
                return "WAIT"
            if s.upper() == "DONE" or s.lower().startswith("finished"):
                return "DONE"
            if s.upper() == "FAIL":
                return "FAIL"

            # 适配 UI-TARS 特殊动作函数写法
            if re.match(r"^wait\s*\(\s*\)\s*$", s, re.IGNORECASE):
                return "WAIT"
            if re.match(r"^finished\s*\(.*\)\s*$", s, re.IGNORECASE):
                return "DONE"
            if re.match(r"^call_user\s*\(\s*\)\s*$", s, re.IGNORECASE):
                # 在 OSWorld 控制器中无对应原生动作，这里退化为等待
                return "WAIT"

            try:
                parsed = parse_action(s)
            except Exception:
                return s

            if not parsed or not isinstance(parsed, dict):
                return s

            func = (parsed.get("function") or "").lower()
            args = parsed.get("args") or {}

            if func in {"click", "left_single"}:
                pt = _parse_point(str(args.get("start_box", "")))
                if pt:
                    x, y = pt
                    return {"action_type": "CLICK", "parameters": {"button": "left", "x": x, "y": y}}
                return {"action_type": "CLICK", "parameters": {}}

            if func in {"left_double"}:
                pt = _parse_point(str(args.get("start_box", "")))
                if pt:
                    x, y = pt
                    return {"action_type": "DOUBLE_CLICK", "parameters": {"x": x, "y": y}}
                return {"action_type": "DOUBLE_CLICK", "parameters": {}}

            if func in {"right_single"}:
                pt = _parse_point(str(args.get("start_box", "")))
                if pt:
                    x, y = pt
                    return {"action_type": "CLICK", "parameters": {"button": "right", "x": x, "y": y}}
                return {"action_type": "CLICK", "parameters": {"button": "right"}}

            if func == "hotkey":
                key = str(args.get("key", "")).strip()
                keys = [k.strip() for k in key.split()] if key else []
                return {"action_type": "HOTKEY", "parameters": {"keys": keys}}

            if func == "type":
                content = str(args.get("content", ""))
                return {"action_type": "TYPING", "parameters": {"text": content}}

            if func == "scroll":
                direction = str(args.get("direction", "")).strip().lower()
                dy = -120 if direction == "down" else 120 if direction == "up" else 0
                return {"action_type": "SCROLL", "parameters": ({"dy": dy} if dy != 0 else {})}

            if func == "drag":
                end_pt = _parse_point(str(args.get("end_box", "")))
                if end_pt:
                    x, y = end_pt
                    return {"action_type": "DRAG_TO", "parameters": {"x": x, "y": y}}
                return {"action_type": "DRAG_TO", "parameters": {}}

            # 其他未覆盖的动作原样返回
            return s

        # 支持 JSON 形式的 {"action_type":..., "parameters": {...}}，否则将 UI-TARS 风格动作转为 computer_13 字典
        act_or_dict = act
        try:
            s = act.strip()
            if s.startswith("{") and s.endswith("}"):
                import json as _json  # local import to avoid top-level dependency
                parsed = _json.loads(s)
                if isinstance(parsed, dict) and ("action_type" in parsed or "parameters" in parsed):
                    act_or_dict = parsed
                else:
                    act_or_dict = _to_computer13(act)
            else:
                act_or_dict = _to_computer13(act)
        except Exception:
            act_or_dict = _to_computer13(act)

        # 兜底：避免把未知字符串透传到 server（server 侧 controller 期望 dict）。
        if isinstance(act_or_dict, str):
            s = act_or_dict.strip().upper()
            if s not in {"WAIT", "DONE", "FAIL"}:
                act_or_dict = "WAIT"
        resp = self._post("step", {"action": act_or_dict, "pause": self._step_pause_seconds})
        self.info = resp
        # 自增步号并保存截图
        try:
            self._step_index += 1
            if self._run_dir:
                out_path = os.path.join(self._run_dir, f"step_{self._step_index:03d}.png")
                self._save_observation_snapshot(resp.get("observation"), out_path)
        except Exception:
            pass

        # IMPORTANT: Keep per-step observation formatting consistent with `observe()`.
        # - For `observe_format="screenshot_base64"`, this ensures the VLM continues to receive screenshots
        #   after the initial reset (otherwise it would only get a text summary).
        # - For `observe_format="text"` / "raw", this matches the user's chosen mode.
        next_state = self.observe()
        return StepOutput(
            state=next_state,
            reward=float(resp["reward"]),
            done=bool(resp["done"]),
        )

    def reset(self, data_idx: int = 0) -> dict[str, Any]:
        # End previous episode recording (if any) before starting a new task.
        self._maybe_end_recording()

        task_config = None
        sel = None
        if self.example_files:
            try:
                sel = self.example_files[data_idx % len(self.example_files)]
                with open(sel, 'r', encoding='utf-8') as f:
                    task_config = json.load(f)
            except Exception:
                task_config = None
        resp = self._post("reset", {"task_config": task_config, "seed": data_idx})
        self.info = resp
        # 初始化任务标识、运行目录并保存 reset 首帧
        try:
            self._current_task_id = None
            if isinstance(task_config, dict):
                tid = task_config.get("id")
                if isinstance(tid, str) and tid.strip():
                    self._current_task_id = tid.strip()
            if not self._current_task_id:
                # 退化用索引或文件名
                if isinstance(sel, str) and sel:
                    try:
                        self._current_task_id = os.path.splitext(os.path.basename(sel))[0]
                    except Exception:
                        self._current_task_id = f"idx_{data_idx}"
                else:
                    self._current_task_id = f"idx_{data_idx}"
            self._step_index = 0
            if self._save_dir:
                self._run_dir = os.path.join(self._save_dir, self._current_task_id)
                os.makedirs(self._run_dir, exist_ok=True)
                self._save_observation_snapshot(resp.get("observation"), os.path.join(self._run_dir, "reset.png"))
                # Start recording for this task; will be stopped on next reset/close.
                self._maybe_start_recording()
        except Exception:
            pass
        # 给 Chrome/应用一点时间完成启动渲染，避免首帧看不到窗口
        try:
            if self._post_reset_delay_seconds > 0:
                time.sleep(self._post_reset_delay_seconds)
                try:
                    refreshed = self._get("observation")
                    if isinstance(self.info, dict) and isinstance(refreshed, dict):
                        self.info["observation"] = refreshed.get("observation", self.info.get("observation"))
                        if self._run_dir:
                            self._save_observation_snapshot(self.info.get("observation"), os.path.join(self._run_dir, "reset_after_delay.png"))
                except Exception:
                    pass
        except Exception:
            pass
        return resp

    def close(self) -> dict[str, Any]:
        self._maybe_end_recording()
        return self._post("close", {})

    @staticmethod
    def _format_observation(observation: Any) -> str:
        if isinstance(observation, str):
            return observation
        if not isinstance(observation, dict):
            return str(observation)
        parts: list[str] = []
        if observation.get("instruction"):
            parts.append(f"Instruction: {observation['instruction']}")
        if observation.get("accessibility_tree"):
            parts.append("A11Y: available")
        if observation.get("terminal"):
            parts.append("Terminal: available")
        if observation.get("screenshot_base64"):
            parts.append("Screenshot: available (base64)")
        return " \n".join(parts) or "Observation received."

    def _save_observation_snapshot(self, observation: Any, out_path: str) -> None:
        try:
            if not isinstance(observation, dict):
                return
            data_url = observation.get("screenshot_base64") or ""
            if not isinstance(data_url, str) or not data_url:
                return
            if data_url.startswith("data:image"):
                b64 = data_url.split(",", 1)[-1]
            else:
                b64 = data_url
            img_bytes = base64.b64decode(b64)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(img_bytes)
        except Exception:
            pass


class OSWorldTask(BaseTask):
    env_client_cls = OSWorldEnvClient
    env_name = "OSWorld"

    def __init__(self, client_args: Mapping[str, Any], *args, n_clients: int = 1, **kwargs) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)
