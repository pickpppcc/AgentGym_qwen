from __future__ import annotations

import re
from typing import Any, Mapping
import os
import time
import json
import base64

import requests

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput
import sys
sys.path.append('/data/pengchen/OSWorld/Osworld/OSWorld')
from mm_agents.uitars15_v1 import parse_action


class OSWorldEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage({
            "from": "human",
            "loss": None,
            "value": "You control a desktop environment. Think and act to complete the instruction.",
        }),
        ConversationMessage({
            "from": "gpt",
            "loss": False,
            "value": "OK. I will act carefully and finish the task.",
        }),
    )

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

        def _parse_point(s: str) -> tuple[float, float] | None:
            try:
                inner = s.strip().strip("'").strip('"').strip()
                if inner.startswith("(") and inner.endswith(")"):
                    x_str, y_str = inner[1:-1].split(",")
                    return float(x_str.strip()), float(y_str.strip())
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
        return StepOutput(
            state=self._format_observation(resp["observation"]),
            reward=float(resp["reward"]),
            done=bool(resp["done"]),
        )

    def reset(self, data_idx: int = 0) -> dict[str, Any]:
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


