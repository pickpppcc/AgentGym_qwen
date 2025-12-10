from __future__ import annotations

import re
from typing import Any

import requests
import os
import sys

# 确保可导入同仓库内的 `agentenv` 包
# 目录结构：AgentGym/agentenv/agentenv/...，因此需要把
# - AgentGym/agentenv   加入到 sys.path（支持 `from agentenv.controller ...`）
# - AgentGym            作为兜底（支持 `from agentenv.agentenv.controller ...`）
_HERE = os.path.abspath(os.path.dirname(__file__))
_AGENTGYM_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))  # AgentGym/
_CANDIDATES = [
    os.path.join(_AGENTGYM_ROOT, "agentenv"),  # AgentGym/agentenv
    _AGENTGYM_ROOT,                             # AgentGym
]
for _p in _CANDIDATES:
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

try:
    # 优先使用标准安装路径
    from agentenv.controller import BaseEnvClient
    from agentenv.controller.types import StepOutput, ConversationMessage
except Exception:  # pragma: no cover - 兼容本地相对路径开发
    from agentenv.controller.env import BaseEnvClient
    from agentenv.controller.types import StepOutput, ConversationMessage


class OSWorldEnvClient(BaseEnvClient):
    """AgentGym 客户端：调用 OSWorld Server 的 HTTP 接口。

    与教程中的 BabyAIEnvClient 对齐：
    - /create -> 记录 env_id
    - /step    -> 返回 observation/reward/done
    - /reset   -> 返回初始 observation
    - /observation -> 可选，取最近状态
    - /close   -> 释放远端实例
    """

    # 对话起始（可后续根据需要调整）
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

    def __init__(self, env_server_base: str, data_len: int = 1, timeout: int = 300, create_cfg: dict[str, Any] | None = None, observe_format: str = "text", sleep_after_execution: float | int = 2, verbose: bool = False):
        super().__init__()
        self.env_server_base = env_server_base.rstrip("/")
        self.timeout = timeout
        self.data_len = data_len
        # observe_format: "text" | "raw" | "screenshot_base64"
        self.observe_format = observe_format
        # 默认每步执行后的等待秒数，映射 OSWorld 的 --sleep_after_execution
        self.pause_seconds: float = float(sleep_after_execution)
        self.verbose = bool(verbose)

        # 允许在创建时传入 OSWorld DesktopEnv 的初始化配置（provider_name、path_to_vm 等）
        if create_cfg is None:
            ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        else:
            ok = requests.post(f"{self.env_server_base}/create", json=create_cfg, timeout=self.timeout)
        ok.raise_for_status()
        j = ok.json()
        self.env_id = j["id"]
        self.info: dict[str, Any] | None = None

    def __len__(self) -> int:
        return self.data_len

    # ------------------ 内部 HTTP 调用 ------------------ #
    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(f"{self.env_server_base}/{path}", json=data, timeout=self.timeout)
        res.raise_for_status()
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(f"{self.env_server_base}/{path}?id={self.env_id}", timeout=self.timeout)
        res.raise_for_status()
        return res.json()

    # ------------------ 交互方法 ------------------ #
    def observe(self) -> str:
        if self.info:
            ob = self.info["observation"]  # type: ignore[index]
        else:
            r = self._get("observation")
            ob = r["observation"]  # type: ignore[index]

        if self.observe_format == "text":
            return self._format_observation(ob)
        elif self.observe_format == "screenshot_base64":
            try:
                return ob.get("screenshot_base64", "")  # type: ignore[union-attr]
            except Exception:
                return ""
        elif self.observe_format == "raw":
            # 为保持 BaseTask 流程，仍返回字符串；如需原始字典，可直接调用 _get("observation")
            return str(ob)
        else:
            return self._format_observation(ob)

    def step(self, action: str) -> StepOutput:
        # 将 LLM 输出中的 Action: 提取出来（若使用 ReAct 格式）
        action_matches = re.findall(r"Action:\s*(.*?)(?=\n|$)", action, re.DOTALL)
        if len(action_matches) > 1:
            return StepOutput(
                state=(
                    "Error: Only one 'Action' is allowed per response. Please adjust your response."
                ),
                reward=0,
                done=False,
            )
        act = action_matches[-1] if action_matches else action
        if self.verbose:
            try:
                print(f"[Client] raw_response={action!r} parsed_action={act!r}")
            except Exception:
                pass
        # 默认将字符串直接转发给 OSWorld；若使用结构化动作，可在此做 JSON 解析
        resp = self._post("step", {"action": act, "pause": self.pause_seconds})
        self.info = resp
        if self.verbose:
            try:
                keys = list(resp.get("observation", {}).keys()) if isinstance(resp.get("observation"), dict) else []
                print(f"[Client] result reward={resp.get('reward')} done={resp.get('done')} obs_keys={keys}")
            except Exception:
                pass
        return StepOutput(
            state=self._format_observation(resp["observation"]),
            reward=float(resp["reward"]),
            done=bool(resp["done"]),
        )

    def reset(self, idx: int = 0, task_config: dict[str, Any] | None = None) -> dict[str, Any]:
        # OSWorld 的 reset 支持 task_config：可传入完整 JSON（instruction/config/evaluator 等）
        resp = self._post("reset", {"task_config": task_config, "seed": idx})
        self.info = resp
        return resp

    def close(self) -> dict[str, Any]:
        resp = self._post("close", {})
        return resp

    # ------------------ 辅助 ------------------ #
    @staticmethod
    def _format_observation(observation: Any) -> str:
        # 将 OSWorld 的 dict 观测（含 screenshot_base64, accessibility_tree, terminal, instruction）转为简洁文本
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

