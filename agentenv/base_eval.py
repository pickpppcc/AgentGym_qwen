from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from agentenv.controller import Agent, Evaluator, APIAgent
from agentenv.envs import OSWorldTask
import re
import json
import sys
sys.path.append('/data/pengchen/OSWorld/Osworld/OSWorld')
from mm_agents.uitars15_v1 import (
    UITARS_ACTION_SPACE,
    UITARS_CALL_USR_ACTION_SPACE,
    UITARS_NORMAL_ACTION_SPACE,
    UITARS_USR_PROMPT_NOTHOUGHT,
    UITARS_USR_PROMPT_THOUGHT,
    UITARS_SYS_PROMPT_THOUGHT_STRICT,
    add_box_token,
)

MODEL_PATH = "THUDM/agentlm-7b"  # 示例

# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map={2,7}, trust_remote_code=True).eval()


class VisionAPIAgent(APIAgent):
    def __init__(self, *args, keep_history: bool = True, history_n: int = 15, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_history: bool = bool(keep_history)
        self.history_n: int = int(history_n)
        # 仍保留，以备后续需要
        self.history_responses: list[str] = []

    def _extract_structured_action(self, text: str) -> str:
        text = (text or "").strip()
        # 若模型已按 UI-TARS 或既有规范直接给出 "Action: ..."，则透传该行
        m_direct = re.search(r"Action:\s*(.*)", text, re.DOTALL)
        if m_direct:
            body = m_direct.group(1).strip()
            if body:
                return f"Action: {body}"
        # 快速路径：WAIT/DONE/FAIL
        if re.search(r"\bWAIT\b", text, re.IGNORECASE):
            return "Action: WAIT"
        if re.search(r"\bDONE\b", text, re.IGNORECASE):
            return "Action: DONE"
        if re.search(r"\bFAIL\b", text, re.IGNORECASE):
            return "Action: FAIL"

        # 提取 JSON 体：优先匹配 "Action: { ... }"
        m = re.search(r"Action:\s*(\{.*\})", text, re.DOTALL)
        candidate = None
        if m:
            candidate = m.group(1)
        else:
            # 回退：尝试抓第一段平衡的大括号
            s = text
            start = s.find("{")
            if start != -1:
                depth = 0
                for i in range(start, len(s)):
                    ch = s[i]
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = s[start:i+1]
                            break

        def valid_and_normalize(obj: dict) -> dict | None:
            if not isinstance(obj, dict):
                return None
            at = obj.get("action_type") or obj.get("type")
            pm = obj.get("parameters") or obj.get("params") or {}
            if not isinstance(pm, dict):
                return None
            if not isinstance(at, str):
                return None
            at_u = at.strip().upper()
            allowed = {
                "MOVE_TO": ["x", "y"],
                "CLICK": ["x", "y"],
                "DOUBLE_CLICK": ["x", "y"],
                "RIGHT_CLICK": ["x", "y"],
                "SCROLL": [],  # dx/dy 可选
                "TYPING": ["text"],
                "PRESS": ["key"],
                "KEY_DOWN": ["key"],
                "KEY_UP": ["key"],
                "HOTKEY": ["keys"],
            }
            if at_u not in allowed:
                return None
            # 基本必需参数校验
            req = allowed[at_u]
            for k in req:
                if k not in pm:
                    return None
            # 简单类型校验
            if at_u in {"MOVE_TO", "CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"}:
                if not (isinstance(pm.get("x"), (int, float)) and isinstance(pm.get("y"), (int, float))):
                    return None
            if at_u == "SCROLL":
                if ("dx" in pm and not isinstance(pm.get("dx"), (int, float))) or ("dy" in pm and not isinstance(pm.get("dy"), (int, float))):
                    return None
            if at_u == "TYPING":
                if not isinstance(pm.get("text"), str):
                    return None
            if at_u in {"PRESS", "KEY_DOWN", "KEY_UP"}:
                if not isinstance(pm.get("key"), str):
                    return None
            if at_u == "HOTKEY":
                if not isinstance(pm.get("keys"), list) or not all(isinstance(x, str) for x in pm.get("keys")):
                    return None
            return {"action_type": at_u, "parameters": pm}

        if candidate:
            try:
                obj = json.loads(candidate)
                norm = valid_and_normalize(obj)
                if norm:
                    return "Action: " + json.dumps(norm, ensure_ascii=False)
            except Exception:
                pass

        # 兜底
        return "Action: WAIT"

    def generate(self, conversation):
        # 1) 构造与 UITARS 一致的 messages 结构
        # 提取 instruction（若无法获得则给出通用占位）
        instruction_text = "Please complete the next step based on the current desktop context."
        try:
            # 如果会话里有包含 Instruction: 的文本，抓取最近一条
            for m in reversed(conversation):
                c = m.get("content")
                if isinstance(c, str) and c.startswith("Instruction:"):
                    instruction_text = c.split("Instruction:", 1)[-1].strip()
                    break
        except Exception:
            pass

        # 选择默认的动作空间与模板（Thought 版本）
        action_space_text = UITARS_ACTION_SPACE
        user_prompt = UITARS_USR_PROMPT_THOUGHT.format(
            instruction=instruction_text,
            action_space=action_space_text,
            language="Chinese"
        )

        msgs = [
            {
                "role": "system",
                "content": [{"type": "text", "text": UITARS_SYS_PROMPT_THOUGHT_STRICT}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}]
            }
        ]

        # 参考 uitars15_v1：使用 self.history_responses 与用户图片交错构造 message 流
        # 1) 从 conversation 中提取所有用户图片（按时间顺序），统一转为 data URL
        image_urls: list[str] = []
        try:
            for m in conversation:
                role = m.get("role")
                content = m.get("content")
                if role != "user" or not isinstance(content, str):
                    continue
                if content.startswith("data:image") or len(content) > 1000:
                    img = content if content.startswith("data:image") else f"data:image/png;base64,{content}"
                    image_urls.append(img)
        except Exception:
            image_urls = []

        # 2) 控制历史窗口长度
        if len(image_urls) > self.history_n:
            image_urls = image_urls[-self.history_n:]

        # 3) 交错拼接：对于最近的历史响应，尽量配对用户图片
        image_num = 0
        if self.keep_history and len(self.history_responses) > 0:
            for history_idx, history_response in enumerate(self.history_responses):
                # 仅为最后 history_n 条历史配图
                if history_idx + self.history_n > len(self.history_responses):
                    if image_num < len(image_urls):
                        msgs.append({
                            "role": "user",
                            "content": [{"type": "image_url", "image_url": {"url": image_urls[image_num]}}]
                        })
                        image_num += 1
                msgs.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": add_box_token(history_response)}]
                })

            # 末尾追加当前图片
            if image_num < len(image_urls):
                msgs.append({
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": image_urls[image_num]}}]
                })
                image_num += 1
        else:
            # 无历史响应时，至少追加一张当前图片
            if len(image_urls) > 0:
                msgs.append({
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": image_urls[0]}}]
                })

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        raw_text = resp.choices[0].message.content or ""
        # 记录历史响应文本（可选）
        try:
            self.history_responses.append(raw_text)
        except Exception:
            pass
        # 提取 Thought 作为 reasoning_content（若存在）
        thought = None
        try:
            text = (raw_text or "").strip()
            if text.startswith("Thought:"):
                m = re.search(r"Thought:\s*(.+?)(?=\n\s*Action:|$)", text, re.DOTALL)
                if m:
                    thought = m.group(1).strip()
            elif text.startswith("Reflection:"):
                m = re.search(r"Reflection:\s*(.+?)Action_Summary:\s*(.+?)(?=\n\s*Action:|$)", text, re.DOTALL)
                if m:
                    reflection = m.group(1).strip()
                    summary = m.group(2).strip()
                    thought = f"{reflection}\n{summary}".strip()
            elif text.startswith("Action_Summary:"):
                m = re.search(r"Action_Summary:\s*(.+?)(?=\n\s*Action:|$)", text, re.DOTALL)
                if m:
                    thought = m.group(1).strip()
            else:
                m = re.search(r"Thought:\s*(.+?)(?=\n\s*Action:|$)", text, re.DOTALL)
                if m:
                    thought = m.group(1).strip()
        except Exception:
            thought = None
        # 严格返回一行 Action（缺失时兜底为 WAIT），避免环境端 500
        action_only = self._extract_structured_action(raw_text)
        return action_only, thought

api_agent = VisionAPIAgent(
    api_key="empty",                 # 看你服务是否需要，很多本地服务忽略
    base_url="http://localhost:7860/v1",  # 你的服务地址（示例来自 uitars15_v1.py）
    model="ui-tars",         # 你的服务里声明的模型名
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
)

evaluator = Evaluator(
    api_agent,
    [
        OSWorldTask(
            client_args={
                "env_server_base": "http://127.0.0.1:36001",
               
                "timeout": 600,
                # DesktopEnv 初始化参数（可选）
                "create_cfg": {
                    "provider_name": "docker",     # 或 vmware/aws/grpc_cluster
                    "path_to_vm": "/data/pengchen/OSWorld/env/Ubuntu.qcow2",
                    
                    "snapshot_name": "init_state", # 视 provider 而定
                    "action_space": "computer_13",   # 或 computer_13
                    "require_a11y_tree": False,
                    "require_terminal": False,
                    "headless": True,
                },
                # 观测返回形式：text | screenshot_base64 | raw
                "observe_format": "screenshot_base64",
                "examples_dir": "/data/pengchen/OSWorld/Osworld/OSWorld/evaluation_examples/examples/chrome",
                "save_dir": "/data/pengchen/OSWorld/Osworld/OSWorld/results/pyautogui/screenshot/ui-tars/agentgym/chrome"
            },
            n_clients=1,
        )
    ],
)

exps = evaluator.eval(
    generation_config=GenerationConfig(
        max_length=4096,
        # eos_token_id=api_agent.client.eos_token_id,
        # pad_token_id=api_agent.client.pad_token_id or api_agent.client.eos_token_id,
    ),
    max_rounds=15,
    idxs=[0],  # 自行决定评测索引集合
)
print(exps.score, exps.success)
