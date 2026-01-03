from typing import Any, Callable, Mapping, Optional, Sequence
import json
import re
import os
import math
from io import BytesIO
import base64

from transformers import GenerationConfig

from . import Agent, APIAgent, BaseEnvClient
from .types import ConversationMessage, APIConversationMessage, ExperienceOutput, APIExperienceOutput

try:
    from PIL import Image  # type: ignore[import]
except Exception:
    # Pillow 仅在视觉语言多模态推理路径下需要；若不存在则退化为纯文本推理。
    Image = None


class BaseTask:
    env_client_cls: Callable
    env_name: str

    def __init__(
        self,
        client_args: Mapping[str, Any],
        n_clients: int = 1,
    ) -> None:
        """
        Initializes the Task object.

        Args:
            client_args (Mapping[str, Any]): A mapping of client arguments.
            n_clients (int, optional): The number of clients. Defaults to 1. Larger than 1 for batch generation. Batch generation is not implemented yet.
        """
        if self.env_client_cls is None or self.env_name is None:
            raise NotImplementedError
        self.clients = [self.env_client_cls(**client_args) for _ in range(n_clients)]
        self.len = len(self.clients[0])

    def _generate_experience_one(
        self,
        agent: Agent | APIAgent,
        client: BaseEnvClient,
        idx: int,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> ExperienceOutput:
        client.reset(idx)
        reward = 0.0
        done = False
        state = client.observe()

        def _uitars_system_prompt() -> str:
            return (
                "You are UITARS, a GUI agent that controls a real desktop environment.\n\n"
                "Your response is parsed by a strict program. Any extra text or formatting may break execution.\n\n"
                "OUTPUT (exactly 2 lines; no Markdown; no code fences):\n"
                "Thought: <1–2 sentences; follow the language requested in the user prompt; include a tiny plan; end with 'Next: ...'>\n"
                "Action: <exactly ONE action call>\n\n"
                "RULES:\n"
                "- Include exactly one 'Thought:' line and exactly one 'Action:' line.\n"
                "- The Action line must be a single function call from the provided action space.\n"
                "- Never output multiple actions, numbered steps, bullet lists, or explanations.\n"
                "- Coordinates use UI-TARS normalized space: x,y in [0,1000] over a 1920x1080 desktop; use numbers like (500,500).\n"
                "- For click/left_double/right_single/drag/scroll: use start_box/end_box as plain '(x,y)' numeric pairs; "
                "do NOT include '<|box_start|>' tokens.\n"
                "- For type(content=...): use a single-quoted string; use \\\\n to press Enter; escape quotes if needed.\n"
                "- If waiting for the UI, output Action: wait().\n"
                "- Only output Action: finished() when you are confident the user instruction is fully satisfied; if unsure, continue interacting instead.\n"
                "- If the observation says finished() was rejected / not successful yet, do NOT repeat finished(); take additional actions to satisfy the requirements.\n"
                "- If impossible or you need user help, output Action: call_user()."
            )

        def _sanitize_obs_text(raw_val: str) -> str:
            """
            OSWorld VL 观测可能是包含 screenshot_base64 的 JSON 字符串；这里抽取 instruction/step_text，
            避免把整段 Base64 token 化造成超长序列。
            """
            if not isinstance(raw_val, str):
                return str(raw_val)
            s = raw_val.strip()
            if not s.startswith("{"):
                return raw_val
            try:
                inner = json.loads(s)
                if not isinstance(inner, dict):
                    return raw_val
                instr = inner.get("instruction") or ""
                step_text = inner.get("step_text") or ""
                merged_txt = (str(instr).strip() + "\n" + str(step_text).strip()).strip()
                return merged_txt or raw_val
            except Exception:
                return raw_val

        def _split_system_from_conversation_start(
            conv_start: Sequence[ConversationMessage],
        ) -> tuple[str | None, list[ConversationMessage]]:
            """
            OSWorld sometimes stores the strict system prompt as the first 'human' message
            for compatibility with text-only templates. Promote it to a real system prompt
            for API/VL paths and drop it from user content.
            """
            if not conv_start:
                return None, []
            first = conv_start[0]
            first_from = str(first.get("from", ""))
            first_val = str(first.get("value", ""))

            if first_from == "system":
                return first_val, list(conv_start[1:])

            looks_like_uitars_sys = (
                first_from == "human"
                and first_val.strip().startswith("You are UITARS")
                and "OUTPUT (exactly 2 lines" in first_val
            )
            if looks_like_uitars_sys:
                return first_val, list(conv_start[1:])

            return None, list(conv_start)

        def _coerce_uitars_format(text: str) -> str:
            """
            Best-effort: force model output into exactly:
              Thought: ...
              Action: ...
            If Action cannot be recovered, fall back to wait().
            """
            if not isinstance(text, str):
                text = str(text)
            raw = text.replace("\r\n", "\n").replace("\r", "\n").strip()

            thought = ""
            action = ""

            m_th = re.search(r"(?is)\bThought:\s*(.*?)(?:\n\s*Action:|\Z)", raw)
            if m_th:
                thought = (m_th.group(1) or "").strip()

            m_act = re.search(r"(?is)\bAction:\s*(.*)", raw)
            if m_act:
                action = (m_act.group(1) or "").strip()

            allowed_action_call = re.compile(
                r"(?is)^(?:"
                r"(?:click|left_double|right_single|drag|hotkey|type|scroll)\s*\([^\n]*\)"
                r"|wait\s*\(\s*\)"
                r"|finished\s*\([^\n]*\)"
                r"|call_user\s*\(\s*\)"
                r")$"
            )

            if action:
                # Prefer the first non-empty line as the single action line.
                first_line = ""
                for line in action.split("\n"):
                    line = line.strip()
                    if line:
                        first_line = line
                        break
                action = first_line

            if action and not allowed_action_call.match(action):
                action = ""

            if not action:
                # Recover the first valid action call found anywhere in the output.
                m_fn = re.search(
                    r"(?is)\b(?:"
                    r"(?:click|left_double|right_single|drag|hotkey|type|scroll)\s*\([^\n]*\)"
                    r"|wait\s*\(\s*\)"
                    r"|finished\s*\([^\n]*\)"
                    r"|call_user\s*\(\s*\)"
                    r")\b",
                    raw,
                )
                if m_fn:
                    cand = m_fn.group(0).strip()
                    if allowed_action_call.match(cand):
                        action = cand

            if not action:
                action = "wait()"

            if not thought:
                thought = "I will follow the required format and perform the next GUI action."

            thought = " ".join(thought.split())
            return f"Thought: {thought}\nAction: {action}"

        # 仅在「视觉语言模型 + OSWorld 截图观测」时，走 AutoProcessor + model.generate 路径；
        # 其他任务/模型保持原有文本-only 流程不变。
        use_vl = (
            isinstance(agent, Agent)
            and getattr(agent, "is_vision_language", False)
            and getattr(agent, "processor", None) is not None
            and getattr(client, "observe_format", None) == "screenshot_base64"
        )

        if isinstance(agent, Agent):
            tokenizer = getattr(agent.tokenizer, "tokenizer", agent.tokenizer)
            conversation_text_only: list[ConversationMessage] | None = None
            conversation = list(client.conversation_start)
            conversation.append(
                ConversationMessage({"from": "human", "loss": None, "value": state})
            )
            # 注意：VL 路径下不要 token 化 screenshot_base64；改为仅抽取可读文本用于 token 统计/回放。
            if use_vl:
                conversation_text_only = list(client.conversation_start)
                conversation_text_only.append(
                    ConversationMessage(
                        {"from": "human", "loss": None, "value": _sanitize_obs_text(state)}
                    )
                )
                conversation_tokenized = agent.chat_template.tokenize_conversation(
                    conversation_text_only, tokenizer, add_generation_prompt=True
                )
            else:
                conversation_tokenized = agent.chat_template.tokenize_conversation(
                    conversation, tokenizer, add_generation_prompt=True
                )
        elif isinstance(agent, APIAgent):
            sys_text, conv_start_rest = _split_system_from_conversation_start(
                list(client.conversation_start)
            )
            conversation = []
            if sys_text is None and self.env_name == "OSWorld":
                sys_text = _uitars_system_prompt()
            if sys_text is not None:
                conversation.append(
                    APIConversationMessage(
                        {"role": "system", "content": sys_text, "reasoning_content": None}
                    )
                )

            role_map = {"system": "system", "human": "user", "gpt": "assistant"}
            for m in conv_start_rest:
                role = role_map.get(m.get("from", "human"), "user")
                conversation.append(
                    APIConversationMessage(
                        {"role": role, "content": str(m.get("value", "")), "reasoning_content": None}
                    )
                )
            conversation.append(
                APIConversationMessage(
                    {
                        "role": "user",
                        "content": _sanitize_obs_text(state) if self.env_name == "OSWorld" else state,
                        "reasoning_content": None,
                    }
                )
            )
        else:
            raise NotImplementedError
        rounds = 0

        def _build_qwen_vl_messages(conv: list[ConversationMessage]) -> tuple[list[dict[str, Any]], list[Any]]:
            """
            将 BaseTask 的会话格式转成 Qwen-VL / Qwen2-VL 期望的 messages + images：
            - messages: [{\"role\": \"user\"|\"assistant\", \"content\": [{\"type\": \"text\"|\"image\", ...}, ...]}, ...]
            - images: 与 messages 对应的 PIL.Image 列表（主要用于调试，可选）
            """
            messages: list[dict[str, Any]] = []
            images_for_conv: list[Any] = []

            sys_text, conv_rest = _split_system_from_conversation_start(conv)
            if sys_text is None:
                sys_text = _uitars_system_prompt()
            # IMPORTANT: Ensure the model always receives the strict UITARS formatting instruction.
            # Without this, VL models often emit only a Thought line and omit the Action line.
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": sys_text}],
                }
            )

            # OSWorld-style: keep only the most recent N screenshots to avoid OOM as the episode grows.
            # Defaults match OSWorld agents (history_n=5).
            try:
                max_image_history = int(os.environ.get("AGENTGYM_OSWORLD_HISTORY_N", "5"))
            except Exception:
                max_image_history = 5
            if max_image_history < 0:
                max_image_history = 0

            # OSWorld-style pixel budget (prevents extremely large frames from exploding visual tokens).
            try:
                max_pixels = int(os.environ.get("AGENTGYM_OSWORLD_MAX_PIXELS", str(16384 * 28 * 28)))
            except Exception:
                max_pixels = 16384 * 28 * 28
            try:
                min_pixels = int(os.environ.get("AGENTGYM_OSWORLD_MIN_PIXELS", str(100 * 28 * 28)))
            except Exception:
                min_pixels = 100 * 28 * 28

            # Pre-scan which turns contain screenshots; only decode the last N to save memory/time.
            screenshot_turn_idxs: list[int] = []
            if max_image_history > 0:
                for i, msg in enumerate(conv_rest):
                    raw_val = msg.get("value")
                    if isinstance(raw_val, str) and raw_val.strip().startswith("{"):
                        try:
                            inner = json.loads(raw_val)
                            if isinstance(inner, dict) and inner.get("screenshot_base64"):
                                screenshot_turn_idxs.append(i)
                        except Exception:
                            pass
            if max_image_history > 0:
                keep_idxs = set(screenshot_turn_idxs[-max_image_history:])
            else:
                keep_idxs = set()

            for i, msg in enumerate(conv_rest):
                role = "user" if msg["from"] == "human" else "assistant"
                raw_val = msg["value"]
                text_val = raw_val
                screenshot_b64 = None

                # 兼容 OSWorld observe_format == "screenshot_base64" 返回的 JSON：
                # {"instruction": ..., "step_text": ..., "screenshot_base64": ...}
                if isinstance(raw_val, str) and raw_val.strip().startswith("{"):
                    try:
                        inner = json.loads(raw_val)
                        if isinstance(inner, dict):
                            instr = inner.get("instruction") or ""
                            step_text = inner.get("step_text") or ""
                            merged_txt = (str(instr).strip() + "\n" + str(step_text).strip()).strip()
                            if merged_txt:
                                text_val = merged_txt
                            screenshot_b64 = inner.get("screenshot_base64")
                    except Exception:
                        pass

                content_parts: list[dict[str, Any]] = []
                img = None
                if screenshot_b64 and Image is not None and (i in keep_idxs):
                    try:
                        b64_str = str(screenshot_b64)
                        # 兼容 data:image/png;base64,xxx 形式
                        if "," in b64_str and b64_str.strip().startswith("data:"):
                            b64_str = b64_str.split(",", 1)[-1]
                        img_bytes = base64.b64decode(b64_str)
                        img = Image.open(BytesIO(img_bytes)).convert("RGB")

                        # Resize to keep pixels within [min_pixels, max_pixels] (OSWorld-style).
                        try:
                            cur_pixels = int(img.width) * int(img.height)
                            if max_pixels and cur_pixels > int(max_pixels):
                                resize_factor = math.sqrt(float(max_pixels) / float(cur_pixels))
                                w = max(1, int(img.width * resize_factor))
                                h = max(1, int(img.height * resize_factor))
                                img = img.resize((w, h))
                            cur_pixels = int(img.width) * int(img.height)
                            if min_pixels and cur_pixels < int(min_pixels):
                                resize_factor = math.sqrt(float(min_pixels) / float(cur_pixels))
                                w = max(1, int(math.ceil(img.width * resize_factor)))
                                h = max(1, int(math.ceil(img.height * resize_factor)))
                                img = img.resize((w, h))
                            if img.mode != "RGB":
                                img = img.convert("RGB")
                        except Exception:
                            pass

                        content_parts.append({"type": "image", "image": img})
                    except Exception:
                        img = None

                if text_val:
                    content_parts.append({"type": "text", "text": str(text_val)})

                if not content_parts:
                    # 至少保证有一个空文本，避免 processor 崩溃
                    content_parts.append({"type": "text", "text": ""})

                messages.append({"role": role, "content": content_parts})
                if img is not None:
                    images_for_conv.append(img)

            return messages, images_for_conv

        while not done:
            if isinstance(agent, Agent):
                if use_vl:
                    # 视觉语言分支：仅在 is_vision_language=True 且存在 AutoProcessor 时启用。
                    try:
                        proc = getattr(agent, "processor", None)
                        if proc is None:
                            raise RuntimeError("VL path requires agent.processor, but got None.")

                        messages, images_for_conv = _build_qwen_vl_messages(conversation)
                        # 优先使用 AutoProcessor 内部的 tokenizer 提供的 chat_template
                        tok_for_template = getattr(proc, "tokenizer", proc)
                        try:
                            prompt = tok_for_template.apply_chat_template(
                                messages,
                                add_generation_prompt=True,
                                tokenize=False,
                            )
                        except Exception:
                            # 回退：简单拼接所有文本片段，保证不为空
                            text_parts: list[str] = []
                            try:
                                if sys_text:
                                    text_parts.append(str(sys_text).strip())
                                for m in messages:
                                    for part in m.get("content", []):
                                        if isinstance(part, dict) and part.get("type") == "text":
                                            t = str(part.get("text", "")).strip()
                                            if t:
                                                text_parts.append(t)
                            except Exception:
                                text_parts = []
                            prompt = "\n".join(text_parts).strip() or (str(state) if state is not None else "")

                        # 使用 AutoProcessor 构造多模态输入；只要当前对话中存在图像，就一起传入
                        if images_for_conv:
                            model_inputs = proc(
                                text=[prompt],
                                images=[images_for_conv],
                                return_tensors="pt",
                            )
                        else:
                            model_inputs = proc(
                                text=[prompt],
                                return_tensors="pt",
                            )

                        # 移动到模型所在设备
                        model_inputs = {
                            k: v.to(agent.model.device) for k, v in model_inputs.items()
                        }
                        prompt_len = model_inputs["input_ids"].shape[1]

                        output = agent.model.generate(
                            **model_inputs,
                            generation_config=generation_config,
                        )
                        # 默认 generate 返回 Tensor；保持与 Agent.generate 一致的处理方式
                        seq = output[0]
                        gen_tokens = seq[prompt_len:]
                        generated_text = agent.tokenizer.decode(
                            gen_tokens,
                            skip_special_tokens=True,
                        )
                    except Exception as e:  # pylint: disable=W0718:broad-exception-caught
                        print(e)
                        break  # 发生异常则中止当前 episode，避免影响其它任务
                else:
                    input_length = len(conversation_tokenized["input_ids"])
                    try:
                        generated_tokens = agent.generate(
                            [conversation_tokenized["input_ids"]], generation_config
                        )[0]
                    except Exception as e:  # pylint: disable=W0718:broad-exception-caught
                        print(e)
                        break  # break if generate method raises exceptions

                    eos_id = getattr(getattr(agent.model, "generation_config", None), "eos_token_id", None)
                    if eos_id is None:
                        eos_id = getattr(tokenizer, "eos_token_id", None)
                    if generated_tokens[-1] != eos_id and eos_id is not None:
                        generated_tokens += [eos_id]

                    generated_text = tokenizer.decode(generated_tokens)
                    conversation_tokenized["text"] += f" {generated_text}"
                    conversation_tokenized["input_ids"] += generated_tokens
                    conversation_tokenized["action_mask"] += [1] * len(generated_tokens)

                    # 移除尾部 eos 文本（如存在）
                    try:
                        eos_tok = getattr(tokenizer, "eos_token", None)
                        if eos_tok and generated_text.endswith(eos_tok):
                            generated_text = generated_text[: -len(eos_tok)]
                    except Exception:
                        pass

                if self.env_name == "OSWorld":
                    generated_text = _coerce_uitars_format(generated_text)

                # 统一在此处追加模型回复到对话记录中
                if isinstance(agent, Agent):
                    conversation.append(
                        ConversationMessage(
                            {"from": "gpt", "loss": True, "value": generated_text}
                        )
                    )
                    if use_vl and conversation_text_only is not None:
                        conversation_text_only.append(
                            ConversationMessage(
                                {"from": "gpt", "loss": True, "value": generated_text}
                            )
                        )
            elif isinstance(agent, APIAgent):
                generated_text, generated_reasoning_text = agent.generate(conversation)
                if generated_reasoning_text:
                    try:
                        print(f"[Thought] {generated_reasoning_text}")
                    except Exception:
                        pass
                if self.env_name == "OSWorld":
                    generated_text = _coerce_uitars_format(generated_text)
                conversation.append(
                    APIConversationMessage(
                        {"role": "assistant", "content": generated_text, "reasoning_content": generated_reasoning_text}
                    )
                )
            else:
                raise NotImplementedError

            step_output = client.step(generated_text)
            state, reward, done = (
                step_output.state,
                step_output.reward,
                step_output.done,
            )

            if isinstance(agent, Agent):
                env_message_raw = ConversationMessage(
                    {"from": "human", "loss": None, "value": state}
                )
                env_message_for_tok = (
                    ConversationMessage(
                        {"from": "human", "loss": None, "value": _sanitize_obs_text(state)}
                    )
                    if use_vl
                    else env_message_raw
                )
                env_message_tokenized = agent.chat_template.tokenize_conversation_one(
                    env_message_for_tok, tokenizer, add_generation_prompt=True
                )

                conversation.append(env_message_raw)
                if use_vl and conversation_text_only is not None:
                    conversation_text_only.append(env_message_for_tok)
                    conversation_tokenized = agent.chat_template.tokenize_conversation(
                        conversation_text_only, tokenizer, add_generation_prompt=True
                    )
                else:
                    conversation_tokenized["text"] += env_message_tokenized["text"]
                    conversation_tokenized["input_ids"] += env_message_tokenized["input_ids"]
                    conversation_tokenized["action_mask"] += env_message_tokenized[
                        "action_mask"
                    ]
            elif isinstance(agent, APIAgent):
                conversation.append(
                    APIConversationMessage(
                        {"role": "user", "content": state, "reasoning_content": None}
                    )
                )
            else:
                raise NotImplementedError

            rounds += 1
            if max_rounds is not None and rounds >= max_rounds:
                break

        if isinstance(agent, Agent):
            # Best-effort per-episode logging for OSWorld: print task id and final score.
            # This is helpful during batch evaluation to see progress beyond tqdm task ids.
            try:
                if self.env_name == "OSWorld":
                    rank = str(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
                    if rank == "0":
                        task_id = getattr(client, "_current_task_id", None)
                        status = "done" if done else "truncated" if (max_rounds is not None and rounds >= max_rounds) else "stopped"
                        print(f"[OSWorld] task_id={task_id} idx={idx} status={status} score={float(reward):.3f}")
            except Exception:
                pass
            return ExperienceOutput(
                conversation=conversation,
                reward=reward,
                text=conversation_tokenized["text"],
                seq_ids=conversation_tokenized["input_ids"],
                attention_mask=[1] * len(conversation_tokenized["input_ids"]),
                action_mask=conversation_tokenized["action_mask"],
            )
        elif isinstance(agent, APIAgent):
            return APIExperienceOutput(
                conversation=conversation,
                reward=reward,
            )
        else:
            raise NotImplementedError

    def _generate_experience_batch(
        self,
        agent: Agent | APIAgent,
        idxs: Sequence[int],
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> list[ExperienceOutput]:
        client = self.clients[0]
        result = [
            self._generate_experience_one(
                agent=agent,
                client=client,
                idx=idx,
                generation_config=generation_config,
                max_rounds=max_rounds,
            )
            for idx in idxs
        ]
        return result

    def generate_experience(
        self,
        agent: Agent | APIAgent,
        idxs: Sequence[int] | int,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> list[ExperienceOutput]:
        if isinstance(idxs, int):
            idxs = [idxs]

        return self._generate_experience_batch(
            agent=agent,
            idxs=idxs,
            generation_config=generation_config,
            max_rounds=max_rounds,
        )
