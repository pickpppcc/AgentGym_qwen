from typing import Any, Callable, Mapping, Optional, Sequence
import json
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
            conversation = list(client.conversation_start)
            conversation.append(
                ConversationMessage({"from": "human", "loss": None, "value": state})
            )
            # 注意：VL 路径下不再对包含 screenshot_base64 的 JSON 文本做 tokenizer.encode，
            # 否则会把整段 Base64 当作纯文本 token 化，导致 10^6 量级的 token 长度。
            if not use_vl:
                conversation_tokenized = agent.chat_template.tokenize_conversation(
                    conversation, tokenizer, add_generation_prompt=True
                )
            else:
                conversation_tokenized = None  # type: ignore[assignment]
        elif isinstance(agent, APIAgent):
            conversation = [APIConversationMessage({"role": "user", "content": client.conversation_start[0]["value"], "reasoning_content": None}),
                            APIConversationMessage({"role": "assistant", "content": client.conversation_start[1]["value"], "reasoning_content": None}),
                            APIConversationMessage({"role": "user", "content": state, "reasoning_content": None})]
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

            for msg in conv:
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
                if screenshot_b64 and Image is not None:
                    try:
                        b64_str = str(screenshot_b64)
                        # 兼容 data:image/png;base64,xxx 形式
                        if "," in b64_str and b64_str.strip().startswith("data:"):
                            b64_str = b64_str.split(",", 1)[-1]
                        img_bytes = base64.b64decode(b64_str)
                        img = Image.open(BytesIO(img_bytes)).convert("RGB")
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
                    # if input_length exceeds max_length, break
                    if input_length >= (generation_config.max_length or 4096):
                        break
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

                # 统一在此处追加模型回复到对话记录中
                if isinstance(agent, Agent):
                    conversation.append(
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
                env_message = ConversationMessage(
                    {"from": "human", "loss": None, "value": state}
                )
                env_message_tokenized = agent.chat_template.tokenize_conversation_one(
                    env_message, tokenizer, add_generation_prompt=True
                )

                conversation.append(env_message)
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
