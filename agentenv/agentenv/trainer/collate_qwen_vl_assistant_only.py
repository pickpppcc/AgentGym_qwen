import json
from collections import defaultdict
from io import BytesIO
import base64
from typing import Any, Dict, List

import torch

try:
    from PIL import Image  # type: ignore[import]
except Exception:
    Image = None


def _parse_conversations_with_images(item: Dict[str, Any]):
    """
    从单条样本中解析：
      - conversations: 原始对话列表
      - 对每条 message 提取纯文本 text_val（展开 JSON 中的 instruction/step_text）
      - 可选的 screenshot_base64
    返回:
      messages: List[dict]，每个元素包含:
        {
          "role": "user" | "assistant",
          "text": str,
          "screenshot_b64": Optional[str],
        }
    """
    conversations = item["conversations"]
    messages: List[Dict[str, Any]] = []

    for message in conversations:
        raw_val = message["value"]
        text_val = raw_val
        screenshot_b64 = None

        # 兼容嵌套 JSON：{"instruction": ..., "step_text": ..., "screenshot_base64": ...}
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

        role = "user" if message["from"] == "human" else "assistant"
        messages.append(
            {
                "role": role,
                "text": str(text_val) if text_val is not None else "",
                "screenshot_b64": screenshot_b64,
            }
        )

    return messages


def _encode_single_sample_assistant_only(
    item: Dict[str, Any],
    tokenizer,
    processor,
    max_input_length: int,
):
    """
    将一条 OSWorld 样本编码为：
      - input_ids: List[int]
      - labels:    List[int]（user 段为 -100，仅 assistant 段有监督）
      - attention_mask: List[int]
      - pixel_values / image_grid_thw: 可选的视觉特征（使用 AutoProcessor.image_processor）

    文本模板采用简化版：
      - user:      "<s>[INST] {text} [/INST]"
      - assistant: " {text}</s>"
    与 BCTrainer.text-only 路径保持一致，方便构造 labels mask。
    """
    messages = _parse_conversations_with_images(item)

    # 使用内部 tokenizer，以兼容 Processor（如 Qwen2VLProcessor）
    tok = getattr(tokenizer, "tokenizer", tokenizer)

    input_ids: List[int] = []
    labels: List[int] = []
    attention_mask: List[int] = []

    vl_images = []

    for msg in messages:
        text_val = msg["text"]
        role = msg["role"]
        screenshot_b64 = msg.get("screenshot_b64")

        if role == "user":
            text = f"<s>[INST] {text_val} [/INST]"
            encode = tok.encode(text, add_special_tokens=False)
            input_ids.extend(encode)
            # user 段 label 全部为 -100
            labels.extend([-100] * len(encode))
        else:
            # assistant
            text = f" {text_val}"
            encode = tok.encode(text, add_special_tokens=False)
            encode += [tok.eos_token_id]
            input_ids.extend(encode)
            labels.extend(encode)

        # 仅在存在 screenshot_base64 时，尝试解码图像
        if screenshot_b64 and Image is not None:
            try:
                b64_str = str(screenshot_b64)
                if "," in b64_str and b64_str.strip().startswith("data:"):
                    b64_str = b64_str.split(",", 1)[-1]
                img_bytes = base64.b64decode(b64_str)
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                vl_images.append(img)
            except Exception:
                pass

    attention_mask = [1] * len(input_ids)

    # 保证非空序列，避免 RoPE / attention 内部 assert
    if len(input_ids) == 0:
        fallback_id = (
            getattr(tok, "eos_token_id", None)
            if getattr(tok, "eos_token_id", None) is not None
            else getattr(tok, "unk_token_id", 0)
        )
        input_ids = [fallback_id]
        labels = [-100]
        attention_mask = [1]

    # No truncation: keep the full token sequence as-is.

    pixel_values = None
    image_grid_thw = None

    # 使用 AutoProcessor.image_processor 生成视觉特征（若存在图像）
    if vl_images:
        try:
            img_processor = getattr(processor, "image_processor", processor)
            # 这里简单地取最后一帧截图作为该样本图像输入
            last_img = vl_images[-1]
            proc_outputs = img_processor(images=last_img, return_tensors="pt")
            pixel_values = proc_outputs["pixel_values"]  # [1, num_patches, dim] 或类似
            grid = proc_outputs.get("image_grid_thw", None)
            image_grid_thw = grid
        except Exception:
            pixel_values = None
            image_grid_thw = None

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }


def collate_qwen_vl_assistant_only(
    batch: List[Dict[str, Any]],
    tokenizer,
    processor,
    max_input_length: int,
):
    """
    针对 Qwen-VL 的 collate_fn，实现：
      - 仅监督 assistant 段（user 段 label 置为 -100）；
      - 可选多模态：若存在 screenshot_base64，则通过 image_processor 生成 pixel_values / image_grid_thw。

    与 BCTrainer 中 text-only 路径的行为保持一致，只是这里直接作用在原始 raw_dataset['train'] 上。
    """
    encoded_items = [
        _encode_single_sample_assistant_only(
            item,
            tokenizer=tokenizer,
            processor=processor,
            max_input_length=max_input_length,
        )
        for item in batch
    ]

    max_len = max(len(x["input_ids"]) for x in encoded_items)

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "unk_token_id", 0) or 0

    input_ids = []
    attention_mask = []
    labels = []

    for x in encoded_items:
        cur_input = x["input_ids"]
        cur_label = x["labels"]
        cur_mask = x["attention_mask"]

        pad_len = max_len - len(cur_input)
        input_ids.append(cur_input + [pad_id] * pad_len)
        attention_mask.append(cur_mask + [0] * pad_len)
        labels.append(cur_label + [-100] * pad_len)

    input_ids_t = torch.LongTensor(input_ids)
    attention_mask_t = torch.LongTensor(attention_mask)
    labels_t = torch.LongTensor(labels)

    forward_kwargs: Dict[str, Any] = {
        "input_ids": input_ids_t,
        "attention_mask": attention_mask_t,
        "labels": labels_t,
    }

    # 多模态：将 pixel_values / image_grid_thw 堆叠成 batch
    pixel_tensors = []
    grid_tensors = []
    for x in encoded_items:
        pv = x["pixel_values"]
        grid = x["image_grid_thw"]
        if pv is None or grid is None:
            continue
        # 保守地保证 batch 维度统一
        pixel_tensors.append(pv.to(dtype=torch.float32))
        grid_tensors.append(grid.to(dtype=torch.long))

    if pixel_tensors and grid_tensors:
        try:
            forward_kwargs["pixel_values"] = torch.cat(pixel_tensors, dim=0)
            forward_kwargs["image_grid_thw"] = torch.cat(grid_tensors, dim=0)
        except Exception:
            # 若拼接失败，则宁可不传图像特征，也不要中断训练
            pass

    return {"forward_kwargs": forward_kwargs}

