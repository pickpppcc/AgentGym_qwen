import json
import shutil
import os
from collections import defaultdict
from dataclasses import asdict
from datetime import timedelta
from functools import partial
from typing import Sequence, Optional

from io import BytesIO
import base64

import jsonlines
import numpy as np
import torch
import wandb
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs, DeepSpeedPlugin

from accelerate.utils import broadcast, gather_object
from agentenv.controller import Agent
from agentenv.controller.agent import Agent
from agentenv.controller.task import BaseTask
from agentenv.controller.utils import BaseTrainer
from agentenv.trainer.utils import set_seed
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
from transformers import GenerationConfig, get_linear_schedule_with_warmup

try:
    from PIL import Image  # type: ignore[import]
except Exception:
    # Pillow 仅在 VL 多模态路径下才需要；若不存在则退化为纯文本训练。
    Image = None


class BCTrainer(BaseTrainer):
    def __init__(
        self,
        agent: Agent,
        tasks: Sequence[BaseTask],
        args,
        processor: Optional[object] = None,
        is_vision_language: bool = False,
    ) -> None:
        self.agent = agent
        self.tasks = tasks
        self.args = asdict(args)

        # 多模态相关：VL 模型会传入 AutoProcessor 和标志位
        self.processor = processor
        self.is_vision_language = bool(is_vision_language and processor is not None)

        # data & loader
        self.train_dataset = None
        self.train_dataloader = None
        self.test_dataloader = None

        # accelerator
        self.accelerator = None

        # train in parallel
        self.optimizer = None
        self.scheduler = None

        # log dict
        self.best_eval_log_dict = {}
        self.summary_log_dict = {}

        self.create_accelerator()
        self.set_seed()
        self.setup_tokenizer()
        self.get_raw_dataset()
        self.get_train_dataloader()
        self.get_inference_test_dataloader()
        self.setup_wandb()
        self.init_train_stuff()

    def create_accelerator(self):
        """
        Create the accelerator.
        """
        common_kwargs = [InitProcessGroupKwargs(timeout=timedelta(seconds=18000))]
        if self.args.get("use_deepspeed", False):
            # 使用 ZeRO-3 + torch_adam，避免编译自定义算子以及 CUDA 版本不匹配
            ds_config = {
                "train_micro_batch_size_per_gpu": self.args["batch_size"],
                "gradient_accumulation_steps": self.args["gradient_accumulation_steps"],
                "zero_optimization": {
                    "stage": 3,
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    # 启用优化器状态 CPU offload 缓解显存压力
                    "offload_optimizer": {"device": "cpu", "pin_memory": True},
                    "offload_param": {"device": "none"},
                    # 允许保存时聚合 16bit 权重，便于 unwrap 并保存
                    "gather_16bit_weights_on_model_save": True,
                },
                # 根据 torch_dtype 自动选择 bf16
                "bf16": {"enabled": True},
                "fp16": {"enabled": False},
                # 激活检查点：用算力换显存，显著降低长序列下的激活占用
                "activation_checkpointing": {
                    "partition_activations": True,
                    "contiguous_memory_optimization": False,
                    "cpu_checkpointing": True,
                    "number_checkpoints": None,
                    "synchronize_checkpoint_boundary": False,
                    "profile": False
                },
                # 顶层启用 torch_adam，强制使用 torch.optim 以避免自定义算子编译
                "torch_adam": True,
                # 强制使用 PyTorch 原生 Adam（AdamW 模式），避免 FusedAdam/CPUAdam 编译
                "optimizer": {
                    "type": "Adam",
                    "params": {
                        "lr": self.args["learning_rate"],
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": self.args["weight_decay"],
                        "adam_w_mode": True
                    }
                },
                "gradient_clipping": float(self.args.get("clip_grad_norm", 1.0)),
                "aio": {"enabled": False},
                "steps_per_print": 0,
                "wall_clock_breakdown": False,
            }
            ds_plugin = DeepSpeedPlugin(
                hf_ds_config=ds_config,
                zero_stage=3,
                gradient_accumulation_steps=self.args["gradient_accumulation_steps"],
                gradient_clipping=float(self.args.get("clip_grad_norm", 1.0)),
            )
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.args["gradient_accumulation_steps"],
                kwargs_handlers=common_kwargs,
                deepspeed_plugin=ds_plugin,
            )
            # 明确打印 DeepSpeed 状态，便于外部日志确认
            try:
                zero_stage = getattr(self.accelerator.state.deepspeed_plugin, "zero_stage", None)
            except Exception:
                zero_stage = None
            self.accelerator.print(f"[Accelerate] DeepSpeed enabled: True | zero_stage={zero_stage} | grad_accum_steps={self.args['gradient_accumulation_steps']}")
        else:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.args["gradient_accumulation_steps"],
                kwargs_handlers=common_kwargs + [
                    DistributedDataParallelKwargs(find_unused_parameters=True)
                ],
            ) 
            self.accelerator.print("[Accelerate] DeepSpeed enabled: False | Using DDP")

    def set_seed(self):
        """
        Set the random seed.
        """
        set_seed(self.args["seed"] + self.accelerator.process_index)

    def setup_tokenizer(self):
        """
        Setup the tokenizer.
        """
        # Avoid hard-coding special token ids; only set when missing
        try:
            if getattr(self.agent.tokenizer, "pad_token_id", None) is None:
                unk_id = getattr(self.agent.tokenizer, "unk_token_id", None)
                eos_id = getattr(self.agent.tokenizer, "eos_token_id", None)
                self.agent.tokenizer.pad_token_id = (
                    unk_id if unk_id is not None else eos_id
                )
            if getattr(self.agent.tokenizer, "eos_token_id", None) is None:
                if hasattr(self.agent.tokenizer, "convert_tokens_to_ids"):
                    maybe_eos = self.agent.tokenizer.convert_tokens_to_ids("</s>")
                    if isinstance(maybe_eos, int) and maybe_eos >= 0:
                        self.agent.tokenizer.eos_token_id = maybe_eos
        except Exception:
            pass
        # Robust vocab size handling for processors without __len__
        tok = self.agent.tokenizer
        vocab_size = None
        if hasattr(tok, "vocab_size"):
            vocab_size = tok.vocab_size
        elif hasattr(tok, "get_vocab"):
            try:
                vocab_size = len(tok.get_vocab())
            except Exception:
                vocab_size = None
        elif hasattr(tok, "tokenizer") and hasattr(tok.tokenizer, "vocab_size"):
            vocab_size = tok.tokenizer.vocab_size
        if vocab_size is None:
            model_config = getattr(getattr(self.agent, "model", None), "config", None)
            vocab_size = getattr(model_config, "vocab_size", None)

        self.accelerator.print(
            f"[Vocab size]: {vocab_size if vocab_size is not None else 'unknown'}"
        )
        # Only expand embeddings; never shrink to avoid OOB indices
        try:
            current_embed = None
            try:
                current_embed = (
                    self.agent.model.get_input_embeddings().weight.shape[0]
                )
            except Exception:
                pass
            if (
                vocab_size is not None
                and isinstance(current_embed, int)
                and vocab_size > current_embed
            ):
                self.agent.model.resize_token_embeddings(vocab_size)
        except Exception:
            pass

    def get_raw_dataset(self):
        # 在单进程/未初始化分布式的情况下，避免进入 barrier
        if getattr(self.accelerator, "num_processes", 1) == 1:
            from contextlib import nullcontext
            mpf_ctx = nullcontext()
        else:
            mpf_ctx = self.accelerator.main_process_first()

        with mpf_ctx:
            def _load_list_from_path(path):
                path = os.path.expanduser(path)
                ext = os.path.splitext(path)[1].lower()
                # JSONL family
                if ext in [".jsonl", ".jsonlines", ".jl"]:
                    items = []
                    with jsonlines.open(path, mode="r") as reader:
                        for obj in reader:
                            if isinstance(obj, dict):
                                items.append(obj)
                    return items
                # JSON
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, list):
                    return [x for x in obj if isinstance(x, dict)]
                if isinstance(obj, dict):
                    # Try common keys
                    for key in ["data", "examples", "train", "inference", "test"]:
                        v = obj.get(key)
                        if isinstance(v, list):
                            return [x for x in v if isinstance(x, dict)]
                    # Fallback: first list value
                    for v in obj.values():
                        if isinstance(v, list):
                            return [x for x in v if isinstance(x, dict)]
                    raise ValueError(
                        f"Unsupported JSON structure in {path}. Expected list[dict] or dict with a list field."
                    )
                raise ValueError(
                    f"Unsupported file content in {path}. Expected JSON/JSONL containing list[dict]."
                )

            train_list = _load_list_from_path(self.args["train_file"])
            infer_list = _load_list_from_path(self.args["inference_file"])
            test_list = _load_list_from_path(self.args["test_file"])

            # limit train samples for quick smoke test if specified
            max_samples = self.args.get("train_max_samples")
            if isinstance(max_samples, int) and max_samples > 0:
                train_list = train_list[: max_samples]

            self.raw_dataset = DatasetDict(
                {
                    "train": Dataset.from_list(train_list),
                    "inference": Dataset.from_list(infer_list),
                    "test": Dataset.from_list(test_list),
                }
            )
            self.accelerator.print(
                "Raw sizes:",
                {k: len(v) for k, v in self.raw_dataset.items()},
            )

    def get_train_dataloader(self):
        """
        create train_dataset and train_dataloader
        """

        def tokenize_fn(batch, args, tokenizer, processor=None, is_vision_language: bool = False):
            # tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content | trim + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content | trim + ' ' + eos_token }}{% endif %}{% endfor %}"
            # 使用内部 tokenizer 以兼容 Processor（如 Qwen2VLProcessor）
            tok = getattr(tokenizer, "tokenizer", tokenizer)
            assert getattr(tok, "eos_token_id", None) is not None, (
                getattr(tok, "eos_token_id", None),
                getattr(tok, "eos_token", None),
            )
            new_batch = defaultdict(list)
            all_keys = list(batch.keys())
            use_vl = bool(is_vision_language and processor is not None)

            for item_values in zip(*(batch[k] for k in all_keys)):
                item = {k: item_values[i] for i, k in enumerate(all_keys)}
                item_id, conversations = (item["item_id"], item["conversations"])

                input_ids = []
                labels = []

                # 对于 VL 模型，我们尝试从对话中解析 screenshot_base64，
                # 将它解码成图像，后续交给 AutoProcessor 生成 pixel_values。
                vl_images = []

                for message in conversations:
                    raw_val = message["value"]
                    text_val = raw_val
                    screenshot_b64 = None

                    if use_vl and isinstance(raw_val, str) and raw_val.strip().startswith("{"):
                        # 尝试解析嵌套 JSON，形如 {"instruction": ..., "step_text": ..., "screenshot_base64": ...}
                        try:
                            inner = json.loads(raw_val)
                            if isinstance(inner, dict):
                                instr = inner.get("instruction") or ""
                                step_text = inner.get("step_text") or ""
                                # 只保留与任务相关的文本，避免把整段巨大 JSON 直接当成 prompt
                                merged_txt = (str(instr).strip() + "\n" + str(step_text).strip()).strip()
                                if merged_txt:
                                    text_val = merged_txt
                                screenshot_b64 = inner.get("screenshot_base64")
                        except Exception:
                            pass

                    # 文本 token 化逻辑保持原样，只是用 text_val 替代原始 value
                    if message["from"] == "human":
                        text = f"<s>[INST] {text_val} [/INST]"
                        input_encode = tok.encode(text, add_special_tokens=False)
                        input_ids.extend(input_encode)
                        labels.extend([-100] * len(input_encode))
                    else:
                        # message["from"] == "gpt":
                        # text = f" {message['value']}</s>"
                        text = f" {text_val}"
                        input_encode = tok.encode(text, add_special_tokens=False)
                        input_encode += [tok.eos_token_id]
                        input_ids.extend(input_encode)
                        labels.extend(input_encode)

                    # 仅在 VL 模式且存在 screenshot_base64 时，尝试解码图像
                    if use_vl and screenshot_b64 and Image is not None:
                        try:
                            b64_str = str(screenshot_b64)
                            # 兼容 data:image/...;base64,xxx 形式
                            if "," in b64_str and b64_str.strip().startswith("data:"):
                                b64_str = b64_str.split(",", 1)[-1]
                            img_bytes = base64.b64decode(b64_str)
                            img = Image.open(BytesIO(img_bytes)).convert("RGB")
                            vl_images.append(img)
                        except Exception:
                            # 解码失败则忽略该帧
                            pass

                attention_mask = [1] * len(input_ids)

                # Ensure non-empty sequence to avoid device-side assert in RoPE indexing
                if len(input_ids) == 0:
                    fallback_id = (
                        getattr(tok, "eos_token_id", None)
                        if getattr(tok, "eos_token_id", None) is not None
                        else getattr(tok, "unk_token_id", 0)
                    )
                    input_ids = [fallback_id]
                    labels = [-100]
                    attention_mask = [1]

                # Truncation
                input_ids_max_length = len(input_ids)
                # assert input_ids_max_length <= args['max_input_length'], input_ids_max_length
                input_ids = input_ids[: args["max_input_length"]]
                labels = labels[: args["max_input_length"]]
                attention_mask = attention_mask[: args["max_input_length"]]

                ##
                new_batch["input_ids"].append(input_ids)
                new_batch["labels"].append(labels)
                new_batch["attention_mask"].append(attention_mask)

                # 多模态：目前简单地使用“最后一帧截图”作为该样本对应的图像输入，
                # 后续在 collate_fn 中堆叠成 batch，并连同 image_grid_thw 一起传给模型。
                if use_vl and vl_images:
                    try:
                        last_img = vl_images[-1]
                        img_processor = getattr(processor, "image_processor", processor)
                        # 使用 Qwen2VLImageProcessor 直接对单张图像做预处理，得到：
                        # - pixel_values: [num_patches, dim]
                        # - image_grid_thw: [1, 3]，记录该图像在视觉网格上的 (T, H, W)
                        proc_outputs = img_processor(images=last_img, return_tensors="pt")
                        pv = proc_outputs["pixel_values"].numpy().tolist()
                        grid = proc_outputs.get("image_grid_thw", None)
                        grid_list = (
                            grid.numpy().tolist() if grid is not None else None
                        )
                        new_batch["pixel_values"].append(pv)
                        new_batch["image_grid_thw"].append(grid_list)
                    except Exception:
                        # processor 失败时，不终止整个 batch 的处理
                        new_batch["pixel_values"].append(None)
                        new_batch["image_grid_thw"].append(None)
                elif use_vl:
                    # 统一字段结构：即便没有图像，也补一个占位，以便后续逻辑判断
                    new_batch["pixel_values"].append(None)
                    new_batch["image_grid_thw"].append(None)

                ##
                new_batch["item_id"].append(item_id)
                new_batch["input_ids_max_length"].append(input_ids_max_length)

            return new_batch

        tokenized_dataset = DatasetDict(
            {
                "train": self.raw_dataset["train"].map(
                    tokenize_fn,
                    fn_kwargs={
                        "args": self.args,
                        "tokenizer": self.agent.tokenizer,
                        "processor": self.processor,
                        "is_vision_language": self.is_vision_language,
                    },
                    batched=True,
                    remove_columns=self.raw_dataset["train"].column_names,
                    num_proc=4,
                    load_from_cache_file=False,
                )
            }
        )
        self.accelerator.print("Processed data:", tokenized_dataset)
        for mode, dataset in tokenized_dataset.items():
            self.accelerator.print(
                f"\n{mode}_input_ids_max_length",
                max(dataset["input_ids_max_length"]),
            )

        def collate_fn(batch, tokenizer, processor=None, is_vision_language: bool = False):
            max_input_length = max([len(item["input_ids"]) for item in batch])
            input_ids = []
            attention_mask = []
            labels = []

            for item in batch:
                input_ids.append(
                    item["input_ids"]
                    + [tokenizer.pad_token_id]
                    * (max_input_length - len(item["input_ids"]))
                )
                attention_mask.append(
                    item["attention_mask"]
                    + [0] * (max_input_length - len(item["attention_mask"]))
                )
                labels.append(
                    item["labels"] + [-100] * (max_input_length - len(item["labels"]))
                )

            # sanitize token ids to avoid out-of-range embedding indices
            try:
                # prefer model embedding size over tokenizer vocab size
                model_embed_size = None
                try:
                    model_embed_size = self.agent.model.get_input_embeddings().weight.shape[0]
                except Exception:
                    model_embed_size = None

                vsz = model_embed_size
                if vsz is None:
                    try:
                        if hasattr(tokenizer, "__len__"):
                            vsz = len(tokenizer)
                    except Exception:
                        vsz = None
                if vsz is None and hasattr(tokenizer, "vocab_size"):
                    vsz = tokenizer.vocab_size
                if vsz is None and hasattr(tokenizer, "get_vocab"):
                    try:
                        vsz = len(tokenizer.get_vocab())
                    except Exception:
                        vsz = None
                if vsz is None and hasattr(tokenizer, "tokenizer"):
                    inner_tok = tokenizer.tokenizer
                    try:
                        if hasattr(inner_tok, "__len__"):
                            vsz = len(inner_tok)
                    except Exception:
                        vsz = getattr(inner_tok, "vocab_size", None)

                unk_id = getattr(tokenizer, "unk_token_id", 0) or 0
                for seq in input_ids:
                    for i, tid in enumerate(seq):
                        if (tid is None) or (tid < 0) or (vsz is not None and tid >= int(vsz)):
                            seq[i] = unk_id

                # ensure at least one valid token per sequence for RoPE indexing
                for i in range(len(attention_mask)):
                    if sum(attention_mask[i]) <= 0:
                        attention_mask[i][0] = 1
                        if input_ids[i]:
                            input_ids[i][0] = unk_id
                        else:
                            input_ids[i] = [unk_id]
                            labels[i] = [-100]
                            attention_mask[i] = [1] + attention_mask[i]
            except Exception:
                pass

            forward_kwargs = {
                "input_ids": torch.LongTensor(input_ids),
                "attention_mask": torch.LongTensor(attention_mask),
                "labels": torch.LongTensor(labels),
            }

            # 多模态：如果样本中已经通过 processor 预处理出了 pixel_values / image_grid_thw，
            # 且当前确认为 VL 模型，则在这里拼接成 batch 传给模型。
            use_vl = bool(is_vision_language and processor is not None)
            if use_vl and "pixel_values" in batch[0]:
                try:
                    pixel_tensors = []
                    grid_tensors = []
                    for item in batch:
                        pv = item.get("pixel_values", None)
                        grid = item.get("image_grid_thw", None)
                        if pv is None or grid is None:
                            continue
                        # 单个样本：
                        # - pv: [num_patches_i, dim]
                        # - grid: [[T_i, H_i, W_i]]
                        pixel_tensors.append(torch.tensor(pv, dtype=torch.float32))
                        grid_tensors.append(torch.tensor(grid, dtype=torch.long))
                    if pixel_tensors:
                        # 拼成：
                        # - pixel_values: [sum_i num_patches_i, dim]
                        # - image_grid_thw: [num_images_in_batch, 3]
                        forward_kwargs["pixel_values"] = torch.cat(
                            pixel_tensors, dim=0
                        )
                        forward_kwargs["image_grid_thw"] = torch.cat(
                            grid_tensors, dim=0
                        )
                except Exception:
                    # 不因为图像失败而中断训练
                    pass
            return {"forward_kwargs": forward_kwargs}

        self.train_dataset = tokenized_dataset["train"]
        self.train_dataloader = DataLoader(
            tokenized_dataset["train"],
            shuffle=True,
            batch_size=self.args["batch_size"],
            num_workers=self.args["num_workers"],
            pin_memory=True,
            collate_fn=partial(
                collate_fn,
                tokenizer=self.agent.tokenizer,
                processor=self.processor,
                is_vision_language=self.is_vision_language,
            ),
        )
        self.accelerator.print("Number of train batches:", len(self.train_dataloader))

    def get_inference_test_dataloader(self):
        """
        create inference_dataloader, test_dataloader
        """

        def collate_fn(batch):
            result = {
                "data_idxs": [int(item["item_id"].split("_")[-1]) for item in batch]
            }
            return result

        # 在单进程/未初始化分布式的情况下，避免进入 barrier
        if getattr(self.accelerator, "num_processes", 1) == 1:
            from contextlib import nullcontext
            mpf_ctx = nullcontext()
        else:
            mpf_ctx = self.accelerator.main_process_first()

        with mpf_ctx:

            self.inference_dataloader = DataLoader(
                self.raw_dataset["inference"],
                batch_size=self.args["eval_batch_size"],
                num_workers=self.args["num_workers"],
                pin_memory=True,
                collate_fn=partial(collate_fn),
            )

            self.test_dataloader = DataLoader(
                self.raw_dataset["test"],
                batch_size=self.args["eval_batch_size"],
                num_workers=self.args["num_workers"],
                pin_memory=True,
                collate_fn=partial(collate_fn),
            )
            self.accelerator.print(
                "Number of inference batches:", len(self.inference_dataloader)
            )
            self.accelerator.print("Number of test batches:", len(self.test_dataloader))

    def setup_wandb(self):
        """
        Set the wandb.
        """
        # os.environ["WANDB_MODE"] = "offline"
        # 使用 accelerate 的进程判断，避免在未初始化分布式时访问 torch.distributed
        if self.accelerator.is_main_process and self.args["wandb_log"]:
            wandb.init(
                project=self.args["wandb_project"],
                name=self.args["wandb_run_name"],
            )
            wandb.config.update(self.args)

        if self.accelerator.is_main_process and self.args["wandb_log"]:
            # compute vocab size robustly for logging
            tok = self.agent.tokenizer
            vocab_size = None
            if hasattr(tok, "vocab_size"):
                vocab_size = tok.vocab_size
            elif hasattr(tok, "get_vocab"):
                try:
                    vocab_size = len(tok.get_vocab())
                except Exception:
                    vocab_size = None
            elif hasattr(tok, "tokenizer") and hasattr(tok.tokenizer, "vocab_size"):
                vocab_size = tok.tokenizer.vocab_size
            if vocab_size is None:
                model_config = getattr(getattr(self.agent, "model", None), "config", None)
                vocab_size = getattr(model_config, "vocab_size", None)

            wandb.run.summary.update(
                {
                    "pad_token_id": self.agent.tokenizer.pad_token_id,
                    "eos_token_id": self.agent.tokenizer.eos_token_id,
                    "unk_token_id": self.agent.tokenizer.unk_token_id,
                    "vocab_size": vocab_size,
                }
            )

    def save_model(self, model, tokenizer, save_path):
        os.makedirs(save_path, exist_ok=True)

        unwrapped_model = self.accelerator.unwrap_model(model)
        try:
            state_dict = None
            # 仅在模型仍为分布式/DeepSpeed 包装对象时使用 accelerator.get_state_dict
            use_ds = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
            if use_ds and hasattr(self.accelerator, "get_state_dict"):
                try:
                    state_dict = self.accelerator.get_state_dict(model)
                except AttributeError:
                    # 如果传入的是已解包的 HF 模型则跳过聚合，走常规保存
                    state_dict = None
            unwrapped_model.save_pretrained(
                save_path,
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save,
                state_dict=state_dict,
            )
        except ValueError:
            # 当未启用 ZeRO-3 聚合保存选项时，退回保存 ZeRO checkpoint
            if hasattr(model, "save_checkpoint"):
                model.save_checkpoint(save_path)
            else:
                raise
        tokenizer.save_pretrained(save_path)

        # 拷贝视觉/多模态处理器配置，便于评估阶段从 checkpoint 目录直接加载 AutoProcessor
        try:
            base_model_path = os.path.abspath(os.path.expanduser(self.args.get("model_train_path", "")))
            if base_model_path and os.path.isdir(base_model_path):
                candidates = [
                    "preprocessor_config.json",
                    "processor_config.json",
                    "image_processor",
                    "image_processor.json",
                ]
                for name in candidates:
                    src = os.path.join(base_model_path, name)
                    dst = os.path.join(save_path, name)
                    if os.path.isfile(src) and not os.path.exists(dst):
                        shutil.copy2(src, dst)
                    elif os.path.isdir(src) and not os.path.exists(dst):
                        shutil.copytree(src, dst)
        except Exception:
            # 非关键失败忽略
            pass

    def init_train_stuff(self):
        """
        Initialize the training stuff, including the optimizer, scheduler, etc.
        Prepare the model, optimizer, and dataloader.
        """
        num_training_steps = (
            len(self.train_dataloader)
            // self.accelerator.num_processes
            * self.args["n_epochs"]
        ) // self.args["gradient_accumulation_steps"]
        warmup_step = (
            self.args["warmup_step"]
            if self.args["warmup_step"] is not None and self.args["warmup_step"] >= 0
            else int(0.1 * num_training_steps)
        )
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.agent.model.named_parameters()
                    if not any(nd in n for nd in ["bias", "LayerNorm.weight"])
                ],
                "weight_decay": self.args["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in self.agent.model.named_parameters()
                    if any(nd in n for nd in ["bias", "LayerNorm.weight"])
                ],
                "weight_decay": 0.0,
            },
        ]

        # 若启用 DeepSpeed，则使用 DummyOptim/DummyScheduler，避免与 DS 配置冲突，并避免 CPUAdam 编译
        use_ds = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        if use_ds:
            from accelerate.utils.deepspeed import DummyOptim, DummyScheduler

            self.optimizer = DummyOptim(
                self.agent.model.parameters(),
                lr=self.args["learning_rate"],
                weight_decay=self.args["weight_decay"],
            )
            self.scheduler = DummyScheduler(
                self.optimizer,
                total_num_steps=num_training_steps,
                warmup_num_steps=warmup_step,
            )
        else:
            self.optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.args["learning_rate"], eps=1e-8
            )
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_step,
                num_training_steps=num_training_steps,
            )

        self.accelerator.print(
            f"***** Running training *****\n"
            f"  Num examples = {len(self.train_dataset)}\n"
            f"  Num Epochs = {self.args['n_epochs']}\n"
            f"  Instantaneous batch size per device = {self.args['batch_size']}\n"
            f"  Total train batch size (w. parallel, distributed & accumulation) = {self.args['batch_size']*self.accelerator.num_processes*self.args['gradient_accumulation_steps']}\n"
            f"  Total optimization steps = {num_training_steps}\n"
            f"  Warm up step: {warmup_step}\n"
            f"  Learning rate: {self.args['learning_rate']}\n"
        )

        (
            self.agent.model,
            self.optimizer,
            self.train_dataloader,
            self.inference_dataloader,
            self.test_dataloader,
        ) = self.accelerator.prepare(
            self.agent.model,
            self.optimizer,
            self.train_dataloader,
            self.inference_dataloader,
            self.test_dataloader,
        )

    def train_one_epoch(self, epoch, global_step):
        clip_grad_norm = self.args.get("clip_grad_norm", None)
        logging_step_freq = self.args.get("logging_step_freq", None)
        self.agent.model.train()
        epoch_result_dict = defaultdict(list)
        with tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            disable=not self.accelerator.is_main_process,
            desc=f"Train Loop | Epoch {epoch}",
        ) as t:
            for idx, batch in t:
                with self.accelerator.accumulate(self.agent.model):
                    output = self.agent.model(**batch["forward_kwargs"])
                    # train_data_idx = batch["item_id"]
                    # # Print train_data_idx
                    # self.accelerator.print("Train data idx:", train_data_idx)
                    # Get some metrics
                    loss = output[0]
                    result_dict, extra = {}, None
                    # Update
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        if clip_grad_norm is not None:
                            self.accelerator.clip_grad_norm_(
                                self.agent.model.parameters(), clip_grad_norm
                            )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.accelerator.sync_gradients and hasattr(self.scheduler, "step"):
                        self.scheduler.step()

                if self.accelerator.sync_gradients:
                    global_step += 1
                    # Step update metric
                    epoch_result_dict["loss"].append(loss.item())
                    for k, v in result_dict.items():
                        epoch_result_dict[k].append(v)

                    # Step logging
                    train_log_dict = {}
                    if (
                        logging_step_freq is not None
                        and global_step % logging_step_freq == 0
                    ):
                        train_log_dict = {
                            f"T.{k}": sum(v) / len(v) if isinstance(v, list) else v
                            for k, v in epoch_result_dict.items()
                        }

                    if train_log_dict:
                        log_dict = dict(train_log_dict)
                        current_lr = None
                        if hasattr(self.scheduler, "get_last_lr"):
                            lrs = self.scheduler.get_last_lr()
                            current_lr = (lrs[0] if isinstance(lrs, (list, tuple)) and lrs else lrs)
                        elif hasattr(self.optimizer, "param_groups") and self.optimizer.param_groups:
                            current_lr = self.optimizer.param_groups[0].get("lr")
                        if current_lr is not None:
                            log_dict["lr"] = current_lr
                        if self.accelerator.is_main_process and self.args["wandb_log"]:
                            wandb.log(log_dict, step=global_step)
                            log_dict = {
                                "wandb": self.args["wandb_project"]
                                + "|"
                                + self.args["wandb_run_name"],
                                **log_dict,
                            }
                        log_dict = {
                            k: f"{v:.5g}" if isinstance(v, float) else v
                            for k, v in log_dict.items()
                        }
                        self.accelerator.print(
                            f"[E={epoch}/{self.args['n_epochs']}, S={global_step}] {log_dict}"
                        )

                    # Keep only max_record items
                    for k, v in epoch_result_dict.items():
                        if len(v) > 1:
                            epoch_result_dict[k] = v[-1:]

        # Metric summary:
        epoch_result_dict = {
            k: (sum(v) / len(v) if isinstance(v, list) else v)
            for k, v in epoch_result_dict.items()
        }
        return epoch_result_dict, global_step

    def train(self):
        """
        Train the model.
        """
        global_step = 0
        n_epochs = self.args["n_epochs"]
        logging_epoch_freq = self.args["logging_epoch_freq"]
        evaluating_epoch_freq = self.args["evaluating_epoch_freq"]
        saving_epoch_freq = self.args["saving_epoch_freq"]
        model_save_path = self.args["model_save_path"]
        os.makedirs(model_save_path, exist_ok=True)
        with tqdm(range(1, n_epochs + 1), total=n_epochs, disable=False) as t:
            for epoch in t:
                train_epoch_result_dict, global_step = self.train_one_epoch(
                    epoch, global_step
                )

                eval_log_dict = {}
                is_best = False
                if (
                    evaluating_epoch_freq is not None
                    and epoch % evaluating_epoch_freq == 0
                ):
                    evaluate_result_dict = {
                        f"Eval.Gen.{k}": v
                        for k, v in self.eval_test_dataloader().items()
                    }
                    eval_log_dict.update(evaluate_result_dict)
                    if eval_log_dict["Eval.Gen.success"] > self.best_eval_log_dict.get(
                        "Eval.Gen.success_best", 0
                    ):
                        is_best = True
                        self.best_eval_log_dict["Eval.Gen.success_best"] = (
                            eval_log_dict["Eval.Gen.success"]
                        )
                    if "Eval.Gen.success" not in self.summary_log_dict:
                        self.summary_log_dict["Eval.Gen.success"] = []
                    self.summary_log_dict["Eval.Gen.success"].append(
                        eval_log_dict["Eval.Gen.success"]
                    )

                train_log_dict = {}
                if logging_epoch_freq is not None and epoch % logging_epoch_freq == 0:
                    train_log_dict = {
                        f"T.{k}": sum(v) / len(v) if isinstance(v, list) else v
                        for k, v in train_epoch_result_dict.items()
                    }

                if train_log_dict or eval_log_dict:
                    log_dict = {
                        **train_log_dict,
                        **eval_log_dict,
                        **self.best_eval_log_dict,
                    }
                    current_lr = None
                    if hasattr(self.scheduler, "get_last_lr"):
                        lrs = self.scheduler.get_last_lr()
                        current_lr = (lrs[0] if isinstance(lrs, (list, tuple)) and lrs else lrs)
                    elif hasattr(self.optimizer, "param_groups") and self.optimizer.param_groups:
                        current_lr = self.optimizer.param_groups[0].get("lr")
                    if current_lr is not None:
                        log_dict["lr"] = current_lr
                    if self.accelerator.is_main_process and self.args["wandb_log"]:
                        wandb.log(log_dict, step=global_step)
                        log_dict = {
                            "wandb": self.args["wandb_project"]
                            + "|"
                            + self.args["wandb_run_name"],
                            **log_dict,
                        }
                    log_dict = {
                        k: f"{v:.5g}" if isinstance(v, float) else v
                        for k, v in log_dict.items()
                    }
                    self.accelerator.print(
                        f"[E={epoch}/{self.args['n_epochs']}, S={global_step}] {log_dict}"
                    )

                if saving_epoch_freq is not None and epoch % saving_epoch_freq == 0:
                    # if is_best:
                    save_path = os.path.join(model_save_path, f"train_epoch_{epoch}")
                    self.save_model(self.agent.model, self.agent.tokenizer, save_path)

    def eval_test_dataloader(
        self,
        dataloader=None,
        do_sample=False,
        temperature=0.3,
        record_to_file=True,
    ):
        # test
        self.agent.model.eval()
        all_rewards = []
        all_success = []
        if dataloader is None:
            dataloader = self.test_dataloader

        for _, batch in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            disable=not self.accelerator.is_main_process,
            desc="Evaluation Gen Loop",
        ):
            data_idxs = batch["data_idxs"]
            self.accelerator.print("==== Batch inference data idxs ====", data_idxs)
            with torch.no_grad():
                exps = self.eval(
                    generation_config=GenerationConfig(
                        max_length=4096,
                        do_sample=do_sample,
                        temperature=temperature,
                        eos_token_id=self.agent.tokenizer.eos_token_id,
                        pad_token_id=(
                            self.agent.tokenizer.pad_token_id
                            if self.agent.tokenizer.pad_token_id is not None
                            else self.agent.tokenizer.unk_token_id
                        ),
                    ),
                    max_rounds=self.args["max_round"],
                    idxs=data_idxs,
                )

                cur_batch_rewards = torch.FloatTensor(
                    [exp.reward for exp in exps.experiences]
                ).to(self.accelerator.device)
                cur_batch_success = torch.FloatTensor(
                    [1 if exp.reward == 1 else 0 for exp in exps.experiences]
                ).to(self.accelerator.device)
                cur_batch_data_idx = torch.tensor(data_idxs).to(self.accelerator.device)
                
                # gather operation
                all_device_batch_rewards = self.accelerator.gather(cur_batch_rewards)
                all_device_batch_success = self.accelerator.gather(cur_batch_success)
                all_device_batch_exp = gather_object(exps.experiences)
                all_device_data_idx = self.accelerator.gather(cur_batch_data_idx)
                all_rewards.extend(all_device_batch_rewards.cpu().numpy().tolist())
                all_success.extend(all_device_batch_success.cpu().numpy().tolist())
                
                # write inference results to file
                if record_to_file and self.accelerator.is_main_process:
                    # write to file
                    inference_file_path = os.path.join(
                        self.args["model_save_path"], "inference.jsonl"
                    )
                    with jsonlines.open(inference_file_path, mode="a") as f:
                        for idx, exp in enumerate(all_device_batch_exp):
                            cur_idx = all_device_data_idx[idx]
                            conversation = exp.conversation
                            cur_reward = exp.reward
                            cur_success = 1 if exp.reward == 1 else 0
                            item_id = f"{self.args['task_name']}_{cur_idx}"
                            f.write(
                                {
                                    "conversations": conversation,
                                    "item_id": item_id,
                                    "reward": cur_reward,
                                    "success": cur_success,
                                }
                            )

        # fix for duplicated data
        all_rewards = all_rewards[: len(dataloader.dataset)]
        all_success = all_success[: len(dataloader.dataset)]

        if self.accelerator.is_main_process and self.accelerator.is_local_main_process:
            mean_reward = torch.FloatTensor([np.mean(all_rewards)]).to(
                self.accelerator.device
            )
            mean_success = torch.FloatTensor([np.mean(all_success)]).to(
                self.accelerator.device
            )
        else:
            mean_reward = torch.FloatTensor([-1.0]).to(self.accelerator.device)
            mean_success = torch.FloatTensor([-1.0]).to(self.accelerator.device)

        mean_reward = broadcast(mean_reward).cpu().numpy().tolist()[0]
        mean_success = broadcast(mean_success).cpu().numpy().tolist()[0]
        self.accelerator.print("\n\n==== Test Evaluation ====\n")
        self.accelerator.print(f"Score: {mean_reward:.5f}")
        self.accelerator.print(f"Success: {mean_success:.5f}")

        return {"score": mean_reward, "success": mean_success}

    def train_and_inference(self):
        self.accelerator.print("[BC Trainer] Start training.")
        self.train()
        self.accelerator.print("[BC Trainer] Start inference.")
        self.eval_test_dataloader(
            dataloader=self.inference_dataloader,
            do_sample=False,
            temperature=0.1,
            record_to_file=True,
        )


class QwenVLBCTrainer(BCTrainer):
    """
    专门针对 Qwen-VL / Qwen2-VL 系列的多模态行为克隆 Trainer。
    与通用 BCTrainer 的主要区别：
      - 不再手工用 tokenizer.encode 拼文本；
      - 在 collate_fn 里直接使用 AutoProcessor.apply_chat_template，一次性生成
        input_ids / attention_mask / pixel_values / image_grid_thw，避免
        “Image features and image tokens do not match” 一类错配。
    训练循环、优化器等逻辑沿用父类。
    """

    def get_train_dataloader(self):
        """
        为 Qwen-VL 构建 train_dataloader：
        - 保留原始 self.raw_dataset['train']；
        - 在 collate_fn 里把一整个 batch 转成 Qwen-VL 所需的对话格式，
          交给 AutoProcessor 统一做多模态处理。
        """

        assert (
            self.processor is not None and self.is_vision_language
        ), "QwenVLBCTrainer 仅在多模态模型（processor + is_vision_language=True）下使用"

        processor = self.processor
        args = self.args
        # 官方推荐由 tokenizer 提供 chat_template；AutoProcessor 通常把 tokenizer 挂在其 tokenizer 属性下
        tok_for_template = getattr(processor, "tokenizer", processor)

        # 预先解析“图像 token id”，用于在 collate_fn 里判断当前 batch 文本中
        # 是否真的包含图像 token；若没有，则不要把 pixel_values 传给模型，
        # 否则会触发 “Image features and image tokens do not match” 报错。
        image_token_ids = set()
        try:
            model_config = getattr(self.agent.model, "config", None)
            if model_config is not None and hasattr(model_config, "image_token_index"):
                idx = getattr(model_config, "image_token_index")
                if isinstance(idx, int):
                    image_token_ids.add(int(idx))
                elif isinstance(idx, (list, tuple)):
                    for v in idx:
                        try:
                            image_token_ids.add(int(v))
                        except Exception:
                            continue
        except Exception:
            image_token_ids = set()

        # 若 config 中没有给出，则从 tokenizer 词表里启发式查找包含 "image" 关键字的 token
        if not image_token_ids:
            try:
                tok = getattr(processor, "tokenizer", None)
                vocab = None
                if tok is not None:
                    if hasattr(tok, "get_vocab"):
                        vocab = tok.get_vocab()
                    elif hasattr(tok, "vocab"):
                        vocab = tok.vocab
                if isinstance(vocab, dict):
                    for k, v in vocab.items():
                        try:
                            if "image" in str(k).lower():
                                image_token_ids.add(int(v))
                        except Exception:
                            continue
            except Exception:
                image_token_ids = set()

        def build_mask_for_single_sequence(
            input_id_row: torch.Tensor,
            assistant_prefix_ids,
            im_end_ids,
            ignore_index: int = -100,
        ):
            """
            针对单条样本：
            - 在 token 序列中查找所有 "<|im_start|>assistant\\n" 片段的位置；
            - 对于每个位置，向后找到下一个 "<|im_end|>" 片段；
            - 把这两个之间的 token 设为有效 label，其余设为 ignore_index。
            """
            target = torch.full_like(input_id_row, ignore_index)
            ids = input_id_row.tolist()
            n = len(ids)

            pre = assistant_prefix_ids
            pre_len = len(pre)

            end = im_end_ids
            end_len = len(end)

            def find_subseq(haystack, needle, start_pos):
                """在 list[int] 里从 start_pos 开始寻找 needle 子序列，返回起始下标或 None"""
                last = len(haystack) - len(needle)
                for s in range(start_pos, last + 1):
                    if haystack[s : s + len(needle)] == needle:
                        return s
                return None

            pos = 0
            while pos <= n - pre_len:
                # 找到一个 "<|im_start|>assistant\\n" 前缀
                if ids[pos : pos + pre_len] == pre:
                    content_start = pos + pre_len

                    # 从 content_start 往后找最近的 "<|im_end|>"
                    end_pos = find_subseq(ids, end, content_start)
                    if end_pos is None:
                        # 理论上不应该发生；保守处理：直接 break
                        break

                    # [content_start, end_pos) 之间的 token 是 assistant 的内容
                    if content_start < end_pos:
                        target[content_start:end_pos] = input_id_row[content_start:end_pos]

                    # 继续往后找下一个 assistant 段
                    pos = end_pos + end_len
                else:
                    pos += 1

            return target

        def make_labels(input_ids: torch.Tensor, tokenizer, ignore_index: int = -100):
            """
            根据 input_ids 生成 labels，Mask 掉 user/system 部分，只保留 assistant 段落：
            即形如 <|im_start|>assistant\\n ... <|im_end|> 中间的 token。
            适用于当前 Qwen2-VL ChatML 模板。
            """
            # 关键 token / 片段的 id 序列：
            # 注意：这里直接构造一个“assistant 段前缀”的 token 序列：
            # 文本就是模板里的片段："<|im_start|>assistant\n"
            assistant_prefix_ids = tokenizer.encode(
                "<|im_start|>assistant\n", add_special_tokens=False
            )
            im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)

            batch_size, _ = input_ids.shape
            labels = torch.full_like(input_ids, ignore_index)

            for i in range(batch_size):
                labels[i] = build_mask_for_single_sequence(
                    input_ids[i],
                    assistant_prefix_ids=assistant_prefix_ids,
                    im_end_ids=im_end_ids,
                    ignore_index=ignore_index,
                )

            return labels

        def build_qwen_vl_conversation(item):
            """
            将单条样本的 conversations 转成 Qwen-VL chat template 期望的格式：
            [
              {
                "role": "user" | "assistant",
                "content": [
                  {"type": "image", "image": PIL.Image},
                  {"type": "text", "text": "..."},
                  ...
                ]
              },
              ...
            ]
            """
            conversations = item["conversations"]
            proc_messages = []
            images_for_conv = []

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
                            merged_txt = (
                                str(instr).strip() + "\n" + str(step_text).strip()
                            ).strip()
                            if merged_txt:
                                text_val = merged_txt
                            screenshot_b64 = inner.get("screenshot_base64")
                    except Exception:
                        pass

                content_parts = []

                # 如果存在截图，则解码为 PIL.Image 并作为 image 内容传入
                img = None
                if screenshot_b64 and Image is not None:
                    try:
                        b64_str = str(screenshot_b64)
                        if "," in b64_str and b64_str.strip().startswith("data:"):
                            b64_str = b64_str.split(",", 1)[-1]
                        img_bytes = base64.b64decode(b64_str)
                        img = Image.open(BytesIO(img_bytes)).convert("RGB")
                        content_parts.append({"type": "image", "image": img})
                    except Exception:
                        # 图像解码失败时，退化为纯文本
                        img = None
                if img is not None:
                    images_for_conv.append(img)

                if text_val:
                    content_parts.append({"type": "text", "text": str(text_val)})

                if not content_parts:
                    # 至少保证有一个空文本，避免 processor 崩溃
                    content_parts.append({"type": "text", "text": ""})

                role = "user" if message["from"] == "human" else "assistant"
                proc_messages.append({"role": role, "content": content_parts})

            return proc_messages, images_for_conv

        def collate_fn(batch):
            """
            对于 Qwen-VL：
            - 直接以多轮 messages（包含 role 和 content 列表）作为单位，
              调用 tokenizer.apply_chat_template(..., tokenize=False) 得到纯文本 prompt；
            - 再统一用 AutoProcessor(text=..., images=...) 编码成批次，自动对齐图像 token 与特征。
            """
            texts = []
            images_batch = []

            for item in batch:
                messages, images_for_conv = build_qwen_vl_conversation(item)

                # 使用自定义的 chat_template，将多轮 role+content 结构转成带 <|vision_start|><|image_pad|><|vision_end|> 的文本。
                try:
                    text = tok_for_template.apply_chat_template(
                        messages,
                        add_generation_prompt=False,
                        tokenize=False,
                    )
                except Exception:
                    text = None

                # 若模板返回空串 / 非字符串，则简单拼接所有文本片段作为回退，避免出现空序列。
                if not isinstance(text, str) or not text.strip():
                    fallback_text_parts = []
                    try:
                        for msg in messages:
                            for part in msg.get("content", []):
                                if (
                                    isinstance(part, dict)
                                    and part.get("type") == "text"
                                ):
                                    t = str(part.get("text", "")).strip()
                                    if t:
                                        fallback_text_parts.append(t)
                    except Exception:
                        fallback_text_parts = []
                        fallback_text = "\n".join(fallback_text_parts).strip()
                        text = fallback_text if fallback_text else " "

                texts.append(text)
                images_batch.append(images_for_conv if images_for_conv else [])

            # 至少有一个样本带图像时，传入 images 参数；否则仅按纯文本处理
            if any(len(imgs) > 0 for imgs in images_batch):
                model_inputs = processor(
                    text=texts,
                    images=images_batch,
                    return_tensors="pt",
                    truncation=False,
                    max_length=args.get("max_input_length", 4096),
                    padding=True,
                )
            else:
                model_inputs = processor(
                    text=texts,
                    return_tensors="pt",
                    truncation=False,
                    max_length=args.get("max_input_length", 4096),
                    padding=True,
                )

            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]

            # 防御性处理：如果某些 batch 里所有样本的 prompt 都被截断成空序列，
            # 会出现 shape = [B, 0]，导致模型内部在 dim=1 上做 max/reduction 时触发
            # “Expected reduction dim 1 to have non-zero size” 的报错。
            # 这里强制为每个样本至少保留 1 个占位 token。
            try:
                if input_ids.size(1) == 0:
                    # 尝试从 processor.tokenizer 或 agent.tokenizer 获取一个合理的占位 token
                    tok = getattr(processor, "tokenizer", None)
                    if tok is None and hasattr(self, "agent"):
                        tok = getattr(self.agent, "tokenizer", None)
                    fallback_id = None
                    if tok is not None:
                        for attr in ["eos_token_id", "pad_token_id", "bos_token_id", "unk_token_id"]:
                            vid = getattr(tok, attr, None)
                            if isinstance(vid, int) and vid >= 0:
                                fallback_id = vid
                                break
                    if fallback_id is None:
                        fallback_id = 0
                    bsz = input_ids.size(0)
                    device = input_ids.device
                    input_ids = torch.full(
                        (bsz, 1),
                        int(fallback_id),
                        dtype=torch.long,
                        device=device,
                    )
                    attention_mask = torch.ones(
                        (bsz, 1),
                        dtype=torch.long,
                        device=device,
                    )
            except Exception:
                # 如果检查失败，则保持原始输出，让上游错误暴露
                pass

            # Qwen2-VL 系列通常会返回 pixel_values / image_grid_thw
            pixel_values = model_inputs.get("pixel_values", None)
            image_grid_thw = model_inputs.get("image_grid_thw", None)

            # label 构造：
            #  - 默认：对所有 token 做 LM 训练（labels = input_ids）
            #  - 若启用 assistant_only_loss：仅对 assistant 轮次的 token 计算 loss，其余位置置为 -100
            assistant_only = bool(args.get("assistant_only_loss", False))
            if not assistant_only:
                labels = input_ids.clone()
            else:
                # 基于 ChatML token 序列，只对 assistant 段落计算 loss
                tok_for_labels = getattr(processor, "tokenizer", processor)
                labels = make_labels(
                    input_ids=input_ids,
                    tokenizer=tok_for_labels,
                    ignore_index=-100,
                )

            forward_kwargs = {
                "input_ids": input_ids.to(dtype=torch.long),
                "attention_mask": attention_mask.to(dtype=torch.long),
                "labels": labels.to(dtype=torch.long),
            }

            # 仅当文本中真实存在图像 token 时，才把图像特征传给模型；
            # 否则会出现 “tokens: 0, features > 0” 的错配。
            attach_images = (
                pixel_values is not None
                and image_grid_thw is not None
            )
            if attach_images and image_token_ids:
                try:
                    flat_ids = input_ids.view(-1).tolist()
                    attach_images = any(tid in image_token_ids for tid in flat_ids)
                except Exception:
                    # 回退为保守策略：若检查失败，则仍然尝试传入图像
                    attach_images = True

            if attach_images:
                forward_kwargs["pixel_values"] = pixel_values
                forward_kwargs["image_grid_thw"] = image_grid_thw

            return {"forward_kwargs": forward_kwargs}

        # 对于 QwenVLBCTrainer，我们直接使用原始 train 集合，让 collate_fn 处理多模态
        self.train_dataset = self.raw_dataset["train"]
        self.train_dataloader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.args["batch_size"],
            num_workers=self.args["num_workers"],
            pin_memory=True,
            collate_fn=collate_fn,
        )
        self.accelerator.print(
            "Number of train batches (Qwen-VL):", len(self.train_dataloader)
        )
