from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import json
import os

import torch
import transformers
from agentenv.controller import Agent
from agentenv.envs import (
    AlfWorldTask,
    SciworldTask,
    TextCraftTask,
    WebarenaTask,
    WebshopTask,
    OSWorldTask,
)
from agentenv.trainer.bc_trainer import BCTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

# 兼容新旧 Transformers：新版本优先使用 AutoModelForImageTextToText；否则回退到 AutoModelForVision2Seq
try:
    from transformers import AutoModelForImageTextToText as AutoModelForVL  # >=4.43
except Exception:
    from transformers import AutoModelForVision2Seq as AutoModelForVL


@dataclass
class TrainingArguments:
    train_file: str = field(metadata={"help": "Training dataset."})
    use_deepspeed: bool = field(default=False, metadata={"help": "Enable DeepSpeed ZeRO-3 via Accelerate"})
    inference_file: str = field(
        default="./data/train/webshop_train.json", metadata={"help": "Inference dataset."}
    )
    test_file: str = field(default="./data/test/webshop_test.json", metadata={"help": "Test dataset."})
    # model path
    model_train_path: str = field(
        default="/mnt/petrelfs/share_data/llm_llama/llama2/llama-2-7b-chat-hf",
        metadata={"help": "Path of initial train model"},
    )
    model_save_path: str = field(
        default="outputs/model",
        metadata={"help": "Directory to save the trained model."},
    )
    task_name: str = field(
        default="webshop", metadata={"help": "Task name for evaluation"}
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for training."},
    )
    eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size for evaluation."}
    )
    n_epochs: int = field(default=40)
    num_workers: int = field(
        default=8, metadata={"help": "Number of subprocesses to use for data loading."}
    )
    learning_rate: float = field(default=2e-5, metadata={"help": "Learning rate."})
    weight_decay: float = field(
        default=1e-6, metadata={"help": "Weight decay for regularization."}
    )
    warmup_step: int = field(
        default=0,
        metadata={"help": "Number of warmup steps for learning rate scheduling."},
    )
    clip_grad_norm: float = field(
        default=1, metadata={"help": "Gradient clipping threshold."}
    )
    gradient_accumulation_steps: int = field(default=1)
    evaluating_epoch_freq: int = field(default=1)
    logging_epoch_freq: int = field(default=1)
    saving_epoch_freq: int = field(default=1)
    logging_step_freq: int = field(default=None)
    seed: int = field(default=42)
    max_input_length: int = field(default=700)

    # environment
    max_round: int = field(
        default=6,
        metadata={"help": "Interaction rounds between agents and environment"},
    )

    # wandb stuff
    wandb_log: bool = field(default=False)
    wandb_project: str = field(default="AgentGym_behavioral_clone")
    wandb_run_name: str = field(default="behavioral_clone")

    # environment parameters
    env_server_base: str = field(default=None)
    data_len: int = field(default=200)
    timeout: int = field(default=2400)

    # OSWorld optional configs
    create_cfg_json: Optional[str] = field(
        default=None, metadata={"help": "JSON string for OSWorld /create payload"}
    )
    create_cfg_file: Optional[str] = field(
        default=None, metadata={"help": "Path to JSON file for OSWorld /create payload"}
    )
    observe_format: Optional[str] = field(
        default=None, metadata={"help": "OSWorld observe format: text | screenshot_base64 | raw"}
    )
    examples_dir: Optional[str] = field(
        default=None, metadata={"help": "OSWorld evaluation examples directory"}
    )
    save_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory to save OSWorld run artifacts/screenshots"}
    )
    post_reset_delay_seconds: Optional[float] = field(
        default=None, metadata={"help": "OSWorld: seconds to wait after reset before first observe"}
    )
    step_pause_seconds: Optional[float] = field(
        default=None, metadata={"help": "OSWorld: seconds to pause between steps"}
    )

    # 开发/调试：限制训练样本数做冒烟测试
    train_max_samples: Optional[int] = field(
        default=None, metadata={"help": "Limit number of training samples for quick smoke test"}
    )


def main():
    parser = transformers.HfArgumentParser(TrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()

    # 根据路径与 config.json 识别是否为视觉语言模型（Qwen2-VL / Qwen3-VL 等）
    model_path_lower = args.model_train_path.lower()
    is_vl_by_name = any(k in model_path_lower for k in ["qwen2-vl", "qwen3-vl", "qwen-vl"])  # 名称启发式

    is_vl_by_config = False
    try:
        cfg_path = Path(args.model_train_path) / "config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                _cfg = json.load(f)
            mt = str(_cfg.get("model_type", "")).lower()
            archs = " ".join(_cfg.get("architectures", [])).lower()
            if any(tag in mt for tag in ["qwen2_vl", "qwen3_vl"]) or "vl" in archs:
                is_vl_by_config = True
    except Exception:
        pass

    is_vision_language = is_vl_by_name or is_vl_by_config

    if is_vision_language:
        # 视觉语言模型（如 Qwen2-VL / Qwen3-VL）
        # 1) 先尝试本地
        try:
            processor = AutoProcessor.from_pretrained(
                args.model_train_path,
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception:
            processor = AutoProcessor.from_pretrained(
                args.model_train_path,
                trust_remote_code=True,
            )
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_train_path,
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_train_path,
                trust_remote_code=True,
            )
        # 2) 模型本地优先，再回退到远程，以及回退关闭 safetensors
        model = None
        last_exc = None
        for load_kwargs in (
            dict(local_files_only=True, use_safetensors=True),
            dict(local_files_only=True, use_safetensors=False),
            dict(local_files_only=False, use_safetensors=True),
            dict(local_files_only=False, use_safetensors=False),
        ):
            try:
                model = AutoModelForVL.from_pretrained(
                    args.model_train_path,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    **load_kwargs,
                )
                break
            except Exception as e:
                last_exc = e
                model = None
                # 回退到 AutoModelForCausalLM（部分环境/版本对 VL Auto 类兼容性欠佳）
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model_train_path,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                        **load_kwargs,
                    )
                    break
                except Exception as e2:
                    last_exc = e2
                    model = None
        if model is None:
            raise RuntimeError(
                f"Failed to load VL model from {args.model_train_path}. Please check weights availability. Last error: {repr(last_exc)}"
            )
    else:
        # 纯文本模型
        tokenizer = AutoTokenizer.from_pretrained(args.model_train_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_train_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
        )

    # Enable gradient checkpointing to reduce activation memory
    try:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    except Exception:
        pass

    # Ensure use_cache is disabled explicitly
    try:
        if hasattr(model, "config"):
            setattr(model.config, "use_cache", False)
    except Exception:
        pass

    # Freeze vision branch if using Qwen2-VL for text-only training
    try:
        if is_vision_language:
            for name, p in model.named_parameters():
                ln = name.lower()
                if ("vision" in ln) or ("visual" in ln) or ("image" in ln):
                    p.requires_grad = False
    except Exception:
        pass

    # task_name - task dict
    task_classes = {
        "webshop": WebshopTask,
        "alfworld": AlfWorldTask,
        "sciworld": SciworldTask,
        "textcraft": TextCraftTask,
        "webarena": WebarenaTask,
        "osworld": OSWorldTask,
    }

    # select task according to the name
    task_class = task_classes.get(args.task_name.lower(), None)
    if task_class is None:
        raise ValueError(f"Unsupported task name: {args.task_name}")

    # set environment parameters
    env_args = {
        "env_server_base": args.env_server_base,
        "data_len": args.data_len,
        "timeout": args.timeout,
    }

    # Enrich OSWorld-specific args only when needed
    if args.task_name and args.task_name.lower() == "osworld":
        # create_cfg can come from inline JSON or file
        create_cfg = None
        if args.create_cfg_json:
            try:
                create_cfg = json.loads(args.create_cfg_json)
            except Exception:
                create_cfg = None
        if create_cfg is None and args.create_cfg_file:
            try:
                with open(os.path.expanduser(args.create_cfg_file), "r", encoding="utf-8") as f:
                    create_cfg = json.load(f)
            except Exception:
                create_cfg = None
        if create_cfg is not None:
            env_args["create_cfg"] = create_cfg

        if args.observe_format:
            env_args["observe_format"] = args.observe_format
        if args.examples_dir:
            env_args["examples_dir"] = args.examples_dir
        if args.save_dir:
            env_args["save_dir"] = args.save_dir
        if isinstance(args.post_reset_delay_seconds, (int, float)):
            env_args["post_reset_delay_seconds"] = args.post_reset_delay_seconds
        if isinstance(args.step_pause_seconds, (int, float)):
            env_args["step_pause_seconds"] = args.step_pause_seconds

    trainer = BCTrainer(
        Agent(model, tokenizer),
        [task_class(client_args=env_args, n_clients=1)],
        args,
    )

    trainer.train()


if __name__ == "__main__":
    main()
