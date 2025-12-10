import time
from dataclasses import asdict, dataclass, field
from typing import Optional
import json
import os

import torch
import transformers
from agentenv.controller import Agent
from agentenv.envs import (
    AcademiaTask,
    AlfWorldTask,
    BabyAITask,
    MazeTask,
    MovieTask,
    SciworldTask,
    SheetTask,
    SqlGymTask,
    TextCraftTask,
    TodoTask,
    WeatherTask,
    WebarenaTask,
    WebshopTask,
    WordleTask,
    OSWorldTask
)
from agentenv.trainer.distributed_evaluator import DistributedEvaluator
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq, AutoConfig


@dataclass
class EvalArguments:
    model_path: str
    # required path
    output_file: str
    inference_file: str = field(metadata={"help": "Inference dataset."})
    task_name: str = field(default="webshop", metadata={"help": "Task name for evaluation"})
    # optional: base model path for loading processor when checkpoint lacks it
    base_model_path: Optional[str] = field(default=None, metadata={"help": "Base pretrained model path for processor"})

    # eval config
    eval_batch_size: int = field(default=8, metadata={"help": "Batch size for evaluation."})
    num_workers: int = field(default=8, metadata={"help": "Number of subprocesses to use for data loading."})
    seed: int = field(default=42)
    do_sample: bool = field(default=False, metadata={"help": "Do sampling or not."})
    temperature: float = field(default=1.0, metadata={"help": "Sampling temperature."})

    # conversation rounds
    max_round: int = field(
        default=6,
        metadata={"help": "Interaction rounds between agents and environment"},
    )

    weight_decay: float = field(default=1e-6, metadata={"help": "Weight decay for regularization."})

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


def main():
    parser = transformers.HfArgumentParser(EvalArguments)
    (args,) = parser.parse_args_into_dataclasses()

    # 规范化本地模型路径，优先本地加载，避免误判为 HuggingFace Hub 仓库名
    args.model_path = os.path.abspath(os.path.expanduser(args.model_path))
    # 通过 AutoConfig 检测模型类型是否为 Qwen2-VL
    load_kwargs = {"local_files_only": True}
    try:
        cfg = AutoConfig.from_pretrained(args.model_path, **load_kwargs)
        model_type = getattr(cfg, "model_type", "") or cfg.__class__.__name__.lower()
        is_qwen2vl = ("qwen2_vl" in model_type) or ("qwen2-vl" in model_type)
    except Exception:
        # 回退到路径关键字判断
        is_qwen2vl = "qwen2-vl" in args.model_path.lower()
    print(f"is_qwen2vl: {args.model_path}")
    # exit()
    if is_qwen2vl:
        # Qwen2-VL 使用 AutoProcessor + AutoModelForVision2Seq
        # 优先从 checkpoint 目录加载，如果没有预处理配置则退回到 base_model_path
        processor_source = args.model_path
        preprocessor_cfg = os.path.join(args.model_path, "preprocessor_config.json")
        if not os.path.exists(preprocessor_cfg) and args.base_model_path:
            processor_source = os.path.abspath(os.path.expanduser(args.base_model_path))
        processor = AutoProcessor.from_pretrained(processor_source, **load_kwargs)
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, **load_kwargs
        )
        tokenizer = processor
    else:
        # 纯文本模型使用 AutoTokenizer + AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, **load_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, **load_kwargs
        )

    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    # task_name - task dict
    task_classes = {
        "webshop": WebshopTask,
        "alfworld": AlfWorldTask,
        "babyai": BabyAITask,
        "sciworld": SciworldTask,
        "textcraft": TextCraftTask,
        "webarena": WebarenaTask,
        "sqlgym": SqlGymTask,
        "lmrlgym_maze": MazeTask,
        "lmrlgym_wordle": WordleTask,
        "weather": WeatherTask,
        "todo": TodoTask,
        "movie": MovieTask,
        "sheet": SheetTask,
        "academia": AcademiaTask,
        "babyai": BabyAITask,
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

    distributed_evaluator = DistributedEvaluator(
        Agent(model, tokenizer),
        [task_class(client_args=env_args, n_clients=1)],
        args,
    )
    start_time = time.time()
    distributed_evaluator.generate()
    process_time = time.time() - start_time
    print(f"==== {process_time} seconds ====")


if __name__ == "__main__":
    main()
