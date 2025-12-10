import json
import os
from dataclasses import asdict
from datetime import timedelta
from functools import partial
from typing import Sequence

import jsonlines
import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import broadcast, gather_object
from agentenv.controller import Agent
from agentenv.controller.agent import Agent
from agentenv.controller.task import BaseTask, GenerationConfig
from agentenv.controller.utils import BaseTrainer
from agentenv.trainer.utils import set_seed
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
from transformers import GenerationConfig


class DistributedEvaluator(BaseTrainer):
    def __init__(self, agent: Agent, tasks: Sequence[BaseTask], args) -> None:
        self.agent = agent
        self.tasks = tasks
        self.args = asdict(args)

        # data & loader
        self.raw_dataset = None

        # accelerator
        self.accelerator = None

        self.create_accelerator()
        self.set_seed()
        self.setup_tokenizer()
        self.get_raw_dataset()
        self.get_inference_dataloader()

    def create_accelerator(self):
        """
        Create the accelerator.
        """
        self.accelerator = Accelerator(
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))],
        )

    def set_seed(self):
        """
        Set the random seed.
        """
        set_seed(self.args["seed"] + self.accelerator.process_index)

    def setup_tokenizer(self):
        """
        Setup the tokenizer.
        Avoid overriding model-specific special tokens.
        """
        try:
            if getattr(self.agent.tokenizer, "pad_token_id", None) is None:
                unk_id = getattr(self.agent.tokenizer, "unk_token_id", None)
                eos_id = getattr(self.agent.tokenizer, "eos_token_id", None)
                self.agent.tokenizer.pad_token_id = unk_id if unk_id is not None else eos_id
            if getattr(self.agent.tokenizer, "eos_token_id", None) is None:
                if hasattr(self.agent.tokenizer, "convert_tokens_to_ids"):
                    maybe_eos = self.agent.tokenizer.convert_tokens_to_ids("</s>")
                    if isinstance(maybe_eos, int) and maybe_eos >= 0:
                        self.agent.tokenizer.eos_token_id = maybe_eos
        except Exception:
            pass
        # Handle processors without __len__ (e.g., Qwen2VLProcessor)
        tok = self.agent.tokenizer
        vocab_size = None
        # Prefer len(tokenizer) to include added tokens if available
        try:
            if hasattr(tok, "__len__"):
                vocab_size = len(tok)
        except Exception:
            vocab_size = None
        if vocab_size is None and hasattr(tok, "vocab_size"):
            vocab_size = tok.vocab_size
        if vocab_size is None and hasattr(tok, "get_vocab"):
            try:
                vocab_size = len(tok.get_vocab())
            except Exception:
                vocab_size = None
        if vocab_size is None and hasattr(tok, "tokenizer"):
            inner_tok = tok.tokenizer
            try:
                if hasattr(inner_tok, "__len__"):
                    vocab_size = len(inner_tok)
            except Exception:
                vocab_size = getattr(inner_tok, "vocab_size", None)
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
        with self.accelerator.main_process_first():
            path = os.path.expanduser(self.args["inference_file"])
            ext = os.path.splitext(path)[1].lower()
            records = []
            if ext in [".jsonl", ".jsonlines", ".jl"]:
                with jsonlines.open(path, mode="r") as reader:
                    for obj in reader:
                        if isinstance(obj, dict):
                            records.append(obj)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, list):
                    for it in obj:
                        if isinstance(it, dict):
                            records.append(it)
                elif isinstance(obj, dict):
                    # Flatten dict-of-lists. If list elements are dicts, keep them;
                    # if elements are strings/IDs, we only need the count for indexing.
                    tmp = []
                    for v in obj.values():
                        if isinstance(v, list):
                            if v and isinstance(v[0], dict):
                                tmp.extend([x for x in v if isinstance(x, dict)])
                            else:
                                tmp.extend([{"_raw_id": x} for x in v if isinstance(x, (str, int))])
                    records = tmp

            # Minimal dataset: evaluator只用到 item_id 的索引后缀，构造占位 item_id
            items = []
            for i, _ in enumerate(records):
                items.append({"item_id": f"{self.args['task_name']}_{i}"})

            self.raw_dataset = DatasetDict({"inference": Dataset.from_list(items)})
            self.accelerator.print("Raw data:", {k: len(v) for k, v in self.raw_dataset.items()})

    def get_inference_dataloader(self):
        def collate_fn(batch):
            result = {
                "data_idxs": [int(item["item_id"].split("_")[-1]) for item in batch]
            }
            return result

        with self.accelerator.main_process_first():
            self.inference_dataloader = DataLoader(
                self.raw_dataset["inference"],
                batch_size=self.args["eval_batch_size"],
                num_workers=self.args["num_workers"],
                pin_memory=True,
                collate_fn=partial(collate_fn),
            )

            self.accelerator.print(
                "Number of inference batches:", len(self.inference_dataloader)
            )

    def generate(self, dataloader=None):
        self.optimizer = AdamW(self.agent.model.parameters())
        self.agent.model, self.optimizer, self.inference_dataloader = (
            self.accelerator.prepare(
                self.agent.model, self.optimizer, self.inference_dataloader
            )
        )
        self.agent.model.eval()
        all_rewards = []
        all_success = []
        if dataloader is None:
            dataloader = self.inference_dataloader

        for _, batch in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            disable=not self.accelerator.is_main_process,
            desc="Inference Gen Loop",
        ):
            data_idxs = batch["data_idxs"]
            with torch.no_grad():
                # Robustly resolve eos/pad ids to support processors (e.g., Qwen2VLProcessor)
                eos_id = None
                pad_id = None
                gen_cfg = getattr(self.agent.model, "generation_config", None)
                if gen_cfg is not None:
                    eos_id = getattr(gen_cfg, "eos_token_id", None)
                    pad_id = getattr(gen_cfg, "pad_token_id", None)

                tok = getattr(self.agent, "tokenizer", None)
                def _get(obj, name):
                    return getattr(obj, name, None) if obj is not None else None

                if eos_id is None:
                    eos_id = _get(tok, "eos_token_id")
                    if eos_id is None and _get(tok, "tokenizer") is not None:
                        eos_id = _get(tok.tokenizer, "eos_token_id")

                if pad_id is None:
                    pad_id = _get(tok, "pad_token_id")
                    if pad_id is None and _get(tok, "tokenizer") is not None:
                        pad_id = _get(tok.tokenizer, "pad_token_id")
                    if pad_id is None:
                        # Fallback to unk -> eos
                        unk_id = _get(tok, "unk_token_id") or (
                            _get(_get(tok, "tokenizer"), "unk_token_id") if tok is not None else None
                        )
                        pad_id = unk_id if unk_id is not None else eos_id

                exps = self.eval(
                    generation_config=GenerationConfig(
                        max_length=4096,
                        do_sample=self.args["do_sample"],
                        temperature=self.args["temperature"],
                        eos_token_id=eos_id,
                        pad_token_id=pad_id,
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
                if self.accelerator.is_main_process:
                    with jsonlines.open(self.args["output_file"], mode="a") as f:
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
        self.accelerator.print("\n\n==== Inference Evaluation ====\n")
        self.accelerator.print(f"Score: {mean_reward:.5f}")
        self.accelerator.print(f"Success: {mean_success:.5f}")

