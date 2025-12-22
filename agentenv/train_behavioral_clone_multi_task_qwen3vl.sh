#!/usr/bin/env bash
set -euo pipefail

# Run behavioral cloning (OSWorld) with the same args as .vscode/launch.json.
# Assumes agent_osworld server is already running (e.g. http://127.0.0.1:36001).
#
# Examples:
#   bash AgentGym2/AgentGym/agentenv/train_behavioral_clone_multi_task_qwen3vl.sh
#   GPUS=5 bash AgentGym2/AgentGym/agentenv/train_behavioral_clone_multi_task_qwen3vl.sh
#   GPUS=0,1 bash AgentGym2/AgentGym/agentenv/train_behavioral_clone_multi_task_qwen3vl.sh

ROOT_DIR="/data/pengchen/OSWorld/Osworld/AgentGym2/AgentGym/agentenv"
cd "${ROOT_DIR}"

# ---- Environment (match launch.json) ----
export PATH="/home/pengchen/miniconda3/envs/osworld/bin:${PATH}"
export PYTHONPATH="/data/pengchen/OSWorld/Osworld/AgentGym2/AgentGym/agentenv:${PYTHONPATH:-}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DEEPSPEED_COMM_AUTO_MPI_DISCOVERY="${DEEPSPEED_COMM_AUTO_MPI_DISCOVERY:-0}"
export DS_BUILD_OPS="${DS_BUILD_OPS:-0}"
export DS_BUILD_SPARSE_ATTN="${DS_BUILD_SPARSE_ATTN:-0}"
export DS_BUILD_AIO="${DS_BUILD_AIO:-0}"
export DS_BUILD_FUSED_ADAM="${DS_BUILD_FUSED_ADAM:-0}"
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# GPUs: can pass GPUS="5" or "0,1". Defaults to 5 to match your launch.json.
GPUS="${GPUS:-5}"
CUDA_VISIBLE_DEVICES_VAL="$(echo "${GPUS}" | tr ' ' ',' | tr -d '\n' )"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VAL}"

# accelerate
GPU_LIST_NORMALIZED="$(echo "${GPUS}" | tr ',' ' ' | xargs)"
set -- ${GPU_LIST_NORMALIZED}
NUM_GPUS="${NUM_GPUS:-$#}"
main_process_port="${MAIN_PROCESS_PORT:-8897}"

# ---- Args (match launch.json) ----
train_file="/data/pengchen/OSWorld/Osworld/AgentGym2/AgentGym/data/test_real_smoke_4turn.json"
inference_file="/data/pengchen/OSWorld/Osworld/AgentGym2/AgentGym/data/test_real_smoke_4turn.json"
test_file="/data/pengchen/OSWorld/Osworld/AgentGym2/AgentGym/data/test_real_smoke_4turn.json"

model_train_path="/data/pengchen/OSWorld/Osworld/models/Qwen/Qwen3-VL-4B-Instruct"
model_save_path="/data/pengchen/OSWorld/Osworld/AgentGym2/AgentGym/agentenv/outputs/behavioral_clone_osworld_smoke"

env_server_base="http://127.0.0.1:36001"
examples_dir="/data/pengchen/OSWorld/Osworld/OSWorld/evaluation_examples/examples/chrome"
save_dir="/data/pengchen/OSWorld/Osworld/OSWorld/results/pyautogui/screenshot/ui-tars/agentgym/chrome"

create_cfg_json='{"provider_name":"docker","auto_evaluate":"True","path_to_vm":"/data/pengchen/OSWorld/env/Ubuntu.qcow2","snapshot_name":"init_state","action_space":"computer_13","require_a11y_tree":false,"require_terminal":false,"headless":true}'

accelerate launch \
  --num_processes="${NUM_GPUS}" \
  --num_machines=1 --machine_rank=0 \
  --main_process_port="${main_process_port}" \
  train_behavioral_clone.py \
  --train_file "${train_file}" \
  --inference_file "${inference_file}" \
  --test_file "${test_file}" \
  --model_train_path "${model_train_path}" \
  --model_save_path "${model_save_path}" \
  --task_name "osworld" \
  --batch_size "4" \
  --eval_batch_size "1" \
  --n_epochs "2000" \
  --num_workers "0" \
  --learning_rate "1e-5" \
  --weight_decay "0" \
  --warmup_step "-100" \
  --clip_grad_norm "1" \
  --evaluating_epoch_freq "100" \
  --logging_epoch_freq "1" \
  --saving_epoch_freq "100" \
  --logging_step_freq "5" \
  --seed "42" \
  --max_input_length "7200" \
  --max_round "15" \
  --gradient_accumulation_steps "4" \
  --wandb_log "False" \
  --wandb_project "agentenv" \
  --wandb_run_name "behavioral_clone_osworld_smoke" \
  --env_server_base "${env_server_base}" \
  --data_len "15" \
  --timeout "2400" \
  --observe_format "screenshot_base64" \
  --create_cfg_json "${create_cfg_json}" \
  --use_deepspeed "true" \
  --examples_dir "${examples_dir}" \
  --save_dir "${save_dir}" \
  --post_reset_delay_seconds "3" \
  --step_pause_seconds "3" \
  --train_max_samples "20000" \
  --debug "true" \
  --debug_max_tokens "512" \
  --assistant_only_loss "true"

