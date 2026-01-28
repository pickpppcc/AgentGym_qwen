#!/usr/bin/env bash
set -euo pipefail

# Cluster launcher (OSWorld, Qwen3-VL) with multi-GPU ZeRO-3 sharding via Accelerate + DeepSpeed.
# Notes:
# - This uses ZeRO-3 sharding across GPUs (often what people mean by "model parallel" for training).
# - True tensor/pipeline parallel is NOT implemented in this repo; ZeRO-3 is the supported path.
#
# Examples:
#   GPUS=0 bash AgentGym2/AgentGym/agentenv/train_behavioral_clone_multi_task_qwen3vl_cluster.sh
#   GPUS=0,1,2,3 bash AgentGym2/AgentGym/agentenv/train_behavioral_clone_multi_task_qwen3vl_cluster.sh
#   GPUS="0 1 2 3" MAIN_PROCESS_PORT=8897 bash AgentGym2/AgentGym/agentenv/train_behavioral_clone_multi_task_qwen3vl_cluster.sh

ROOT_DIR="${ROOT_DIR:-/dfs/data/AgentGym_qwen/agentenv}"
cd "${ROOT_DIR}"

# ---- Environment ----
export DEEPSPEED_COMM_AUTO_MPI_DISCOVERY="${DEEPSPEED_COMM_AUTO_MPI_DISCOVERY:-0}"
export DS_BUILD_OPS="${DS_BUILD_OPS:-0}"
export DS_BUILD_SPARSE_ATTN="${DS_BUILD_SPARSE_ATTN:-0}"
export DS_BUILD_AIO="${DS_BUILD_AIO:-0}"
export DS_BUILD_FUSED_ADAM="${DS_BUILD_FUSED_ADAM:-0}"
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- GPU selection ----
# GPUS can be: "0" | "0,1" | "0 1" (defaults to "0")
GPUS="${GPUS:-0}"
GPU_LIST_NORMALIZED="$(echo "${GPUS}" | tr ',' ' ' | xargs)"
CUDA_VISIBLE_DEVICES_VAL="$(echo "${GPU_LIST_NORMALIZED}" | tr ' ' ',')"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VAL}"

# Count GPUs
set -- ${GPU_LIST_NORMALIZED}
NUM_GPUS="${NUM_GPUS:-$#}"
if [ "${NUM_GPUS}" -lt 1 ]; then
  echo "[ERR] NUM_GPUS resolved to <1. Check GPUS='${GPUS}'." 1>&2
  exit 2
fi

main_process_port="${MAIN_PROCESS_PORT:-8897}"
mixed_precision="${MIXED_PRECISION:-bf16}"

# ---- Args (cluster paths; keep as-is) ----
train_file="/dfs/data/AgentGym_qwen/data/processed_conversations_vl_train.json"
inference_file="/dfs/data/AgentGym_qwen/data/processed_conversations_vl_validation.json"
test_file="/dfs/data/AgentGym_qwen/data/processed_conversations_vl_test.json"

model_train_path="/dfs/data/models/Qwen3-VL-4B-Instruct"
model_save_path="/dfs/data/AgentGym_qwen/agentenv/outputs"

env_server_base="http://10.110.146.172:36002"
examples_dir="/data/pengchen/OSWorld/Osworld/OSWorld/evaluation_examples/examples/chrome"
save_dir="/data/pengchen/OSWorld/Osworld/OSWorld/results/pyautogui/screenshot/ui-tars/agentgym/chrome"

create_cfg_json='{"provider_name":"docker","auto_evaluate":"True","path_to_vm":"/data/pengchen/OSWorld/env/Ubuntu.qcow2","snapshot_name":"init_state","action_space":"computer_13","require_a11y_tree":false,"require_terminal":false,"headless":true}'

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} | NUM_GPUS=${NUM_GPUS} | PORT=${main_process_port} | MP=${mixed_precision}"

accelerate launch \
  --num_processes "${NUM_GPUS}" \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_port "${main_process_port}" \
  --mixed_precision "${mixed_precision}" \
  train_behavioral_clone.py \
  --train_file "${train_file}" \
  --inference_file "${inference_file}" \
  --test_file "${test_file}" \
  --model_train_path "${model_train_path}" \
  --model_save_path "${model_save_path}" \
  --task_name "osworld" \
  --batch_size "2" \
  --eval_batch_size "1" \
  --n_epochs "2000" \
  --num_workers "0" \
  --learning_rate "1e-5" \
  --weight_decay "0" \
  --warmup_step "-100" \
  --clip_grad_norm "1" \
  --evaluating_epoch_freq "100" \
  --logging_epoch_freq "1" \
  --saving_epoch_freq "1" \
  --save_latest_only "true" \
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
