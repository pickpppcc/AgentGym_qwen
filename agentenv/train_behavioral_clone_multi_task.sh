#!/usr/bin/env bash
set -euo pipefail

# Multi-task launcher (OSWorld) - tuned from .vscode/launch.json.
# Run from: /data/pengchen/OSWorld/Osworld/AgentGym2/AgentGym/agentenv

exp_name="${EXP_NAME:-behavioral_clone_osworld_smoke}"

n_epochs="${N_EPOCHS:-2000}"
# 可选：通过 CUDA_VISIBLE_DEVICES 指定卡列表，或在外部传入
# export CUDA_VISIBLE_DEVICES=0
# GPU 选择：支持命令行传入（例如："1 2 3" 或 "1,2,3"），也支持环境变量 GPUS
# 如遇 OOM 可减少卡数/切单卡
GPUS="${GPUS:-5}"
export GPUS
export DEEPSPEED_COMM_AUTO_MPI_DISCOVERY=0
GPU_LIST_INPUT="${GPUS:-$*}"
if [ -n "${GPU_LIST_INPUT}" ]; then
  GPU_LIST_NORMALIZED="$(echo "${GPU_LIST_INPUT}" | tr ',' ' ' | xargs)"
  CUDA_VISIBLE_DEVICES_VAL="$(echo "${GPU_LIST_NORMALIZED}" | tr ' ' ',')"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VAL}"
  set -- ${GPU_LIST_NORMALIZED}
  NUM_GPUS_CALC=$#
else
  NUM_GPUS_CALC="${NUM_GPUS:-1}"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# accelerator config（支持多卡：通过 NUM_GPUS 控制进程数，默认 1）
NUM_GPUS="${NUM_GPUS_CALC}"
num_processes="${NUM_GPUS}"
main_process_port='8897'
# 生成 accelerate 配置文件（MULTI_GPU + DeepSpeed），确保不触发 MPI 探测
ACC_CFG_DIR="/data/pengchen/.cache/huggingface/accelerate"
ACC_CFG_FILE="${ACC_CFG_DIR}/default_config.yaml"
mkdir -p "${ACC_CFG_DIR}"
cat > "${ACC_CFG_FILE}" <<'YAML'
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 1
machine_rank: 0
num_processes: REPLACE_NUM_PROCESSES
main_process_port: REPLACE_PORT
mixed_precision: bf16
YAML
sed -i "s/REPLACE_NUM_PROCESSES/${num_processes}/" "${ACC_CFG_FILE}"
sed -i "s/REPLACE_PORT/${main_process_port}/" "${ACC_CFG_FILE}"

# 单卡时显式设置分布式环境，避免 DeepSpeed 触发 MPI 探测；多卡交由 accelerate/deepspeed 处理
if [ "${num_processes}" = "1" ]; then
  export RANK=0
  export WORLD_SIZE=1
  export LOCAL_RANK=0
  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT=${main_process_port}
else
  unset RANK WORLD_SIZE LOCAL_RANK
fi

# 显式使用该配置
config_file="${ACC_CFG_FILE}"
LAUNCH_CFG=(--config_file "${config_file}")

# training arguments
train_file='/data/pengchen/OSWorld/Osworld/AgentGym2/AgentGym/data/test_real_smoke_4turn.json'
inference_file='/data/pengchen/OSWorld/Osworld/AgentGym2/AgentGym/data/test_real_smoke_4turn.json'
model_train_path="/data/pengchen/OSWorld/Osworld/models/Qwen/Qwen3-VL-4B-Instruct"
# use absolute path to avoid cwd issues
model_save_path="$(pwd)/outputs/${exp_name}"

batch_size="4"
eval_batch_size="1"
gradient_accumulation_steps="4"
max_input_length="7200"
num_workers="0"
learning_rate="1e-5"
weight_decay="0"
warmup_step="-100"
clip_grad_norm="1"
seed="42"

logging_epoch_freq="1"
evaluating_epoch_freq="100"
saving_epoch_freq="100"
logging_step_freq="5"

# wandb config
wandb_log="False"
wandb_project="agentenv"
wandb_run_name="${exp_name}"

# environment parameters
data_len="15"
timeout="2400"

# eval
task_list=("osworld")
test_file_list=("/data/pengchen/OSWorld/Osworld/AgentGym2/AgentGym/data/test_real_smoke_4turn.json")
# task_list=("webshop" "alfworld" "textcraft" "sciworld")
# # eval parameters
# test_file_list=("PATH/TO/webshop_test.json" "PATH/TO/alfworld_test.json" "PATH/TO/textcraft_test.json" "PATH/TO/sciworld_test_small.json")
do_sample="False"
temperature="1.0"
sample_num="2"
max_round_list=("15" "30" "20" "30")
env_server_base_list=("http://127.0.0.1:36001" "http://127.0.0.1:36002" "http://127.0.0.1:36008" "http://127.0.0.1:36010")

mkdir -p "${model_save_path}"
# 确保使用 /data 前缀下的 osworld 环境，可找到 accelerate
export PATH=/data/pengchen/miniconda3/envs/osworld/bin:$PATH
export DS_BUILD_OPS=0
export DS_BUILD_SPARSE_ATTN=0
export DS_BUILD_AIO=0
export DEEPSPEED_COMM_AUTO_MPI_DISCOVERY=0
# 禁用 FusedAdam 编译并优先使用系统 libstdc++.so.6
export DS_BUILD_FUSED_ADAM=0
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:${LD_PRELOAD:-}
# 避免主机 IB/NVLink 环境差异导致 NCCL 初始化失败（单机调通为主）
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
# step1: train         # --config_file "${config_file}" \
accelerate launch \
        "${LAUNCH_CFG[@]}" \
        --num_processes=${num_processes} \
        --num_machines=1 --machine_rank=0 \
        --main_process_port=${main_process_port} \
    train_behavioral_clone.py \
        --train_file "${train_file}" \
        --inference_file "${inference_file}" \
        --test_file "${test_file_list[0]}" \
        --model_train_path "${model_train_path}" \
        --model_save_path "${model_save_path}" \
        --task_name "${task_list[0]}" \
        --batch_size "${batch_size}" \
        --eval_batch_size "${eval_batch_size}" \
        --n_epochs "${n_epochs}" \
        --num_workers "${num_workers}" \
        --learning_rate "${learning_rate}" \
        --weight_decay "${weight_decay}" \
        --warmup_step "${warmup_step}" \
        --clip_grad_norm "${clip_grad_norm}" \
        --evaluating_epoch_freq "${evaluating_epoch_freq}" \
        --logging_epoch_freq "${logging_epoch_freq}" \
        --saving_epoch_freq "${saving_epoch_freq}" \
        --logging_step_freq "${logging_step_freq}" \
        --seed "${seed}" \
        --max_input_length "${max_input_length}" \
        --max_round "${max_round_list[0]}" \
        --gradient_accumulation_steps "${gradient_accumulation_steps}" \
        --wandb_log "${wandb_log}" \
        --wandb_project "${wandb_project}" \
        --wandb_run_name "${wandb_run_name}" \
        --env_server_base "http://127.0.0.1:36001" \
        --data_len "${data_len}" \
        --timeout "${timeout}" \
        --observe_format "screenshot_base64" \
        --create_cfg_json '{"provider_name":"docker","auto_evaluate":"True","path_to_vm":"/data/pengchen/OSWorld/env/Ubuntu.qcow2","snapshot_name":"init_state","action_space":"computer_13","require_a11y_tree":false,"require_terminal":false,"headless":true}' \
        --use_deepspeed true \
        --examples_dir "/data/pengchen/OSWorld/Osworld/OSWorld/evaluation_examples/examples/chrome" \
        --save_dir "/data/pengchen/OSWorld/Osworld/OSWorld/results/pyautogui/screenshot/ui-tars/agentgym/chrome" \
        --post_reset_delay_seconds "3" \
        --step_pause_seconds "3" \
        --train_max_samples "20000" \
        --debug "true" \
        --debug_max_tokens "512" \
        --assistant_only_loss "true"
                
# step2: eval on test dataset (only if trained model folder exists)
if [ -d "${model_save_path}/train_epoch_${n_epochs}" ]; then
  for index in "${!task_list[@]}";
  do
      cur_task=${task_list[$index]}
      cur_test_file="${test_file_list[$index]}"
      cur_max_round=${max_round_list[$index]}
      cur_env_server_base=${env_server_base_list[$index]}
      cur_eval_output_file="${model_save_path}/eval_${cur_task}.jsonl"

      accelerate launch \
              "${LAUNCH_CFG[@]}" \
              --num_processes=${num_processes} \
              --main_process_port=${main_process_port} \
              utils/distributed_eval_task.py \
              --model_path "${model_save_path}/train_epoch_${n_epochs}" \
              --output_file "${cur_eval_output_file}" \
              --inference_file "${cur_test_file}" \
              --task_name "${cur_task}" \
              --eval_batch_size "${eval_batch_size}" \
              --num_workers "${num_workers}" \
              --seed "${seed}" \
              --do_sample "${do_sample}" \
              --max_round "${cur_max_round}" \
              --env_server_base "${cur_env_server_base}" \
              --data_len "${data_len}" \
              --timeout "${timeout}" \
              --observe_format "screenshot_base64" \
              --create_cfg_json '{
                "provider_name":"docker",
                "path_to_vm":"/data/pengchen/OSWorld/env/Ubuntu.qcow2",
                "snapshot_name":"init_state",
                "action_space":"computer_13",
                "require_a11y_tree":false,
                "require_terminal":false,
                "headless":true
              }' \
              --examples_dir "/data/pengchen/OSWorld/Osworld/OSWorld/evaluation_examples/examples/chrome" \
              --save_dir "/data/pengchen/OSWorld/Osworld/OSWorld/results/pyautogui/screenshot/ui-tars/agentgym/chrome" \
              --post_reset_delay_seconds "3" \
              --step_pause_seconds "3"
  done
else
  echo "[WARN] Skipping eval: model directory '${model_save_path}/train_epoch_${n_epochs}' not found."
fi
