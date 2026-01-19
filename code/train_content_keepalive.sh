#!/usr/bin/env bash
set -euo pipefail

B4M_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

CONTENT_DIR="${B4M_ROOT}/assets/content/corgi"
STYLE1_DIR="${B4M_ROOT}/assets/style/ChineseInkPainting"
STYLE2_DIR="${B4M_ROOT}/assets/style/glass"
STYLE3_DIR="${B4M_ROOT}/assets/style/yarn"

OUT_DIR="${B4M_ROOT}/outputs/exp_a6000_keepalive_20260119"
mkdir -p "${OUT_DIR}"

cd "${B4M_ROOT}"
export PYTHONPATH="${B4M_ROOT}/src:${PYTHONPATH:-}"

for d in "${CONTENT_DIR}" "${STYLE1_DIR}" "${STYLE2_DIR}" "${STYLE3_DIR}"; do
  test -d "${d}" || { echo "Missing dir: ${d}"; exit 1; }
done

# 初始 max steps（你要的）
MAX_STEPS="${MAX_STEPS:-10000000}"
# 每次「正常結束」後自動把 max steps 往上加，避免秒退→秒重跑
MAX_STEPS_INCREMENT="${MAX_STEPS_INCREMENT:-1000000}"

LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# 無人值守：失敗會重啟；成功結束也會重啟（並自動提高 max steps 以便繼續訓練）
while true; do
  ts="$(date +%Y%m%d_%H%M%S)"
  log_file="${LOG_DIR}/train_${ts}.log"

  RESUME_ARGS=()
  if compgen -G "${OUT_DIR}/checkpoint-*" > /dev/null; then
    echo "[${ts}] [resume] found checkpoints, resume_from_checkpoint=latest" | tee -a "${log_file}"
    RESUME_ARGS+=(--resume_from_checkpoint="latest")
  else
    echo "[${ts}] [resume] no checkpoint found, start fresh" | tee -a "${log_file}"
  fi

  echo "[${ts}] [run] starting training (max_steps=${MAX_STEPS}, inc=${MAX_STEPS_INCREMENT})" | tee -a "${log_file}"

  set +e
  accelerate launch code/train_content.py \
    --multi_style \
    --pretrained_model_name_or_path="${MODEL_NAME}" \
    --pretrained_vae_model_name_or_path="${VAE_PATH}" \
    --instance_data_dir="${CONTENT_DIR}" \
    --instance_data_dir_2="${STYLE1_DIR}" \
    --instance_data_dir_3="${STYLE2_DIR}" \
    --instance_data_dir_4="${STYLE3_DIR}" \
    --output_dir="${OUT_DIR}" \
    --instance_prompt="a photo of snq dog" \
    --instance_prompt_2="w@z Chinese ink painting style" \
    --instance_prompt_3="w@z crafted from clean glass" \
    --instance_prompt_4="w@z yarn art style" \
    --resolution=256 \
    --rank=64 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --learning_rate_2=1e-5 \
    --learning_rate=1e-4 \
    --report_to="tensorboard" \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps="${MAX_STEPS}" \
    --mixed_precision=fp16 \
    --seed=0 \
    --checkpointing_steps=1000 \
    --checkpoints_total_limit=2 \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee -a "${log_file}"

  exit_code="${PIPESTATUS[0]}"
  set -e

  ts2="$(date +%Y%m%d_%H%M%S)"
  if [[ "${exit_code}" -eq 0 ]]; then
    echo "[${ts2}] [ok] training exited normally (exit_code=0). Will restart and extend max_steps." | tee -a "${log_file}"
    MAX_STEPS=$((MAX_STEPS + MAX_STEPS_INCREMENT))
    echo "[${ts2}] [next] max_steps increased to ${MAX_STEPS}" | tee -a "${log_file}"
    sleep 10
  else
    echo "[${ts2}] [warn] training crashed (exit_code=${exit_code}). Restarting in 30s..." | tee -a "${log_file}"
    sleep 30
  fi
done
