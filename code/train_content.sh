#!/usr/bin/env bash
set -euo pipefail

# repo root（自動抓：code/ 的上一層）
B4M_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

CONTENT_DIR="${B4M_ROOT}/assets/content/corgi"

STYLE1_DIR="${B4M_ROOT}/assets/style/ChineseInkPainting"
STYLE2_DIR="${B4M_ROOT}/assets/style/glass"
STYLE3_DIR="${B4M_ROOT}/assets/style/yarn"

OUT_DIR="${B4M_ROOT}/outputs/content_run_$(date +%Y%m%d_%H%M%S)"

cd "${B4M_ROOT}"

# Force this repo's src/ to take precedence (avoid env editable diffusers from other repo)
export PYTHONPATH="${B4M_ROOT}/src:${PYTHONPATH:-}"


# 檢查資料夾存在
for d in "${CONTENT_DIR}" "${STYLE1_DIR}" "${STYLE2_DIR}" "${STYLE3_DIR}"; do
  test -d "${d}" || { echo "Missing dir: ${d}"; exit 1; }
done

# 重要注意事項：參數之間不能有任何註解，請把註解寫在整個指令的下方，否則可能導致參數沒有正確吃到進而發生異常
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
  --max_train_steps=50 \
  --mixed_precision=fp16 \
  --seed=0 \
  --checkpointing_steps=10

# 原本設定
#  --resolution=512
#  --gradient_accumulation_steps=4
#  --max_train_steps=1000
#  --checkpointing_steps=100

# 當前設定僅測試是否能train
#  --resolution=256    
#  --gradient_accumulation_steps=1
#  --max_train_steps=50
#  --checkpointing_steps=10
