#!/usr/bin/env bash
set -euo pipefail

# repo root（自動抓：code/ 的上一層）
B4M_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

# ===== 三份 content（你要的三種）=====
CONTENT_DIR="${B4M_ROOT}/assets/content"
CONTENT1_DIR="${CONTENT_DIR}/corgi"
CONTENT2_DIR="${CONTENT_DIR}/eagle"
CONTENT3_DIR="${CONTENT_DIR}/elephant"

# ===== 每個 content 各用一個 token，避免混在同一個 snq =====
# content tokens:
#   corgi    -> snq
#   eagle    -> snr
#   elephant -> snp
CONTENT1_PROMPT="a photo of snq corgi"
CONTENT2_PROMPT="a photo of snr eagle"
CONTENT3_PROMPT="a photo of snp elephant"

# ===== 三個 style（你現在的批次跑法）=====
STYLE1_NAME="ChineseInkPainting"
STYLE1_DIR="${B4M_ROOT}/assets/style/ChineseInkPainting"
STYLE1_PROMPT="w@z Chinese ink painting style, ink wash painting, rice paper texture, expressive brush strokes, monochrome"

STYLE2_NAME="glass"
STYLE2_DIR="${B4M_ROOT}/assets/style/glass"
STYLE2_PROMPT="w@z crafted from clean glass, transparent material, glossy reflections, refraction"

STYLE3_NAME="yarn"
STYLE3_DIR="${B4M_ROOT}/assets/style/yarn"
STYLE3_PROMPT="w@z yarn art style, knitted or crochet texture, wool fibers, handcrafted"

OUT_ROOT="${B4M_ROOT}/outputs/style_run_$(date +%Y%m%d_%H%M%S)"

cd "${B4M_ROOT}"

# Force this repo's src/ to take precedence (avoid env editable diffusers from other repo)
export PYTHONPATH="${B4M_ROOT}/src:${PYTHONPATH:-}"

# ===== 防呆：資料夾存在 + 不含子資料夾（避免 .cache 事故）=====
check_dir_clean () {
  local d="$1"
  test -d "$d" || { echo "Missing dir: $d"; exit 1; }
  if find "$d" -mindepth 1 -maxdepth 1 -type d | grep -q .; then
    echo "ERROR: Found subdirectories inside: $d"
    echo "Please remove/move them (e.g., .cache) so the folder contains only image files."
    find "$d" -mindepth 1 -maxdepth 1 -type d -print
    exit 1
  fi
  if ! find "$d" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" \) | head -n 1 | grep -q .; then
    echo "ERROR: No image files found in: $d"
    exit 1
  fi
}

check_dir_clean "$CONTENT1_DIR"
check_dir_clean "$CONTENT2_DIR"
check_dir_clean "$CONTENT3_DIR"
check_dir_clean "$STYLE1_DIR"
check_dir_clean "$STYLE2_DIR"
check_dir_clean "$STYLE3_DIR"

mkdir -p "${OUT_ROOT}"

# ===== 超參數：先跑起來驗證（對齊你 train_content.sh 可跑通配置）=====
RESOLUTION=256
RANK=64
TRAIN_BS=1
GRAD_ACC=1
LR=1e-4
LR2=1e-5
MAX_STEPS=50
SEED=0
CKPT_STEPS=10
REPORT_TO="tensorboard"

run_one_style () {
  local style_name="$1"
  local style_dir="$2"
  local style_prompt="$3"
  local out_dir="${OUT_ROOT}/${style_name}"

  mkdir -p "$out_dir"

  echo "================================================="
  echo "[Style Train] ${style_name}"
  echo "  style   : ${style_dir}"
  echo "  out     : ${out_dir}"
  echo "  style prompt : ${style_prompt}"
  echo "  content1: ${CONTENT1_DIR} | ${CONTENT1_PROMPT}"
  echo "  content2: ${CONTENT2_DIR} | ${CONTENT2_PROMPT}"
  echo "  content3: ${CONTENT3_DIR} | ${CONTENT3_PROMPT}"
  echo "================================================="

  accelerate launch code/train_style.py \
    --multi_style \
    --pretrained_model_name_or_path="${MODEL_NAME}" \
    --pretrained_vae_model_name_or_path="${VAE_PATH}" \
    --instance_data_dir="${style_dir}" \
    --instance_data_dir_2="${CONTENT1_DIR}" \
    --instance_data_dir_3="${CONTENT2_DIR}" \
    --instance_data_dir_4="${CONTENT3_DIR}" \
    --output_dir="${out_dir}" \
    --instance_prompt="${style_prompt}" \
    --instance_prompt_2="${CONTENT1_PROMPT}" \
    --instance_prompt_3="${CONTENT2_PROMPT}" \
    --instance_prompt_4="${CONTENT3_PROMPT}" \
    --resolution="${RESOLUTION}" \
    --rank="${RANK}" \
    --train_batch_size="${TRAIN_BS}" \
    --gradient_accumulation_steps="${GRAD_ACC}" \
    --gradient_checkpointing \
    --learning_rate_2="${LR2}" \
    --learning_rate="${LR}" \
    --report_to="${REPORT_TO}" \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps="${MAX_STEPS}" \
    --mixed_precision=fp16 \
    --seed="${SEED}" \
    --checkpointing_steps="${CKPT_STEPS}"
}

# 逐一訓練三種 style
run_one_style "${STYLE1_NAME}" "${STYLE1_DIR}" "${STYLE1_PROMPT}"
run_one_style "${STYLE2_NAME}" "${STYLE2_DIR}" "${STYLE2_PROMPT}"
run_one_style "${STYLE3_NAME}" "${STYLE3_DIR}" "${STYLE3_PROMPT}"

echo "All style trainings done. Output root: ${OUT_ROOT}"
