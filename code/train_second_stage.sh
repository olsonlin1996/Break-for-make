#!/usr/bin/env bash
set -euo pipefail

B4M_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${B4M_ROOT}"

export PYTHONPATH="${B4M_ROOT}/src:${PYTHONPATH:-}"

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

# =========================
# 1) 三種 content（資料夾 + token + prompt）
# =========================
CONTENT_DIR="${B4M_ROOT}/assets/content"

CONTENT1_NAME="corgi"
CONTENT1_DIR="${CONTENT_DIR}/corgi"
CONTENT1_TOKEN="snq"
CONTENT1_PROMPT="a photo of ${CONTENT1_TOKEN} corgi"

CONTENT2_NAME="eagle"
CONTENT2_DIR="${CONTENT_DIR}/eagle"
CONTENT2_TOKEN="snr"
CONTENT2_PROMPT="a photo of ${CONTENT2_TOKEN} eagle"

CONTENT3_NAME="elephant"
CONTENT3_DIR="${CONTENT_DIR}/elephant"
CONTENT3_TOKEN="snp"
CONTENT3_PROMPT="a photo of ${CONTENT3_TOKEN} elephant"

# =========================
# 2) 三種 style（資料夾 + prompt）
# =========================
STYLE1_NAME="ChineseInkPainting"
STYLE1_DIR="${B4M_ROOT}/assets/style/ChineseInkPainting"
STYLE1_PROMPT="w@z Chinese ink painting style, ink wash painting, rice paper texture, expressive brush strokes, monochrome"

STYLE2_NAME="glass"
STYLE2_DIR="${B4M_ROOT}/assets/style/glass"
STYLE2_PROMPT="w@z crafted from clean glass, transparent material, glossy reflections, refraction"

STYLE3_NAME="yarn"
STYLE3_DIR="${B4M_ROOT}/assets/style/yarn"
STYLE3_PROMPT="w@z yarn art style, knitted or crochet texture, wool fibers, handcrafted"

# =========================
# 3) 輸出資料夾
# =========================
OUT_ROOT="/mnt/cglab/olson/Break-for-make-outputs/second_stage_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_ROOT}"

# =========================
# 4) 超參數（模式 A：先跑起來驗證）
# =========================
# Paper-aligned override: LR=1e-4, BS=1, rank=64; RESOLUTION=512, grad_acc=4; max_steps kept at 50 (~few dozen)

# RESOLUTION=256
RESOLUTION=512
RANK=64
TRAIN_BS=1
# GRAD_ACC=1
GRAD_ACC=4
LR=1e-4
LR2=1e-4
MAX_STEPS=50
SEED=0
# CKPT_STEPS=10
CKPT_STEPS=50
REPORT_TO="tensorboard"

NUM_VAL=2
VAL_EPOCHS=10

# =========================
# 5) 工具
# =========================
check_dir_clean () {
  local d="$1"
  test -d "$d" || { echo "Missing dir: $d"; exit 1; }
  if find "$d" -mindepth 1 -maxdepth 1 -type d | grep -q .; then
    echo "ERROR: Found subdirectories inside: $d"
    find "$d" -mindepth 1 -maxdepth 1 -type d -print
    exit 1
  fi
  if ! find "$d" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" \) | head -n 1 | grep -q .; then
    echo "ERROR: No image files found in: $d"
    exit 1
  fi
  local bad
  bad="$(find "$d" -maxdepth 1 -type f ! \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" \) -print | head -n 20 || true)"
  if [[ -n "$bad" ]]; then
    echo "ERROR: Non-image files found in: $d"
    echo "$bad"
    exit 1
  fi
}

check_file () {
  local f="$1"
  test -f "$f" || { echo "Missing file: $f"; exit 1; }
}

latest_dir_maybe () {
  local pattern="$1"
  local d
  d="$(ls -1dt ${pattern} 2>/dev/null | head -n 1 || true)"
  if [[ -n "$d" ]]; then
    echo "$d"
  fi
}

pick_lora_file_in_dir () {
  local d="$1"
  local f_safe="${d}/pytorch_lora_weights.safetensors"
  local f_bin="${d}/pytorch_lora_weights.bin"
  if [[ -f "$f_safe" ]]; then
    echo "$f_safe"
  elif [[ -f "$f_bin" ]]; then
    echo "$f_bin"
  else
    echo "ERROR: No pytorch_lora_weights.(safetensors|bin) in: $d"
    exit 1
  fi
}

resolve_symlink_file () {
  local p_safe="$1"
  local p_bin="$2"
  if [[ -f "$p_safe" ]]; then
    echo "$p_safe"
    return
  fi
  if [[ -f "$p_bin" ]]; then
    echo "$p_bin"
    return
  fi
}

# =========================
# 6) 優先使用 symlink：latest_style_run / latest_style_* / latest_content_*
# =========================
STYLE_RUN_DIR=""
if [[ -L "/mnt/cglab/olson/Break-for-make-outputs/latest_style_run" || -d "/mnt/cglab/olson/Break-for-make-outputs/latest_style_run" ]]; then
  STYLE_RUN_DIR="$(cd "/mnt/cglab/olson/Break-for-make-outputs/latest_style_run" && pwd)"
else
  STYLE_RUN_DIR="$(latest_dir_maybe "/mnt/cglab/olson/Break-for-make-outputs/style_run_*")"
fi
if [[ -z "${STYLE_RUN_DIR}" ]]; then
  echo "ERROR: Cannot resolve STYLE_RUN_DIR (no latest_style_run and no style_run_*)."
  exit 1
fi

STYLE1_LORA_PATH="$(resolve_symlink_file "/mnt/cglab/olson/Break-for-make-outputs/latest_style_${STYLE1_NAME}.safetensors" "/mnt/cglab/olson/Break-for-make-outputs/latest_style_${STYLE1_NAME}.bin" || true)"
STYLE2_LORA_PATH="$(resolve_symlink_file "/mnt/cglab/olson/Break-for-make-outputs/latest_style_${STYLE2_NAME}.safetensors" "/mnt/cglab/olson/Break-for-make-outputs/latest_style_${STYLE2_NAME}.bin" || true)"
STYLE3_LORA_PATH="$(resolve_symlink_file "/mnt/cglab/olson/Break-for-make-outputs/latest_style_${STYLE3_NAME}.safetensors" "/mnt/cglab/olson/Break-for-make-outputs/latest_style_${STYLE3_NAME}.bin" || true)"

if [[ -z "${STYLE1_LORA_PATH}" ]]; then STYLE1_LORA_PATH="$(pick_lora_file_in_dir "${STYLE_RUN_DIR}/${STYLE1_NAME}")"; fi
if [[ -z "${STYLE2_LORA_PATH}" ]]; then STYLE2_LORA_PATH="$(pick_lora_file_in_dir "${STYLE_RUN_DIR}/${STYLE2_NAME}")"; fi
if [[ -z "${STYLE3_LORA_PATH}" ]]; then STYLE3_LORA_PATH="$(pick_lora_file_in_dir "${STYLE_RUN_DIR}/${STYLE3_NAME}")"; fi

CONTENT1_LORA_PATH="$(resolve_symlink_file "/mnt/cglab/olson/Break-for-make-outputs/latest_content_${CONTENT1_NAME}.safetensors" "/mnt/cglab/olson/Break-for-make-outputs/latest_content_${CONTENT1_NAME}.bin" || true)"
CONTENT2_LORA_PATH="$(resolve_symlink_file "/mnt/cglab/olson/Break-for-make-outputs/latest_content_${CONTENT2_NAME}.safetensors" "/mnt/cglab/olson/Break-for-make-outputs/latest_content_${CONTENT2_NAME}.bin" || true)"
CONTENT3_LORA_PATH="$(resolve_symlink_file "/mnt/cglab/olson/Break-for-make-outputs/latest_content_${CONTENT3_NAME}.safetensors" "/mnt/cglab/olson/Break-for-make-outputs/latest_content_${CONTENT3_NAME}.bin" || true)"

if [[ -z "${CONTENT1_LORA_PATH}" || -z "${CONTENT2_LORA_PATH}" || -z "${CONTENT3_LORA_PATH}" ]]; then
  echo "ERROR: Missing latest_content_<name> symlinks/files."
  echo "Please run the symlink-enabled train_content.sh first so these exist:"
  echo "  outputs/latest_content_${CONTENT1_NAME}.(safetensors|bin)"
  echo "  outputs/latest_content_${CONTENT2_NAME}.(safetensors|bin)"
  echo "  outputs/latest_content_${CONTENT3_NAME}.(safetensors|bin)"
  exit 1
fi

# =========================
# 7) 檢查資料夾/檔案
# =========================
check_dir_clean "${CONTENT1_DIR}"
check_dir_clean "${CONTENT2_DIR}"
check_dir_clean "${CONTENT3_DIR}"

check_dir_clean "${STYLE1_DIR}"
check_dir_clean "${STYLE2_DIR}"
check_dir_clean "${STYLE3_DIR}"

check_file "${STYLE1_LORA_PATH}"
check_file "${STYLE2_LORA_PATH}"
check_file "${STYLE3_LORA_PATH}"

check_file "${CONTENT1_LORA_PATH}"
check_file "${CONTENT2_LORA_PATH}"
check_file "${CONTENT3_LORA_PATH}"

run_one_pair () {
  local style_name="$1"
  local style_dir="$2"
  local style_prompt="$3"
  local style_lora="$4"
  local content_name="$5"
  local content_dir="$6"
  local content_prompt="$7"
  local content_lora="$8"

  local out_dir="${OUT_ROOT}/${style_name}__${content_name}"
  mkdir -p "${out_dir}"

  local val_prompt="${content_prompt}, ${style_prompt}"

  echo "================================================="
  echo "[Second Stage] ${style_name} + ${content_name}"
  echo "  style_run_dir  : ${STYLE_RUN_DIR}"
  echo "  out            : ${out_dir}"
  echo "  style_prompt   : ${style_prompt}"
  echo "  content_prompt : ${content_prompt}"
  echo "  lora_style     : ${style_lora}"
  echo "  lora_obj       : ${content_lora}"
  echo "  val_prompt     : ${val_prompt}"
  echo "================================================="

  accelerate launch --num_processes=1 --num_machines=1 code/train_second_stage.py \
    --pretrained_model_name_or_path="${MODEL_NAME}" \
    --pretrained_vae_model_name_or_path="${VAE_PATH}" \
    --instance_data_dir="${style_dir}" \
    --instance_data_dir_2="${content_dir}" \
    --output_dir="${out_dir}" \
    --instance_prompt="${style_prompt}" \
    --instance_prompt_2="${content_prompt}" \
    --validation_prompt="${val_prompt}" \
    --num_validation_images="${NUM_VAL}" \
    --validation_epochs="${VAL_EPOCHS}" \
    --resolution="${RESOLUTION}" \
    --rank="${RANK}" \
    --train_batch_size="${TRAIN_BS}" \
    --gradient_accumulation_steps="${GRAD_ACC}" \
    --gradient_checkpointing \
    --learning_rate="${LR}" \
    --learning_rate_2="${LR2}" \
    --report_to="${REPORT_TO}" \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps="${MAX_STEPS}" \
    --mixed_precision=fp16 \
    --seed="${SEED}" \
    --checkpointing_steps="${CKPT_STEPS}" \
    --exchange_finetune \
    --lora_path_obj="${content_lora}" \
    --lora_path_style="${style_lora}"

    # --mixed_precision=bf16
}

run_one_pair "${STYLE1_NAME}" "${STYLE1_DIR}" "${STYLE1_PROMPT}" "${STYLE1_LORA_PATH}" "${CONTENT1_NAME}" "${CONTENT1_DIR}" "${CONTENT1_PROMPT}" "${CONTENT1_LORA_PATH}"
run_one_pair "${STYLE1_NAME}" "${STYLE1_DIR}" "${STYLE1_PROMPT}" "${STYLE1_LORA_PATH}" "${CONTENT2_NAME}" "${CONTENT2_DIR}" "${CONTENT2_PROMPT}" "${CONTENT2_LORA_PATH}"
run_one_pair "${STYLE1_NAME}" "${STYLE1_DIR}" "${STYLE1_PROMPT}" "${STYLE1_LORA_PATH}" "${CONTENT3_NAME}" "${CONTENT3_DIR}" "${CONTENT3_PROMPT}" "${CONTENT3_LORA_PATH}"

run_one_pair "${STYLE2_NAME}" "${STYLE2_DIR}" "${STYLE2_PROMPT}" "${STYLE2_LORA_PATH}" "${CONTENT1_NAME}" "${CONTENT1_DIR}" "${CONTENT1_PROMPT}" "${CONTENT1_LORA_PATH}"
run_one_pair "${STYLE2_NAME}" "${STYLE2_DIR}" "${STYLE2_PROMPT}" "${STYLE2_LORA_PATH}" "${CONTENT2_NAME}" "${CONTENT2_DIR}" "${CONTENT2_PROMPT}" "${CONTENT2_LORA_PATH}"
run_one_pair "${STYLE2_NAME}" "${STYLE2_DIR}" "${STYLE2_PROMPT}" "${STYLE2_LORA_PATH}" "${CONTENT3_NAME}" "${CONTENT3_DIR}" "${CONTENT3_PROMPT}" "${CONTENT3_LORA_PATH}"

run_one_pair "${STYLE3_NAME}" "${STYLE3_DIR}" "${STYLE3_PROMPT}" "${STYLE3_LORA_PATH}" "${CONTENT1_NAME}" "${CONTENT1_DIR}" "${CONTENT1_PROMPT}" "${CONTENT1_LORA_PATH}"
run_one_pair "${STYLE3_NAME}" "${STYLE3_DIR}" "${STYLE3_PROMPT}" "${STYLE3_LORA_PATH}" "${CONTENT2_NAME}" "${CONTENT2_DIR}" "${CONTENT2_PROMPT}" "${CONTENT2_LORA_PATH}"
run_one_pair "${STYLE3_NAME}" "${STYLE3_DIR}" "${STYLE3_PROMPT}" "${STYLE3_LORA_PATH}" "${CONTENT3_NAME}" "${CONTENT3_DIR}" "${CONTENT3_PROMPT}" "${CONTENT3_LORA_PATH}"

echo "All second-stage trainings done."
echo "  style_run_dir : ${STYLE_RUN_DIR}"
echo "  output root   : ${OUT_ROOT}"
