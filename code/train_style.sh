#!/usr/bin/env bash
set -euo pipefail

# repo root（自動抓：code/ 的上一層）
B4M_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

cd "${B4M_ROOT}"

# Force this repo's src/ to take precedence
export PYTHONPATH="${B4M_ROOT}/src:${PYTHONPATH:-}"

# =========================
# 三份 content（style 訓練時的 content reference）
# =========================
CONTENT_DIR="${B4M_ROOT}/assets/content"

CONTENT1_DIR="${CONTENT_DIR}/corgi"
CONTENT2_DIR="${CONTENT_DIR}/eagle"
CONTENT3_DIR="${CONTENT_DIR}/elephant"

CONTENT1_PROMPT="a photo of snq corgi"
CONTENT2_PROMPT="a photo of snr eagle"
CONTENT3_PROMPT="a photo of snp elephant"

# =========================
# 三個 style（你要訓練的）
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
# 輸出根目錄（整批 style 的 run）
# =========================
OUT_ROOT="${B4M_ROOT}/outputs/style_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_ROOT}"

# =========================
# 超參數（模式 A：先跑起來驗證）
# =========================
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

# =========================
# 防呆工具
# =========================
check_dir_images_only () {
  local d="$1"
  test -d "${d}" || { echo "Missing dir: ${d}"; exit 1; }

  if find "${d}" -mindepth 1 -maxdepth 1 -type d | grep -q .; then
    echo "ERROR: Found subdirectories inside: ${d}"
    find "${d}" -mindepth 1 -maxdepth 1 -type d -print
    exit 1
  fi

  if ! find "${d}" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" \) | head -n 1 | grep -q .; then
    echo "ERROR: No image files found in: ${d}"
    exit 1
  fi

  local bad
  bad="$(find "${d}" -maxdepth 1 -type f \
    ! \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" \) \
    -print | head -n 20 || true)"
  if [[ -n "${bad}" ]]; then
    echo "ERROR: Non-image files found in: ${d}"
    echo "${bad}"
    echo "Please remove/move them so the folder contains only image files."
    exit 1
  fi
}

pick_lora_file_in_dir () {
  local d="$1"
  local f_safe="${d}/pytorch_lora_weights.safetensors"
  local f_bin="${d}/pytorch_lora_weights.bin"
  if [[ -f "${f_safe}" ]]; then
    echo "${f_safe}"
  elif [[ -f "${f_bin}" ]]; then
    echo "${f_bin}"
  else
    echo "ERROR: No pytorch_lora_weights.(safetensors|bin) in: ${d}"
    exit 1
  fi
}

update_latest_style_symlink () {
  local style_name="$1"
  local out_dir="$2"

  local lora_path
  lora_path="$(pick_lora_file_in_dir "${out_dir}")"

  local link_path
  if [[ "${lora_path}" == *.safetensors ]]; then
    link_path="${B4M_ROOT}/outputs/latest_style_${style_name}.safetensors"
  else
    link_path="${B4M_ROOT}/outputs/latest_style_${style_name}.bin"
  fi

  ln -sfn "${lora_path}" "${link_path}"
  echo "[symlink] ${link_path} -> ${lora_path}"
}

update_latest_style_run_symlink () {
  local run_dir="$1"
  local link_path="${B4M_ROOT}/outputs/latest_style_run"
  ln -sfn "${run_dir}" "${link_path}"
  echo "[symlink] ${link_path} -> ${run_dir}"
}

# 檢查資料夾
check_dir_images_only "${CONTENT1_DIR}"
check_dir_images_only "${CONTENT2_DIR}"
check_dir_images_only "${CONTENT3_DIR}"

check_dir_images_only "${STYLE1_DIR}"
check_dir_images_only "${STYLE2_DIR}"
check_dir_images_only "${STYLE3_DIR}"

run_one_style () {
  local style_name="$1"
  local style_dir="$2"
  local style_prompt="$3"
  local out_dir="${OUT_ROOT}/${style_name}"

  mkdir -p "${out_dir}"

  echo "================================================="
  echo "[Style Train] ${style_name}"
  echo "  style   : ${style_dir}"
  echo "  out     : ${out_dir}"
  echo "  style prompt : ${style_prompt}"
  echo "  content1: ${CONTENT1_DIR} | ${CONTENT1_PROMPT}"
  echo "  content2: ${CONTENT2_DIR} | ${CONTENT2_PROMPT}"
  echo "  content3: ${CONTENT3_DIR} | ${CONTENT3_PROMPT}"
  echo "================================================="

  # 重要注意事項：參數之間不能有任何註解
  accelerate launch --num_processes=1 --num_machines=1 code/train_style.py \
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

  update_latest_style_symlink "${style_name}" "${out_dir}"
}

run_one_style "${STYLE1_NAME}" "${STYLE1_DIR}" "${STYLE1_PROMPT}"
run_one_style "${STYLE2_NAME}" "${STYLE2_DIR}" "${STYLE2_PROMPT}"
run_one_style "${STYLE3_NAME}" "${STYLE3_DIR}" "${STYLE3_PROMPT}"

update_latest_style_run_symlink "${OUT_ROOT}"

echo "All style trainings done."
echo "Output root: ${OUT_ROOT}"
echo "Latest symlinks:"
ls -lah "${B4M_ROOT}/outputs"/latest_style_* "${B4M_ROOT}/outputs"/latest_style_run 2>/dev/null || true
