#!/usr/bin/env bash
set -euo pipefail

# repo root（自動抓：code/ 的上一層）
B4M_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

CONTENT_ROOT="${B4M_ROOT}/assets/content"

STYLE1_DIR="${B4M_ROOT}/assets/style/ChineseInkPainting"
STYLE2_DIR="${B4M_ROOT}/assets/style/glass"
STYLE3_DIR="${B4M_ROOT}/assets/style/yarn"

cd "${B4M_ROOT}"

# Force this repo's src/ to take precedence (avoid env editable diffusers from other repo)
export PYTHONPATH="${B4M_ROOT}/src:${PYTHONPATH:-}"

# -------------------------
# Mode A（先跑起來驗證）超參數：沿用你目前可跑通設定
# -------------------------
# Paper-aligned override: LR=1e-4, BS=1, rank=64; RESOLUTION=512, grad_acc=4, max_steps=1000

# RESOLUTION=256
RESOLUTION=512
RANK=64
TRAIN_BS=1
# GRAD_ACC=1
GRAD_ACC=4
LR=1e-4
# LR2=1e-5
LR2=1e-4
# MAX_STEPS=50
MAX_STEPS=1000
SEED=0
# CKPT_STEPS=10
CKPT_STEPS=100
REPORT_TO="tensorboard"

# -------------------------
# 防呆：資料夾存在 + 僅允許圖片檔（避免 PIL 讀到非圖片炸掉）
# -------------------------
check_dir_images_only () {
  local d="$1"
  test -d "${d}" || { echo "Missing dir: ${d}"; exit 1; }

  # 不允許子資料夾
  if find "${d}" -mindepth 1 -maxdepth 1 -type d | grep -q .; then
    echo "ERROR: Found subdirectories inside: ${d}"
    find "${d}" -mindepth 1 -maxdepth 1 -type d -print
    exit 1
  fi

  # 至少要有一張圖片
  if ! find "${d}" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" \) | head -n 1 | grep -q .; then
    echo "ERROR: No image files found in: ${d}"
    exit 1
  fi

  # 若存在非圖片檔，直接 fail（避免 Dataset/PIL 讀取時爆）
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

update_latest_symlink () {
  local name="$1"
  local out_dir="$2"

  local lora_path
  lora_path="$(pick_lora_file_in_dir "${out_dir}")"

  local link_path
  if [[ "${lora_path}" == *.safetensors ]]; then
    link_path="/mnt/cglab/olson/Break-for-make-outputs/latest_content_${name}.safetensors"
  else
    link_path="/mnt/cglab/olson/Break-for-make-outputs/latest_content_${name}.bin"
  fi

  ln -sfn "${lora_path}" "${link_path}"
  echo "[symlink] ${link_path} -> ${lora_path}"
}

# 檢查 style 資料夾存在且乾淨（train_content.py 會把它們當成 reference）
check_dir_images_only "${STYLE1_DIR}"
check_dir_images_only "${STYLE2_DIR}"
check_dir_images_only "${STYLE3_DIR}"

run_one_content () {
  local content_name="$1"
  local content_dir="$2"
  local content_prompt="$3"

  check_dir_images_only "${content_dir}"

  local out_dir="/mnt/cglab/olson/Break-for-make-outputs/content_${content_name}_run_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${out_dir}"

  echo "================================================="
  echo "[Content Train] ${content_name}"
  echo "  data  : ${content_dir}"
  echo "  out   : ${out_dir}"
  echo "  prompt: ${content_prompt}"
  echo "================================================="

  # 重要注意事項：參數之間不能有任何註解
  accelerate launch code/train_content.py \
    --multi_style \
    --pretrained_model_name_or_path="${MODEL_NAME}" \
    --pretrained_vae_model_name_or_path="${VAE_PATH}" \
    --instance_data_dir="${content_dir}" \
    --instance_data_dir_2="${STYLE1_DIR}" \
    --instance_data_dir_3="${STYLE2_DIR}" \
    --instance_data_dir_4="${STYLE3_DIR}" \
    --output_dir="${out_dir}" \
    --instance_prompt="${content_prompt}" \
    --instance_prompt_2="w@z Chinese ink painting style" \
    --instance_prompt_3="w@z crafted from clean glass" \
    --instance_prompt_4="w@z yarn art style" \
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

  update_latest_symlink "${content_name}" "${out_dir}"
}

# -------------------------
# 三個 content（各自 token）
# -------------------------
run_one_content "corgi"    "${CONTENT_ROOT}/corgi"    "a photo of snq corgi"
run_one_content "eagle"    "${CONTENT_ROOT}/eagle"    "a photo of snr eagle"
run_one_content "elephant" "${CONTENT_ROOT}/elephant" "a photo of snp elephant"

echo "All content trainings done."
echo "Latest symlinks:"
ls -lah "/mnt/cglab/olson/Break-for-make-outputs"/latest_content_* 2>/dev/null || true
