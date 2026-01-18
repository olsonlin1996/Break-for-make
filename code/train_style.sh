#!/usr/bin/env bash
set -euo pipefail

# repo root（自動抓：code/ 的上一層）
B4M_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_ROOT="/home/cglab/project/olson/diffusers/examples/dreambooth"

# ===== 模型設定 =====
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

# ===== style 資料夾（請確保「只有圖片檔」，不要有 .cache / 子資料夾）=====
STYLE1_NAME="ChineseInkPainting"
STYLE1_DIR="${B4M_ROOT}/assets/ChineseInkPainting"

STYLE2_NAME="glass"
STYLE2_DIR="${B4M_ROOT}/assets/glass"

STYLE3_NAME="yarn"
STYLE3_DIR="${B4M_ROOT}/assets/yarn"

# ===== prompt（style token 你目前用 w@z，就沿用；可以再加細節描述）=====
STYLE1_PROMPT="w@z Chinese ink painting style, ink wash painting, rice paper texture, expressive brush strokes, monochrome"
STYLE2_PROMPT="w@z crafted from clean glass, transparent material, glossy reflections, refraction"
STYLE3_PROMPT="w@z yarn art style, knitted or crochet texture, wool fibers, handcrafted"

# ===== 輸出根目錄 =====
OUT_ROOT="${B4M_ROOT}/outputs/style_run_$(date +%Y%m%d_%H%M%S)"

cd "${B4M_ROOT}"

# Force this repo's src/ to take precedence (avoid env editable diffusers from other repo)
export PYTHONPATH="${B4M_ROOT}/src:${PYTHONPATH:-}"


# ===== (可選) 前置 sanity check：確保目前環境用的是 repo 內 diffusers =====
python - <<'PY'
import diffusers
print("diffusers file:", diffusers.__file__)
assert "/Break-for-make_runA/src/diffusers" in diffusers.__file__, "diffusers not from Break-for-make_runA/src/diffusers"
print("OK: using Break-for-make_runA diffusers")
PY

# ===== 防呆：資料夾存在 + 不含子資料夾（避免 .cache 事故）=====
check_dir_clean () {
  local d="$1"
  test -d "$d" || { echo "Missing dir: $d"; exit 1; }
  # 任何子資料夾都視為危險（最常見是 .cache）
  if find "$d" -mindepth 1 -maxdepth 1 -type d | grep -q .; then
    echo "ERROR: Found subdirectories inside: $d"
    echo "Please remove/move them (e.g., .cache) so the folder contains only image files."
    find "$d" -mindepth 1 -maxdepth 1 -type d -print
    exit 1
  fi
  # 至少要有一張圖
  if ! find "$d" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" \) | head -n 1 | grep -q .; then
    echo "ERROR: No image files found in: $d"
    exit 1
  fi
}

check_dir_clean "$STYLE1_DIR"
check_dir_clean "$STYLE2_DIR"
check_dir_clean "$STYLE3_DIR"

mkdir -p "${OUT_ROOT}"

# ===== 訓練參數（先跟 content 對齊，之後再微調）=====
RES=512
RANK=64
BS=1
GAS=4
LR=1e-4
MAX_STEPS=1000
SEED=0
CKPT=100

# 你若要 tensorboard 記錄：先 pip install tensorboard，不然 accelerate 會警告沒 tracker
REPORT_TO="tensorboard"

run_one_style () {
  local style_name="$1"
  local style_dir="$2"
  local style_prompt="$3"
  local out_dir="${OUT_ROOT}/${style_name}"

  mkdir -p "$out_dir"

  echo "================================================="
  echo "[Style Train] ${style_name}"
  echo "  data : ${style_dir}"
  echo "  out  : ${out_dir}"
  echo "  prompt: ${style_prompt}"
  echo "================================================="

  accelerate launch code/train_style.py \
    --pretrained_model_name_or_path="${MODEL_NAME}" \
    --pretrained_vae_model_name_or_path="${VAE_PATH}" \
    --instance_data_dir="${style_dir}" \
    --output_dir="${out_dir}" \
    --instance_prompt="${style_prompt}" \
    --resolution="${RES}" \
    --rank="${RANK}" \
    --train_batch_size="${BS}" \
    --gradient_accumulation_steps="${GAS}" \
    --learning_rate="${LR}" \
    --report_to="${REPORT_TO}" \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps="${MAX_STEPS}" \
    --seed="${SEED}" \
    --checkpointing_steps="${CKPT}"
}

# 逐一訓練三種 style
run_one_style "${STYLE1_NAME}" "${STYLE1_DIR}" "${STYLE1_PROMPT}"
run_one_style "${STYLE2_NAME}" "${STYLE2_DIR}" "${STYLE2_PROMPT}"
run_one_style "${STYLE3_NAME}" "${STYLE3_DIR}" "${STYLE3_PROMPT}"

echo "All style trainings done. Output root: ${OUT_ROOT}"
