#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-b4m}"
SNAPSHOT_DIR="${2:-env_snapshot}"

SPEC_FILE="${SNAPSHOT_DIR}/conda-spec-linux-64.txt"
YML_FILE="${SNAPSHOT_DIR}/environment.yml"
PIP_FILE="${SNAPSHOT_DIR}/requirements-pip.txt"

echo "ENV_NAME=${ENV_NAME}"
echo "SNAPSHOT_DIR=${SNAPSHOT_DIR}"

# 先確保 conda 可用（視你的 server 安裝方式可自行調整）
command -v conda >/dev/null 2>&1 || { echo "conda not found"; exit 1; }

# 若環境已存在，避免覆蓋
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Env ${ENV_NAME} already exists. Abort to avoid overwrite."
  exit 1
fi

# 優先使用 explicit spec：同平台最容易 1:1 還原
if [[ -f "${SPEC_FILE}" ]]; then
  echo "[create] using explicit spec: ${SPEC_FILE}"
  # explicit spec 就是給 conda create --file 用 :contentReference[oaicite:2]{index=2}
  conda create -n "${ENV_NAME}" --file "${SPEC_FILE}" -y
elif [[ -f "${YML_FILE}" ]]; then
  echo "[create] using environment.yml: ${YML_FILE}"
  conda env create -n "${ENV_NAME}" -f "${YML_FILE}"
else
  echo "No spec/yml found in ${SNAPSHOT_DIR}"
  exit 1
fi

# pip 補齊
if [[ -f "${PIP_FILE}" ]]; then
  echo "[pip] installing ${PIP_FILE}"
  conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
  conda run -n "${ENV_NAME}" python -m pip install -r "${PIP_FILE}"
else
  echo "[pip] no requirements-pip.txt found, skip"
fi

echo "Done. To use:"
echo "  conda activate ${ENV_NAME}"
echo "[smoke] python/pip/accelerate paths"
conda run -n "${ENV_NAME}" which python
conda run -n "${ENV_NAME}" which pip
conda run -n "${ENV_NAME}" which accelerate || true

echo "[smoke] import key libs"
conda run -n "${ENV_NAME}" python - <<'PY'
import sys, os
print("sys.executable:", sys.executable)
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
import transformers, accelerate
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
PY

echo "[smoke] diffusers should come from repo src (if you intend that)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONPATH="${REPO_ROOT}/src" conda run -n "${ENV_NAME}" python - <<'PY'
import os, diffusers
print("diffusers file:", os.path.realpath(diffusers.__file__))
PY
