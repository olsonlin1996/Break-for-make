#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-b4m}"
OUT_DIR="${2:-env_snapshot}"

mkdir -p "${OUT_DIR}"

echo "[1/6] conda info/config"
conda info -a > "${OUT_DIR}/conda_info.txt"
conda config --show-sources > "${OUT_DIR}/conda_config_sources.txt" || true
conda config --show > "${OUT_DIR}/conda_config_show.txt" || true

echo "[2/6] conda env export (portable)"
# --from-history: 只保留你顯式裝過的項目
# --no-builds: 不鎖 build string（較不易因 build 不存在而卡）
conda env export -n "${ENV_NAME}" --from-history --no-builds \
  | grep -v '^prefix:' > "${OUT_DIR}/environment.yml"

echo "[3/6] conda explicit spec (exact, same platform best)"
conda list -n "${ENV_NAME}" --explicit > "${OUT_DIR}/conda-spec-linux-64.txt"

echo "[4/6] pip freeze"
conda run -n "${ENV_NAME}" python -m pip freeze > "${OUT_DIR}/requirements-pip.txt"

echo "[5/6] key runtime versions (torch/cuda/etc.)"
TMP_PY="$(mktemp /tmp/${ENV_NAME}_runtime_versions.XXXXXX.py)"

cat > "${TMP_PY}" <<'PY'
import os, sys
print("sys.executable:", sys.executable)
print("python:", sys.version)

def show(mod, name=None):
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "unknown")
        f = getattr(m, "__file__", "")
        print(f"{name or mod}: {ver} ({f})")
    except Exception as e:
        print(f"{name or mod}: import failed: {e}")

try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("torch.version.cuda:", getattr(torch.version, "cuda", None))
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("cap:", torch.cuda.get_device_capability(0))
except Exception as e:
    print("torch import failed:", e)

show("diffusers")
show("transformers")
show("accelerate")
show("peft")
show("huggingface_hub", "huggingface_hub")
PY

conda run -n "${ENV_NAME}" python "${TMP_PY}" > "${OUT_DIR}/runtime_versions.txt" 2>&1
rm -f "${TMP_PY}"


echo "[6/6] done -> ${OUT_DIR}/"
ls -lah "${OUT_DIR}"
