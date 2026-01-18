# 環境重建說明

本專案使用 Conda 建立可重現的 Python 環境，並把當前可跑通的環境快照放在 `env_snapshot/`。
重建時會優先用「Conda explicit spec」做 1:1 還原（同平台最穩），再用 `pip` 補齊少數套件。

---

## A. 你會用到的檔案/目錄
- `env_snapshot/`
  - `conda-spec-linux-64.txt`：最重要，Linux 同平台最佳還原方式
  - `environment.yml`：備援（較可攜，但不一定 100% 還原）
  - `requirements-pip.txt`：pip 補齊清單
  - `runtime_versions.txt`：快照當下的 python/torch/cuda 等版本記錄
- `scripts/`
  - `create_env.sh`：用快照重建環境
  - `snapshot_env.sh`：重新產生/更新快照（你之後若升級環境可再跑）

---

## B. 前置需求（建議照做）
### B1. 系統/硬體
- Linux（此快照由 linux-64 產生）
- 若要訓練：需 NVIDIA GPU 驅動、CUDA runtime（由 PyTorch 套件帶的 cu12 runtime）
- 硬體與驅動版本不同不一定會失敗，但可能造成：
  - torch/cuda 不相容
  - xformers/bitsandbytes 等套件需要重裝或改版本

### B2. 建議避免 pyenv 干擾（重要）
如果你的 shell 有啟用 pyenv shims，可能導致：
- `python` / `pip` / `accelerate` 指到 `~/.pyenv/shims/...`
- conda env 內實際安裝的套件被「路徑」蓋過，出現「套件存在但 import 找不到」或版本混亂

建議確認：
```bash
command -v python
command -v pip
command -v accelerate
```

- 正常情況：應該指向 .../miniconda3/... 或 .../envs/<env>/...
- 若看到 ~/.pyenv/shims/...：代表 pyenv 仍在搶優先權，需要關掉或移出 PATH

#### B2-2. 快速把 pyenv 從當前 shell PATH 暫時移除（必要時）
（僅影響當前 shell，不會改你的設定檔）

```
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v "$HOME/.pyenv" | paste -sd ':' -)"
hash -r 2>/dev/null || true
```

---

## C. 還原流程（一步一步指令教學）
以下指令假設你在 Linux server 上操作，並且已經安裝好 conda（miniconda/anaconda 皆可）。
若你用 SSH 登入，請在同一個 shell 內完成整段流程，避免 PATH/conda 初始化狀態不一致。

### C1. 取得專案原始碼
```
git clone <你的repo-url>
cd Break-for-make
```

確認目錄內應該要看到：

```
ls
# 你應該會看到：env_snapshot/ scripts/ src/ code/ ... 等
```

### C2. 確認 conda 可用
```
command -v conda
conda --version
conda info -a | head
```

若 conda 找不到，請先把你的 conda 初始化（視你機器安裝方式而定）。

### C3. 建立環境（建議用 scripts/create_env.sh）

這會：
- 優先用 env_snapshot/conda-spec-linux-64.txt 還原
- 若沒有 spec，才退回用 environment.yml
- 最後再用 pip 安裝 requirements-pip.txt

建立 b4m 環境：
```
bash scripts/create_env.sh b4m env_snapshot
```

常見情況：
- 如果它說環境已存在：你需要換環境名，或刪掉舊環境再重建
- 如果 explicit spec 下載不到某些檔：通常是 channel/鏡像不同或版本下架
- 這種情況可以改用 environment.yml（可攜性較高，但不保證完全一致）

### C4. 啟用環境
```conda activate b4m```

啟用後立刻確認 python/pip/accelerate 指到哪裡：
```
command -v python
command -v pip
command -v accelerate
python -c "import sys; print(sys.executable)"
```

### C5. 驗證核心套件版本（必做）
```
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('torch cuda', getattr(torch.version,'cuda',None))"
python -c "import transformers, accelerate; print('transformers', transformers.__version__); print('accelerate', accelerate.__version__)"
```

如果 torch.cuda.is_available() 是 False：
- 可能是你在無 GPU 的機器上（可接受，但訓練會非常慢）
- 或是 driver 沒裝好 / CUDA runtime 不相容

### C6. 驗證 diffusers 來源（非常重要）
本專案希望以 repo 內 src/diffusers 為準，避免外部 pip editable diffusers 造成版本混亂。
在 repo root 執行：
```
PYTHONPATH="$PWD/src" python -c "import os, diffusers; print(os.path.realpath(diffusers.__file__))"
```

你應該看到類似：
- `.../Break-for-make/src/diffusers/__init__.py`

若顯示的是：
- `.../site-packages/diffusers/...` 或指到別的 repo
代表你現在不是用本 repo 的 diffusers，需要檢查：
- `PYTHONPATH` 是否有帶 `$PWD/src`
- 你是否曾 `pip install -e` 安裝過 diffusers（並指向其它位置）

### C7. 先跑一次最小訓練測試（確認可跑）
```
bash code/train_content.sh
```

建議保留完整 log 以方便排錯：
```
bash code/train_content.sh 2>&1 | tee /tmp/train_content_$(date +%Y%m%d_%H%M%S).log
```

---

## D. 成功/失敗的判斷方式
### D1. 成功的基本判斷

- terminal 進度條跑到 `100%|...| N/N`
- `outputs/.../` 會產生：
  - `pytorch_lora_weights.safetensors`
  - `checkpoint-*`（若有設定 checkpointing_steps）
  - `logs`/（tensorboard event files）

### D2. 常見失敗與快速排查
#### D2-1. OOM（CUDA out of memory）

先用 log 搜：
```
grep -nE "OutOfMemoryError|CUDA out of memory|Killed|Traceback" /tmp/train_content_*.log | tail -n 120
```

常見快速降載方式：
- 降 --resolution
- 降 --rank
- 確保 --mixed_precision=fp16 或 accelerate config 的 mixed_precision: fp16
- 開 --gradient_checkpointing

#### D2-2. 進程看起來停住但沒有報錯

確認是否仍在跑：
```
ps -ef | grep -E "train_content.py|accelerate launch" | grep -v grep
```

---

## E. 需要更新快照時（你改動環境後才需要）

若你有新增/移除套件並且希望更新快照：
```
bash scripts/snapshot_env.sh b4m env_snapshot
```

更新後建議檢查 env_snapshot/runtime_versions.txt 是否有內容：
```
sed -n '1,160p' env_snapshot/runtime_versions.txt
```

---

## F. 注意事項（重現性的現實限制）

- `conda-spec-linux-64.txt` 是「同平台」最穩的還原方式；不同 distro/不同 driver/不同 glibc 仍可能需要微調
- 若你變更了 PyTorch 的 CUDA 版本（例如 cu118/cu121/cu12x），建議重新快照並同步更新 `runtime_versions.txt`
- 若你有 pyenv 習慣，建議維持「預設 conda、pyenv 手動開啟」的策略，避免 shims 長期干擾
