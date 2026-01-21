import os
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

LORA_PATH = "/home/cglab/project/olson/Break-for-make/outputs/second_stage_run_20260120_163109/glass__corgi/pytorch_lora_weights.safetensors"
OUT_DIR = "/home/cglab/project/olson/Break-for-make/outputs/infer_glass_debug"
os.makedirs(OUT_DIR, exist_ok=True)

CONTENT_TOKEN = "snq"   # content token
STYLE_TOKEN = "w@z"     # style token

BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"

# 為了穩定先用 fp16
# 若你確定要 bf16：把 DTYPE 改 torch.bfloat16，並且 VAE/pipe 都用同一個 dtype
DTYPE = torch.float16

SEED = 77
NUM_STEPS = 30
GUIDANCE = 5.0
H, W = 1024, 1024

# -------------------------
# Prompt 設計（關鍵）
# 1) prompt 走「style encoder」
# 2) prompt_2 走「content encoder」
# 但為了把玻璃感拉回來：先在 prompt_2 也放少量玻璃關鍵字（保守做法）
# -------------------------
style_prompt = (
    f"{STYLE_TOKEN} glass sculpture style, clean glass, transparent material, glossy reflections, strong refraction, "
    "studio lighting, product photo, sharp highlights"
)

# A：忠於訓練語境（含 a photo of）
content_prompt_A = f"a photo of {CONTENT_TOKEN} corgi, glass sculpture, transparent glass, refraction"

# B：降低「照片語境」拉扯（不含 a photo of）
content_prompt_B = f"{CONTENT_TOKEN} corgi, glass sculpture, transparent glass, refraction"

negative_common = (
    "multiple dogs, puppies, crowd, group, extra heads, extra faces, extra limbs, "
    "deformed, disfigured, bad anatomy, blurry, low quality, text, watermark"
)
negative_style = "plastic wrap, cellophane, crumpled film, latex"
neg = f"{negative_common}, {negative_style}"

# -------------------------
# Build pipeline
# -------------------------
vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=DTYPE)

pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL_ID,
    vae=vae,
    torch_dtype=DTYPE,
    use_safetensors=True,
    variant="fp16" if DTYPE == torch.float16 else None,
).to("cuda")

# 有些環境會吵進度條，關掉讓 log 乾淨
#pipe.set_progress_bar_config(disable=True)

def gen_one(tag: str, prompt_2: str):
    g = torch.Generator(device="cuda").manual_seed(SEED)
    image = pipe(
        prompt=style_prompt,
        prompt_2=prompt_2,
        negative_prompt=neg,
        negative_prompt_2=neg,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE,
        height=H,
        width=W,
        generator=g,
    ).images[0]

    out_path = os.path.join(OUT_DIR, f"{tag}_seed{SEED}_gs{GUIDANCE}_steps{NUM_STEPS}_{W}x{H}.png")
    image.save(out_path)
    print("saved:", out_path)

# =========================
# 1) BASE（不載 LoRA）
# =========================
gen_one("BASE_A", content_prompt_A)
gen_one("BASE_B", content_prompt_B)

# =========================
# 2) +LoRA（明確命名 adapter + 明確 set_adapters 權重掃描）
# =========================
ADAPTER = "b4m"

# 用官方建議的 load_lora_weights 方式載入 LoRA（可處理 UNet + text encoder）:contentReference[oaicite:3]{index=3}
pipe.load_lora_weights(LORA_PATH, adapter_name=ADAPTER)

# 先用一個「偏強」的權重確認玻璃能不能回來，再往下掃
for w in [1.0, 0.8, 0.6, 0.4]:
    # set_adapters 會「啟用」並設定 adapter 權重 :contentReference[oaicite:4]{index=4}
    pipe.set_adapters([ADAPTER], adapter_weights=[w])

    gen_one(f"LORA_A_w{w}", content_prompt_A)
    gen_one(f"LORA_B_w{w}", content_prompt_B)

print("done.")
