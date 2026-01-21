import os
import time
import argparse
import json
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, EulerDiscreteScheduler


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--vae_id", default="madebyollin/sdxl-vae-fp16-fix")
    p.add_argument("--lora_path", required=True)
    p.add_argument("--task_name", default="glass__corgi_quick")
    p.add_argument("--out_root", default="./outputs/infer_quick")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--seed", type=int, default=77)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)

    # 兩個 prompt：token vs plain（最值得報告的對照）
    p.add_argument(
        "--prompt_token",
        default="a photo of snq corgi, w@z crafted from clean glass, transparent material, glossy reflections, refraction",
    )
    p.add_argument(
        "--prompt_plain",
        default="a photo of a corgi, crafted from clean glass, transparent material, glossy reflections, refraction",
    )
    p.add_argument(
        "--negative_prompt",
        default="low quality, blurry, deformed, extra limbs, text, watermark, logo, jpeg artifacts",
    )
    args = p.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    # ---- output dir ----
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_root, f"{args.task_name}__{ts}")
    ensure_dir(out_dir)

    # ---- build pipeline (fast + stable) ----
    vae = AutoencoderKL.from_pretrained(args.vae_id, torch_dtype=dtype)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model_id,
        vae=vae,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    ).to(args.device)

    # optional speed
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[opt] xformers enabled")
    except Exception:
        print("[opt] xformers not available, skip")

    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=False)

    prompts = {
        "PROMPT_TOKEN": args.prompt_token,
        "PROMPT_PLAIN": args.prompt_plain,
    }

    manifest = []

    def run_one(tag: str, prompt: str, lora_scale, out_path: str):
        # fixed seed for reproducibility
        g = torch.Generator(device=args.device).manual_seed(args.seed)

        kwargs = dict(
            prompt=prompt,
            prompt_2=prompt,  # reduce variables (fast + stable)
            negative_prompt=args.negative_prompt,
            negative_prompt_2=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            width=args.width,
            height=args.height,
            generator=g,
        )

        if lora_scale is not None:
            kwargs["cross_attention_kwargs"] = {"scale": float(lora_scale)}

        with torch.inference_mode():
            img = pipe(**kwargs).images[0]

        img.save(out_path)
        print(f"[{tag}] saved: {out_path}")

        manifest.append(
            dict(
                tag=tag,
                out_path=out_path,
                prompt=prompt,
                scheduler="euler",
                steps=args.steps,
                cfg=args.cfg,
                lora_scale=lora_scale,
                seed=args.seed,
                width=args.width,
                height=args.height,
                dtype=args.dtype,
            )
        )

    # =========
    # 1) BASE (2 imgs)
    # =========
    for pkey, ptxt in prompts.items():
        out_path = os.path.join(
            out_dir, f"{pkey}__BASE__euler__s{args.steps}__cfg{args.cfg}__seed{args.seed}.png"
        )
        run_one(f"{pkey}__BASE", ptxt, None, out_path)

    # =========
    # 2) Load LoRA once
    # =========
    lora_dir = os.path.dirname(args.lora_path)
    lora_name = os.path.basename(args.lora_path)
    print(f"[lora] load from dir={lora_dir} weight_name={lora_name}")
    pipe.load_lora_weights(lora_dir, weight_name=lora_name)

    # =========
    # 3) LoRA scale sweep (4 imgs)
    # =========
    for pkey, ptxt in prompts.items():
        for s in [0.6, 1.0]:
            out_path = os.path.join(
                out_dir, f"{pkey}__LORA__euler__s{args.steps}__cfg{args.cfg}__lora{s}__seed{args.seed}.png"
            )
            run_one(f"{pkey}__LORA__lora{s}", ptxt, s, out_path)

    # ---- manifest ----
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("[manifest] saved:", manifest_path)
    print("[done] outputs:", out_dir)


if __name__ == "__main__":
    main()
