import os
import re
import json
import time
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont

from diffusers import StableDiffusionXLPipeline, AutoencoderKL, EulerDiscreteScheduler


# -------------------------
# Utilities
# -------------------------
def slugify(s: str, max_len: int = 140) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "", s)
    return s[:max_len]


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_contact_sheet(
    images: List[Tuple[str, Image.Image]],
    out_path: str,
    cols: int = 5,
    pad: int = 12,
    title: Optional[str] = None,
) -> None:
    if not images:
        return

    w = max(im.size[0] for _, im in images)
    h = max(im.size[1] for _, im in images)

    rows = math.ceil(len(images) / cols)
    label_h = 42
    title_h = 54 if title else 0

    sheet_w = cols * w + (cols + 1) * pad
    sheet_h = title_h + rows * (h + label_h) + (rows + 1) * pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    y = pad
    if title:
        draw.text((pad, y), title, fill=(235, 235, 235), font=font)
        y += title_h

    for idx, (label, im) in enumerate(images):
        r = idx // cols
        c = idx % cols
        x0 = pad + c * (w + pad)
        y0 = y + r * (h + label_h + pad)

        sheet.paste(im, (x0, y0))
        ly = y0 + h + 8
        draw.text((x0, ly), label[:120], fill=(235, 235, 235), font=font)

    ensure_dir(os.path.dirname(out_path))
    sheet.save(out_path)
    print(f"[contact-sheet] saved: {out_path}")


# -------------------------
# Data structures
# -------------------------
@dataclass
class Task:
    name: str
    lora_path: str
    prompts: Dict[str, str]
    negative_prompt: str


# -------------------------
# Scheduler (conservative: Euler only)
# -------------------------
def build_euler_scheduler(pipe: StableDiffusionXLPipeline):
    return EulerDiscreteScheduler.from_config(pipe.scheduler.config)


# -------------------------
# Main sweep (conservative)
# -------------------------
def run_sweep(
    task: Task,
    base_model_id: str,
    vae_id: str,
    out_root: str,
    device: str,
    dtype: torch.dtype,
    width: int,
    height: int,
    seed: int,
    steps: int,                  # fixed: 25
    cfg_list: List[float],       # fixed: [3.5, 5.0]
    lora_scales: List[float],    # fixed: [0.6, 1.0]
    do_base_baseline: bool,
    skip_existing: bool,
) -> None:
    ensure_dir(out_root)

    # 1) Build pipeline
    vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_id,
        vae=vae,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    ).to(device)

    # Optional speed-ups
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[opt] xformers enabled")
    except Exception:
        print("[opt] xformers not available, skip")

    pipe.set_progress_bar_config(disable=False)
    pipe.scheduler = build_euler_scheduler(pipe)

    # 2) Output dir per run
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"{task.name}__{ts}")
    ensure_dir(out_dir)

    manifest_rows: List[dict] = []
    contact_buckets: Dict[str, List[Tuple[str, Image.Image]]] = {k: [] for k in task.prompts.keys()}

    def generate_one(
        tag: str,
        prompt: str,
        cfg: float,
        lora_scale: Optional[float],
        out_path: str,
    ) -> None:
        if skip_existing and os.path.exists(out_path):
            print(f"[skip] {out_path}")
            return

        g = torch.Generator(device=device).manual_seed(seed)

        kwargs = dict(
            prompt=prompt,
            prompt_2=prompt,  # conservative: keep same to reduce variables
            negative_prompt=task.negative_prompt,
            negative_prompt_2=task.negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=g,
            width=width,
            height=height,
        )

        if lora_scale is not None:
            kwargs["cross_attention_kwargs"] = {"scale": float(lora_scale)}

        with torch.inference_mode():
            img = pipe(**kwargs).images[0]

        img.save(out_path)
        print(f"[{tag}] saved: {out_path}")

        manifest_rows.append(
            {
                "task": task.name,
                "tag": tag,
                "out_path": out_path,
                "prompt": prompt,
                "negative_prompt": task.negative_prompt,
                "scheduler": "euler",
                "steps": steps,
                "cfg": cfg,
                "lora_scale": lora_scale,
                "seed": seed,
                "width": width,
                "height": height,
                "dtype": "fp16" if dtype == torch.float16 else "bf16",
            }
        )

        key = tag.split("__", 1)[0]
        if key in contact_buckets and lora_scale is not None:
            label = f"euler|s{steps}|cfg{cfg}|lora{lora_scale}"
            contact_buckets[key].append((label, img.copy()))

    # 3) BASE baseline
    if do_base_baseline:
        base_cfg = cfg_list[len(cfg_list) // 2]  # 5.0
        for pkey, ptxt in task.prompts.items():
            fname = f"{pkey}__BASE__euler__s{steps}__cfg{base_cfg}__seed{seed}.png"
            generate_one(
                tag=f"{pkey}__BASE",
                prompt=ptxt,
                cfg=base_cfg,
                lora_scale=None,
                out_path=os.path.join(out_dir, fname),
            )

    # 4) Load LoRA once
    lora_dir = os.path.dirname(task.lora_path)
    lora_name = os.path.basename(task.lora_path)
    print(f"[lora] load from dir={lora_dir} weight_name={lora_name}")
    pipe.load_lora_weights(lora_dir, weight_name=lora_name)

    # 5) Sweep: prompt x cfg x lora_scale (Euler, steps fixed)
    for pkey, ptxt in task.prompts.items():
        for cfg in cfg_list:
            for lora_scale in lora_scales:
                fname = f"{pkey}__LORA__euler__s{steps}__cfg{cfg}__lora{lora_scale}__seed{seed}.png"
                generate_one(
                    tag=f"{pkey}__LORA",
                    prompt=ptxt,
                    cfg=cfg,
                    lora_scale=lora_scale,
                    out_path=os.path.join(out_dir, fname),
                )

    # 6) manifest + contact sheets
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    save_jsonl(manifest_path, manifest_rows)
    print(f"[manifest] saved: {manifest_path}")

    for pkey, imgs in contact_buckets.items():
        sheet_path = os.path.join(out_dir, f"CONTACT__{pkey}.png")
        make_contact_sheet(
            imgs,
            sheet_path,
            cols=5,
            pad=12,
            title=f"{task.name} | {pkey} | euler | steps={steps} | cfg={cfg_list} | lora={lora_scales} | seed={seed}",
        )

    print(f"[done] outputs: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--vae_id", default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--task_name", default="glass__corgi")
    parser.add_argument("--out_root", default="./outputs/infer_sweep")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=77)

    # conservative fixed params
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", default="3.5,5.0")
    parser.add_argument("--lora_scales", default="0.6,1.0")

    parser.add_argument("--no_base", action="store_true")
    parser.add_argument("--no_skip_existing", action="store_true")

    parser.add_argument(
        "--prompt_token",
        default="a photo of snq corgi, w@z crafted from clean glass, transparent material, glossy reflections, refraction",
    )
    parser.add_argument(
        "--prompt_plain",
        default="a photo of a corgi, crafted from clean glass, transparent material, glossy reflections, refraction",
    )
    parser.add_argument(
        "--negative_prompt",
        default="low quality, blurry, deformed, extra limbs, text, watermark, logo, jpeg artifacts",
    )

    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    cfg_list = [float(x.strip()) for x in args.cfg.split(",") if x.strip()]
    lora_scales = [float(x.strip()) for x in args.lora_scales.split(",") if x.strip()]

    task = Task(
        name=args.task_name,
        lora_path=args.lora_path,
        prompts={
            "PROMPT_TOKEN": args.prompt_token,
            "PROMPT_PLAIN": args.prompt_plain,
        },
        negative_prompt=args.negative_prompt,
    )

    torch.backends.cuda.matmul.allow_tf32 = True

    run_sweep(
        task=task,
        base_model_id=args.base_model_id,
        vae_id=args.vae_id,
        out_root=args.out_root,
        device=args.device,
        dtype=dtype,
        width=args.width,
        height=args.height,
        seed=args.seed,
        steps=args.steps,
        cfg_list=cfg_list,
        lora_scales=lora_scales,
        do_base_baseline=(not args.no_base),
        skip_existing=(not args.no_skip_existing),
    )


if __name__ == "__main__":
    main()
