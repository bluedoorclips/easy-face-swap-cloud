"""
LoRA trainer for a character. Wraps diffusers' DreamBooth-LoRA-SDXL training.
Usage: python train_lora.py <character_name> <images_dir> [--max_train_steps N]
Saves the LoRA to /workspace/loras/<name>/pytorch_lora_weights.safetensors
"""
import argparse, os, subprocess, sys
from pathlib import Path

BASE_MODEL = "SG161222/RealVisXL_V4.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="character trigger name, e.g. 'baileyy'")
    ap.add_argument("images_dir", help="directory of training images")
    ap.add_argument("--max_train_steps", type=int, default=1200)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    name = args.name.strip().lower()
    if not name.isalnum():
        print(f"ERROR: name must be alphanumeric, got '{name}'", flush=True)
        sys.exit(1)

    images_dir = Path(args.images_dir)
    n_images = sum(1 for p in images_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if n_images < 5:
        print(f"ERROR: need at least 5 training images, got {n_images}", flush=True)
        sys.exit(1)

    out_dir = Path(f"/workspace/loras/{name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_script = "/workspace/diffusers_train_dreambooth_lora_sdxl.py"
    if not Path(train_script).exists():
        print(f"ERROR: training script not found at {train_script}", flush=True)
        sys.exit(1)

    instance_prompt = f"a photo of {name} woman"

    cmd = [
        "accelerate", "launch", "--mixed_precision=fp16",
        train_script,
        f"--pretrained_model_name_or_path={BASE_MODEL}",
        f"--instance_data_dir={images_dir}",
        f"--output_dir={out_dir}",
        f"--instance_prompt={instance_prompt}",
        "--resolution=1024",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=2",
        f"--learning_rate={args.lr}",
        "--lr_scheduler=cosine",
        "--lr_warmup_steps=0",
        f"--max_train_steps={args.max_train_steps}",
        "--seed=42",
        f"--rank={args.rank}",
        "--gradient_checkpointing",
        "--use_8bit_adam",
        "--checkpointing_steps=999999",
        "--mixed_precision=fp16",
        "--variant=fp16",
    ]

    print(f"[train_lora] Training character '{name}' with {n_images} images for {args.max_train_steps} steps", flush=True)
    print(f"[train_lora] Output: {out_dir}", flush=True)
    print(f"[train_lora] Command: {' '.join(cmd)}", flush=True)

    env = os.environ.copy()
    env["HF_HOME"] = "/workspace/hf_cache"
    # Fix MKL threading-layer conflict between bitsandbytes' MKL and pytorch's OpenMP
    env["MKL_THREADING_LAYER"] = "GNU"
    env["MKL_SERVICE_FORCE_INTEL"] = "1"
    # Stream stdout/stderr live
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env)
    for line in proc.stdout:
        print(line, end='', flush=True)
    rc = proc.wait()

    lora_file = out_dir / "pytorch_lora_weights.safetensors"
    if rc == 0 and lora_file.exists():
        size_mb = lora_file.stat().st_size / (1024*1024)
        print(f"[train_lora] SUCCESS - saved {lora_file} ({size_mb:.0f} MB)", flush=True)
        sys.exit(0)
    else:
        print(f"[train_lora] FAILED (rc={rc})", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
