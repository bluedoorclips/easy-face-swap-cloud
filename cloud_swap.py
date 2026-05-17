"""
Easy Face Swap — cloud edition.
Tab 1: Swap (InstantID img2img + optional character LoRA)
Tab 2: Train Character (DreamBooth-LoRA SDXL via subprocess)
"""
import os, sys, time, traceback, subprocess, threading, shutil
from pathlib import Path

sys.path.insert(0, "/workspace/InstantID")

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
import gradio as gr
from insightface.app import FaceAnalysis
from huggingface_hub import snapshot_download
from diffusers.models import ControlNetModel

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

from pipeline_stable_diffusion_xl_instantid_img2img import StableDiffusionXLInstantIDImg2ImgPipeline
from pipeline_stable_diffusion_xl_instantid import draw_kps

OUTPUT_DIR  = Path("/workspace/outputs")
LORAS_DIR   = Path("/workspace/loras")
TRAINING_DIR= Path("/workspace/training")
for d in (OUTPUT_DIR, LORAS_DIR, TRAINING_DIR):
    d.mkdir(parents=True, exist_ok=True)

DTYPE = torch.float16
BASE_MODEL    = "SG161222/RealVisXL_V4.0"

IP_SCALE      = 0.85
CN_SCALE      = 0.80
STRENGTH      = 0.60
STEPS         = 32
GUIDANCE      = 2.5
TARGET_SIM    = 0.55
MAX_ATTEMPTS  = 2
GEN_SIZE      = 1024
MAX_INPUT_DIM = 2048
CROP_PAD      = 1.35
LORA_SCALE    = 0.8   # how strongly the character LoRA influences generation

BASE_PROMPT = "candid photograph, natural soft lighting, sharp focus, high resolution, film grain"
NEG_PROMPT = (
    "AI generated, CGI, 3d render, plastic skin, airbrushed, doll face, perfect symmetry, "
    "glossy, cartoon, illustration, painting, deformed, ugly, blurry, lowres, "
    "beauty filter, instagram filter, fake, oversaturated, posterized"
)

print("=" * 60)
print("Easy Face Swap (cloud) — loading...")
print("=" * 60)

print("[1/2] Face analyzer (antelopev2)...")
face_app = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

print(f"[2/2] InstantID img2img on {BASE_MODEL}...")
instantid_dir = snapshot_download("InstantX/InstantID", allow_patterns=["ControlNetModel/*", "ip-adapter.bin"])
controlnet = ControlNetModel.from_pretrained(os.path.join(instantid_dir, "ControlNetModel"), torch_dtype=DTYPE)
pipe = StableDiffusionXLInstantIDImg2ImgPipeline.from_pretrained(
    BASE_MODEL, controlnet=controlnet, torch_dtype=DTYPE,
    variant="fp16", use_safetensors=True,
)
pipe.load_ip_adapter_instantid(os.path.join(instantid_dir, "ip-adapter.bin"))

vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"GPU VRAM: {vram_gb:.0f} GB")
if vram_gb >= 20:
    pipe.to("cuda")
else:
    pipe.enable_model_cpu_offload(); pipe.enable_vae_tiling(); pipe.enable_vae_slicing()
pipe.set_progress_bar_config(disable=True)

# Track currently-loaded LoRA so we can unload before loading another
_current_lora = {"name": None}

print("Ready.")


def list_loras():
    """List trained characters (folders under /workspace/loras with safetensors inside)."""
    out = []
    if not LORAS_DIR.exists():
        return out
    for d in sorted(LORAS_DIR.iterdir()):
        if d.is_dir() and (d / "pytorch_lora_weights.safetensors").exists():
            out.append(d.name)
    return out


def ensure_lora_loaded(name):
    """Load the named LoRA into the pipe if not already loaded. Pass name=None to unload."""
    if _current_lora["name"] == name:
        return
    # Unload any current LoRA
    if _current_lora["name"] is not None:
        try:
            pipe.unload_lora_weights()
        except Exception as e:
            print(f"[lora-unload] {e}", flush=True)
        _current_lora["name"] = None
    if name:
        lora_file = LORAS_DIR / name / "pytorch_lora_weights.safetensors"
        if not lora_file.exists():
            raise RuntimeError(f"LoRA '{name}' not found at {lora_file}")
        pipe.load_lora_weights(str(LORAS_DIR / name), weight_name="pytorch_lora_weights.safetensors")
        _current_lora["name"] = name
        print(f"[lora] loaded '{name}'", flush=True)


def normalize_image(img):
    if img is None or img.size == 0:
        raise RuntimeError("Empty image array")
    if img.dtype == np.uint16:
        img = (img.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim != 3 or img.shape[2] != 3:
        raise RuntimeError(f"Unsupported image shape: {img.shape}")
    h, w = img.shape[:2]
    if h < 32 or w < 32:
        raise RuntimeError(f"Image too small: {w}x{h}")
    if max(h, w) > MAX_INPUT_DIM:
        scale = MAX_INPUT_DIM / max(h, w)
        nw, nh = max(64, int(round(w*scale))), max(64, int(round(h*scale)))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return img


def load_image_bgr(path):
    img = cv2.imread(path)
    if img is None or img.size == 0:
        with Image.open(path) as pil:
            pil = ImageOps.exif_transpose(pil).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return normalize_image(img)


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def biggest_face(img_bgr):
    faces = face_app.get(img_bgr)
    if not faces:
        try:
            face_app.prepare(ctx_id=0, det_size=(1024, 1024), det_thresh=0.2)
            faces = face_app.get(img_bgr)
        finally:
            face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])) if faces else None


def compute_source_embedding(source_images):
    embs = []
    for img_rgb in source_images:
        if img_rgb is None: continue
        try:
            bgr = normalize_image(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            face = biggest_face(bgr)
            if face is not None:
                embs.append(face.embedding)
        except Exception as e:
            print(f"[source-skip] {e}", flush=True)
    if not embs:
        return None
    return np.mean(np.stack(embs, axis=0), axis=0)


def swap_one(source_emb, target_path, lora_name=None):
    target_bgr = load_image_bgr(target_path)
    tgt_face = biggest_face(target_bgr)
    if tgt_face is None:
        raise RuntimeError("[step:detect-face] No face detected in target")

    H, W = target_bgr.shape[:2]
    x1, y1, x2, y2 = tgt_face.bbox.astype(int)
    w, h = max(1, x2-x1), max(1, y2-y1)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    side = int(max(w, h) * CROP_PAD)
    side = side if side % 2 == 0 else side + 1
    sx1 = max(0, cx - side//2); sy1 = max(0, cy - side//2)
    sx2 = min(W, sx1+side); sy2 = min(H, sy1+side)
    if sx2-sx1 < 64 or sy2-sy1 < 64:
        raise RuntimeError("face crop too small")
    crop_bgr = target_bgr[sy1:sy2, sx1:sx2]
    ch, cw = crop_bgr.shape[:2]
    crop_resized = cv2.resize(crop_bgr, (GEN_SIZE, GEN_SIZE), interpolation=cv2.INTER_LANCZOS4)
    crop_pil = Image.fromarray(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))
    kps_in_crop = tgt_face.kps - np.array([sx1, sy1])
    kps_scaled = kps_in_crop * np.array([GEN_SIZE/cw, GEN_SIZE/ch])
    kps_image = draw_kps(crop_pil.copy(), kps_scaled)

    # Build prompt with trigger word if LoRA is loaded
    prompt = BASE_PROMPT
    cross_attn = {}
    if lora_name:
        prompt = f"a photo of {lora_name} woman, " + BASE_PROMPT
        cross_attn = {"scale": LORA_SCALE}

    best = None; best_sim = -1.0; cur_ip = IP_SCALE
    for attempt in range(MAX_ATTEMPTS):
        pipe.set_ip_adapter_scale(cur_ip)
        seed = int(time.time()*1000) % (2**31) + attempt*7919
        gen = torch.Generator(device="cuda").manual_seed(seed)
        result = pipe(
            prompt=prompt, negative_prompt=NEG_PROMPT,
            image=crop_pil, control_image=kps_image,
            image_embeds=torch.from_numpy(source_emb).unsqueeze(0),
            strength=STRENGTH, controlnet_conditioning_scale=CN_SCALE,
            num_inference_steps=STEPS, guidance_scale=GUIDANCE,
            width=GEN_SIZE, height=GEN_SIZE, generator=gen,
            cross_attention_kwargs=cross_attn if cross_attn else None,
        ).images[0]
        gen_full = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        if best is None or best.size == 0:
            best = gen_full
        gf = face_app.get(gen_full)
        if gf:
            biggest = max(gf, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            src_norm = source_emb / (np.linalg.norm(source_emb) + 1e-9)
            sim = cosine_sim(src_norm, biggest.normed_embedding)
        else:
            sim = -1.0
        if sim > best_sim:
            best_sim = sim; best = gen_full
        if sim >= TARGET_SIM: break
        cur_ip = min(1.0, cur_ip + 0.05)

    gen_bgr = cv2.resize(best, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
    result_bgr = target_bgr.copy()
    crop_orig = result_bgr[sy1:sy2, sx1:sx2].copy()
    mask = np.zeros((ch, cw), dtype=np.uint8)
    cv2.ellipse(mask, (cw//2, ch//2), (int(cw*0.42), int(ch*0.48)), 0, 0, 360, 255, -1)
    mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=2)
    try:
        cloned = cv2.seamlessClone(gen_bgr, crop_orig, mask, (cw//2, ch//2), cv2.NORMAL_CLONE)
        result_bgr[sy1:sy2, sx1:sx2] = cloned
    except cv2.error:
        soft = cv2.GaussianBlur(mask.astype(np.float32)/255.0, (0,0), 18)[..., np.newaxis]
        result_bgr[sy1:sy2, sx1:sx2] = (gen_bgr.astype(np.float32) * soft + crop_orig.astype(np.float32) * (1-soft)).astype(np.uint8)
    return result_bgr, best_sim


def swap_batch(source_img, source_extras, target_files, character_lora, progress=gr.Progress(track_tqdm=False)):
    sources = [source_img]
    if source_extras:
        for f in source_extras:
            path = f if isinstance(f, str) else f.name
            try:
                img_bgr = load_image_bgr(path)
                sources.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            except Exception as e:
                print(f"[extra-source-skip] {path}: {e}", flush=True)

    if not any(s is not None for s in sources):
        return None, "Drop a source face first."
    if not target_files:
        return None, "Drop at least one target."

    # Load/unload character LoRA
    lora_name = (character_lora or "").strip()
    if lora_name == "(none)":
        lora_name = ""
    try:
        ensure_lora_loaded(lora_name if lora_name else None)
    except Exception as e:
        return None, f"LoRA load error: {e}"

    try:
        source_emb = compute_source_embedding(sources)
    except Exception as e:
        return None, f"Source image error: {e}"
    if source_emb is None:
        return None, "No face detected in any source image."

    run_dir = OUTPUT_DIR / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(exist_ok=True)

    results, failed, sim_log = [], [], []
    n = len(target_files)
    for i, f in enumerate(target_files):
        progress((i+1)/n, desc=f"Generating {i+1}/{n}")
        path = f if isinstance(f, str) else f.name
        try:
            out_bgr, sim = swap_one(source_emb, path, lora_name or None)
            sim_pct = max(0.0, sim) * 100
            tag = "[GOOD]" if sim >= TARGET_SIM else "[OK]"
            stem = Path(path).stem
            suffix = f"_{lora_name}" if lora_name else ""
            out_path = run_dir / f"swap_{i:03d}_sim{int(sim_pct):02d}{suffix}_{stem}.png"
            cv2.imwrite(str(out_path), out_bgr)
            results.append(str(out_path))
            sim_log.append(f"{tag} {Path(path).name}: {sim_pct:.0f}%")
        except Exception as e:
            traceback.print_exc()
            failed.append(f"{Path(path).name}: {e}")

    if sim_log:
        sims = [float(line.split(': ')[1].split('%')[0]) for line in sim_log]
        avg = sum(sims) / len(sims)
        n_src = sum(1 for s in sources if s is not None)
        msg = f"Done: {len(results)} swapped to {run_dir}\n\nSources used: {n_src}\nLoRA: {lora_name or '(none)'}\nAvg identity match: {avg:.0f}%\n\n" + "\n".join(sim_log[:20])
    else:
        msg = "No images were successfully swapped."
    if failed:
        msg += "\n\nSkipped:\n" + "\n".join(failed[:10])
    return results, msg


def refresh_lora_list():
    loras = list_loras()
    return gr.update(choices=["(none)"] + loras, value="(none)")


# ---- Training tab ----
_train_status = {"running": False, "log": "", "last_lora": None}


def train_character(character_name, photos, max_steps, progress=gr.Progress(track_tqdm=False)):
    name = (character_name or "").strip().lower()
    if not name or not name.isalnum():
        return "Character name must be alphanumeric (letters/numbers only), no spaces."
    if not photos or len(photos) < 5:
        return f"Need at least 5 photos, got {len(photos) if photos else 0}."

    if _train_status["running"]:
        return "A training job is already running. Wait for it to finish."

    images_dir = TRAINING_DIR / name / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True)

    # Copy all uploaded photos
    n_saved = 0
    for i, f in enumerate(photos):
        src = Path(f if isinstance(f, str) else f.name)
        try:
            with Image.open(src) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                img.save(images_dir / f"{name}_{i:04d}.png")
            n_saved += 1
        except Exception as e:
            print(f"[copy-skip] {src.name}: {e}", flush=True)

    if n_saved < 5:
        return f"Only {n_saved} photos were usable. Try clearer images."

    _train_status["running"] = True
    _train_status["log"] = f"Starting training for '{name}' with {n_saved} images, {max_steps} steps...\n"

    # Run training in subprocess
    cmd = [sys.executable, "/workspace/app/train_lora.py", name, str(images_dir),
           f"--max_train_steps={int(max_steps)}"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        line_count = 0
        for line in proc.stdout:
            _train_status["log"] += line
            line_count += 1
            # Update gradio progress periodically
            if line_count % 20 == 0:
                # Try to find step info in log to estimate progress
                progress(0.5, desc=f"Training... {line_count} log lines")
        proc.wait()
        rc = proc.returncode
    finally:
        _train_status["running"] = False

    lora_file = LORAS_DIR / name / "pytorch_lora_weights.safetensors"
    if rc == 0 and lora_file.exists():
        size_mb = lora_file.stat().st_size / (1024*1024)
        _train_status["last_lora"] = name
        return (f"Success. LoRA '{name}' saved ({size_mb:.0f} MB).\n"
                f"Switch to the Swap tab and pick '{name}' from the Character dropdown.\n\n"
                f"Last 20 log lines:\n" + "\n".join(_train_status["log"].splitlines()[-20:]))
    else:
        return f"Training failed (rc={rc}).\n\nLast 30 log lines:\n" + "\n".join(_train_status["log"].splitlines()[-30:])


with gr.Blocks(title="Easy Face Swap (Cloud)") as demo:
    gr.Markdown("# Easy Face Swap — Cloud")
    with gr.Tab("Swap"):
        gr.Markdown("Drop source face on the left, target images on the right, optionally pick a trained character, hit **Swap All**.")
        with gr.Row():
            with gr.Column():
                source = gr.Image(label="Source face (main)", type="numpy", height=320)
                source_extras = gr.Files(label="Extra source photos (optional)", file_types=["image"], file_count="multiple")
                character_lora = gr.Dropdown(label="Character LoRA (optional)", choices=["(none)"] + list_loras(), value="(none)")
                refresh_btn = gr.Button("Refresh character list", size="sm")
            with gr.Column():
                targets = gr.Files(label="Target images", file_types=["image"], file_count="multiple")
        btn = gr.Button("Swap All", variant="primary", size="lg")
        status = gr.Markdown("")
        gallery = gr.Gallery(label="Results", columns=3, height=600)
        btn.click(swap_batch, [source, source_extras, targets, character_lora], [gallery, status])
        refresh_btn.click(refresh_lora_list, [], [character_lora])
    with gr.Tab("Train Character"):
        gr.Markdown("**Train a character LoRA.** Pick a short name (e.g. `baileyy`), drop 15-50 clear face photos, click Train. Takes ~30-40 min on the GPU.")
        train_name = gr.Textbox(label="Character name (lowercase, letters/numbers only)", placeholder="e.g. baileyy")
        train_photos = gr.Files(label="Training photos (15-50)", file_types=["image"], file_count="multiple")
        train_steps = gr.Slider(label="Training steps", minimum=400, maximum=2000, value=1200, step=100)
        train_btn = gr.Button("Train LoRA", variant="primary", size="lg")
        train_log = gr.Markdown("")
        train_btn.click(train_character, [train_name, train_photos, train_steps], [train_log])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        allowed_paths=[str(OUTPUT_DIR)],
    )
