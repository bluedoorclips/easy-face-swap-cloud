"""
Easy Face Swap v2 — cloud edition.

Tabs:
  1. Swap          — InstantID img2img + optional character LoRA
  2. Generate      — T2I with character LoRA + traits + bulk + random prompts (NSFW levels)
  3. Approval/Swap — Pick approved generations, auto face-swap them
  4. Characters    — Library of traits per character
  5. Train         — Stage photos + train LoRA
  6. Status        — Quick overview
"""
import os, sys, time, traceback, subprocess, shutil, json, gc, random
from pathlib import Path

sys.path.insert(0, "/workspace/InstantID")
sys.path.insert(0, "/workspace/app")  # so we can import prompts_library

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
import gradio as gr
from insightface.app import FaceAnalysis
from huggingface_hub import snapshot_download
from diffusers import StableDiffusionXLPipeline
from diffusers.models import ControlNetModel

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

from pipeline_stable_diffusion_xl_instantid_img2img import StableDiffusionXLInstantIDImg2ImgPipeline
from pipeline_stable_diffusion_xl_instantid import draw_kps

try:
    from prompts_library import make_n_prompts
except Exception:
    def make_n_prompts(n, nsfw_level="off", trigger_name=None):
        return [f"a photo of {trigger_name or 'a woman'} in a candid pose"] * n

OUTPUT_DIR       = Path("/workspace/outputs")
LORAS_DIR        = Path("/workspace/loras")
TRAINING_DIR     = Path("/workspace/training")
CHARACTERS_FILE  = Path("/workspace/characters.json")
APPROVED_DIR     = Path("/workspace/approved")
for d in (OUTPUT_DIR, LORAS_DIR, TRAINING_DIR, APPROVED_DIR):
    d.mkdir(parents=True, exist_ok=True)

DTYPE = torch.float16
BASE_MODEL = "SG161222/RealVisXL_V4.0"

# Swap defaults (no LoRA)
IP_SCALE = 0.85
CN_SCALE = 0.80
STRENGTH = 0.60
STEPS    = 32
GUIDANCE = 2.5

# Swap with LoRA — LoRA leads, InstantID follows
LORA_SCALE         = 1.2
IP_SCALE_WITH_LORA = 0.5
CN_SCALE_WITH_LORA = 0.65

# T2I generation defaults
T2I_STEPS    = 30
T2I_GUIDANCE = 4.0

TARGET_SIM   = 0.55
MAX_ATTEMPTS = 2
GEN_SIZE     = 1024
MAX_INPUT_DIM = 2048
CROP_PAD     = 1.35

BASE_PROMPT_SUFFIX = "ultra high resolution, sharp focus, photorealistic, fine skin pores, natural lighting, raw photo, film grain"
NEG_PROMPT = (
    "AI generated, CGI, 3d render, plastic skin, airbrushed, doll face, perfect symmetry, "
    "glossy, cartoon, illustration, painting, deformed hands, extra fingers, missing fingers, "
    "fused fingers, deformed face, extra limbs, ugly, blurry, lowres, "
    "beauty filter, instagram filter, fake, oversaturated, posterized"
)

print("=" * 60)
print("Easy Face Swap v2 (cloud) — loading...")
print("=" * 60)

print("[1/3] Face analyzer (antelopev2)...")
face_app = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

print(f"[2/3] InstantID img2img on {BASE_MODEL}...")
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
    _PIPE_ON_GPU = True
else:
    pipe.enable_model_cpu_offload(); pipe.enable_vae_tiling(); pipe.enable_vae_slicing()
    _PIPE_ON_GPU = False
pipe.set_progress_bar_config(disable=True)

print("[3/3] T2I pipeline (shares components with swap pipe)...")
t2i_pipe = StableDiffusionXLPipeline(
    vae=pipe.vae,
    text_encoder=pipe.text_encoder,
    text_encoder_2=pipe.text_encoder_2,
    tokenizer=pipe.tokenizer,
    tokenizer_2=pipe.tokenizer_2,
    unet=pipe.unet,
    scheduler=pipe.scheduler,
)
t2i_pipe.set_progress_bar_config(disable=True)

_current_lora = {"name": None}
print("Ready.")


# ============ HELPERS ============

def _move_pipe_to(device):
    global _PIPE_ON_GPU
    try:
        pipe.to(device)
        gc.collect()
        if device == "cpu":
            torch.cuda.empty_cache()
        _PIPE_ON_GPU = (device == "cuda")
        free, total = torch.cuda.mem_get_info()
        print(f"[gpu] pipe -> {device}, free={free/(1024**3):.1f}GB / {total/(1024**3):.1f}GB", flush=True)
    except Exception as e:
        print(f"[gpu] move pipe to {device} failed: {e}", flush=True)


def _valid_name(name):
    return name and name.isalnum() and name == name.lower()


def list_loras():
    out = []
    if not LORAS_DIR.exists():
        return out
    for d in sorted(LORAS_DIR.iterdir()):
        if d.is_dir() and (d / "pytorch_lora_weights.safetensors").exists():
            out.append(d.name)
    return out


def list_staged():
    out = []
    if not TRAINING_DIR.exists():
        return out
    for d in sorted(TRAINING_DIR.iterdir()):
        if d.is_dir():
            imgs_dir = d / "images"
            if imgs_dir.exists():
                n = sum(1 for f in imgs_dir.iterdir() if f.suffix.lower() in ('.png','.jpg','.jpeg'))
                if n > 0:
                    out.append(f"{d.name} ({n} photos)")
    return out


def load_characters():
    if CHARACTERS_FILE.exists():
        try:
            return json.loads(CHARACTERS_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_characters_json(chars):
    CHARACTERS_FILE.write_text(json.dumps(chars, indent=2))


def list_characters():
    return sorted(load_characters().keys())


def ensure_lora_loaded(name):
    if _current_lora["name"] == name:
        return
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


# ============ SWAP (unchanged structurally) ============

def swap_one(source_emb, target_path, lora_name=None):
    if not _PIPE_ON_GPU:
        _move_pipe_to("cuda")
    target_bgr = load_image_bgr(target_path)
    tgt_face = biggest_face(target_bgr)
    if tgt_face is None:
        raise RuntimeError("No face detected in target")

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

    if lora_name:
        prompt = f"a photo of {lora_name} woman, " + BASE_PROMPT_SUFFIX
        cross_attn = {"scale": LORA_SCALE}
        ip = IP_SCALE_WITH_LORA
        cn = CN_SCALE_WITH_LORA
    else:
        prompt = BASE_PROMPT_SUFFIX
        cross_attn = None
        ip = IP_SCALE
        cn = CN_SCALE

    best = None; best_sim = -1.0; cur_ip = ip
    for attempt in range(MAX_ATTEMPTS):
        pipe.set_ip_adapter_scale(cur_ip)
        seed = int(time.time()*1000) % (2**31) + attempt*7919
        gen = torch.Generator(device="cuda").manual_seed(seed)
        result = pipe(
            prompt=prompt, negative_prompt=NEG_PROMPT,
            image=crop_pil, control_image=kps_image,
            image_embeds=torch.from_numpy(source_emb).unsqueeze(0),
            strength=STRENGTH, controlnet_conditioning_scale=cn,
            num_inference_steps=STEPS, guidance_scale=GUIDANCE,
            width=GEN_SIZE, height=GEN_SIZE, generator=gen,
            cross_attention_kwargs=cross_attn,
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
    results, failed = [], []
    n = len(target_files)
    for i, f in enumerate(target_files):
        progress((i+1)/n, desc=f"Swapping {i+1}/{n}")
        path = f if isinstance(f, str) else f.name
        try:
            out_bgr, sim = swap_one(source_emb, path, lora_name or None)
            stem = Path(path).stem
            out_path = run_dir / f"swap_{i:03d}_{stem}.png"
            cv2.imwrite(str(out_path), out_bgr)
            results.append(str(out_path))
        except Exception as e:
            failed.append(f"{Path(path).name}: {e}")
    msg = f"Done: {len(results)} swapped → {run_dir}"
    if failed:
        msg += "\n\nSkipped:\n" + "\n".join(failed[:10])
    return results, msg


# ============ T2I GENERATE ============

def deformity_check(image_bgr):
    """Returns (passed: bool, reason: str). Simple checks: exactly 1 face, image not all-black."""
    if image_bgr is None or image_bgr.size == 0:
        return False, "empty"
    if np.mean(image_bgr) < 5:
        return False, "all-black image"
    faces = face_app.get(image_bgr)
    if len(faces) == 0:
        return False, "no face detected"
    if len(faces) > 2:
        return False, f"too many faces ({len(faces)})"
    return True, "ok"


def generate_images(character, custom_scenario, n_images, aspect, nsfw_level, use_random_scenes, steps, guidance, progress=gr.Progress(track_tqdm=False)):
    """T2I generation with optional character LoRA + traits.
    If use_random_scenes is True, build scenes from the prompt library.
    Otherwise use custom_scenario."""
    chars = load_characters()
    character = (character or "").strip()
    if character == "(none)":
        character = ""

    lora_name = None
    char_traits = ""
    char_neg = ""
    if character and character in chars:
        c = chars[character]
        char_traits = (c.get("traits") or "").strip()
        char_neg    = (c.get("negative_traits") or "").strip()
        lora_name   = (c.get("preferred_lora") or character).strip() or None

    # If character not in library but has a LoRA folder, use that
    if not lora_name and character in list_loras():
        lora_name = character

    try:
        ensure_lora_loaded(lora_name)
    except Exception as e:
        return [], f"LoRA load error: {e}"
    cross_attn = {"scale": LORA_SCALE} if lora_name else None

    aspect_map = {
        "832x1216 (portrait)": (832, 1216),
        "1024x1024 (square)":  (1024, 1024),
        "1216x832 (landscape)":(1216, 832),
    }
    width, height = aspect_map.get(aspect, (832, 1216))

    # Build N prompts
    n = int(n_images)
    if use_random_scenes:
        prompts = make_n_prompts(n, nsfw_level=nsfw_level, trigger_name=lora_name or character or None)
        if char_traits:
            prompts = [f"{char_traits}, {p}" for p in prompts]
    else:
        scene = (custom_scenario or "").strip() or "portrait"
        trigger = f"a photo of {lora_name} woman, " if lora_name else ""
        base = f"{trigger}{char_traits + ', ' if char_traits else ''}{scene}, " + BASE_PROMPT_SUFFIX
        prompts = [base] * n

    full_neg = (char_neg + ", " + NEG_PROMPT) if char_neg else NEG_PROMPT

    out_dir = OUTPUT_DIR / time.strftime("%Y%m%d_%H%M%S_t2i")
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    pass_log = []
    for i, prompt in enumerate(prompts):
        progress((i+1)/n, desc=f"Generating {i+1}/{n}")
        seed = int(time.time()*1000) % (2**31) + i*7919
        gen = torch.Generator(device="cuda").manual_seed(seed)
        try:
            result = t2i_pipe(
                prompt=prompt, negative_prompt=full_neg,
                num_inference_steps=int(steps), guidance_scale=float(guidance),
                width=width, height=height, generator=gen,
                cross_attention_kwargs=cross_attn,
            ).images[0]
            img_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            ok, reason = deformity_check(img_bgr)
            tag = "OK" if ok else f"FAIL ({reason})"
            pass_log.append(tag)
            stem = f"t2i_{character or 'free'}_{i:02d}_seed{seed}"
            if ok:
                p = out_dir / f"{stem}.png"
            else:
                p = out_dir / f"_rejected_{stem}.png"
            cv2.imwrite(str(p), img_bgr)
            if ok:
                saved.append(str(p))
        except Exception as e:
            pass_log.append(f"ERROR: {e}")

    msg = f"Generated {len(saved)}/{n} (kept after deformity check)\nDir: {out_dir}\n\nResults per image:\n" + "\n".join(f"  [{i+1}] {pass_log[i]}" for i in range(len(pass_log)))
    return saved, msg


# ============ APPROVED → SWAP PIPELINE ============

def stage_approved_for_swap(images, character):
    """Take selected/approved images and copy them to APPROVED_DIR/{character}/
    so they can be face-swapped against. Returns paths."""
    character = (character or "free").strip().lower()
    if not character.isalnum() and character != "free":
        return [], "bad character name"
    dest = APPROVED_DIR / character
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for img in images:
        path = Path(img if isinstance(img, str) else img.name)
        try:
            new_path = dest / f"approved_{int(time.time()*1000)}_{path.name}"
            shutil.copy(path, new_path)
            out.append(str(new_path))
        except Exception as e:
            print(f"[approve-skip] {path.name}: {e}", flush=True)
    return out, f"Approved {len(out)} images saved to {dest}"


# ============ CHARACTER LIBRARY ============

def save_character(name, traits, negative_traits, preferred_lora):
    name = (name or "").strip().lower()
    if not _valid_name(name):
        return f"Bad name '{name}'. Must be lowercase alphanumeric."
    chars = load_characters()
    chars[name] = {
        "display_name": name,
        "traits": (traits or "").strip(),
        "negative_traits": (negative_traits or "").strip(),
        "preferred_lora": (preferred_lora or "").strip(),
    }
    save_characters_json(chars)
    return f"Saved character '{name}'. Library now has {len(chars)} characters."


def delete_character(name):
    name = (name or "").strip().lower()
    chars = load_characters()
    if name in chars:
        del chars[name]
        save_characters_json(chars)
        return f"Deleted '{name}'."
    return f"'{name}' not in library."


def load_character_to_form(name):
    if not name or name == "(new)":
        return "", "", "(none)", ""
    chars = load_characters()
    c = chars.get(name, {})
    return (c.get("traits", ""),
            c.get("negative_traits", ""),
            c.get("preferred_lora", "") or "(none)",
            c.get("display_name", name))


# ============ TRAINING ============

def stage_character(character_name, photos):
    name = (character_name or "").strip().lower()
    if not _valid_name(name):
        return f"Bad name '{name}'."
    if not photos:
        return "No photos provided."
    images_dir = TRAINING_DIR / name / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True)
    n_saved = 0
    for i, f in enumerate(photos):
        src = Path(f if isinstance(f, str) else f.name)
        try:
            with Image.open(src) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                img.save(images_dir / f"{name}_{i:04d}.png")
            n_saved += 1
        except Exception as e:
            print(f"[stage-skip] {src.name}: {e}", flush=True)
    return f"Staged {n_saved} photos for '{name}'."


def train_character(character_name, photos, max_steps, progress=gr.Progress(track_tqdm=False)):
    name = (character_name or "").strip().lower()
    if not _valid_name(name):
        return f"Bad name '{name}'."
    images_dir = TRAINING_DIR / name / "images"
    if photos:
        if images_dir.exists():
            shutil.rmtree(images_dir)
        images_dir.mkdir(parents=True)
        for i, f in enumerate(photos):
            src = Path(f if isinstance(f, str) else f.name)
            try:
                with Image.open(src) as img:
                    img = ImageOps.exif_transpose(img).convert("RGB")
                    img.save(images_dir / f"{name}_{i:04d}.png")
            except Exception as e:
                print(f"[stage-skip] {src.name}: {e}", flush=True)
    if not images_dir.exists():
        return f"No photos staged for '{name}'."
    n_images = sum(1 for p in images_dir.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg'))
    if n_images < 5:
        return f"Only {n_images} photos. Need at least 5."
    out_dir = LORAS_DIR / name
    progress(0.01, desc=f"Training '{name}' with {n_images} photos...")
    _move_pipe_to("cpu")
    cmd = [sys.executable, "/workspace/app/train_lora.py", name, str(images_dir),
           f"--max_train_steps={int(max_steps)}"]
    print(f"[train] {' '.join(cmd)}", flush=True)
    log_lines = []
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        line_count = 0
        for line in proc.stdout:
            log_lines.append(line.rstrip())
            print(line, end='', flush=True)
            line_count += 1
            if line_count % 30 == 0:
                progress(min(0.98, line_count/2000), desc=f"Training... {line_count} log lines")
        proc.wait()
        rc = proc.returncode
    finally:
        _move_pipe_to("cuda")
    lora_file = out_dir / "pytorch_lora_weights.safetensors"
    if rc == 0 and lora_file.exists():
        size_mb = lora_file.stat().st_size / (1024*1024)
        return f"SUCCESS: '{name}' LoRA saved ({size_mb:.0f} MB)."
    else:
        return f"FAILED (rc={rc}).\n\nLast 30 lines:\n" + "\n".join(log_lines[-30:])


def show_status():
    loras = list_loras()
    staged = list_staged()
    chars = load_characters()
    return (f"**Trained LoRAs ({len(loras)}):** {', '.join(loras) if loras else '(none)'}\n\n"
            f"**Staged but not trained:** {', '.join(staged) if staged else '(none)'}\n\n"
            f"**Characters in library ({len(chars)}):** {', '.join(sorted(chars.keys())) if chars else '(none)'}")


def refresh_all_dropdowns():
    loras = ["(none)"] + list_loras()
    chars = ["(none)"] + list_characters()
    char_edit = ["(new)"] + list_characters()
    # 5 dropdowns: swap_lora, t2i_char, char_edit_select, char_form_lora, approval_char
    return (gr.update(choices=loras, value="(none)"),
            gr.update(choices=chars, value="(none)"),
            gr.update(choices=char_edit, value="(new)"),
            gr.update(choices=loras, value="(none)"),
            gr.update(choices=chars, value="(none)"))


# ============ UI ============

with gr.Blocks(title="Easy Face Swap v2 (Cloud)") as demo:
    gr.Markdown("# Easy Face Swap — Cloud v2")

    with gr.Tab("Swap"):
        gr.Markdown("Drop source face on the left, target images on the right, optionally pick a trained character.")
        with gr.Row():
            with gr.Column():
                source = gr.Image(label="Source face (main)", type="numpy", height=320)
                source_extras = gr.Files(label="Extra source photos (optional)", file_types=["image"], file_count="multiple")
                swap_lora = gr.Dropdown(label="Character LoRA", choices=["(none)"] + list_loras(), value="(none)")
            with gr.Column():
                targets = gr.Files(label="Target images", file_types=["image"], file_count="multiple")
        swap_btn = gr.Button("Swap All", variant="primary", size="lg")
        swap_status = gr.Markdown("")
        swap_gallery = gr.Gallery(label="Results", columns=3, height=600)
        swap_btn.click(swap_batch, [source, source_extras, targets, swap_lora], [swap_gallery, swap_status])

    with gr.Tab("Generate"):
        gr.Markdown("Pick a character, choose a scenario (or random), generate.")
        with gr.Row():
            with gr.Column():
                t2i_char = gr.Dropdown(label="Character", choices=["(none)"] + list_characters(), value="(none)")
                t2i_scenario = gr.Textbox(label="Custom scene (used if 'Random scenes' is OFF)", lines=2,
                                          placeholder="e.g. at the beach in a white bikini at sunset")
                t2i_random = gr.Checkbox(label="Use random scenes from library (recommended)", value=True)
                t2i_nsfw = gr.Radio(label="NSFW level", choices=["off", "tasteful", "explicit"], value="off")
                t2i_n = gr.Slider(label="How many images", minimum=1, maximum=20, value=10, step=1)
                t2i_aspect = gr.Dropdown(label="Aspect", value="832x1216 (portrait)",
                                         choices=["832x1216 (portrait)", "1024x1024 (square)", "1216x832 (landscape)"])
                with gr.Accordion("Advanced", open=False):
                    t2i_steps = gr.Slider(label="Steps", minimum=15, maximum=50, value=T2I_STEPS, step=1)
                    t2i_guidance = gr.Slider(label="Prompt strictness", minimum=1.5, maximum=8.0, value=T2I_GUIDANCE, step=0.1)
                t2i_btn = gr.Button("Generate", variant="primary", size="lg")
            with gr.Column():
                t2i_gallery = gr.Gallery(label="Results (only deformity-passed shown)", columns=2, height=500)
                t2i_status = gr.Markdown("")
        t2i_btn.click(generate_images,
                      [t2i_char, t2i_scenario, t2i_n, t2i_aspect, t2i_nsfw, t2i_random, t2i_steps, t2i_guidance],
                      [t2i_gallery, t2i_status])

    with gr.Tab("Approve & Swap"):
        gr.Markdown("Upload (or paste paths to) generated images you've approved, pick the same character, "
                    "and run the swap pipeline to refine the face with that character's LoRA.")
        with gr.Row():
            with gr.Column():
                approval_char = gr.Dropdown(label="Character", choices=["(none)"] + list_characters(), value="(none)")
                approval_source = gr.Image(label="Source face (optional — if blank, uses LoRA only)", type="numpy", height=300)
                approval_images = gr.Files(label="Approved images (drag from Generate output folder)", file_types=["image"], file_count="multiple")
            with gr.Column():
                approval_status = gr.Markdown("")
                approval_gallery = gr.Gallery(label="Refined results", columns=2, height=500)
        approval_btn = gr.Button("Swap approved images", variant="primary", size="lg")
        # Reuse swap_batch (sources: just the supplied source_img, no extras, targets: approved images)
        approval_btn.click(swap_batch,
                           [approval_source, gr.State(None), approval_images, approval_char],
                           [approval_gallery, approval_status])

    with gr.Tab("Characters"):
        gr.Markdown("Library of trait profiles. When you pick a character on the Generate tab, these traits auto-apply.")
        with gr.Row():
            with gr.Column():
                char_edit_select = gr.Dropdown(label="Edit existing or (new)", choices=["(new)"] + list_characters(), value="(new)")
                char_name = gr.Textbox(label="Character name (lowercase, alphanumeric)", placeholder="e.g. baileyy")
                char_traits = gr.Textbox(label="Physical traits", lines=2,
                                         placeholder="e.g. dark blonde, slim build, blue eyes, freckles, small breasts")
                char_neg = gr.Textbox(label="Negative traits to avoid", lines=1,
                                      placeholder="e.g. older, gray hair, heavy makeup")
                char_form_lora = gr.Dropdown(label="LoRA to use",
                                             choices=["(none)"] + list_loras(), value="(none)")
                with gr.Row():
                    save_char_btn = gr.Button("Save character", variant="primary")
                    delete_char_btn = gr.Button("Delete")
            with gr.Column():
                char_msg = gr.Markdown("")
                gr.Markdown(
                    "**Tips:**\n"
                    "- Name should match the LoRA name for auto-trigger\n"
                    "- Traits are appended to every generation prompt\n"
                    "- Keep traits concise — model handles details if you're terse"
                )
        save_char_btn.click(save_character, [char_name, char_traits, char_neg, char_form_lora], [char_msg])
        delete_char_btn.click(delete_character, [char_name], [char_msg])
        char_edit_select.change(load_character_to_form, [char_edit_select],
                                [char_traits, char_neg, char_form_lora, char_name])

    with gr.Tab("Train"):
        gr.Markdown("Train a new character LoRA (~25 min).")
        train_name = gr.Textbox(label="Character name", placeholder="e.g. baileyy")
        train_photos = gr.Files(label="Training photos (15-50)", file_types=["image"], file_count="multiple")
        train_steps = gr.Slider(label="Training steps", minimum=400, maximum=2000, value=1200, step=100)
        with gr.Row():
            stage_btn = gr.Button("Stage photos only (fast)")
            train_btn = gr.Button("Train LoRA", variant="primary")
        train_log = gr.Markdown("")
        stage_btn.click(stage_character, [train_name, train_photos], [train_log], api_name="stage_character")
        train_btn.click(train_character, [train_name, train_photos, train_steps], [train_log], api_name="train_character")

    with gr.Tab("Status"):
        status_btn = gr.Button("Refresh status + dropdowns")
        status_md = gr.Markdown(show_status())
        status_btn.click(show_status, [], [status_md]).then(
            refresh_all_dropdowns, [],
            [swap_lora, t2i_char, char_edit_select, char_form_lora, approval_char]
        )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        allowed_paths=[str(OUTPUT_DIR), str(APPROVED_DIR)],
    )
