"""
Easy Face Swap v2 — cloud edition.
"""
import os, sys, time, traceback, subprocess, shutil, json, gc, random, base64, io
from pathlib import Path

CRASH_LOG = "/workspace/v2_startup_error.log"
try:
    Path("/workspace").mkdir(exist_ok=True)
except Exception:
    pass

def _log_crash_and_reraise(stage):
    tb = traceback.format_exc()
    msg = f"\n{'='*60}\nCRASH at stage: {stage}\n{'='*60}\n{tb}\n"
    print(msg, flush=True)
    try:
        with open(CRASH_LOG, "a") as f:
            f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
    except Exception:
        pass
    raise

try:
    sys.path.insert(0, "/workspace/InstantID")
    sys.path.insert(0, "/workspace/app")

    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    import gradio as gr
    from insightface.app import FaceAnalysis
    from huggingface_hub import snapshot_download
    from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image
    from diffusers.models import ControlNetModel
except Exception:
    _log_crash_and_reraise("imports")

try:
    import anthropic
    _HAS_ANTHROPIC = True
except Exception:
    _HAS_ANTHROPIC = False
    print("[anthropic] SDK not installed", flush=True)

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

try:
    from pipeline_stable_diffusion_xl_instantid_img2img import StableDiffusionXLInstantIDImg2ImgPipeline
    from pipeline_stable_diffusion_xl_instantid import draw_kps
except Exception:
    _log_crash_and_reraise("InstantID pipeline import")

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

IP_SCALE = 0.85
CN_SCALE = 0.80
# KEY CHANGE: STRENGTH 0.60 -> 0.42. Less denoising = more of the ORIGINAL real photo
# preserved = original skin texture/lighting kept = less AI sheen.
STRENGTH = 0.42
STEPS    = 32
GUIDANCE = 2.5

LORA_SCALE_SWAP     = 1.0
LORA_SCALE_GENERATE = 1.05
IP_SCALE_WITH_LORA  = 0.65
CN_SCALE_WITH_LORA  = 0.65

T2I_STEPS    = 30
T2I_GUIDANCE = 2.0

PHONE_FILTER_DEFAULT = 0.85

TARGET_SIM   = 0.55
MAX_ATTEMPTS = 3
GEN_SIZE     = 1024
MAX_INPUT_DIM = 2048
CROP_PAD     = 1.35

SMART_SWAP_VARIATIONS = 3

BASE_PROMPT_SUFFIX = "candid amateur iPhone photograph, matte skin without makeup highlights, real skin texture with pores and minor blemishes, natural skin tone, no retouching, no filter, slight ISO grain, soft uneven lighting"

NEG_PROMPT = (
    "AI generated, CGI, 3d render, plastic skin, doll face, perfect symmetry, "
    "airbrushed skin, smooth perfect skin, glowing skin, glossy skin, shiny skin, "
    "oily skin, dewy skin, highlighter on cheekbones, contoured makeup, "
    "beauty filter, instagram filter, glamour shot, magazine cover, "
    "professional fashion photography, studio softbox lighting, ring light, "
    "retouched, photoshopped, flawless, model in skincare ad, "
    "cartoon, illustration, painting, anime, oil painting, "
    "deformed hands, extra fingers, missing fingers, fused fingers, deformed face, "
    "deformed head, low forehead, hair on forehead, stray hair strand, "
    "hair artifact, dark line on face, hair bleeding into skin, "
    "extra limbs, ugly, blurry, lowres, fake, oversaturated, posterized"
)

_ANTHROPIC_KEY = {"value": os.environ.get("ANTHROPIC_API_KEY", "")}

print("=" * 60)
print("Easy Face Swap v2 (cloud) — loading...")
print("=" * 60)

try:
    print("[1/3] Face analyzer (antelopev2)...", flush=True)
    face_app = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
except Exception:
    _log_crash_and_reraise("FaceAnalysis init")

try:
    print(f"[2/3] InstantID img2img on {BASE_MODEL}...", flush=True)
    instantid_dir = snapshot_download("InstantX/InstantID", allow_patterns=["ControlNetModel/*", "ip-adapter.bin"])
    controlnet = ControlNetModel.from_pretrained(os.path.join(instantid_dir, "ControlNetModel"), torch_dtype=DTYPE)
    pipe = StableDiffusionXLInstantIDImg2ImgPipeline.from_pretrained(
        BASE_MODEL, controlnet=controlnet, torch_dtype=DTYPE,
        variant="fp16", use_safetensors=True,
    )
    pipe.load_ip_adapter_instantid(os.path.join(instantid_dir, "ip-adapter.bin"))

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU VRAM: {vram_gb:.0f} GB", flush=True)
    if vram_gb >= 20:
        pipe.to("cuda")
        _PIPE_ON_GPU = True
    else:
        pipe.enable_model_cpu_offload(); pipe.enable_vae_tiling(); pipe.enable_vae_slicing()
        _PIPE_ON_GPU = False
    pipe.set_progress_bar_config(disable=True)
except Exception:
    _log_crash_and_reraise("InstantID pipeline load")

try:
    print("[3/3] T2I pipeline...", flush=True)
    t2i_pipe = None
    try:
        t2i_pipe = AutoPipelineForText2Image.from_pipe(pipe)
        print("  -> via from_pipe (shared components)", flush=True)
    except Exception as e:
        print(f"  -> from_pipe failed ({e}); loading fresh", flush=True)
    if t2i_pipe is None:
        t2i_pipe = StableDiffusionXLPipeline.from_pretrained(
            BASE_MODEL, torch_dtype=DTYPE, variant="fp16", use_safetensors=True,
        )
        if vram_gb >= 20:
            t2i_pipe.to("cuda")
        print("  -> loaded fresh from_pretrained", flush=True)
    t2i_pipe.set_progress_bar_config(disable=True)
except Exception:
    _log_crash_and_reraise("T2I pipeline init")

_current_lora = {"name": None}
print("Ready.", flush=True)


def apply_phone_filter(img_bgr, strength=0.6):
    if strength <= 0 or img_bgr is None:
        return img_bgr
    s = float(np.clip(strength, 0, 1))
    img = img_bgr.astype(np.float32)
    threshold = 200
    over = np.maximum(img - threshold, 0)
    compression = 1.0 - 0.45 * s
    img = np.minimum(img, threshold) + over * compression
    grain_std = 3.0 * s
    noise = np.random.normal(0, grain_std, img.shape).astype(np.float32)
    img = img + noise
    desat = 1.0 - 0.08 * s
    img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = hsv[..., 1] * desat
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    warm = np.array([0, 1.5, 3.0]) * s
    img = img + warm
    return np.clip(img, 0, 255).astype(np.uint8)


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


def list_chars_and_loras():
    chars = set(load_characters().keys())
    loras = set(list_loras())
    out = []
    for name in sorted(chars | loras):
        if name in chars and name in loras:
            out.append(name)
        elif name in loras:
            out.append(f"{name} (no traits)")
        else:
            out.append(f"{name} (no LoRA)")
    return out


def parse_char_choice(choice):
    if not choice or choice == "(none)":
        return ""
    return choice.split(" (")[0]


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


def swap_one_single(source_emb, target_path, lora_name=None, seed_offset=0, filter_strength=PHONE_FILTER_DEFAULT):
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
        cross_attn = {"scale": LORA_SCALE_SWAP}
        ip = IP_SCALE_WITH_LORA
        cn = CN_SCALE_WITH_LORA
    else:
        prompt = BASE_PROMPT_SUFFIX
        cross_attn = None
        ip = IP_SCALE
        cn = CN_SCALE

    pipe.set_ip_adapter_scale(ip)
    seed = int(time.time()*1000) % (2**31) + seed_offset*7919 + random.randint(0, 1000)
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

    gf = face_app.get(gen_full)
    if gf:
        biggest = max(gf, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        src_norm = source_emb / (np.linalg.norm(source_emb) + 1e-9)
        sim = cosine_sim(src_norm, biggest.normed_embedding)
    else:
        sim = -1.0

    gen_bgr = cv2.resize(gen_full, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
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

    result_bgr = apply_phone_filter(result_bgr, strength=filter_strength)
    return result_bgr, sim


def swap_one(source_emb, target_path, lora_name=None, filter_strength=PHONE_FILTER_DEFAULT):
    best = None; best_sim = -1.0
    for attempt in range(MAX_ATTEMPTS):
        try:
            out_bgr, sim = swap_one_single(source_emb, target_path, lora_name, seed_offset=attempt, filter_strength=filter_strength)
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1 and best is None:
                raise
            continue
        if sim > best_sim:
            best_sim = sim; best = out_bgr
        if sim >= TARGET_SIM:
            break
    return best, best_sim


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

    lora_name = parse_char_choice(character_lora)
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


def _encode_jpeg_b64(img_bgr, max_dim=1024):
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def claude_pick_best(images_bgr, anthropic_key=None, reference_bgr=None):
    if not _HAS_ANTHROPIC or not images_bgr:
        return 0
    key = anthropic_key or _ANTHROPIC_KEY.get("value") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return 0
    try:
        client = anthropic.Anthropic(api_key=key)
        n = len(images_bgr)

        if reference_bgr is not None:
            prompt_text = (
                f"You are judging {n} variants of a face-swapped photograph.\n\n"
                "The FIRST image is the REFERENCE FACE — what the person SHOULD look like.\n"
                f"The next {n} images are VARIANTS where the face was swapped onto a target body/scene.\n\n"
                "Pick the variant that BEST satisfies BOTH:\n"
                "  (a) Identity match: same eyes, nose, jaw, skin tone, facial structure as the reference\n"
                "  (b) Realism: looks like a real candid photo, matte skin, no plastic AI glow, no deformities, "
                "no hair artifacts on the forehead\n\n"
                f"Reply with ONLY the variant number 1 through {n}. No words, no explanation, just the digit."
            )
            content = [{"type": "text", "text": prompt_text}]
            content.append({"type": "text", "text": "REFERENCE FACE:"})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": _encode_jpeg_b64(reference_bgr)}
            })
            content.append({"type": "text", "text": f"VARIANTS (pick one, 1 to {n}):"})
            for img in images_bgr:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": _encode_jpeg_b64(img)}
                })
        else:
            prompt_text = (
                f"You are judging {n} variants of a face-swapped photograph. "
                "Pick the one that looks MOST like a real candid photograph (matte skin, natural lighting, "
                "no obvious AI artifacts, no plastic glow, no deformities). "
                f"Reply with ONLY the number 1 through {n}. No explanation."
            )
            content = [{"type": "text", "text": prompt_text}]
            for img in images_bgr:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": _encode_jpeg_b64(img)}
                })

        msg = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=10,
            messages=[{"role": "user", "content": content}]
        )
        text = msg.content[0].text.strip()
        for ch in text:
            if ch.isdigit():
                idx = int(ch) - 1
                if 0 <= idx < n:
                    return idx
        return 0
    except Exception as e:
        print(f"[claude-judge] failed: {e}, falling back to index 0", flush=True)
        return 0


def smart_swap_batch(anthropic_key, source_img, target_files, character_lora, progress=gr.Progress(track_tqdm=False)):
    if anthropic_key and anthropic_key != "***SAVED***":
        _ANTHROPIC_KEY["value"] = anthropic_key.strip()

    if source_img is None:
        return None, "Drop a source face first (top left of this tab)."
    if not target_files:
        return None, "Drop at least one target photo."

    lora_name = parse_char_choice(character_lora)
    try:
        ensure_lora_loaded(lora_name if lora_name else None)
    except Exception as e:
        return None, f"LoRA load error: {e}"

    try:
        source_emb = compute_source_embedding([source_img])
    except Exception as e:
        return None, f"Source image error: {e}"
    if source_emb is None:
        return None, "No face detected in your source photo."

    try:
        ref_bgr = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR) if source_img is not None else None
    except Exception:
        ref_bgr = None

    run_dir = OUTPUT_DIR / time.strftime("%Y%m%d_%H%M%S_smart")
    run_dir.mkdir(exist_ok=True)

    face_photos = []
    skipped = []
    total = len(target_files)
    for i, f in enumerate(target_files):
        progress(i / (total * 2), desc=f"Filtering {i+1}/{total}")
        path = f if isinstance(f, str) else f.name
        try:
            img_bgr = load_image_bgr(path)
            if biggest_face(img_bgr) is not None:
                face_photos.append(path)
            else:
                skipped.append(f"{Path(path).name}: no face detected")
        except Exception as e:
            skipped.append(f"{Path(path).name}: {e}")

    if not face_photos:
        return None, f"None of the {total} photos contained a detectable face."

    results = []
    judge_log = []
    use_claude = _HAS_ANTHROPIC and bool(_ANTHROPIC_KEY["value"])
    judge_name = "Claude Haiku (w/ reference face)" if use_claude else "face-sim score"

    for i, path in enumerate(face_photos):
        prog_frac = 0.5 + (i / (len(face_photos) * 2))
        progress(prog_frac, desc=f"Swapping {i+1}/{len(face_photos)} ({SMART_SWAP_VARIATIONS} variations)")
        variations = []
        sims = []
        for v in range(SMART_SWAP_VARIATIONS):
            try:
                out_bgr, sim = swap_one_single(source_emb, path, lora_name or None, seed_offset=v)
                variations.append(out_bgr)
                sims.append(sim)
            except Exception as e:
                print(f"[smart-swap variation {v}] {Path(path).name}: {e}", flush=True)

        if not variations:
            judge_log.append(f"  {Path(path).name}: ALL VARIATIONS FAILED")
            continue

        if use_claude and len(variations) > 1:
            best_idx = claude_pick_best(variations, anthropic_key=_ANTHROPIC_KEY["value"], reference_bgr=ref_bgr)
            judge_log.append(f"  {Path(path).name}: {len(variations)} variants, Claude picked #{best_idx+1} (sim {sims[best_idx]:.2f})")
        elif len(variations) > 1:
            best_idx = int(np.argmax(sims))
            judge_log.append(f"  {Path(path).name}: {len(variations)} variants, math picked #{best_idx+1} (sim {sims[best_idx]:.2f})")
        else:
            best_idx = 0
            judge_log.append(f"  {Path(path).name}: only 1 variant succeeded")

        stem = Path(path).stem
        out_path = run_dir / f"smart_{i:03d}_{stem}.png"
        cv2.imwrite(str(out_path), variations[best_idx])
        results.append(str(out_path))

    msg = (f"**Smart Swap done**\n\n"
           f"- Photos in: {total}\n"
           f"- With face: {len(face_photos)}\n"
           f"- Swapped: {len(results)}\n"
           f"- Judge: {judge_name}\n"
           f"- Output dir: `{run_dir}`\n\n"
           f"**Per-image:**\n" + "\n".join(judge_log[:30]))
    if skipped:
        msg += f"\n\n**Skipped (no face):**\n" + "\n".join(skipped[:10])
    return results, msg


def deformity_check(image_bgr):
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


def generate_images(character, custom_scenario, n_images, aspect, nsfw_level, use_random_scenes, steps, guidance, filter_strength, progress=gr.Progress(track_tqdm=False)):
    chars = load_characters()
    character = parse_char_choice(character)

    lora_name = None
    char_traits = ""
    char_neg = ""
    if character and character in chars:
        c = chars[character]
        char_traits = (c.get("traits") or "").strip()
        char_neg    = (c.get("negative_traits") or "").strip()
        lora_name   = (c.get("preferred_lora") or character).strip() or None

    if not lora_name and character in list_loras():
        lora_name = character

    try:
        ensure_lora_loaded(lora_name)
    except Exception as e:
        return [], f"LoRA load error: {e}"
    cross_attn = {"scale": LORA_SCALE_GENERATE} if lora_name else None

    aspect_map = {
        "832x1216 (portrait)": (832, 1216),
        "1024x1024 (square)":  (1024, 1024),
        "1216x832 (landscape)":(1216, 832),
    }
    width, height = aspect_map.get(aspect, (832, 1216))

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
            img_bgr = apply_phone_filter(img_bgr, strength=float(filter_strength))
            stem = f"t2i_{character or 'free'}_{i:02d}_seed{seed}"
            if ok:
                p = out_dir / f"{stem}.png"
            else:
                p = out_dir / f"_rejected_{stem}.png"
            cv2.imwrite(str(p), img_bgr)
            if ok:
                saved.append(str(p))
        except Exception as e:
            traceback.print_exc()
            pass_log.append(f"ERROR: {e}")

    msg = f"Generated {len(saved)}/{n} (kept after deformity check)\nDir: {out_dir}\n\nResults per image:\n" + "\n".join(f"  [{i+1}] {pass_log[i]}" for i in range(len(pass_log)))
    return saved, msg


def stage_approved_for_swap(images, character):
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
    has_anthropic = "yes" if _ANTHROPIC_KEY["value"] else "no (Smart Swap will use math judging)"
    return (f"**Trained LoRAs ({len(loras)}):** {', '.join(loras) if loras else '(none)'}\n\n"
            f"**Staged but not trained:** {', '.join(staged) if staged else '(none)'}\n\n"
            f"**Characters in library ({len(chars)}):** {', '.join(sorted(chars.keys())) if chars else '(none)'}\n\n"
            f"**Anthropic key loaded:** {has_anthropic}")


def refresh_all_dropdowns():
    loras = ["(none)"] + list_loras()
    chars_and_loras = ["(none)"] + list_chars_and_loras()
    char_edit = ["(new)"] + list_characters()
    return (gr.update(choices=loras, value="(none)"),
            gr.update(choices=chars_and_loras, value="(none)"),
            gr.update(choices=char_edit, value="(new)"),
            gr.update(choices=loras, value="(none)"),
            gr.update(choices=chars_and_loras, value="(none)"),
            gr.update(choices=loras, value="(none)"))


with gr.Blocks(title="Easy Face Swap v2 (Cloud)") as demo:
    gr.Markdown("# Easy Face Swap — Cloud v2")

    with gr.Tab("Smart Swap"):
        gr.Markdown("**Easiest workflow.** Drop a bunch of photos, pick a character, hit Smart Swap. "
                    "I filter face-photos automatically, run 3 variations per photo, and Claude Haiku picks "
                    "the variation that best matches your source face AND looks most like a real photo.")
        with gr.Row():
            with gr.Column():
                smart_anthropic = gr.Textbox(label="Anthropic API Key (kept in memory only)",
                                             type="password",
                                             placeholder="sk-ant-api03-...",
                                             value=("***SAVED***" if _ANTHROPIC_KEY["value"] else ""))
                smart_source = gr.Image(label="Source face (your face) — drag here", type="numpy", height=300)
                smart_char = gr.Dropdown(label="Character LoRA (optional)",
                                         choices=["(none)"] + list_loras(), value="(none)")
            with gr.Column():
                smart_targets = gr.Files(label="Target photos — drop a bunch (real photos, sfw or nsfw)",
                                         file_types=["image"], file_count="multiple")
        smart_btn = gr.Button("Smart Swap All", variant="primary", size="lg")
        smart_status = gr.Markdown("")
        smart_gallery = gr.Gallery(label="Best of each batch (one per face-photo)", columns=3, height=600)
        smart_btn.click(smart_swap_batch,
                        [smart_anthropic, smart_source, smart_targets, smart_char],
                        [smart_gallery, smart_status])

    with gr.Tab("Swap"):
        gr.Markdown("Drop source face, target images, optionally pick a trained character.")
        with gr.Row():
            with gr.Column():
                source = gr.Image(label="Source face (main) — drag a photo here", type="numpy", height=320)
                source_extras = gr.Files(label="Extra source photos (optional) — drag-drop or click", file_types=["image"], file_count="multiple")
                swap_lora = gr.Dropdown(label="Character LoRA", choices=["(none)"] + list_loras(), value="(none)")
            with gr.Column():
                targets = gr.Files(label="Target images — drag-drop or click", file_types=["image"], file_count="multiple")
        swap_btn = gr.Button("Swap All", variant="primary", size="lg")
        swap_status = gr.Markdown("")
        swap_gallery = gr.Gallery(label="Results", columns=3, height=600)
        swap_btn.click(swap_batch, [source, source_extras, targets, swap_lora], [swap_gallery, swap_status])

    with gr.Tab("Generate"):
        gr.Markdown("Pick a character (LoRA folders auto-listed), choose a scenario (or random), generate.")
        with gr.Row():
            with gr.Column():
                t2i_char = gr.Dropdown(label="Character", choices=["(none)"] + list_chars_and_loras(), value="(none)")
                t2i_scenario = gr.Textbox(label="Custom scene", lines=2)
                t2i_random = gr.Checkbox(label="Use random scenes from library", value=True)
                t2i_nsfw = gr.Radio(label="NSFW level", choices=["off", "tasteful", "explicit"], value="off")
                t2i_n = gr.Slider(label="How many images", minimum=1, maximum=20, value=10, step=1)
                t2i_aspect = gr.Dropdown(label="Aspect", value="832x1216 (portrait)",
                                         choices=["832x1216 (portrait)", "1024x1024 (square)", "1216x832 (landscape)"])
                t2i_filter = gr.Slider(label="Phone-photo filter", minimum=0.0, maximum=1.0, value=PHONE_FILTER_DEFAULT, step=0.05)
                with gr.Accordion("Advanced", open=False):
                    t2i_steps = gr.Slider(label="Steps", minimum=15, maximum=50, value=T2I_STEPS, step=1)
                    t2i_guidance = gr.Slider(label="Prompt strictness", minimum=1.0, maximum=8.0, value=T2I_GUIDANCE, step=0.1)
                t2i_btn = gr.Button("Generate", variant="primary", size="lg")
            with gr.Column():
                t2i_gallery = gr.Gallery(label="Results", columns=2, height=500)
                t2i_status = gr.Markdown("")
        t2i_btn.click(generate_images,
                      [t2i_char, t2i_scenario, t2i_n, t2i_aspect, t2i_nsfw, t2i_random, t2i_steps, t2i_guidance, t2i_filter],
                      [t2i_gallery, t2i_status])

    with gr.Tab("Approve & Swap"):
        with gr.Row():
            with gr.Column():
                approval_char = gr.Dropdown(label="Character", choices=["(none)"] + list_chars_and_loras(), value="(none)")
                approval_source = gr.Image(label="Source face", type="numpy", height=300)
                approval_images = gr.Files(label="Approved images", file_types=["image"], file_count="multiple")
            with gr.Column():
                approval_status = gr.Markdown("")
                approval_gallery = gr.Gallery(label="Refined results", columns=2, height=500)
        approval_btn = gr.Button("Swap approved", variant="primary", size="lg")
        approval_btn.click(swap_batch,
                           [approval_source, gr.State(None), approval_images, approval_char],
                           [approval_gallery, approval_status])

    with gr.Tab("Characters"):
        with gr.Row():
            with gr.Column():
                char_edit_select = gr.Dropdown(label="Edit existing or (new)", choices=["(new)"] + list_characters(), value="(new)")
                char_name = gr.Textbox(label="Character name")
                char_traits = gr.Textbox(label="Physical traits", lines=2)
                char_neg = gr.Textbox(label="Negative traits", lines=1)
                char_form_lora = gr.Dropdown(label="LoRA", choices=["(none)"] + list_loras(), value="(none)")
                with gr.Row():
                    save_char_btn = gr.Button("Save character", variant="primary")
                    delete_char_btn = gr.Button("Delete")
            with gr.Column():
                char_msg = gr.Markdown("")
        save_char_btn.click(save_character, [char_name, char_traits, char_neg, char_form_lora], [char_msg])
        delete_char_btn.click(delete_character, [char_name], [char_msg])
        char_edit_select.change(load_character_to_form, [char_edit_select],
                                [char_traits, char_neg, char_form_lora, char_name])

    with gr.Tab("Train"):
        train_name = gr.Textbox(label="Character name")
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
            [swap_lora, t2i_char, char_edit_select, char_form_lora, approval_char, smart_char]
        )


if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        allowed_paths=[str(OUTPUT_DIR), str(APPROVED_DIR)],
        max_file_size="100mb",
        ssr_mode=False,
    )
