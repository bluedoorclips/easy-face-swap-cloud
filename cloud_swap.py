"""
Easy Face Swap — cloud edition. Pure drag-drop UI, no slider tuning exposed.
"""
import os, sys, time, traceback
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

from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

OUTPUT_DIR = Path("/workspace/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DTYPE = torch.float16
IP_SCALE, CN_SCALE, STEPS, TARGET_SIM, MAX_ATTEMPTS, GEN_SIZE, MAX_INPUT_DIM = 0.85, 0.85, 22, 0.65, 2, 1024, 2048

print("=" * 60)
print("Easy Face Swap (cloud) — loading...")
print("=" * 60)

print("[1/2] Face analyzer (antelopev2)...")
face_app = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

print("[2/2] InstantID + RealVisXL...")
instantid_dir = snapshot_download("InstantX/InstantID", allow_patterns=["ControlNetModel/*", "ip-adapter.bin"])
controlnet = ControlNetModel.from_pretrained(os.path.join(instantid_dir, "ControlNetModel"), torch_dtype=DTYPE)
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0", controlnet=controlnet, torch_dtype=DTYPE,
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
print("Ready.")


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


def swap_one(source_emb, target_path):
    # Each step wrapped — error message will say which one died
    try:
        target_bgr = load_image_bgr(target_path)
    except Exception as e:
        raise RuntimeError(f"[step:load] {type(e).__name__}: {e}")
    print(f"target shape: {target_bgr.shape} dtype: {target_bgr.dtype}", flush=True)

    try:
        tgt_face = biggest_face(target_bgr)
    except Exception as e:
        raise RuntimeError(f"[step:detect-face] {type(e).__name__}: {e}")
    if tgt_face is None:
        raise RuntimeError("[step:detect-face] No face detected in target")

    try:
        H, W = target_bgr.shape[:2]
        x1, y1, x2, y2 = tgt_face.bbox.astype(int)
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        side = int(max(w, h) * 1.3)
        side = side if side % 2 == 0 else side + 1
        sx1 = max(0, cx - side//2); sy1 = max(0, cy - side//2)
        sx2 = min(W, sx1 + side);   sy2 = min(H, sy1 + side)
        if sx2 - sx1 < 64 or sy2 - sy1 < 64:
            raise RuntimeError(f"face crop {sx2-sx1}x{sy2-sy1} too small (face near edge)")
        crop_bgr = target_bgr[sy1:sy2, sx1:sx2]
        ch, cw = crop_bgr.shape[:2]
        print(f"crop shape: {crop_bgr.shape}", flush=True)
        crop_resized = cv2.resize(crop_bgr, (GEN_SIZE, GEN_SIZE), interpolation=cv2.INTER_LANCZOS4)
        crop_pil = Image.fromarray(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))
        kps_in_crop = tgt_face.kps - np.array([sx1, sy1])
        kps_scaled = kps_in_crop * np.array([GEN_SIZE/cw, GEN_SIZE/ch])
        kps_image = draw_kps(crop_pil, kps_scaled)
    except Exception as e:
        raise RuntimeError(f"[step:crop+kps] {type(e).__name__}: {e}")

    best = None; best_sim = -1.0; cur_ip = IP_SCALE

    for attempt in range(MAX_ATTEMPTS):
        try:
            pipe.set_ip_adapter_scale(cur_ip)
            seed = int(time.time()*1000) % (2**31) + attempt*7919
            gen = torch.Generator(device="cuda").manual_seed(seed)
            result = pipe(
                prompt="raw photo of a person, ultra realistic photo, natural skin with pores, detailed eyes, "
                       "natural lighting, sharp focus, high resolution, candid",
                negative_prompt="cartoon, painting, 3d, plastic, smooth, airbrushed, blurry, lowres, deformed, ugly",
                image_embeds=torch.from_numpy(source_emb).unsqueeze(0),
                image=kps_image,
                controlnet_conditioning_scale=CN_SCALE,
                num_inference_steps=STEPS, guidance_scale=5.0,
                width=GEN_SIZE, height=GEN_SIZE, generator=gen,
            ).images[0]
        except Exception as e:
            raise RuntimeError(f"[step:diffuse attempt={attempt}] {type(e).__name__}: {e}")
        try:
            gen_full = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            gf = face_app.get(gen_full)
            if gf:
                biggest = max(gf, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                sim = cosine_sim(source_emb, biggest.normed_embedding)
            else:
                sim = -1.0
            if sim > best_sim:
                best_sim = sim; best = gen_full
            if sim >= TARGET_SIM:
                break
            cur_ip = min(1.0, cur_ip + 0.05)
        except Exception as e:
            raise RuntimeError(f"[step:score attempt={attempt}] {type(e).__name__}: {e}")

    try:
        gen_bgr = cv2.resize(best, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
        mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.ellipse(mask, (cw//2, ch//2), (int(cw*0.40), int(ch*0.46)), 0, 0, 360, 255, -1)
        mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=2)
        result_bgr = target_bgr.copy()
        crop_orig = result_bgr[sy1:sy2, sx1:sx2].copy()
        try:
            cloned = cv2.seamlessClone(gen_bgr, crop_orig, mask, (cw//2, ch//2), cv2.NORMAL_CLONE)
            result_bgr[sy1:sy2, sx1:sx2] = cloned
        except cv2.error:
            soft = cv2.GaussianBlur(mask.astype(np.float32)/255.0, (0,0), 12)[..., np.newaxis]
            result_bgr[sy1:sy2, sx1:sx2] = (gen_bgr * soft + crop_orig * (1-soft)).astype(np.uint8)
        return result_bgr, best_sim
    except Exception as e:
        raise RuntimeError(f"[step:composite] {type(e).__name__}: {e}")


def swap_batch(source_img, target_files, progress=gr.Progress(track_tqdm=False)):
    if source_img is None:
        return None, "Drop a source face first."
    if not target_files:
        return None, "Drop at least one target."
    try:
        src_bgr = normalize_image(cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR))
        src_face = biggest_face(src_bgr)
    except Exception as e:
        return None, f"Source image error: {e}"
    if src_face is None:
        return None, "No face detected in the source image. Try a clearer front-facing photo."
    source_emb = src_face.normed_embedding

    run_dir = OUTPUT_DIR / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(exist_ok=True)

    results, failed, sim_log = [], [], []
    n = len(target_files)
    for i, f in enumerate(target_files):
        progress((i+1)/n, desc=f"Generating {i+1}/{n}")
        path = f if isinstance(f, str) else f.name
        try:
            out_bgr, sim = swap_one(source_emb, path)
            sim_pct = max(0.0, sim) * 100
            tag = "[GOOD]" if sim >= TARGET_SIM else "[WEAK]"
            stem = Path(path).stem
            out_path = run_dir / f"swap_{i:03d}_sim{int(sim_pct):02d}_{stem}.png"
            cv2.imwrite(str(out_path), out_bgr)
            results.append(str(out_path))
            sim_log.append(f"{tag} {Path(path).name}: {sim_pct:.0f}%")
        except Exception as e:
            traceback.print_exc()
            failed.append(f"{Path(path).name}: {e}")

    if sim_log:
        sims = [float(line.split(': ')[1].split('%')[0]) for line in sim_log]
        avg = sum(sims) / len(sims)
        msg = f"Done: {len(results)} swapped → {run_dir}\n\n**Avg identity match: {avg:.0f}%**\n\n" + "\n".join(sim_log[:20])
    else:
        msg = "No images were successfully swapped."
    if failed:
        msg += "\n\nSkipped:\n" + "\n".join(failed[:10])
    return results, msg


with gr.Blocks(title="Easy Face Swap (Cloud)") as demo:
    gr.Markdown("# Easy Face Swap — Cloud")
    gr.Markdown("Drop your source face on the left, drop target images on the right, hit **Swap All**.")
    with gr.Row():
        with gr.Column():
            source = gr.Image(label="Source face", type="numpy", height=400)
        with gr.Column():
            targets = gr.Files(label="Target images", file_types=["image"], file_count="multiple")
    btn = gr.Button("Swap All", variant="primary", size="lg")
    status = gr.Markdown("")
    gallery = gr.Gallery(label="Results", columns=3, height=600)
    btn.click(swap_batch, [source, targets], [gallery, status])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
