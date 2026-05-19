#!/bin/bash
# NOTE: removed `set -e` and added trap so the diagnostic HTTP fallback ALWAYS runs.
# If anything fails (pip install, model download, python crash), the trap fires
# and serves /workspace via HTTP on 7860 so we can read the logs in the browser.
LOG=/workspace/boot.log
exec > >(tee -a "$LOG") 2>&1

# === FALLBACK: always run on exit ===
fallback() {
    EXIT_CODE=$?
    echo "" | tee -a /workspace/runtime.log
    echo "=== setup.sh exited with code $EXIT_CODE at $(date -u) ===" | tee -a /workspace/runtime.log
    echo "Last 5 lines of boot.log:" | tee -a /workspace/runtime.log
    tail -5 /workspace/boot.log 2>/dev/null | tee -a /workspace/runtime.log
    echo "Starting diagnostic HTTP server on port 7860..." | tee -a /workspace/runtime.log

    cat > /workspace/index.html <<'HTML'
<!DOCTYPE html>
<html><head><title>DIAGNOSTIC MODE</title>
<style>
body { font-family: monospace; padding: 20px; background: #111; color: #eee; }
h1 { color: #f55; } h2 { color: #fa5; margin-top: 24px; }
a { color: #5af; } pre { background: #222; padding: 12px; border-radius: 4px; max-height: 50vh; overflow: auto; white-space: pre-wrap; }
</style></head>
<body>
<h1>Easy Face Swap - DIAGNOSTIC MODE</h1>
<p>Setup or v2 crashed. Logs below:</p>
<p>
<a href="/boot.log">boot.log</a> | 
<a href="/runtime.log">runtime.log</a> | 
<a href="/v2_startup_error.log">v2_startup_error.log</a>
</p>
<h2>boot.log (tail)</h2><pre id="boot">loading...</pre>
<h2>runtime.log (tail)</h2><pre id="rt">loading...</pre>
<h2>v2_startup_error.log</h2><pre id="v2">loading...</pre>
<script>
const tail = (path, id) => fetch(path).then(r => r.ok ? r.text() : `[${r.status}]`)
  .then(t => { const ls = t.split('\n'); document.getElementById(id).textContent = ls.slice(-200).join('\n'); })
  .catch(e => document.getElementById(id).textContent = String(e));
tail('/boot.log','boot');
tail('/runtime.log','rt');
tail('/v2_startup_error.log','v2');
</script>
</body></html>
HTML

    cd /workspace
    exec python -m http.server 7860 2>&1 | tee -a /workspace/runtime.log
}
trap fallback EXIT

echo "=== Easy Face Swap boot $(date -u) ==="
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "--- apt-get ---"
apt-get update -qq || echo "apt-get update FAILED"
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 wget unzip 2>&1 | tail -3 || echo "apt-get install FAILED"

echo "--- pip install ---"
pip install -q -U pip || echo "pip upgrade FAILED"
pip install -q -U 'gradio>=5,<6' || { echo "GRADIO INSTALL FAILED"; exit 1; }
pip install -q 'transformers==4.46.3' 'diffusers==0.31.0' 'peft==0.13.2' 'accelerate==1.1.1' 'huggingface_hub==0.25.2' || { echo "core libs FAILED"; exit 1; }
pip install -q insightface onnxruntime-gpu 'opencv-python<5' einops 'numpy<2' pillow-heif || { echo "image libs FAILED"; exit 1; }
# Training-only deps - failure not critical for app
pip install -q bitsandbytes datasets prodigyopt tensorboard || echo "training libs FAILED (non-critical)"

echo "--- versions ---"
python -c "import gradio, huggingface_hub, diffusers, transformers, peft, accelerate, torch; print(f'gradio={gradio.__version__} hf_hub={huggingface_hub.__version__} diffusers={diffusers.__version__} transformers={transformers.__version__} peft={peft.__version__} accelerate={accelerate.__version__} torch={torch.__version__}')" || { echo "VERSION CHECK FAILED"; exit 1; }

echo "--- antelopev2 face model ---"
PERSIST_DIR=/workspace/insightface_models/antelopev2
mkdir -p /workspace/insightface_models
mkdir -p /root/.insightface/models
rm -rf /root/.insightface/models/antelopev2
ln -sfn "$PERSIST_DIR" /root/.insightface/models/antelopev2

if [ ! -f "$PERSIST_DIR/glintr100.onnx" ]; then
  mkdir -p "$PERSIST_DIR"
  cd /tmp
  echo "Downloading antelopev2 zip..."
  wget -q -O antelopev2.zip "https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip" || \
  wget -q -O antelopev2.zip "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/antelopev2.zip" || { echo "antelopev2 download FAILED"; exit 1; }
  unzip -q -o antelopev2.zip -d /tmp/antelopev2_extract
  if [ -d /tmp/antelopev2_extract/antelopev2 ]; then
    mv /tmp/antelopev2_extract/antelopev2/* "$PERSIST_DIR/"
  else
    mv /tmp/antelopev2_extract/* "$PERSIST_DIR/"
  fi
  rm -rf /tmp/antelopev2.zip /tmp/antelopev2_extract
  ls -la "$PERSIST_DIR/"
else
  echo "antelopev2 already cached."
fi

echo "--- clone InstantID pipeline ---"
if [ ! -d /workspace/InstantID ]; then
  git clone --depth 1 https://github.com/instantX-research/InstantID.git /workspace/InstantID || { echo "InstantID clone FAILED"; exit 1; }
fi

echo "--- download diffusers training script ---"
if [ ! -f /workspace/diffusers_train_dreambooth_lora_sdxl.py ]; then
  wget -q -O /workspace/diffusers_train_dreambooth_lora_sdxl.py \
    "https://raw.githubusercontent.com/huggingface/diffusers/v0.31.0-release/examples/dreambooth/train_dreambooth_lora_sdxl.py" || echo "training script download FAILED"
fi

echo "--- configure accelerate ---"
mkdir -p /root/.cache/huggingface/accelerate
cat > /root/.cache/huggingface/accelerate/default_config.yaml <<'YAML'
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: '0'
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
YAML

mkdir -p /workspace/hf_cache /workspace/outputs /workspace/loras /workspace/training /workspace/approved
export HF_HOME=/workspace/hf_cache

echo "--- launching app ==="
cd "$(dirname "$0")"
# Run cloud_swap.py. If it exits/crashes, the trap fires and runs the fallback.
# Use exec so cloud_swap.py replaces this shell — but exec WON'T trigger trap on success.
# So we DON'T exec; we run it and let the trap handle whatever happens.
python cloud_swap.py 2>&1 | tee -a /workspace/runtime.log
echo "=== cloud_swap.py returned (this is unusual - gradio should not exit) ==="
