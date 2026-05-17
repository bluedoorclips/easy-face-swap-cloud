#!/bin/bash
# RunPod boot script: install deps, launch Gradio on 7860.
# Output is teed to /workspace/boot.log so we can debug crashes.
set -e
LOG=/workspace/boot.log
exec > >(tee -a "$LOG") 2>&1

echo "=== Easy Face Swap boot $(date -u) ==="
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "--- apt-get ---"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 wget 2>&1 | tail -3

echo "--- pip install ---"
pip install -q -U pip
# gradio 5+ avoids the old HfFolder import. Pin numpy<2 for opencv/insightface compatibility.
pip install -q -U 'gradio>=5,<6' 'huggingface_hub>=0.26' diffusers transformers accelerate peft \
    insightface onnxruntime-gpu 'opencv-python<5' einops 'numpy<2'

echo "--- versions ---"
python -c "import gradio, huggingface_hub, diffusers, torch; print(f'gradio={gradio.__version__} hf_hub={huggingface_hub.__version__} diffusers={diffusers.__version__} torch={torch.__version__} cuda={torch.cuda.is_available()}')"

echo "--- clone InstantID pipeline ---"
if [ ! -d /workspace/InstantID ]; then
  git clone --depth 1 https://github.com/instantX-research/InstantID.git /workspace/InstantID
fi

mkdir -p /workspace/hf_cache /workspace/outputs
export HF_HOME=/workspace/hf_cache

echo "--- launching app ---"
cd "$(dirname "$0")"
exec python cloud_swap.py
