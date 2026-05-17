#!/bin/bash
# RunPod boot script: install deps, fetch helper repo, launch Gradio on 7860.
# Output is teed to /workspace/boot.log so you can debug crashes.
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
pip install -q diffusers transformers accelerate peft 'gradio<5' \
    insightface onnxruntime-gpu 'opencv-python<5' huggingface_hub einops 'numpy<2'

echo "--- clone InstantID pipeline ---"
if [ ! -d /workspace/InstantID ]; then
  git clone --depth 1 https://github.com/instantX-research/InstantID.git /workspace/InstantID
fi

mkdir -p /workspace/hf_cache /workspace/outputs
export HF_HOME=/workspace/hf_cache

echo "--- launching app ---"
cd "$(dirname "$0")"
exec python cloud_swap.py
