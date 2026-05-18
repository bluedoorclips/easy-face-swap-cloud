#!/bin/bash
# Pod bootstrap. Designed to be the entire dockerArgs:
#   bash -c "wget -q -O /tmp/boot.sh https://raw.githubusercontent.com/bluedoorclips/easy-face-swap-cloud/main/boot.sh && bash /tmp/boot.sh"
#
# Pulls latest cloud_swap.py + prompts_library.py + train_lora.py from github
# on every container start, then exec python. Stop/start picks up new code
# automatically — no redeploy needed.
set -e
echo "=== Easy Face Swap v2 pod boot ==="

REPO=https://raw.githubusercontent.com/bluedoorclips/easy-face-swap-cloud/main

# System deps
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 wget 2>&1 | tail -3

# Python deps
pip install -q -U pip
pip install -q diffusers transformers accelerate peft 'gradio<5' insightface \
    onnxruntime-gpu 'opencv-python<5' huggingface_hub einops 'numpy<2' pillow-heif

# InstantID pipeline source (only clone once - lives on persistent /workspace)
if [ ! -d /workspace/InstantID ]; then
    git clone --depth 1 https://github.com/instantX-research/InstantID.git /workspace/InstantID
fi

# Persistent dirs (preserved across stop/start)
mkdir -p /workspace/hf_cache /workspace/outputs /workspace/loras \
         /workspace/training /workspace/approved /workspace/app
export HF_HOME=/workspace/hf_cache

# Pull v2 code from github on every boot
cd /workspace
wget -q -O cloud_swap.py     $REPO/cloud_swap.py
wget -q -O prompts_library.py $REPO/prompts_library.py
cp prompts_library.py app/prompts_library.py
wget -q -O app/train_lora.py $REPO/train_lora.py
echo "=== code fetched ==="
ls -l /workspace/cloud_swap.py /workspace/prompts_library.py /workspace/app/train_lora.py

# Launch (RunPod exposes port 7860)
exec python cloud_swap.py
