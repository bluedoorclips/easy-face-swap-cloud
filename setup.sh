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
# Pin transformers/diffusers/peft to versions that work with torch 2.4 + InstantID.
# transformers 4.50+ uses torch.library APIs only in torch 2.5+; diffusers 0.38+ pulls in peft 0.18 which depends on those.
# These pins are the known-good combo for InstantID + RealVisXL on torch 2.4.
pip install -q -U 'gradio>=5,<6' 'huggingface_hub>=0.26'
pip install -q 'transformers==4.46.3' 'diffusers==0.31.0' 'peft==0.13.2' 'accelerate==1.1.1'
pip install -q insightface onnxruntime-gpu 'opencv-python<5' einops 'numpy<2'

echo "--- versions ---"
python -c "import gradio, huggingface_hub, diffusers, transformers, peft, torch; print(f'gradio={gradio.__version__} hf_hub={huggingface_hub.__version__} diffusers={diffusers.__version__} transformers={transformers.__version__} peft={peft.__version__} torch={torch.__version__} cuda={torch.cuda.is_available()}')"

echo "--- clone InstantID pipeline ---"
if [ ! -d /workspace/InstantID ]; then
  git clone --depth 1 https://github.com/instantX-research/InstantID.git /workspace/InstantID
fi

mkdir -p /workspace/hf_cache /workspace/outputs
export HF_HOME=/workspace/hf_cache

echo "--- launching app ---"
cd "$(dirname "$0")"
exec python cloud_swap.py
