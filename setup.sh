#!/bin/bash
# RunPod boot script: install deps, launch Gradio on 7860.
set -e
LOG=/workspace/boot.log
exec > >(tee -a "$LOG") 2>&1

echo "=== Easy Face Swap boot $(date -u) ==="
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "--- apt-get ---"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 wget unzip 2>&1 | tail -3

echo "--- pip install ---"
pip install -q -U pip
pip install -q -U 'gradio>=5,<6'
pip install -q 'transformers==4.46.3' 'diffusers==0.31.0' 'peft==0.13.2' 'accelerate==1.1.1' 'huggingface_hub==0.25.2'
pip install -q insightface onnxruntime-gpu 'opencv-python<5' einops 'numpy<2'

echo "--- versions ---"
python -c "import gradio, huggingface_hub, diffusers, transformers, peft, torch; print(f'gradio={gradio.__version__} hf_hub={huggingface_hub.__version__} diffusers={diffusers.__version__} transformers={transformers.__version__} peft={peft.__version__} torch={torch.__version__} cuda={torch.cuda.is_available()}')"

echo "--- antelopev2 face model (cached on persistent volume) ---"
# Cache the model on the persistent volume so it survives pod stops/restarts.
PERSIST_DIR=/workspace/insightface_models/antelopev2
mkdir -p /workspace/insightface_models
mkdir -p /root/.insightface/models
# Symlink antelopev2 under the path insightface expects.
rm -rf /root/.insightface/models/antelopev2
ln -sfn "$PERSIST_DIR" /root/.insightface/models/antelopev2

if [ ! -f "$PERSIST_DIR/glintr100.onnx" ]; then
  mkdir -p "$PERSIST_DIR"
  cd /tmp
  echo "Downloading antelopev2 zip..."
  wget -q --show-progress -O antelopev2.zip "https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip" || \
  wget -q --show-progress -O antelopev2.zip "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/antelopev2.zip"
  unzip -q -o antelopev2.zip -d /tmp/antelopev2_extract
  # Models may live in either /tmp/antelopev2_extract/antelopev2/ or directly in /tmp/antelopev2_extract/
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
  git clone --depth 1 https://github.com/instantX-research/InstantID.git /workspace/InstantID
fi

mkdir -p /workspace/hf_cache /workspace/outputs
export HF_HOME=/workspace/hf_cache

echo "--- launching app ---"
cd "$(dirname "$0")"
exec python cloud_swap.py
