#!/bin/bash
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
pip install -q insightface onnxruntime-gpu 'opencv-python<5' einops 'numpy<2' pillow-heif
# Training-only deps
pip install -q bitsandbytes datasets prodigyopt tensorboard

echo "--- versions ---"
python -c "import gradio, huggingface_hub, diffusers, transformers, peft, accelerate, torch; print(f'gradio={gradio.__version__} hf_hub={huggingface_hub.__version__} diffusers={diffusers.__version__} transformers={transformers.__version__} peft={peft.__version__} accelerate={accelerate.__version__} torch={torch.__version__}')"

echo "--- antelopev2 face model (cached on persistent volume) ---"
PERSIST_DIR=/workspace/insightface_models/antelopev2
mkdir -p /workspace/insightface_models
mkdir -p /root/.insightface/models
rm -rf /root/.insightface/models/antelopev2
ln -sfn "$PERSIST_DIR" /root/.insightface/models/antelopev2

if [ ! -f "$PERSIST_DIR/glintr100.onnx" ]; then
  mkdir -p "$PERSIST_DIR"
  cd /tmp
  echo "Downloading antelopev2 zip..."
  wget -q --show-progress -O antelopev2.zip "https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip" || \
  wget -q --show-progress -O antelopev2.zip "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/antelopev2.zip"
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
  git clone --depth 1 https://github.com/instantX-research/InstantID.git /workspace/InstantID
fi

echo "--- download diffusers training script ---"
if [ ! -f /workspace/diffusers_train_dreambooth_lora_sdxl.py ]; then
  wget -q -O /workspace/diffusers_train_dreambooth_lora_sdxl.py \
    "https://raw.githubusercontent.com/huggingface/diffusers/v0.31.0-release/examples/dreambooth/train_dreambooth_lora_sdxl.py"
  echo "Training script size: $(wc -l < /workspace/diffusers_train_dreambooth_lora_sdxl.py) lines"
fi

echo "--- configure accelerate ---"
# Single-GPU non-distributed config
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

mkdir -p /workspace/hf_cache /workspace/outputs /workspace/loras /workspace/training
export HF_HOME=/workspace/hf_cache

echo "--- launching app ---"
cd "$(dirname "$0")"
exec python cloud_swap.py
