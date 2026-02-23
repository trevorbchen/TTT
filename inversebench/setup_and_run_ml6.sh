#!/bin/bash
# Setup and run TTT-LoRA on InverseBench (ML6 cluster)
#
# Usage:
#   cd ~/projects/ttt/TTT
#   git pull
#   bash inversebench/setup_and_run_ml6.sh [gpu_id]
#
# This script:
#   1. Clones InverseBench (if needed) next to TTT
#   2. Copies our TTT-LoRA files into it
#   3. Sets up the Python environment
#   4. Downloads the inv-scatter model checkpoint
#   5. Runs TTT training on inverse scattering

set -e

GPU=${1:-0}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TTT_DIR="$(dirname "$SCRIPT_DIR")"
IB_DIR="$TTT_DIR/../InverseBench"

echo "=== TTT-LoRA on InverseBench (ML6 Setup) ==="
echo "  TTT repo:        $TTT_DIR"
echo "  InverseBench dir: $IB_DIR"
echo "  GPU:              $GPU"
echo ""

# --- 1. Clone InverseBench if needed ---
if [ ! -d "$IB_DIR" ]; then
    echo ">>> Cloning InverseBench..."
    git clone https://github.com/devzhk/InverseBench.git "$IB_DIR"
else
    echo ">>> InverseBench already exists, pulling latest..."
    cd "$IB_DIR" && git pull || true
fi

# --- 2. Copy our TTT-LoRA files ---
echo ">>> Copying TTT-LoRA files into InverseBench..."
cp "$SCRIPT_DIR/algo/lora.py"                   "$IB_DIR/algo/lora.py"
cp "$SCRIPT_DIR/algo/ttt_lora.py"               "$IB_DIR/algo/ttt_lora.py"
cp "$SCRIPT_DIR/train_ttt.py"                    "$IB_DIR/train_ttt.py"
mkdir -p "$IB_DIR/configs/ttt" "$IB_DIR/configs/algorithm"
cp "$SCRIPT_DIR/configs/ttt/default.yaml"        "$IB_DIR/configs/ttt/default.yaml"
cp "$SCRIPT_DIR/configs/algorithm/ttt_lora.yaml" "$IB_DIR/configs/algorithm/ttt_lora.yaml"
echo "  Copied: algo/lora.py, algo/ttt_lora.py, train_ttt.py, configs"

# --- 3. Setup Python environment ---
cd "$IB_DIR"
if [ ! -d ".venv" ]; then
    echo ">>> Creating venv..."
    uv venv .venv --python python3
fi
source .venv/bin/activate
echo ">>> Installing dependencies..."
uv pip install -e . --extra-index-url https://download.pytorch.org/whl/cu124

# --- 4. Download inv-scatter model checkpoint ---
mkdir -p checkpoints
if [ ! -f "checkpoints/inv-scatter-5m.pt" ]; then
    echo ">>> Downloading inv-scatter model checkpoint..."
    wget -O checkpoints/inv-scatter-5m.pt \
        https://github.com/devzhk/InverseBench/releases/download/diffusion-prior/inv-scatter-5m.pt
else
    echo ">>> Model checkpoint already exists"
fi

# --- 5. Check for training data ---
if [ ! -d "../data/inv-scatter-train" ]; then
    echo ""
    echo "WARNING: Training data not found at ../data/inv-scatter-train/"
    echo "Download from: https://data.caltech.edu/records/zg89b-mpv16"
    echo "Extract inv-scatter-train/ and inv-scatter-test/ into $(realpath ../data/)/"
    echo ""
    echo "After downloading, re-run this script or run manually:"
    echo "  cd $IB_DIR"
    echo "  source .venv/bin/activate"
    echo "  CUDA_VISIBLE_DEVICES=$GPU python train_ttt.py problem=inv-scatter pretrain=inv-scatter +ttt.method=direct +ttt.train_pct=80 +ttt.num_epochs=10"
    exit 1
fi

# --- 6. Verify GPU ---
echo ">>> Checking GPU..."
python -c "import torch; print(f'GPU {$GPU}: {torch.cuda.get_device_name($GPU)}')"

# --- 7. Run TTT training ---
echo ""
echo "=== Starting TTT-LoRA training on inv-scatter ==="
echo "  Method: direct (DRaFT)"
echo "  Train %: 80"
echo "  Epochs: 10"
echo "  LR: 1e-3"
echo "  LoRA rank: 64"
echo ""

CUDA_VISIBLE_DEVICES=$GPU python train_ttt.py \
    problem=inv-scatter pretrain=inv-scatter \
    +ttt.method=direct \
    +ttt.train_pct=80 \
    +ttt.lr=1e-3 \
    +ttt.lora_rank=64 \
    +ttt.num_epochs=10

echo ""
echo "=== Done! Check results in exps/ttt/ ==="
