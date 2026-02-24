#!/bin/bash
# Setup and run CBG sweep + training on InverseBench (ML7)
#
# Usage:
#   cd ~/projects/ttt/TTT
#   git pull
#   bash inversebench/setup_and_run_cbg_ml7.sh [gpu_id]
#
# This script:
#   1. Clones InverseBench (if needed) next to TTT
#   2. Copies CBG files into it
#   3. Sets up the Python environment
#   4. Downloads the inv-scatter model checkpoint
#   5. Runs 1-epoch CBG sweep over lr x base_channels
#   6. Picks best config by val_loss
#   7. Runs full training with best config

set -e

# --- GPU selection ---
GPU=${1:-auto}
if [ "$GPU" = "auto" ]; then
    GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
    echo "Auto-selected GPU $GPU (least memory used)"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TTT_DIR="$(dirname "$SCRIPT_DIR")"
IB_DIR="$TTT_DIR/../InverseBench"

echo "=== CBG on InverseBench (ML7 Setup) ==="
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

# --- 2. Copy CBG files ---
echo ">>> Copying CBG files into InverseBench..."
# Core classifier (needed by both training and inference)
cp "$TTT_DIR/classifier.py"                           "$IB_DIR/classifier.py"
# Algo plugin
cp "$SCRIPT_DIR/algo/ttt_cbg.py"                      "$IB_DIR/algo/ttt_cbg.py"
# Training script
cp "$SCRIPT_DIR/train_cbg.py"                          "$IB_DIR/train_cbg.py"
# Configs
mkdir -p "$IB_DIR/configs/cbg" "$IB_DIR/configs/algorithm"
cp "$SCRIPT_DIR/configs/cbg/default.yaml"              "$IB_DIR/configs/cbg/default.yaml"
cp "$SCRIPT_DIR/configs/algorithm/ttt_cbg.yaml"        "$IB_DIR/configs/algorithm/ttt_cbg.yaml"
echo "  Copied: classifier.py, algo/ttt_cbg.py, train_cbg.py, configs"

# --- 3. Setup Python environment ---
cd "$IB_DIR"
if ! command -v uv &>/dev/null; then
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
if [ ! -d ".venv" ]; then
    echo ">>> Creating venv..."
    uv venv .venv --python python3
fi
source .venv/bin/activate
echo ">>> Installing dependencies..."
uv pip install torch torchvision hydra-core omegaconf lmdb numpy tqdm pyyaml matplotlib piq requests scipy sigpy h5py accelerate wandb

# --- 3b. Symlink scattering matrix cache (avoids 40+ min recomputation) ---
CACHE_SRC="/home/wenda/InverseBench/cache/inv-scatter_numT_20_numR_360"
CACHE_DST="cache/inv-scatter_numT_20_numR_360"
if [ -d "$CACHE_SRC" ] && [ ! -e "$CACHE_DST" ]; then
    mkdir -p cache
    ln -sf "$CACHE_SRC" "$CACHE_DST"
    echo "  Symlinked scattering cache from $CACHE_SRC"
fi

# --- 4. Download inv-scatter model checkpoint ---
mkdir -p checkpoints
if [ ! -f "checkpoints/inv-scatter-5m.pt" ]; then
    echo ">>> Downloading inv-scatter model checkpoint..."
    wget -q --show-progress -O checkpoints/inv-scatter-5m.pt \
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
    exit 1
fi

# --- 6. Verify GPU ---
echo ">>> Checking GPU..."
CUDA_VISIBLE_DEVICES=$GPU python -c "import torch; print(f'GPU $GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')"

# --- 7. Sweep: lr x base_channels, 1 epoch ---
echo ""
echo "=== Phase 1: CBG Sweep (1 epoch each) ==="
LR_LIST="1e-3 3e-3 1e-2"
BC_LIST="32 64"
TRAIN_PCT=25

for lr in $LR_LIST; do
    for bc in $BC_LIST; do
        echo ""
        echo ">>> Sweep config: lr=$lr, base_channels=$bc, train_pct=$TRAIN_PCT"
        CUDA_VISIBLE_DEVICES=$GPU python train_cbg.py \
            problem=inv-scatter pretrain=inv-scatter \
            +cbg.lr=$lr \
            +cbg.base_channels=$bc \
            +cbg.train_pct=$TRAIN_PCT \
            +cbg.num_epochs=1 \
            +cbg.batch_size=8 \
            +cbg.val_fraction=0.1 \
            +cbg.save_dir=exps/cbg_sweep \
            || echo "  FAILED: lr=$lr, bc=$bc"
    done
done

# --- 8. Find best config ---
echo ""
echo "=== Finding best sweep config ==="
read BEST_LR BEST_BC BEST_LOSS BEST_DIR <<< $(python -c "
import json, glob, os
best_loss, best_lr, best_bc, best_dir = float('inf'), '', '', ''
for p in sorted(glob.glob('exps/cbg_sweep/*/progress.json')):
    try:
        state = json.load(open(p))
        vl = state.get('best_val_loss') or state.get('val_loss')
        if vl is None: continue
        d = os.path.dirname(p)
        # parse lr and bc from dir name: ..._lr{lr}_ch{bc}
        name = os.path.basename(d)
        lr_str = name.split('_lr')[1].split('_ch')[0]
        bc_str = name.split('_ch')[1] if '_ch' in name else '64'
        if vl < best_loss:
            best_loss, best_lr, best_bc, best_dir = vl, lr_str, bc_str, d
    except Exception as e:
        print(f'  skip {p}: {e}')
print(f'{best_lr} {best_bc} {best_loss} {best_dir}')
")

echo "  Best: lr=$BEST_LR, base_channels=$BEST_BC, val_loss=$BEST_LOSS"
echo "  Dir:  $BEST_DIR"

if [ -z "$BEST_LR" ] || [ "$BEST_LR" = "inf" ]; then
    echo "ERROR: No valid sweep results found!"
    exit 1
fi

# --- 9. Full training with best config ---
echo ""
echo "=== Phase 2: Full training (50 epochs) ==="
echo "  lr=$BEST_LR, base_channels=$BEST_BC, train_pct=$TRAIN_PCT"
echo ""

CUDA_VISIBLE_DEVICES=$GPU python train_cbg.py \
    problem=inv-scatter pretrain=inv-scatter \
    +cbg.lr=$BEST_LR \
    +cbg.base_channels=$BEST_BC \
    +cbg.train_pct=$TRAIN_PCT \
    +cbg.num_epochs=50 \
    +cbg.batch_size=8 \
    +cbg.val_fraction=0.1 \
    +cbg.save_dir=exps/cbg_full

echo ""
echo "=== Done! ==="
echo "  Sweep results:  exps/cbg_sweep/"
echo "  Full training:  exps/cbg_full/"
echo "  Best checkpoint: exps/cbg_full/*/classifier_best.pt"
