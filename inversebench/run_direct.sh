#!/bin/bash
# Two-Target CBG Experiment: Direct target
# target = A(x_t) - y  (apply operator to noisy x_t, no denoiser call)
#
# Pipeline:
#   Phase 1: HP sweep (1 epoch, 10% data)
#   Phase 2: Full training (best HP, 80% data, 50 epochs)
#   Phase 3: Evaluation (CBG vs DPS, L2 with 95% CIs)
#
# Usage:
#   cd ~/projects/ttt/TTT && git pull
#   nohup bash inversebench/run_direct.sh 1 > direct.log 2>&1 &

set -e

GPU=${1:-1}
TARGET_MODE="direct"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TTT_DIR="$(dirname "$SCRIPT_DIR")"
IB_DIR="$TTT_DIR/../InverseBench"

echo "=== Two-Target CBG: $TARGET_MODE (GPU $GPU) ==="
echo "  TTT repo:        $TTT_DIR"
echo "  InverseBench dir: $IB_DIR"
echo "  Started:         $(date)"
echo ""

# --- 1. Clone InverseBench if needed ---
if [ ! -d "$IB_DIR" ]; then
    echo ">>> Cloning InverseBench..."
    git clone https://github.com/devzhk/InverseBench.git "$IB_DIR"
else
    echo ">>> InverseBench already exists, pulling latest..."
    cd "$IB_DIR" && git pull || true
fi

# --- 2. Copy files ---
echo ">>> Copying files into InverseBench..."
cp "$TTT_DIR/classifier.py"                           "$IB_DIR/classifier.py"
cp "$SCRIPT_DIR/algo/ttt_cbg.py"                      "$IB_DIR/algo/ttt_cbg.py"
cp "$SCRIPT_DIR/algo/dps.py"                           "$IB_DIR/algo/dps.py"
cp "$SCRIPT_DIR/train_cbg.py"                          "$IB_DIR/train_cbg.py"
cp "$SCRIPT_DIR/eval_cbg.py"                           "$IB_DIR/eval_cbg.py"
mkdir -p "$IB_DIR/configs/cbg" "$IB_DIR/configs/algorithm"
cp "$SCRIPT_DIR/configs/cbg/default.yaml"              "$IB_DIR/configs/cbg/default.yaml"
cp "$SCRIPT_DIR/configs/algorithm/ttt_cbg.yaml"        "$IB_DIR/configs/algorithm/ttt_cbg.yaml"
cp "$SCRIPT_DIR/configs/algorithm/dps.yaml"            "$IB_DIR/configs/algorithm/dps.yaml"
echo "  Copied: classifier.py, algo/{ttt_cbg,dps}.py, {train,eval}_cbg.py, configs"

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

# --- 3b. Symlink scattering matrix cache ---
CACHE_SRC="/home/wenda/InverseBench/cache/inv-scatter_numT_20_numR_360"
CACHE_DST="cache/inv-scatter_numT_20_numR_360"
if [ -d "$CACHE_SRC" ] && [ ! -e "$CACHE_DST" ]; then
    mkdir -p cache
    ln -sf "$CACHE_SRC" "$CACHE_DST"
    echo "  Symlinked scattering cache from $CACHE_SRC"
fi

# --- 4. Download checkpoint ---
mkdir -p checkpoints
if [ ! -f "checkpoints/inv-scatter-5m.pt" ]; then
    echo ">>> Downloading inv-scatter model checkpoint..."
    wget -q --show-progress -O checkpoints/inv-scatter-5m.pt \
        https://github.com/devzhk/InverseBench/releases/download/diffusion-prior/inv-scatter-5m.pt
else
    echo ">>> Model checkpoint already exists"
fi

# --- 5. Check data ---
if [ ! -d "../data/inv-scatter-train" ]; then
    echo "ERROR: Training data not found at ../data/inv-scatter-train/"
    exit 1
fi

# --- 6. Verify GPU ---
echo ">>> Checking GPU..."
CUDA_VISIBLE_DEVICES=$GPU python -c "import torch; print(f'GPU $GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')"

# =========================================================================
# Phase 1: HP Sweep (1 epoch, 10% data)
# =========================================================================
echo ""
echo "=========================================="
echo "Phase 1: $TARGET_MODE Sweep (1 epoch each)"
echo "=========================================="

LR_LIST="1e-3 3e-3 1e-2"
BC_LIST="32 64"
SWEEP_DIR="exps/cbg_${TARGET_MODE}_sweep"
TRAIN_PCT=10

for lr in $LR_LIST; do
    for bc in $BC_LIST; do
        echo ""
        echo ">>> Sweep: target=$TARGET_MODE lr=$lr bc=$bc pct=$TRAIN_PCT"
        CUDA_VISIBLE_DEVICES=$GPU python train_cbg.py \
            problem=inv-scatter pretrain=inv-scatter \
            +cbg.target_mode=$TARGET_MODE \
            +cbg.lr=$lr \
            +cbg.base_channels=$bc \
            +cbg.train_pct=$TRAIN_PCT \
            +cbg.num_epochs=1 \
            +cbg.batch_size=8 \
            +cbg.val_fraction=0.1 \
            +cbg.save_dir=$SWEEP_DIR \
            || echo "  FAILED: lr=$lr, bc=$bc"
    done
done

# =========================================================================
# Phase 2: Find best HP, full training (80% data, 50 epochs)
# =========================================================================
echo ""
echo "=========================================="
echo "Finding best sweep config..."
echo "=========================================="

read BEST_LR BEST_BC BEST_LOSS BEST_DIR <<< $(python -c "
import json, glob, os
best_loss, best_lr, best_bc, best_dir = float('inf'), '', '', ''
for p in sorted(glob.glob('$SWEEP_DIR/*/progress.json')):
    try:
        state = json.load(open(p))
        vl = state.get('best_val_loss') or state.get('val_loss')
        if vl is None: continue
        d = os.path.dirname(p)
        name = os.path.basename(d)
        # parse lr and bc from dir name: ..._lr{lr}_ch{bc}
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

echo ""
echo "=========================================="
echo "Phase 2: Full $TARGET_MODE training (50 epochs, 80% data)"
echo "  lr=$BEST_LR, base_channels=$BEST_BC"
echo "=========================================="

FULL_DIR="exps/cbg_${TARGET_MODE}_full"

CUDA_VISIBLE_DEVICES=$GPU python train_cbg.py \
    problem=inv-scatter pretrain=inv-scatter \
    +cbg.target_mode=$TARGET_MODE \
    +cbg.lr=$BEST_LR \
    +cbg.base_channels=$BEST_BC \
    +cbg.train_pct=80 \
    +cbg.num_epochs=50 \
    +cbg.batch_size=8 \
    +cbg.val_fraction=0.1 \
    +cbg.save_dir=$FULL_DIR

# =========================================================================
# Phase 3: Evaluation (CBG vs DPS)
# =========================================================================
echo ""
echo "=========================================="
echo "Phase 3: Evaluation (CBG vs DPS)"
echo "=========================================="

# Find the best classifier from full training
CLASSIFIER_PATH=$(find $FULL_DIR -name "classifier_best.pt" | head -1)
if [ -z "$CLASSIFIER_PATH" ]; then
    echo "ERROR: No classifier_best.pt found in $FULL_DIR"
    exit 1
fi
echo "  Using classifier: $CLASSIFIER_PATH"

EVAL_DIR="exps/eval_${TARGET_MODE}"

CUDA_VISIBLE_DEVICES=$GPU python eval_cbg.py \
    problem=inv-scatter pretrain=inv-scatter \
    +eval.classifier_path=$CLASSIFIER_PATH \
    +eval.guidance_scale=1.0 \
    +eval.num_steps=200 \
    +eval.num_test=100 \
    +eval.run_dps=True \
    +eval.dps_guidance_scale=1.0 \
    +eval.out_dir=$EVAL_DIR

echo ""
echo "=========================================="
echo "=== $TARGET_MODE pipeline complete! ==="
echo "=========================================="
echo "  Sweep:      $SWEEP_DIR/"
echo "  Full train: $FULL_DIR/"
echo "  Evaluation: $EVAL_DIR/"
echo "  Finished:   $(date)"
