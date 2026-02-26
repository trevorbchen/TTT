#!/bin/bash
# Large CBG model: bc=64, 2 ResBlocks, 64 cross-attention tokens
#
# Pipeline:
#   Phase 1: HP sweep (1 epoch/pass, 10% data)
#   Phase 2: Full training (best HP, 80% data)
#   Phase 3: Guidance scale tuning + eval vs DPS
#
# Usage:
#   cd ~/projects/ttt/TTT && git pull
#   nohup bash inversebench/run_large.sh 0 > large.log 2>&1 &

set -e

GPU=${1:-0}
TARGET_MODE="tweedie"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TTT_DIR="$(dirname "$SCRIPT_DIR")"
IB_DIR="$TTT_DIR/../InverseBench"

echo "=== Large CBG Model (GPU $GPU) ==="
echo "  bc=64, num_res_blocks=2, num_tokens=64"
echo "  TTT repo:        $TTT_DIR"
echo "  InverseBench dir: $IB_DIR"
echo "  Started:         $(date)"
echo ""

# --- 1. Setup ---
if [ ! -d "$IB_DIR" ]; then
    git clone https://github.com/devzhk/InverseBench.git "$IB_DIR"
else
    cd "$IB_DIR" && git pull || true
fi

echo ">>> Copying files..."
cp "$TTT_DIR/classifier.py"              "$IB_DIR/classifier.py"
cp "$SCRIPT_DIR/train_cbg.py"            "$IB_DIR/train_cbg.py"
cp "$SCRIPT_DIR/tune_guidance.py"        "$IB_DIR/tune_guidance.py"
mkdir -p "$IB_DIR/configs/cbg"
cp "$SCRIPT_DIR/configs/cbg/default.yaml" "$IB_DIR/configs/cbg/default.yaml"
for f in eval_cbg.py; do
    [ -f "$SCRIPT_DIR/$f" ] && cp "$SCRIPT_DIR/$f" "$IB_DIR/$f"
done
for f in ttt_cbg.py dps.py; do
    [ -f "$SCRIPT_DIR/algo/$f" ] && cp "$SCRIPT_DIR/algo/$f" "$IB_DIR/algo/$f"
done

cd "$IB_DIR"
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
[ ! -d ".venv" ] && uv venv .venv --python python3
source .venv/bin/activate
uv pip install torch torchvision hydra-core omegaconf lmdb numpy tqdm pyyaml matplotlib piq requests scipy sigpy h5py accelerate wandb

CACHE_SRC="/home/wenda/InverseBench/cache/inv-scatter_numT_20_numR_360"
CACHE_DST="cache/inv-scatter_numT_20_numR_360"
[ -d "$CACHE_SRC" ] && [ ! -e "$CACHE_DST" ] && mkdir -p cache && ln -sf "$CACHE_SRC" "$CACHE_DST"

mkdir -p checkpoints
[ ! -f "checkpoints/inv-scatter-5m.pt" ] && wget -q --show-progress -O checkpoints/inv-scatter-5m.pt \
    https://github.com/devzhk/InverseBench/releases/download/diffusion-prior/inv-scatter-5m.pt

[ ! -d "../data/inv-scatter-train" ] && echo "ERROR: No training data" && exit 1

echo ">>> Checking GPU..."
CUDA_VISIBLE_DEVICES=$GPU python -c "import torch; print(f'GPU $GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')"

# =========================================================================
# Phase 1: HP Sweep
# =========================================================================
echo ""
echo "=========================================="
echo "Phase 1: HP Sweep (large model)"
echo "  lr x [1e-4, 3e-4, 1e-3]"
echo "  mode x [epoch, sequential]"
echo "  bc=64, num_res_blocks=2, num_tokens=64"
echo "=========================================="

SWEEP_DIR="exps/cbg_large_sweep"
TRAIN_PCT=10

# Sweep: 3 LRs x 2 modes = 6 configs
for lr in 1e-4 3e-4 1e-3; do
    # Epoch-based (1 epoch)
    echo ""
    echo ">>> Sweep: lr=$lr mode=epoch"
    CUDA_VISIBLE_DEVICES=$GPU python train_cbg.py \
        problem=inv-scatter pretrain=inv-scatter \
        +cbg.target_mode=$TARGET_MODE \
        +cbg.lr=$lr \
        +cbg.base_channels=64 \
        +cbg.num_res_blocks=2 \
        +cbg.num_tokens=64 \
        +cbg.decoder_hidden=2048 \
        +cbg.train_pct=$TRAIN_PCT \
        +cbg.num_epochs=1 \
        +cbg.batch_size=8 \
        +cbg.val_fraction=0.1 \
        +cbg.sequential_sigma=false \
        +cbg.grad_clip=10.0 \
        +cbg.save_dir=$SWEEP_DIR \
        || echo "  FAILED: lr=$lr mode=epoch"

    # Sequential sigma (1 pass)
    echo ""
    echo ">>> Sweep: lr=$lr mode=sequential"
    CUDA_VISIBLE_DEVICES=$GPU python train_cbg.py \
        problem=inv-scatter pretrain=inv-scatter \
        +cbg.target_mode=$TARGET_MODE \
        +cbg.lr=$lr \
        +cbg.base_channels=64 \
        +cbg.num_res_blocks=2 \
        +cbg.num_tokens=64 \
        +cbg.decoder_hidden=2048 \
        +cbg.train_pct=$TRAIN_PCT \
        +cbg.batch_size=8 \
        +cbg.val_fraction=0.1 \
        +cbg.sequential_sigma=true \
        +cbg.num_sigma_steps=200 \
        +cbg.sigma_batch_size=8 \
        +cbg.num_passes=1 \
        +cbg.val_every_steps=200 \
        +cbg.save_every_steps=5000 \
        +cbg.grad_clip=10.0 \
        +cbg.save_dir=$SWEEP_DIR \
        || echo "  FAILED: lr=$lr mode=sequential"
done

# =========================================================================
# Phase 2: Find best HP, full training
# =========================================================================
echo ""
echo "=========================================="
echo "Finding best sweep config..."
echo "=========================================="

read BEST_LR BEST_MODE BEST_LOSS BEST_DIR <<< $(python -c "
import json, glob, os
best_loss, best_lr, best_mode, best_dir = float('inf'), '', '', ''
for p in sorted(glob.glob('$SWEEP_DIR/*/progress.json')):
    try:
        state = json.load(open(p))
        vl = state.get('best_val_loss') or state.get('val_loss')
        if vl is None: continue
        d = os.path.dirname(p)
        name = os.path.basename(d)
        lr_str = name.split('_lr')[1].split('_ch')[0]
        mode = 'seq' if '_seq_' in name else 'cbg'
        if vl < best_loss:
            best_loss, best_lr, best_mode, best_dir = vl, lr_str, mode, d
    except Exception as e:
        print(f'  skip {p}: {e}', flush=True)
print(f'{best_lr} {best_mode} {best_loss} {best_dir}')
")

echo "  Best: lr=$BEST_LR, mode=$BEST_MODE, val_loss=$BEST_LOSS"
echo "  Dir:  $BEST_DIR"

if [ -z "$BEST_LR" ] || [ "$BEST_LR" = "inf" ]; then
    echo "ERROR: No valid sweep results found!"
    exit 1
fi

SEQ_FLAG="false"
[ "$BEST_MODE" = "seq" ] && SEQ_FLAG="true"

echo ""
echo "=========================================="
echo "Phase 2: Full training (80% data)"
echo "  lr=$BEST_LR, mode=$BEST_MODE"
echo "  bc=64, num_res_blocks=2, num_tokens=64"
echo "=========================================="

FULL_DIR="exps/cbg_large_full"

if [ "$SEQ_FLAG" = "true" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python train_cbg.py \
        problem=inv-scatter pretrain=inv-scatter \
        +cbg.target_mode=$TARGET_MODE \
        +cbg.lr=$BEST_LR \
        +cbg.base_channels=64 \
        +cbg.num_res_blocks=2 \
        +cbg.num_tokens=64 \
        +cbg.decoder_hidden=2048 \
        +cbg.train_pct=80 \
        +cbg.batch_size=8 \
        +cbg.val_fraction=0.1 \
        +cbg.sequential_sigma=true \
        +cbg.num_sigma_steps=200 \
        +cbg.sigma_batch_size=8 \
        +cbg.num_passes=1 \
        +cbg.val_every_steps=500 \
        +cbg.save_every_steps=5000 \
        +cbg.grad_clip=10.0 \
        +cbg.save_dir=$FULL_DIR
else
    CUDA_VISIBLE_DEVICES=$GPU python train_cbg.py \
        problem=inv-scatter pretrain=inv-scatter \
        +cbg.target_mode=$TARGET_MODE \
        +cbg.lr=$BEST_LR \
        +cbg.base_channels=64 \
        +cbg.num_res_blocks=2 \
        +cbg.num_tokens=64 \
        +cbg.decoder_hidden=2048 \
        +cbg.train_pct=80 \
        +cbg.num_epochs=50 \
        +cbg.batch_size=8 \
        +cbg.val_fraction=0.1 \
        +cbg.sequential_sigma=false \
        +cbg.grad_clip=10.0 \
        +cbg.save_dir=$FULL_DIR
fi

# =========================================================================
# Phase 3: Guidance tuning + eval
# =========================================================================
echo ""
echo "=========================================="
echo "Phase 3: Guidance tuning + eval"
echo "=========================================="

CLASSIFIER_PATH=$(find $FULL_DIR -name "classifier_best.pt" | head -1)
if [ -z "$CLASSIFIER_PATH" ]; then
    echo "ERROR: No classifier_best.pt found in $FULL_DIR"
    exit 1
fi
echo "  Using classifier: $CLASSIFIER_PATH"

EVAL_DIR="exps/eval_large"

CUDA_VISIBLE_DEVICES=$GPU python tune_guidance.py \
    problem=inv-scatter pretrain=inv-scatter \
    +eval.classifier_path=$CLASSIFIER_PATH \
    +eval.num_test=50 \
    +eval.num_steps=200 \
    +eval.run_dps=True \
    "+eval.scales=\"0.5,1.0,1.5,2.0,3.0,5.0\"" \
    +eval.out_dir=$EVAL_DIR

echo ""
echo "=========================================="
echo "=== Large CBG pipeline complete! ==="
echo "=========================================="
echo "  Sweep:      $SWEEP_DIR/"
echo "  Full train: $FULL_DIR/"
echo "  Evaluation: $EVAL_DIR/"
echo "  Finished:   $(date)"
