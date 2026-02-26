#!/bin/bash
# Tweedie vs Direct target mode sweep for CBG — 2-GPU parallel.
#
# Runs tweedie on GPU_A and direct on GPU_B simultaneously, syncing
# at phase boundaries where cross-mode data is needed.
#
# Pipeline:
#   Phase 0:  Setup (copy files, venv, download checkpoint)
#   Phase 1:  HP sweep — tweedie(GPU_A) || direct(GPU_B), 3 LRs each
#   Phase 1b: Select best LR per mode (sequential, fast)
#   Phase 2:  Full training — tweedie(GPU_A) || direct(GPU_B)
#   Phase 3:  Guidance tuning — tweedie(GPU_A) || direct(GPU_B)
#   Phase 4:  Final eval — tweedie+DPS+Plain(GPU_A) || direct CBG-only(GPU_B)
#   Phase 5:  Summary table
#
# Usage:
#   cd ~/projects/ttt/TTT && git pull
#   nohup bash inversebench/run_target_mode_sweep.sh 0 1 > target_sweep.log 2>&1 &

set -e

GPU_A=${1:-0}
GPU_B=${2:-1}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TTT_DIR="$(dirname "$SCRIPT_DIR")"
IB_DIR="$TTT_DIR/../InverseBench"

LOG_DIR="$IB_DIR/logs/target_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== Tweedie vs Direct Target Mode Sweep ==="
echo "  GPU_A (tweedie): $GPU_A"
echo "  GPU_B (direct):  $GPU_B"
echo "  TTT repo:        $TTT_DIR"
echo "  InverseBench dir: $IB_DIR"
echo "  Logs:            $LOG_DIR"
echo "  Started:         $(date)"
echo ""

# =========================================================================
# Phase 0: Setup
# =========================================================================
echo "=========================================="
echo "Phase 0: Setup"
echo "=========================================="

if [ ! -d "$IB_DIR" ]; then
    git clone https://github.com/devzhk/InverseBench.git "$IB_DIR"
else
    cd "$IB_DIR" && git pull || true
fi

echo ">>> Copying files..."
cp "$TTT_DIR/classifier.py"              "$IB_DIR/classifier.py"
cp "$SCRIPT_DIR/train_cbg.py"            "$IB_DIR/train_cbg.py"
cp "$SCRIPT_DIR/tune_guidance.py"        "$IB_DIR/tune_guidance.py"
cp "$SCRIPT_DIR/eval_cbg.py"             "$IB_DIR/eval_cbg.py"
mkdir -p "$IB_DIR/configs/cbg"
cp "$SCRIPT_DIR/configs/cbg/default.yaml" "$IB_DIR/configs/cbg/default.yaml"
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

echo ">>> Checking GPUs..."
CUDA_VISIBLE_DEVICES=$GPU_A python -c "import torch; print(f'GPU_A ($GPU_A): {torch.cuda.get_device_name(0)}')"
CUDA_VISIBLE_DEVICES=$GPU_B python -c "import torch; print(f'GPU_B ($GPU_B): {torch.cuda.get_device_name(0)}')"

# =========================================================================
# Phase 1: HP Sweep — parallel across GPUs
# =========================================================================
echo ""
echo "=========================================="
echo "Phase 1: HP Sweep (parallel)"
echo "  tweedie x 3 LRs on GPU $GPU_A"
echo "  direct  x 3 LRs on GPU $GPU_B"
echo "  10% data, 10 epochs, warmup=2"
echo "=========================================="

SWEEP_DIR="exps/cbg_target_sweep"
TRAIN_PCT=10

# --- Tweedie sweep on GPU_A (background) ---
(
    for lr in 1e-4 3e-4 1e-3; do
        echo "[GPU_A] Sweep: tweedie lr=$lr"
        CUDA_VISIBLE_DEVICES=$GPU_A python train_cbg.py \
            problem=inv-scatter pretrain=inv-scatter \
            +cbg.target_mode=tweedie \
            +cbg.lr=$lr \
            +cbg.base_channels=64 \
            +cbg.num_res_blocks=2 \
            +cbg.num_tokens=64 \
            +cbg.decoder_hidden=2048 \
            +cbg.train_pct=$TRAIN_PCT \
            +cbg.num_epochs=10 \
            +cbg.warmup_epochs=2 \
            +cbg.batch_size=8 \
            +cbg.val_fraction=0.1 \
            +cbg.sequential_sigma=false \
            +cbg.grad_clip=10.0 \
            +cbg.save_dir=$SWEEP_DIR \
            || echo "  FAILED: tweedie lr=$lr"
    done
) > "$LOG_DIR/sweep_tweedie.log" 2>&1 &
PID_TWEEDIE_SWEEP=$!

# --- Direct sweep on GPU_B (background) ---
(
    for lr in 1e-4 3e-4 1e-3; do
        echo "[GPU_B] Sweep: direct lr=$lr"
        CUDA_VISIBLE_DEVICES=$GPU_B python train_cbg.py \
            problem=inv-scatter pretrain=inv-scatter \
            +cbg.target_mode=direct \
            +cbg.lr=$lr \
            +cbg.base_channels=64 \
            +cbg.num_res_blocks=2 \
            +cbg.num_tokens=64 \
            +cbg.decoder_hidden=2048 \
            +cbg.train_pct=$TRAIN_PCT \
            +cbg.num_epochs=10 \
            +cbg.warmup_epochs=2 \
            +cbg.batch_size=8 \
            +cbg.val_fraction=0.1 \
            +cbg.sequential_sigma=false \
            +cbg.grad_clip=10.0 \
            +cbg.save_dir=$SWEEP_DIR \
            || echo "  FAILED: direct lr=$lr"
    done
) > "$LOG_DIR/sweep_direct.log" 2>&1 &
PID_DIRECT_SWEEP=$!

echo "  Tweedie sweep PID: $PID_TWEEDIE_SWEEP"
echo "  Direct  sweep PID: $PID_DIRECT_SWEEP"
echo "  Waiting for both..."

wait $PID_TWEEDIE_SWEEP
TWEEDIE_SWEEP_RC=$?
wait $PID_DIRECT_SWEEP
DIRECT_SWEEP_RC=$?

echo "  Tweedie sweep exit: $TWEEDIE_SWEEP_RC"
echo "  Direct  sweep exit: $DIRECT_SWEEP_RC"

if [ $TWEEDIE_SWEEP_RC -ne 0 ] || [ $DIRECT_SWEEP_RC -ne 0 ]; then
    echo "WARNING: One or more sweep jobs failed. Check logs:"
    echo "  $LOG_DIR/sweep_tweedie.log"
    echo "  $LOG_DIR/sweep_direct.log"
fi

# =========================================================================
# Phase 1b: Select best LR per target mode
# =========================================================================
echo ""
echo "=========================================="
echo "Phase 1b: Selecting best LR per mode"
echo "=========================================="

read BEST_LR_TWEEDIE BEST_LOSS_TWEEDIE BEST_DIR_TWEEDIE <<< $(python -c "
import json, glob, os
best_loss, best_lr, best_dir = float('inf'), '', ''
for p in sorted(glob.glob('$SWEEP_DIR/*/progress.json')):
    try:
        d = os.path.dirname(p)
        name = os.path.basename(d)
        if '_tweedie_' not in name:
            continue
        state = json.load(open(p))
        vl = state.get('best_val_loss') or state.get('val_loss')
        if vl is None:
            continue
        lr_str = name.split('_lr')[1].split('_ch')[0]
        if vl < best_loss:
            best_loss, best_lr, best_dir = vl, lr_str, d
    except Exception as e:
        print(f'  skip {p}: {e}', flush=True)
print(f'{best_lr} {best_loss} {best_dir}')
")

read BEST_LR_DIRECT BEST_LOSS_DIRECT BEST_DIR_DIRECT <<< $(python -c "
import json, glob, os
best_loss, best_lr, best_dir = float('inf'), '', ''
for p in sorted(glob.glob('$SWEEP_DIR/*/progress.json')):
    try:
        d = os.path.dirname(p)
        name = os.path.basename(d)
        if '_direct_' not in name:
            continue
        state = json.load(open(p))
        vl = state.get('best_val_loss') or state.get('val_loss')
        if vl is None:
            continue
        lr_str = name.split('_lr')[1].split('_ch')[0]
        if vl < best_loss:
            best_loss, best_lr, best_dir = vl, lr_str, d
    except Exception as e:
        print(f'  skip {p}: {e}', flush=True)
print(f'{best_lr} {best_loss} {best_dir}')
")

echo "  Tweedie best: lr=$BEST_LR_TWEEDIE, val_loss=$BEST_LOSS_TWEEDIE"
echo "  Direct  best: lr=$BEST_LR_DIRECT,  val_loss=$BEST_LOSS_DIRECT"

if [ -z "$BEST_LR_TWEEDIE" ] || [ "$BEST_LR_TWEEDIE" = "inf" ]; then
    echo "ERROR: No valid tweedie sweep results!"
    echo "Check: $LOG_DIR/sweep_tweedie.log"
    exit 1
fi
if [ -z "$BEST_LR_DIRECT" ] || [ "$BEST_LR_DIRECT" = "inf" ]; then
    echo "ERROR: No valid direct sweep results!"
    echo "Check: $LOG_DIR/sweep_direct.log"
    exit 1
fi

# =========================================================================
# Phase 2: Full Training — parallel across GPUs
# =========================================================================
echo ""
echo "=========================================="
echo "Phase 2: Full training (parallel, 80% data, 50 epochs)"
echo "  Tweedie lr=$BEST_LR_TWEEDIE on GPU $GPU_A"
echo "  Direct  lr=$BEST_LR_DIRECT  on GPU $GPU_B"
echo "=========================================="

FULL_DIR="exps/cbg_target_full"

# --- Tweedie full train on GPU_A ---
(
    echo "[GPU_A] Full train: tweedie lr=$BEST_LR_TWEEDIE"
    CUDA_VISIBLE_DEVICES=$GPU_A python train_cbg.py \
        problem=inv-scatter pretrain=inv-scatter \
        +cbg.target_mode=tweedie \
        +cbg.lr=$BEST_LR_TWEEDIE \
        +cbg.base_channels=64 \
        +cbg.num_res_blocks=2 \
        +cbg.num_tokens=64 \
        +cbg.decoder_hidden=2048 \
        +cbg.train_pct=80 \
        +cbg.num_epochs=50 \
        +cbg.warmup_epochs=5 \
        +cbg.batch_size=8 \
        +cbg.val_fraction=0.1 \
        +cbg.sequential_sigma=false \
        +cbg.grad_clip=10.0 \
        +cbg.save_dir=$FULL_DIR
) > "$LOG_DIR/full_tweedie.log" 2>&1 &
PID_TWEEDIE_FULL=$!

# --- Direct full train on GPU_B ---
(
    echo "[GPU_B] Full train: direct lr=$BEST_LR_DIRECT"
    CUDA_VISIBLE_DEVICES=$GPU_B python train_cbg.py \
        problem=inv-scatter pretrain=inv-scatter \
        +cbg.target_mode=direct \
        +cbg.lr=$BEST_LR_DIRECT \
        +cbg.base_channels=64 \
        +cbg.num_res_blocks=2 \
        +cbg.num_tokens=64 \
        +cbg.decoder_hidden=2048 \
        +cbg.train_pct=80 \
        +cbg.num_epochs=50 \
        +cbg.warmup_epochs=5 \
        +cbg.batch_size=8 \
        +cbg.val_fraction=0.1 \
        +cbg.sequential_sigma=false \
        +cbg.grad_clip=10.0 \
        +cbg.save_dir=$FULL_DIR
) > "$LOG_DIR/full_direct.log" 2>&1 &
PID_DIRECT_FULL=$!

echo "  Tweedie full PID: $PID_TWEEDIE_FULL"
echo "  Direct  full PID: $PID_DIRECT_FULL"
echo "  Waiting for both..."

wait $PID_TWEEDIE_FULL
wait $PID_DIRECT_FULL
echo "  Full training done."

# =========================================================================
# Phase 3: Guidance scale tuning — parallel across GPUs
# =========================================================================
echo ""
echo "=========================================="
echo "Phase 3: Guidance tuning (parallel)"
echo "=========================================="

TUNE_DIR="exps/cbg_target_tune"

CLASSIFIER_TWEEDIE=$(find $FULL_DIR -path "*_tweedie_*" -name "classifier_best.pt" | head -1)
CLASSIFIER_DIRECT=$(find $FULL_DIR -path "*_direct_*" -name "classifier_best.pt" | head -1)

if [ -z "$CLASSIFIER_TWEEDIE" ]; then
    echo "ERROR: No classifier_best.pt for tweedie in $FULL_DIR"
    exit 1
fi
if [ -z "$CLASSIFIER_DIRECT" ]; then
    echo "ERROR: No classifier_best.pt for direct in $FULL_DIR"
    exit 1
fi

echo "  Tweedie classifier: $CLASSIFIER_TWEEDIE"
echo "  Direct  classifier: $CLASSIFIER_DIRECT"

# --- Tweedie tune on GPU_A ---
(
    echo "[GPU_A] Guidance tuning: tweedie"
    CUDA_VISIBLE_DEVICES=$GPU_A python tune_guidance.py \
        problem=inv-scatter pretrain=inv-scatter \
        +eval.classifier_path=$CLASSIFIER_TWEEDIE \
        +eval.num_test=20 \
        +eval.num_steps=200 \
        +eval.run_dps=False \
        "+eval.scales=\"0.5,1.0,1.5,2.0,3.0,5.0\"" \
        +eval.out_dir=$TUNE_DIR/tweedie
) > "$LOG_DIR/tune_tweedie.log" 2>&1 &
PID_TUNE_TW=$!

# --- Direct tune on GPU_B ---
(
    echo "[GPU_B] Guidance tuning: direct"
    CUDA_VISIBLE_DEVICES=$GPU_B python tune_guidance.py \
        problem=inv-scatter pretrain=inv-scatter \
        +eval.classifier_path=$CLASSIFIER_DIRECT \
        +eval.num_test=20 \
        +eval.num_steps=200 \
        +eval.run_dps=False \
        "+eval.scales=\"0.5,1.0,1.5,2.0,3.0,5.0\"" \
        +eval.out_dir=$TUNE_DIR/direct
) > "$LOG_DIR/tune_direct.log" 2>&1 &
PID_TUNE_DR=$!

echo "  Tweedie tune PID: $PID_TUNE_TW"
echo "  Direct  tune PID: $PID_TUNE_DR"
echo "  Waiting for both..."

wait $PID_TUNE_TW
wait $PID_TUNE_DR
echo "  Guidance tuning done."

# Extract best scales
BEST_SCALE_TWEEDIE=$(python -c "
import json
d = json.load(open('$TUNE_DIR/tweedie/guidance_sweep.json'))
print(d['best_scale'])
")
BEST_SCALE_DIRECT=$(python -c "
import json
d = json.load(open('$TUNE_DIR/direct/guidance_sweep.json'))
print(d['best_scale'])
")

echo "  Tweedie best scale: $BEST_SCALE_TWEEDIE"
echo "  Direct  best scale: $BEST_SCALE_DIRECT"

# =========================================================================
# Phase 4: Final Eval — parallel across GPUs
# =========================================================================
echo ""
echo "=========================================="
echo "Phase 4: Final evaluation (parallel, 100 test samples)"
echo "  Tweedie (CBG+DPS+Plain) on GPU $GPU_A"
echo "  Direct  (CBG only)      on GPU $GPU_B"
echo "=========================================="

EVAL_DIR="exps/cbg_target_eval"

# --- Tweedie eval on GPU_A: CBG + DPS + Plain, save images ---
(
    echo "[GPU_A] Eval: tweedie (CBG + DPS + Plain)"
    CUDA_VISIBLE_DEVICES=$GPU_A python eval_cbg.py \
        problem=inv-scatter pretrain=inv-scatter \
        +eval.classifier_path=$CLASSIFIER_TWEEDIE \
        +eval.guidance_scale=$BEST_SCALE_TWEEDIE \
        +eval.num_steps=200 \
        +eval.num_test=100 \
        +eval.run_dps=True \
        +eval.dps_guidance_scale=1.0 \
        +eval.run_plain=True \
        +eval.save_images=True \
        +eval.num_vis=8 \
        +eval.out_dir=$EVAL_DIR/tweedie
) > "$LOG_DIR/eval_tweedie.log" 2>&1 &
PID_EVAL_TW=$!

# --- Direct eval on GPU_B: CBG only, save images ---
(
    echo "[GPU_B] Eval: direct (CBG only)"
    CUDA_VISIBLE_DEVICES=$GPU_B python eval_cbg.py \
        problem=inv-scatter pretrain=inv-scatter \
        +eval.classifier_path=$CLASSIFIER_DIRECT \
        +eval.guidance_scale=$BEST_SCALE_DIRECT \
        +eval.num_steps=200 \
        +eval.num_test=100 \
        +eval.run_dps=False \
        +eval.run_plain=False \
        +eval.save_images=True \
        +eval.num_vis=8 \
        +eval.out_dir=$EVAL_DIR/direct
) > "$LOG_DIR/eval_direct.log" 2>&1 &
PID_EVAL_DR=$!

echo "  Tweedie eval PID: $PID_EVAL_TW"
echo "  Direct  eval PID: $PID_EVAL_DR"
echo "  Waiting for both..."

wait $PID_EVAL_TW
wait $PID_EVAL_DR
echo "  Evaluation done."

# =========================================================================
# Phase 5: Summary table
# =========================================================================
echo ""
echo "=========================================="
echo "Phase 5: Combined Summary"
echo "=========================================="

python -c "
import json
import numpy as np
from scipy import stats

def ci95(vals):
    arr = np.array(vals)
    n = len(arr)
    m = arr.mean()
    s = arr.std(ddof=1)
    if n < 2:
        return m, 0.0, s
    hw = stats.t.ppf(0.975, df=n-1) * s / np.sqrt(n)
    return m, hw, s

rows = []

# Tweedie CBG
tw = json.load(open('$EVAL_DIR/tweedie/eval_results.json'))
m, hw, s = ci95(tw['cbg']['per_sample'])
rows.append(('CBG-tweedie', m, hw, s, tw['cbg']['time_per_sample_sec']))

# Direct CBG
dr = json.load(open('$EVAL_DIR/direct/eval_results.json'))
m, hw, s = ci95(dr['cbg']['per_sample'])
rows.append(('CBG-direct', m, hw, s, dr['cbg']['time_per_sample_sec']))

# DPS (from tweedie eval)
if 'dps' in tw:
    m, hw, s = ci95(tw['dps']['per_sample'])
    rows.append(('DPS', m, hw, s, tw['dps']['time_per_sample_sec']))

# Plain (from tweedie eval)
if 'plain' in tw:
    m, hw, s = ci95(tw['plain']['per_sample'])
    rows.append(('Plain', m, hw, s, tw['plain']['time_per_sample_sec']))

print()
print(f\"{'Method':<16} {'Mean L2':>12} {'95% CI':>20} {'Time/sample':>14}\")
print(f\"{'-'*64}\")
for name, mean, ci, std, tps in rows:
    ci_str = f'[{mean-ci:.4f}, {mean+ci:.4f}]'
    print(f'{name:<16} {mean:>12.6f} {ci_str:>20} {tps:>12.2f}s')
print()

# Highlight best CBG
cbg_rows = [r for r in rows if r[0].startswith('CBG')]
best = min(cbg_rows, key=lambda r: r[1])
print(f'Best CBG method: {best[0]} (mean L2={best[1]:.6f})')
"

echo ""
echo "=========================================="
echo "=== Target mode sweep complete! ==="
echo "=========================================="
echo "  Sweep:      $SWEEP_DIR/"
echo "  Full train: $FULL_DIR/"
echo "  Tuning:     $TUNE_DIR/"
echo "  Evaluation: $EVAL_DIR/"
echo "  Logs:       $LOG_DIR/"
echo "  Finished:   $(date)"
