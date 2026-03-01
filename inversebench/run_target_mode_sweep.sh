#!/bin/bash
# CBG HP sweep — tweedie mode, Min-SNR weighting, single GPU.
#
# Sweeps over: lr × snr_gamma × normalize_target × num_passes
# All configs use tweedie target (x̂₀ = E[x₀|x_t], residual = A(x̂₀) - y).
#
# Pipeline:
#   Phase 0:  Setup (copy files, venv, download checkpoint)
#   Phase 1:  HP sweep — 36 configs sequential
#   Phase 1b: Select best config
#   Phase 2:  Full training (80% data)
#   Phase 3:  Guidance tuning
#   Phase 4:  Final eval (CBG + DPS + Plain)
#   Phase 5:  Summary table
#
# Usage:
#   cd ~/projects/ttt/TTT && git pull
#   nohup bash inversebench/run_target_mode_sweep.sh 0 > snr_sweep.log 2>&1 &

set -e

GPU=${1:-0}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TTT_DIR="$(dirname "$SCRIPT_DIR")"
IB_DIR="$TTT_DIR/../InverseBench"

LOG_DIR="$IB_DIR/logs/snr_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== CBG HP Sweep (tweedie + Min-SNR) ==="
echo "  GPU: $GPU"
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

echo ">>> Checking GPU..."
CUDA_VISIBLE_DEVICES=$GPU python -c "import torch; print(f'GPU ($GPU): {torch.cuda.get_device_name(0)}')"

# =========================================================================
# Phase 1: HP Sweep — sequential on single GPU
# =========================================================================
echo ""
echo "=========================================="
echo "Phase 1: HP Sweep (sequential on GPU $GPU)"
echo "  lr:               1e-4, 3e-4, 1e-3"
echo "  snr_gamma:        0, 1, 5"
echo "  normalize_target: true, false"
echo "  num_passes:       1, 2"
echo "  = 36 configs, 10% data, tweedie mode"
echo "=========================================="

SWEEP_DIR="exps/cbg_snr_sweep"
TRAIN_PCT=10
CFG_IDX=0

for lr in 1e-4 3e-4 1e-3; do
  for snrg in 0 1 5; do
    for norm in true false; do
      for passes in 1 2; do
        CFG_IDX=$((CFG_IDX + 1))
        echo "[${CFG_IDX}/36] lr=$lr snrg=$snrg norm=$norm passes=$passes"
        CUDA_VISIBLE_DEVICES=$GPU python train_cbg.py \
            problem=inv-scatter pretrain=inv-scatter \
            +cbg.target_mode=tweedie \
            +cbg.lr=$lr \
            +cbg.snr_gamma=$snrg \
            +cbg.normalize_target=$norm \
            +cbg.num_passes=$passes \
            +cbg.base_channels=64 \
            +cbg.num_res_blocks=2 \
            +cbg.num_tokens=64 \
            +cbg.decoder_hidden=2048 \
            +cbg.train_pct=$TRAIN_PCT \
            +cbg.num_sigma_steps=200 \
            +cbg.sigma_batch_size=8 \
            +cbg.val_every_steps=300 \
            +cbg.save_every_steps=999999 \
            +cbg.save_final=false \
            +cbg.val_fraction=0.1 \
            +cbg.grad_clip=10.0 \
            +cbg.save_dir=$SWEEP_DIR \
            >> "$LOG_DIR/sweep.log" 2>&1 \
            || echo "  FAILED: lr=$lr snrg=$snrg norm=$norm passes=$passes"
      done
    done
  done
done

echo "  Sweep done ($CFG_IDX configs)."

# =========================================================================
# Phase 1b: Select best config
# =========================================================================
echo ""
echo "=========================================="
echo "Phase 1b: Selecting best config"
echo "=========================================="

# Print leaderboard (display only)
python -c "
import json, glob, os, yaml
results = []
for p in sorted(glob.glob('$SWEEP_DIR/*/progress.json')):
    try:
        d = os.path.dirname(p)
        state = json.load(open(p))
        vl = state.get('best_val_loss') or state.get('val_loss')
        if vl is None: continue
        cfg = yaml.safe_load(open(os.path.join(d, 'config.yaml'))).get('cbg', {})
        results.append((vl, cfg.get('lr'), cfg.get('snr_gamma',0), cfg.get('normalize_target',True), cfg.get('num_passes',1)))
    except: pass
results.sort()
for vl, lr, snrg, norm, ps in results:
    tag = ' <-- BEST' if vl == results[0][0] else ''
    print(f'    lr={lr} snrg={snrg} norm={norm} passes={ps} -> val={vl:.6f}{tag}')
"

# Extract best config (single clean line to stdout)
read BEST_LR BEST_SNRG BEST_NORM BEST_PASSES BEST_LOSS BEST_DIR <<< $(python -c "
import json, glob, os, yaml
best_loss = float('inf')
best_lr, best_snrg, best_norm, best_passes, best_dir = '', '', '', '', ''
for p in sorted(glob.glob('$SWEEP_DIR/*/progress.json')):
    try:
        d = os.path.dirname(p)
        state = json.load(open(p))
        vl = state.get('best_val_loss') or state.get('val_loss')
        if vl is None: continue
        cfg = yaml.safe_load(open(os.path.join(d, 'config.yaml'))).get('cbg', {})
        if vl < best_loss:
            best_loss = vl
            best_lr = str(cfg.get('lr', ''))
            best_snrg = str(cfg.get('snr_gamma', 0))
            best_norm = str(cfg.get('normalize_target', True)).lower()
            best_passes = str(cfg.get('num_passes', 1))
            best_dir = d
    except: pass
print(f'{best_lr} {best_snrg} {best_norm} {best_passes} {best_loss} {best_dir}')
")

echo ""
echo "  Best config:"
echo "    lr=$BEST_LR"
echo "    snr_gamma=$BEST_SNRG"
echo "    normalize_target=$BEST_NORM"
echo "    num_passes=$BEST_PASSES"
echo "    val_loss=$BEST_LOSS"
echo "    dir=$BEST_DIR"

if [ -z "$BEST_LR" ] || [ "$BEST_LR" = "inf" ]; then
    echo "ERROR: No valid sweep results!"
    echo "Check: $LOG_DIR/sweep.log"
    exit 1
fi

# =========================================================================
# Phase 2: Full Training (80% data)
# =========================================================================
echo ""
echo "=========================================="
echo "Phase 2: Full training (80% data)"
echo "  lr=$BEST_LR snr_gamma=$BEST_SNRG norm=$BEST_NORM passes=$BEST_PASSES"
echo "  on GPU $GPU"
echo "=========================================="

FULL_DIR="exps/cbg_snr_full"

CUDA_VISIBLE_DEVICES=$GPU python train_cbg.py \
    problem=inv-scatter pretrain=inv-scatter \
    +cbg.target_mode=tweedie \
    +cbg.lr=$BEST_LR \
    +cbg.snr_gamma=$BEST_SNRG \
    +cbg.normalize_target=$BEST_NORM \
    +cbg.num_passes=$BEST_PASSES \
    +cbg.base_channels=64 \
    +cbg.num_res_blocks=2 \
    +cbg.num_tokens=64 \
    +cbg.decoder_hidden=2048 \
    +cbg.train_pct=80 \
    +cbg.num_sigma_steps=200 \
    +cbg.sigma_batch_size=8 \
    +cbg.val_every_steps=1000 \
    +cbg.save_every_steps=4000 \
    +cbg.val_fraction=0.1 \
    +cbg.grad_clip=10.0 \
    +cbg.save_dir=$FULL_DIR \
    > "$LOG_DIR/full_train.log" 2>&1

echo "  Full training done."

# =========================================================================
# Phase 3: Guidance scale tuning
# =========================================================================
echo ""
echo "=========================================="
echo "Phase 3: Guidance tuning"
echo "=========================================="

TUNE_DIR="exps/cbg_snr_tune"

CLASSIFIER=$(find $FULL_DIR -name "classifier_best.pt" | head -1)
if [ -z "$CLASSIFIER" ]; then
    echo "ERROR: No classifier_best.pt found in $FULL_DIR"
    exit 1
fi
echo "  Classifier: $CLASSIFIER"

CUDA_VISIBLE_DEVICES=$GPU python tune_guidance.py \
    problem=inv-scatter pretrain=inv-scatter \
    +eval.classifier_path=$CLASSIFIER \
    +eval.num_test=20 \
    +eval.num_steps=200 \
    +eval.run_dps=False \
    "+eval.scales=\"0.5,1.0,2.0,5.0,10.0,20.0\"" \
    +eval.out_dir=$TUNE_DIR \
    > "$LOG_DIR/tune.log" 2>&1

BEST_SCALE=$(python -c "
import json
d = json.load(open('$TUNE_DIR/guidance_sweep.json'))
print(d['best_scale'])
")
echo "  Best guidance scale: $BEST_SCALE"

# =========================================================================
# Phase 4: Final Eval (CBG + DPS + Plain)
# =========================================================================
echo ""
echo "=========================================="
echo "Phase 4: Final evaluation (100 test samples)"
echo "  CBG + DPS + Plain on GPU $GPU"
echo "=========================================="

EVAL_DIR="exps/cbg_snr_eval"

CUDA_VISIBLE_DEVICES=$GPU python eval_cbg.py \
    problem=inv-scatter pretrain=inv-scatter \
    +eval.classifier_path=$CLASSIFIER \
    +eval.guidance_scale=$BEST_SCALE \
    +eval.num_steps=200 \
    +eval.num_test=100 \
    +eval.run_dps=True \
    +eval.dps_guidance_scale=1.0 \
    +eval.run_plain=True \
    +eval.save_images=True \
    +eval.num_vis=8 \
    +eval.out_dir=$EVAL_DIR \
    > "$LOG_DIR/eval.log" 2>&1

echo "  Evaluation done."

# =========================================================================
# Phase 5: Summary table
# =========================================================================
echo ""
echo "=========================================="
echo "Phase 5: Summary"
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

ev = json.load(open('$EVAL_DIR/eval_results.json'))
rows = []

# CBG
m, hw, s = ci95(ev['cbg']['per_sample'])
rows.append(('CBG', m, hw, s, ev['cbg']['time_per_sample_sec']))

# DPS
if 'dps' in ev:
    m, hw, s = ci95(ev['dps']['per_sample'])
    rows.append(('DPS', m, hw, s, ev['dps']['time_per_sample_sec']))

# Plain
if 'plain' in ev:
    m, hw, s = ci95(ev['plain']['per_sample'])
    rows.append(('Plain', m, hw, s, ev['plain']['time_per_sample_sec']))

print()
print(f'Best sweep config: lr=$BEST_LR, snr_gamma=$BEST_SNRG, normalize=$BEST_NORM, passes=$BEST_PASSES')
print(f'Guidance scale: $BEST_SCALE')
print()
print(f\"{'Method':<10} {'Mean L2':>10} {'95% CI':>20} {'Std':>8} {'Time/samp':>12}\")
print(f\"{'-'*62}\")
for name, mean, ci, std, tps in rows:
    ci_str = f'[{mean-ci:.4f}, {mean+ci:.4f}]'
    print(f'{name:<10} {mean:>10.6f} {ci_str:>20} {std:>8.4f} {tps:>10.2f}s')
print()
"

echo ""
echo "=========================================="
echo "=== SNR sweep complete! ==="
echo "=========================================="
echo "  Sweep:      $SWEEP_DIR/"
echo "  Full train: $FULL_DIR/"
echo "  Tuning:     $TUNE_DIR/"
echo "  Evaluation: $EVAL_DIR/"
echo "  Logs:       $LOG_DIR/"
echo "  Finished:   $(date)"
