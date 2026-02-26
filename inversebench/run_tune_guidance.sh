#!/bin/bash
# Tune guidance scale for existing CBG classifier
# Usage: bash inversebench/run_tune_guidance.sh [GPU]

set -e
GPU=${1:-0}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TTT_DIR="$(dirname "$SCRIPT_DIR")"
IB_DIR="$TTT_DIR/../InverseBench"

echo "=== Guidance Scale Tuner (GPU $GPU) ==="

# Copy latest files
cp "$TTT_DIR/classifier.py"              "$IB_DIR/classifier.py"
cp "$SCRIPT_DIR/tune_guidance.py"        "$IB_DIR/tune_guidance.py"
cp "$SCRIPT_DIR/eval_cbg.py"             "$IB_DIR/eval_cbg.py"

cd "$IB_DIR"
source .venv/bin/activate

# Find best classifier from tweedie full training
CLASSIFIER_PATH=$(find exps/cbg_tweedie_full -name "classifier_best.pt" | head -1)
if [ -z "$CLASSIFIER_PATH" ]; then
    echo "ERROR: No classifier_best.pt found in exps/cbg_tweedie_full/"
    echo "Available classifiers:"
    find exps/ -name "classifier_best.pt" 2>/dev/null
    exit 1
fi
echo "Using classifier: $CLASSIFIER_PATH"

CUDA_VISIBLE_DEVICES=$GPU python tune_guidance.py \
    problem=inv-scatter pretrain=inv-scatter \
    +eval.classifier_path=$CLASSIFIER_PATH \
    +eval.num_test=20 \
    +eval.num_steps=200 \
    +eval.run_dps=True \
    +eval.scales="0.1,0.3,0.5,1.0,2.0,3.0,5.0,10.0,20.0,50.0" \
    +eval.out_dir=exps/tune_guidance

echo ""
echo "=== Done! ==="
echo "Results: exps/tune_guidance/guidance_sweep.json"
