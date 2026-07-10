#!/usr/bin/env bash

set -u

ROOT="/root/autodl-tmp/omniquant_final"
CODE="$ROOT/code"
MODEL="/root/autodl-tmp/llama-2-7b-hf"
CACHE="$ROOT/restore/OmniQuant/cache"

BASELINE="$ROOT/experiments/7b_w2g128_n32e10_baseline"
ROBUST="$ROOT/experiments/7b_w2g128_n32e10_robust"

source "$ROOT/env/bin/activate"

mkdir -p "$ROOT/experiments"

run_case() {
    local name="$1"
    local output_dir="$2"
    shift 2

    rm -rf "$output_dir"
    mkdir -p "$output_dir"

    {
        echo "CASE=$name"
        echo "START_TIME=$(date --iso-8601=seconds)"
        echo "GIT_COMMIT=$(git -C "$CODE" rev-parse HEAD)"
        nvidia-smi
    } > "$output_dir/environment.txt" 2>&1

    echo "[$(date)] Starting $name"

    cd "$CODE"

    python -u main.py \
        --model "$MODEL" \
        --cache_dir "$CACHE" \
        --output_dir "$output_dir" \
        --calib_dataset wikitext2 \
        --nsamples 32 \
        --batch_size 1 \
        --seed 2 \
        --wbits 2 \
        --abits 16 \
        --group_size 128 \
        --epochs 10 \
        --lwc \
        --lwc_lr 0.005 \
        --eval_ppl \
        "$@" \
        > "$output_dir/run.log" 2>&1

    local status=$?

    echo "RUN_EXIT_CODE=$status" >> "$output_dir/run.log"
    echo "END_TIME=$(date --iso-8601=seconds)" >> "$output_dir/environment.txt"
    echo "RUN_EXIT_CODE=$status" >> "$output_dir/environment.txt"

    echo "[$(date)] Finished $name, exit code=$status"

    # 即使 baseline 失败，也继续运行 robust。
    return 0
}

run_case \
    "baseline" \
    "$BASELINE"

run_case \
    "robust" \
    "$ROBUST" \
    --robust_mode \
    --grad_clip 1.0 \
    --max_retries 3 \
    --nan_lr_decay 0.5 \
    --min_lr 1e-6 \
    --progress_file "$ROBUST/quant_progress.json"

echo "ALL_EXPERIMENTS_FINISHED=$(date --iso-8601=seconds)"
