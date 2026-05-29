#!/usr/bin/env bash
set -e

CUDA_VISIBLE_DEVICES=0 python main.py \
  --model /path/to/llama-2-13b-hf \
  --cache_dir ./cache \
  --output_dir ./log/rerun_w3a16g128_ppl_evalonly \
  --resume /path/to/llama-2-13b-w3a16g128/omni_parameters.pth \
  --wbits 3 \
  --abits 16 \
  --group_size 128 \
  --lwc \
  --epochs 0 \
  --eval_ppl \
  --attn_implementation eager
