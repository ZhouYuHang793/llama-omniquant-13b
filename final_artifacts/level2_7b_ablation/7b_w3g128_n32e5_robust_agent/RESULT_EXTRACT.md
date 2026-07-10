# 7b_w3g128_n32e5_robust_agent

## Directory listing
total 194M
-rw-r--r-- 1 root root  627 Jul  7 01:58 command.sh
-rw-r--r-- 1 root root  132 Jul  7 02:12 environment.txt
-rw-r--r-- 1 root root  39K Jul  7 02:12 log_rank0_1783360699.txt
-rw-r--r-- 1 root root 194M Jul  7 02:09 omni_parameters.pth
-rw-r--r-- 1 root root  434 Jul  7 02:09 quant_progress.json
-rw-r--r-- 1 root root  65K Jul  7 02:12 run.log

## Command
/root/autodl-tmp/omniquant_final/env/bin/python -u /root/autodl-tmp/omniquant_final/code/main.py --model /root/autodl-tmp/llama-2-7b-hf --cache_dir /root/autodl-tmp/omniquant_final/restore/OmniQuant/cache --calib_dataset wikitext2 --batch_size 1 --seed 2 --abits 16 --eval_ppl --output_dir /root/autodl-tmp/omniquant_final/experiments/7b_w3g128_n32e5_robust_agent --nsamples 32 --wbits 3 --group_size 128 --epochs 5 --lwc --lwc_lr 0.001 --robust_mode --grad_clip 1.0 --max_retries 3 --nan_lr_decay 0.5 --min_lr 1e-6 --progress_file /root/autodl-tmp/omniquant_final/experiments/7b_w3g128_n32e5_robust_agent/quant_progress.json 

## Environment
START_EPOCH=1783360694
END_EPOCH=1783361558
ELAPSED_SECONDS=864
RUN_EXIT_CODE=0
GIT_COMMIT=3b25db6413c62581664adf0e6c2dd2f814396bab

## Key result lines
[2026-07-07 01:58:22 root](omniquant.py 247): INFO === Start quantize layer 0 ===
[2026-07-07 01:58:42 root](omniquant.py 247): INFO === Start quantize layer 1 ===
[2026-07-07 01:58:43 root](omniquant.py 525): WARNING [ROBUST] rollback layer=1 epoch=0 retry=1/3: non-finite grad_norm=nan at batch=3
[2026-07-07 01:58:43 root](omniquant.py 565): WARNING [ROBUST] FP32 fallback retry with mode=fp32, let_lr=2.500e-03, lwc_lr=5.000e-04
[2026-07-07 01:59:04 root](omniquant.py 247): INFO === Start quantize layer 2 ===
[2026-07-07 01:59:24 root](omniquant.py 247): INFO === Start quantize layer 3 ===
[2026-07-07 01:59:44 root](omniquant.py 247): INFO === Start quantize layer 4 ===
[2026-07-07 02:00:04 root](omniquant.py 247): INFO === Start quantize layer 5 ===
[2026-07-07 02:00:24 root](omniquant.py 247): INFO === Start quantize layer 6 ===
[2026-07-07 02:00:44 root](omniquant.py 247): INFO === Start quantize layer 7 ===
[2026-07-07 02:01:04 root](omniquant.py 247): INFO === Start quantize layer 8 ===
[2026-07-07 02:01:24 root](omniquant.py 247): INFO === Start quantize layer 9 ===
[2026-07-07 02:01:44 root](omniquant.py 247): INFO === Start quantize layer 10 ===
[2026-07-07 02:02:04 root](omniquant.py 247): INFO === Start quantize layer 11 ===
[2026-07-07 02:02:24 root](omniquant.py 247): INFO === Start quantize layer 12 ===
[2026-07-07 02:02:44 root](omniquant.py 247): INFO === Start quantize layer 13 ===
[2026-07-07 02:03:03 root](omniquant.py 247): INFO === Start quantize layer 14 ===
[2026-07-07 02:03:23 root](omniquant.py 247): INFO === Start quantize layer 15 ===
[2026-07-07 02:03:43 root](omniquant.py 247): INFO === Start quantize layer 16 ===
[2026-07-07 02:04:03 root](omniquant.py 247): INFO === Start quantize layer 17 ===
[2026-07-07 02:04:23 root](omniquant.py 247): INFO === Start quantize layer 18 ===
[2026-07-07 02:04:43 root](omniquant.py 247): INFO === Start quantize layer 19 ===
[2026-07-07 02:05:03 root](omniquant.py 247): INFO === Start quantize layer 20 ===
[2026-07-07 02:05:23 root](omniquant.py 247): INFO === Start quantize layer 21 ===
[2026-07-07 02:05:43 root](omniquant.py 247): INFO === Start quantize layer 22 ===
[2026-07-07 02:06:03 root](omniquant.py 247): INFO === Start quantize layer 23 ===
[2026-07-07 02:06:24 root](omniquant.py 247): INFO === Start quantize layer 24 ===
[2026-07-07 02:06:44 root](omniquant.py 247): INFO === Start quantize layer 25 ===
[2026-07-07 02:07:04 root](omniquant.py 247): INFO === Start quantize layer 26 ===
[2026-07-07 02:07:24 root](omniquant.py 247): INFO === Start quantize layer 27 ===
[2026-07-07 02:07:44 root](omniquant.py 247): INFO === Start quantize layer 28 ===
[2026-07-07 02:08:04 root](omniquant.py 247): INFO === Start quantize layer 29 ===
[2026-07-07 02:08:24 root](omniquant.py 247): INFO === Start quantize layer 30 ===
[2026-07-07 02:08:44 root](omniquant.py 247): INFO === Start quantize layer 31 ===
[2026-07-07 02:10:31 root](main.py 144): INFO wikitext2 : 6.615912437438965
[2026-07-07 02:12:38 root](main.py 144): INFO c4 : 8.373472213745117
RUN_EXIT_CODE=0
