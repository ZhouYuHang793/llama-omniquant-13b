# 7b_w2g64_n32e10_robust_lr1e3_agent

## Directory listing
total 387M
-rw-r--r-- 1 root root  639 Jul  7 01:17 command.sh
-rw-r--r-- 1 root root  133 Jul  7 01:42 environment.txt
-rw-r--r-- 1 root root  73K Jul  7 01:42 log_rank0_1783358280.txt
-rw-r--r-- 1 root root 387M Jul  7 01:38 omni_parameters.pth
-rw-r--r-- 1 root root  434 Jul  7 01:38 quant_progress.json
-rw-r--r-- 1 root root  99K Jul  7 01:42 run.log

## Command
/root/autodl-tmp/omniquant_final/env/bin/python -u /root/autodl-tmp/omniquant_final/code/main.py --model /root/autodl-tmp/llama-2-7b-hf --cache_dir /root/autodl-tmp/omniquant_final/restore/OmniQuant/cache --calib_dataset wikitext2 --batch_size 1 --seed 2 --abits 16 --eval_ppl --output_dir /root/autodl-tmp/omniquant_final/experiments/7b_w2g64_n32e10_robust_lr1e3_agent --nsamples 32 --wbits 2 --group_size 64 --epochs 10 --lwc --lwc_lr 0.001 --robust_mode --grad_clip 1.0 --max_retries 3 --nan_lr_decay 0.5 --min_lr 1e-6 --progress_file /root/autodl-tmp/omniquant_final/experiments/7b_w2g64_n32e10_robust_lr1e3_agent/quant_progress.json 

## Environment
START_EPOCH=1783358275
END_EPOCH=1783359743
ELAPSED_SECONDS=1468
RUN_EXIT_CODE=0
GIT_COMMIT=3b25db6413c62581664adf0e6c2dd2f814396bab

## Key result lines
[2026-07-07 01:18:02 root](omniquant.py 247): INFO === Start quantize layer 0 ===
[2026-07-07 01:18:42 root](omniquant.py 247): INFO === Start quantize layer 1 ===
[2026-07-07 01:18:43 root](omniquant.py 525): WARNING [ROBUST] rollback layer=1 epoch=0 retry=1/3: non-finite grad_norm=nan at batch=1
[2026-07-07 01:18:43 root](omniquant.py 565): WARNING [ROBUST] FP32 fallback retry with mode=fp32, let_lr=2.500e-03, lwc_lr=5.000e-04
[2026-07-07 01:19:22 root](omniquant.py 247): INFO === Start quantize layer 2 ===
[2026-07-07 01:20:01 root](omniquant.py 247): INFO === Start quantize layer 3 ===
[2026-07-07 01:20:39 root](omniquant.py 247): INFO === Start quantize layer 4 ===
[2026-07-07 01:21:18 root](omniquant.py 247): INFO === Start quantize layer 5 ===
[2026-07-07 01:21:56 root](omniquant.py 247): INFO === Start quantize layer 6 ===
[2026-07-07 01:22:35 root](omniquant.py 247): INFO === Start quantize layer 7 ===
[2026-07-07 01:23:14 root](omniquant.py 247): INFO === Start quantize layer 8 ===
[2026-07-07 01:23:52 root](omniquant.py 247): INFO === Start quantize layer 9 ===
[2026-07-07 01:24:31 root](omniquant.py 247): INFO === Start quantize layer 10 ===
[2026-07-07 01:25:09 root](omniquant.py 247): INFO === Start quantize layer 11 ===
[2026-07-07 01:25:48 root](omniquant.py 247): INFO === Start quantize layer 12 ===
[2026-07-07 01:26:27 root](omniquant.py 247): INFO === Start quantize layer 13 ===
[2026-07-07 01:27:05 root](omniquant.py 247): INFO === Start quantize layer 14 ===
[2026-07-07 01:27:44 root](omniquant.py 247): INFO === Start quantize layer 15 ===
[2026-07-07 01:28:23 root](omniquant.py 247): INFO === Start quantize layer 16 ===
[2026-07-07 01:29:01 root](omniquant.py 247): INFO === Start quantize layer 17 ===
[2026-07-07 01:29:40 root](omniquant.py 247): INFO === Start quantize layer 18 ===
[2026-07-07 01:30:19 root](omniquant.py 247): INFO === Start quantize layer 19 ===
[2026-07-07 01:30:57 root](omniquant.py 247): INFO === Start quantize layer 20 ===
[2026-07-07 01:31:36 root](omniquant.py 247): INFO === Start quantize layer 21 ===
[2026-07-07 01:32:15 root](omniquant.py 247): INFO === Start quantize layer 22 ===
[2026-07-07 01:32:54 root](omniquant.py 247): INFO === Start quantize layer 23 ===
[2026-07-07 01:33:33 root](omniquant.py 247): INFO === Start quantize layer 24 ===
[2026-07-07 01:34:11 root](omniquant.py 247): INFO === Start quantize layer 25 ===
[2026-07-07 01:34:50 root](omniquant.py 247): INFO === Start quantize layer 26 ===
[2026-07-07 01:35:30 root](omniquant.py 247): INFO === Start quantize layer 27 ===
[2026-07-07 01:36:09 root](omniquant.py 247): INFO === Start quantize layer 28 ===
[2026-07-07 01:36:48 root](omniquant.py 247): INFO === Start quantize layer 29 ===
[2026-07-07 01:37:27 root](omniquant.py 247): INFO === Start quantize layer 30 ===
[2026-07-07 01:37:28 root](omniquant.py 525): WARNING [ROBUST] rollback layer=30 epoch=0 retry=1/3: non-finite grad_norm=nan at batch=2
[2026-07-07 01:37:28 root](omniquant.py 565): WARNING [ROBUST] FP32 fallback retry with mode=fp32, let_lr=2.500e-03, lwc_lr=5.000e-04
[2026-07-07 01:38:08 root](omniquant.py 247): INFO === Start quantize layer 31 ===
[2026-07-07 01:38:09 root](omniquant.py 525): WARNING [ROBUST] rollback layer=31 epoch=0 retry=1/3: non-finite grad_norm=nan at batch=8
[2026-07-07 01:38:09 root](omniquant.py 565): WARNING [ROBUST] FP32 fallback retry with mode=fp32, let_lr=2.500e-03, lwc_lr=5.000e-04
[2026-07-07 01:40:16 root](main.py 144): INFO wikitext2 : 455.3946228027344
[2026-07-07 01:42:23 root](main.py 144): INFO c4 : 667.1389770507812
RUN_EXIT_CODE=0
