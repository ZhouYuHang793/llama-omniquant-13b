# 7b_w2g128_n32e0_agent

## Directory listing
total 48K
-rw-r--r-- 1 root root  410 Jul  7 00:49 command.sh
-rw-r--r-- 1 root root  132 Jul  7 00:53 environment.txt
-rw-r--r-- 1 root root 4.3K Jul  7 00:53 log_rank0_1783356593.txt
-rw-r--r-- 1 root root  31K Jul  7 00:53 run.log

## Command
/root/autodl-tmp/omniquant_final/env/bin/python -u /root/autodl-tmp/omniquant_final/code/main.py --model /root/autodl-tmp/llama-2-7b-hf --cache_dir /root/autodl-tmp/omniquant_final/restore/OmniQuant/cache --calib_dataset wikitext2 --batch_size 1 --seed 2 --abits 16 --eval_ppl --output_dir /root/autodl-tmp/omniquant_final/experiments/7b_w2g128_n32e0_agent --nsamples 32 --wbits 2 --group_size 128 --epochs 0 

## Environment
START_EPOCH=1783356589
END_EPOCH=1783356816
ELAPSED_SECONDS=227
RUN_EXIT_CODE=0
GIT_COMMIT=3b25db6413c62581664adf0e6c2dd2f814396bab

## Key result lines
[2026-07-07 00:49:55 root](omniquant.py 247): INFO === Start quantize layer 0 ===
[2026-07-07 00:49:55 root](omniquant.py 247): INFO === Start quantize layer 1 ===
[2026-07-07 00:49:55 root](omniquant.py 247): INFO === Start quantize layer 2 ===
[2026-07-07 00:49:56 root](omniquant.py 247): INFO === Start quantize layer 3 ===
[2026-07-07 00:49:56 root](omniquant.py 247): INFO === Start quantize layer 4 ===
[2026-07-07 00:49:56 root](omniquant.py 247): INFO === Start quantize layer 5 ===
[2026-07-07 00:49:56 root](omniquant.py 247): INFO === Start quantize layer 6 ===
[2026-07-07 00:49:57 root](omniquant.py 247): INFO === Start quantize layer 7 ===
[2026-07-07 00:49:57 root](omniquant.py 247): INFO === Start quantize layer 8 ===
[2026-07-07 00:49:57 root](omniquant.py 247): INFO === Start quantize layer 9 ===
[2026-07-07 00:49:58 root](omniquant.py 247): INFO === Start quantize layer 10 ===
[2026-07-07 00:49:58 root](omniquant.py 247): INFO === Start quantize layer 11 ===
[2026-07-07 00:49:58 root](omniquant.py 247): INFO === Start quantize layer 12 ===
[2026-07-07 00:49:59 root](omniquant.py 247): INFO === Start quantize layer 13 ===
[2026-07-07 00:49:59 root](omniquant.py 247): INFO === Start quantize layer 14 ===
[2026-07-07 00:49:59 root](omniquant.py 247): INFO === Start quantize layer 15 ===
[2026-07-07 00:50:00 root](omniquant.py 247): INFO === Start quantize layer 16 ===
[2026-07-07 00:50:00 root](omniquant.py 247): INFO === Start quantize layer 17 ===
[2026-07-07 00:50:00 root](omniquant.py 247): INFO === Start quantize layer 18 ===
[2026-07-07 00:50:00 root](omniquant.py 247): INFO === Start quantize layer 19 ===
[2026-07-07 00:50:01 root](omniquant.py 247): INFO === Start quantize layer 20 ===
[2026-07-07 00:50:01 root](omniquant.py 247): INFO === Start quantize layer 21 ===
[2026-07-07 00:50:01 root](omniquant.py 247): INFO === Start quantize layer 22 ===
[2026-07-07 00:50:02 root](omniquant.py 247): INFO === Start quantize layer 23 ===
[2026-07-07 00:50:02 root](omniquant.py 247): INFO === Start quantize layer 24 ===
[2026-07-07 00:50:02 root](omniquant.py 247): INFO === Start quantize layer 25 ===
[2026-07-07 00:50:03 root](omniquant.py 247): INFO === Start quantize layer 26 ===
[2026-07-07 00:50:03 root](omniquant.py 247): INFO === Start quantize layer 27 ===
[2026-07-07 00:50:03 root](omniquant.py 247): INFO === Start quantize layer 28 ===
[2026-07-07 00:50:04 root](omniquant.py 247): INFO === Start quantize layer 29 ===
[2026-07-07 00:50:04 root](omniquant.py 247): INFO === Start quantize layer 30 ===
[2026-07-07 00:50:04 root](omniquant.py 247): INFO === Start quantize layer 31 ===
[2026-07-07 00:51:29 root](main.py 144): INFO wikitext2 : 4272.63623046875
[2026-07-07 00:53:35 root](main.py 144): INFO c4 : 4907.05078125
RUN_EXIT_CODE=0
