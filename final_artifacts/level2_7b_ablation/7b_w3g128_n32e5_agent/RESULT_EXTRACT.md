# 7b_w3g128_n32e5_agent

## Directory listing
total 6.1M
-rw-r--r-- 1 root root  431 Jul  7 00:49 command.sh
-rw-r--r-- 1 root root  131 Jul  7 00:49 environment.txt
-rw-r--r-- 1 root root 2.1K Jul  7 00:49 log_rank0_1783356565.txt
-rw-r--r-- 1 root root 6.1M Jul  7 00:49 omni_parameters.pth
-rw-r--r-- 1 root root  196 Jul  7 00:49 quant_progress.json
-rw-r--r-- 1 root root 3.3K Jul  7 00:49 run.log

## Command
/root/autodl-tmp/omniquant_final/env/bin/python -u /root/autodl-tmp/omniquant_final/code/main.py --model /root/autodl-tmp/llama-2-7b-hf --cache_dir /root/autodl-tmp/omniquant_final/restore/OmniQuant/cache --calib_dataset wikitext2 --batch_size 1 --seed 2 --abits 16 --eval_ppl --output_dir /root/autodl-tmp/omniquant_final/experiments/7b_w3g128_n32e5_agent --nsamples 32 --wbits 3 --group_size 128 --epochs 5 --lwc --lwc_lr 0.005 

## Environment
START_EPOCH=1783356562
END_EPOCH=1783356589
ELAPSED_SECONDS=27
RUN_EXIT_CODE=1
GIT_COMMIT=3b25db6413c62581664adf0e6c2dd2f814396bab

## Key result lines
[2026-07-07 00:49:27 root](omniquant.py 247): INFO === Start quantize layer 0 ===
[2026-07-07 00:49:47 root](omniquant.py 247): INFO === Start quantize layer 1 ===
Traceback (most recent call last):
    raise FloatingPointError(
FloatingPointError: non-finite gradient norm in baseline mode: layer=1, epoch=0, batch=3, norm=nan
RUN_EXIT_CODE=1
