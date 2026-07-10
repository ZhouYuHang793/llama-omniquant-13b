# 7b_w2g128_n32e10_robust

## Directory listing
total 194M
-rw-r--r-- 1 root root 1.9K Jul  5 16:15 environment.txt
-rw-r--r-- 1 root root  73K Jul  5 16:15 log_rank0_1783237855.txt
-rw-r--r-- 1 root root 194M Jul  5 16:11 omni_parameters.pth
-rw-r--r-- 1 root root  435 Jul  5 16:11 quant_progress.json
-rw-r--r-- 1 root root  99K Jul  5 16:15 run.log

## Command

## Environment
CASE=robust
START_TIME=2026-07-05T15:50:51+08:00
GIT_COMMIT=3b25db6413c62581664adf0e6c2dd2f814396bab
Sun Jul  5 15:50:51 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.120                Driver Version: 550.120        CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA vGPU-32GB               On  |   00000000:82:00.0 Off |                  N/A |
| 68%   55C    P0             62W /  320W |       1MiB /  32760MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
END_TIME=2026-07-05T16:15:18+08:00
RUN_EXIT_CODE=0

## Key result lines
[2026-07-05 15:50:56 root](omniquant.py 247): INFO === Start quantize layer 0 ===
[2026-07-05 15:51:35 root](omniquant.py 247): INFO === Start quantize layer 1 ===
[2026-07-05 15:51:36 root](omniquant.py 525): WARNING [ROBUST] rollback layer=1 epoch=0 retry=1/3: non-finite grad_norm=nan at batch=1
[2026-07-05 15:51:36 root](omniquant.py 565): WARNING [ROBUST] FP32 fallback retry with mode=fp32, let_lr=2.500e-03, lwc_lr=2.500e-03
[2026-07-05 15:52:15 root](omniquant.py 247): INFO === Start quantize layer 2 ===
[2026-07-05 15:52:53 root](omniquant.py 247): INFO === Start quantize layer 3 ===
[2026-07-05 15:53:32 root](omniquant.py 247): INFO === Start quantize layer 4 ===
[2026-07-05 15:54:10 root](omniquant.py 247): INFO === Start quantize layer 5 ===
[2026-07-05 15:54:49 root](omniquant.py 247): INFO === Start quantize layer 6 ===
[2026-07-05 15:55:27 root](omniquant.py 247): INFO === Start quantize layer 7 ===
[2026-07-05 15:56:06 root](omniquant.py 247): INFO === Start quantize layer 8 ===
[2026-07-05 15:56:44 root](omniquant.py 247): INFO === Start quantize layer 9 ===
[2026-07-05 15:57:23 root](omniquant.py 247): INFO === Start quantize layer 10 ===
[2026-07-05 15:58:02 root](omniquant.py 247): INFO === Start quantize layer 11 ===
[2026-07-05 15:58:40 root](omniquant.py 247): INFO === Start quantize layer 12 ===
[2026-07-05 15:59:19 root](omniquant.py 247): INFO === Start quantize layer 13 ===
[2026-07-05 15:59:57 root](omniquant.py 247): INFO === Start quantize layer 14 ===
[2026-07-05 16:00:36 root](omniquant.py 247): INFO === Start quantize layer 15 ===
[2026-07-05 16:01:15 root](omniquant.py 247): INFO === Start quantize layer 16 ===
[2026-07-05 16:01:53 root](omniquant.py 247): INFO === Start quantize layer 17 ===
[2026-07-05 16:02:32 root](omniquant.py 247): INFO === Start quantize layer 18 ===
[2026-07-05 16:03:11 root](omniquant.py 247): INFO === Start quantize layer 19 ===
[2026-07-05 16:03:49 root](omniquant.py 247): INFO === Start quantize layer 20 ===
[2026-07-05 16:04:28 root](omniquant.py 247): INFO === Start quantize layer 21 ===
[2026-07-05 16:05:07 root](omniquant.py 247): INFO === Start quantize layer 22 ===
[2026-07-05 16:05:45 root](omniquant.py 247): INFO === Start quantize layer 23 ===
[2026-07-05 16:06:25 root](omniquant.py 247): INFO === Start quantize layer 24 ===
[2026-07-05 16:07:04 root](omniquant.py 247): INFO === Start quantize layer 25 ===
[2026-07-05 16:07:42 root](omniquant.py 247): INFO === Start quantize layer 26 ===
[2026-07-05 16:08:21 root](omniquant.py 247): INFO === Start quantize layer 27 ===
[2026-07-05 16:09:00 root](omniquant.py 247): INFO === Start quantize layer 28 ===
[2026-07-05 16:09:39 root](omniquant.py 247): INFO === Start quantize layer 29 ===
[2026-07-05 16:10:18 root](omniquant.py 247): INFO === Start quantize layer 30 ===
[2026-07-05 16:10:21 root](omniquant.py 525): WARNING [ROBUST] rollback layer=30 epoch=0 retry=1/3: non-finite grad_norm=nan at batch=21
[2026-07-05 16:10:21 root](omniquant.py 565): WARNING [ROBUST] FP32 fallback retry with mode=fp32, let_lr=2.500e-03, lwc_lr=2.500e-03
[2026-07-05 16:11:01 root](omniquant.py 247): INFO === Start quantize layer 31 ===
[2026-07-05 16:11:02 root](omniquant.py 525): WARNING [ROBUST] rollback layer=31 epoch=0 retry=1/3: non-finite grad_norm=nan at batch=4
[2026-07-05 16:11:02 root](omniquant.py 565): WARNING [ROBUST] FP32 fallback retry with mode=fp32, let_lr=2.500e-03, lwc_lr=2.500e-03
[2026-07-05 16:13:09 root](main.py 144): INFO wikitext2 : 1810.511474609375
[2026-07-05 16:15:17 root](main.py 144): INFO c4 : 4155.16552734375
RUN_EXIT_CODE=0
