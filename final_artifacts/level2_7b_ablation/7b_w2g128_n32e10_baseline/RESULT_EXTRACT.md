# 7b_w2g128_n32e10_baseline

## Directory listing
total 6.1M
-rw-r--r-- 1 root root 1.9K Jul  5 15:50 environment.txt
-rw-r--r-- 1 root root 3.6K Jul  5 15:50 log_rank0_1783237548.txt
-rw-r--r-- 1 root root 6.1M Jul  5 15:50 omni_parameters.pth
-rw-r--r-- 1 root root  197 Jul  5 15:50 quant_progress.json
-rw-r--r-- 1 root root 5.5K Jul  5 15:50 run.log

## Command

## Environment
CASE=baseline
START_TIME=2026-07-05T15:45:45+08:00
GIT_COMMIT=3b25db6413c62581664adf0e6c2dd2f814396bab
Sun Jul  5 15:45:45 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.120                Driver Version: 550.120        CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA vGPU-32GB               On  |   00000000:82:00.0 Off |                  N/A |
| 30%   39C    P8             10W /  320W |       1MiB /  32760MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
END_TIME=2026-07-05T15:50:51+08:00
RUN_EXIT_CODE=1

## Key result lines
[2026-07-05 15:50:10 root](omniquant.py 247): INFO === Start quantize layer 0 ===
[2026-07-05 15:50:49 root](omniquant.py 247): INFO === Start quantize layer 1 ===
Traceback (most recent call last):
    raise FloatingPointError(
FloatingPointError: non-finite gradient norm in baseline mode: layer=1, epoch=0, batch=1, norm=nan
RUN_EXIT_CODE=1
