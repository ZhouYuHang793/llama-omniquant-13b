# 自动证据索引

- 日志数量：71
- Git HEAD：3b25db6413c62581664adf0e6c2dd2f814396bab
- Git 是否干净：否

> 说明：上述 Git HEAD 与工作区状态来自 AutoDL 实验环境打包时的历史快照，用于追踪实验源码版本，不代表当前 GitHub main 分支的提交状态。

## 已核实历史结果

| Case | WikiText2 | C4 | 其他 | 来源 |
|---|---:|---:|---:|---|
| 13B W4A16-g128 | 4.9648 | 6.5721 |  | PROJECT_STATE.md / recovered logs |
| 13B W3A16-g128 | 5.3232 | 7.0206 |  | PROJECT_STATE.md / recovered logs |
| 13B W2A16-g128 LWC | 10.7503 | 13.2891 |  | PROJECT_STATE.md / recovered logs |
| 13B W2A16-g128 LET+LWC | 14.1274 | 17.7756 |  | PROJECT_STATE.md / recovered logs |
| 13B W2A16-g128 no-LWC | 122.1143 | 139.6501 |  | PROJECT_STATE.md / recovered logs |
| 13B W4 real quant BoolQ |  |  | 0.663 | PROJECT_STATE.md / boolq score |

## 自动解析日志

| 日志 | Exit | Last layer | WikiText2 | C4 | Rollback | FP32 | NaN |
|---|---:|---:|---:|---:|---:|---:|---:|
| `experiments/7b_w2_robust_smoke/log_rank0_1783236910.txt` | None | 31 | None | None | 2 | 2 | 4 |
| `experiments/7b_w2_robust_smoke/run.log` | 0 | 31 | None | None | 2 | 2 | 4 |
| `experiments/7b_w2_robust_smoke_v1_failed/log_rank0_1783235078.txt` | None | 30 | None | None | 4 | 0 | 8 |
| `experiments/7b_w2_robust_smoke_v1_failed/run.log` | 1 | 30 | None | None | 4 | 0 | 10 |
| `experiments/7b_w2g128_n32e0_agent/log_rank0_1783356593.txt` | None | 31 | 4272.63623046875 | 4907.05078125 | 0 | 0 | 0 |
| `experiments/7b_w2g128_n32e0_agent/run.log` | 0 | 31 | 4272.63623046875 | 4907.05078125 | 0 | 0 | 0 |
| `experiments/7b_w2g128_n32e10_baseline/log_rank0_1783237548.txt` | None | 1 | None | None | 0 | 0 | 0 |
| `experiments/7b_w2g128_n32e10_baseline/run.log` | 1 | 1 | None | None | 0 | 0 | 2 |
| `experiments/7b_w2g128_n32e10_robust/log_rank0_1783237855.txt` | None | 31 | 1810.511474609375 | 4155.16552734375 | 3 | 3 | 6 |
| `experiments/7b_w2g128_n32e10_robust/run.log` | 0 | 31 | 1810.511474609375 | 4155.16552734375 | 3 | 3 | 6 |
| `experiments/7b_w2g128_n32e10_robust_lr1e3_agent/log_rank0_1783356820.txt` | None | 31 | None | 5235.41796875 | 3 | 3 | 7 |
| `experiments/7b_w2g128_n32e10_robust_lr1e3_agent/run.log` | 0 | 31 | None | 5235.41796875 | 3 | 3 | 7 |
| `experiments/7b_w2g64_n32e10_robust_lr1e3_agent/log_rank0_1783358280.txt` | None | 31 | 455.3946228027344 | 667.1389770507812 | 3 | 3 | 6 |
| `experiments/7b_w2g64_n32e10_robust_lr1e3_agent/run.log` | 0 | 31 | 455.3946228027344 | 667.1389770507812 | 3 | 3 | 6 |
| `experiments/7b_w3g128_n32e5_agent/log_rank0_1783356565.txt` | None | 1 | None | None | 0 | 0 | 0 |
| `experiments/7b_w3g128_n32e5_agent/run.log` | 1 | 1 | None | None | 0 | 0 | 2 |
| `experiments/7b_w3g128_n32e5_robust_agent/log_rank0_1783360699.txt` | None | 31 | 6.615912437438965 | 8.373472213745117 | 1 | 1 | 2 |
| `experiments/7b_w3g128_n32e5_robust_agent/run.log` | 0 | 31 | 6.615912437438965 | 8.373472213745117 | 1 | 1 | 2 |
| `experiments/7b_w4g128_n32e5_agent/log_rank0_1783356538.txt` | None | 1 | None | None | 0 | 0 | 0 |
| `experiments/7b_w4g128_n32e5_agent/run.log` | 1 | 1 | None | None | 0 | 0 | 2 |
| `experiments/7b_w4g128_n32e5_robust_agent/log_rank0_1783359829.txt` | None | 31 | 5.724181652069092 | 7.245031833648682 | 2 | 2 | 4 |
| `experiments/7b_w4g128_n32e5_robust_agent/run.log` | 0 | 31 | 5.724181652069092 | 7.245031833648682 | 2 | 2 | 4 |
| `restore/OmniQuant/log/llama-2-13b-w2a16g128/log_rank0_1775791178.txt` | None | None | None | None | 0 | 0 | 0 |
| `restore/OmniQuant/log/llama-2-13b-w2a16g128/log_rank0_1775801665.txt` | None | 39 | 10.75031566619873 | 13.289106369018555 | 0 | 0 | 0 |
| `restore/OmniQuant/log/llama-2-13b-w2a16g128-full/log_rank0_1775895171.txt` | None | None | None | None | 0 | 0 | 0 |
| `restore/OmniQuant/log/llama-2-13b-w2a16g128-full/log_rank0_1775895509.txt` | None | 39 | 14.127449035644531 | 17.77560043334961 | 0 | 0 | 3 |
| `restore/OmniQuant/log/llama-2-13b-w2a16g128-no-lwc/log_rank0_1775893283.txt` | None | 0 | None | None | 0 | 0 | 0 |
| `restore/OmniQuant/log/llama-2-13b-w2a16g128-no-lwc/log_rank0_1775893416.txt` | None | 39 | 122.11434936523438 | 139.650146484375 | 0 | 0 | 0 |
| `restore/OmniQuant/log/llama-2-13b-w3a16g128/log_rank0_1775814207.txt` | None | 39 | 5.323248386383057 | 7.020618915557861 | 0 | 0 | 0 |
| `restore/OmniQuant/log/llama-2-13b-w4a16g128/log_rank0_1775826820.txt` | None | 39 | 4.964791774749756 | 6.572065353393555 | 0 | 0 | 0 |
| `restore/OmniQuant/log/llama-2-13b-w4a16g128-zeroshot/log_rank0_1775877998.txt` | None | 39 | None | None | 0 | 0 | 0 |
| `restore/OmniQuant/log/llama-2-7b-w2a16/log_rank0_1775034704.txt` | None | 31 | 5129.58447265625 | 16626.1953125 | 0 | 0 | 6 |
| `restore/OmniQuant/log/llama-2-7b-w2a16g128/log_rank0_1775619126.txt` | None | 31 | 18.859512329101562 | 26.605562210083008 | 0 | 0 | 1 |
| `restore/OmniQuant/log/llama-2-7b-w2a16g128/log_rank0_1775627473.txt` | None | 0 | None | None | 0 | 0 | 0 |
| `restore/OmniQuant/log/llama-2-7b-w3a16g128/log_rank0_1775631519.txt` | None | 31 | 6.098163604736328 | 7.814173698425293 | 0 | 0 | 1 |
| `restore/OmniQuant/log/llama-2-7b-w4a16/log_rank0_1774923076.txt` | None | None | None | None | 0 | 0 | 0 |
| `restore/OmniQuant/log/llama-2-7b-w4a16/log_rank0_1774924517.txt` | None | 31 | 5.743441581726074 | 7.360231399536133 | 0 | 0 | 1 |
| `restore/OmniQuant/log/llama-2-7b-w4a16g128/log_rank0_1775638309.txt` | None | 31 | 5.593636512756348 | 7.121578693389893 | 0 | 0 | 1 |
| `restore/OmniQuant/log/llama-2-7b-w4a4/log_rank0_1774953543.txt` | None | None | None | None | 0 | 0 | 0 |
| `restore/OmniQuant/log/llama-2-7b-w4a4/log_rank0_1774953944.txt` | None | 31 | None | None | 0 | 0 | 11 |
| `restore/OmniQuant/log/llama-2-7b-w4a4/log_rank0_1774970720.txt` | None | 9 | None | None | 0 | 0 | 4 |
| `restore/OmniQuant/log/llama-2-7b-w4a4/log_rank0_1775020017.txt` | None | 31 | 20.388158798217773 | 26.501079559326172 | 0 | 0 | 11 |
| `restore/OmniQuant/log/w2_optimized/log_rank0_1776843256.txt` | None | 1 | None | None | 0 | 0 | 0 |
| `restore/OmniQuant/log/w2_optimized/log_rank0_1777003737.txt` | None | 4 | None | None | 0 | 0 | 5 |
| `restore/OmniQuant/log/w3_optimized/log_rank0_1776831237.txt` | None | 39 | None | None | 0 | 0 | 1 |
| `restore/OmniQuant/log/w4_optimized/log_rank0_1776779629.txt` | None | 39 | None | None | 0 | 0 | 0 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/llama-2-13b-w2a16g128/log_rank0_1775791178.txt` | None | None | None | None | 0 | 0 | 0 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/llama-2-13b-w2a16g128/log_rank0_1775801665.txt` | None | 39 | 10.75031566619873 | 13.289106369018555 | 0 | 0 | 0 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/llama-2-13b-w2a16g128-full/log_rank0_1775895171.txt` | None | None | None | None | 0 | 0 | 0 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/llama-2-13b-w2a16g128-full/log_rank0_1775895509.txt` | None | 39 | 14.127449035644531 | 17.77560043334961 | 0 | 0 | 3 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/llama-2-13b-w2a16g128-no-lwc/log_rank0_1775893283.txt` | None | 0 | None | None | 0 | 0 | 0 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/llama-2-13b-w2a16g128-no-lwc/log_rank0_1775893416.txt` | None | 39 | 122.11434936523438 | 139.650146484375 | 0 | 0 | 0 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/llama-2-13b-w3a16g128/log_rank0_1775814207.txt` | None | 39 | 5.323248386383057 | 7.020618915557861 | 0 | 0 | 0 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/llama-2-13b-w4a16g128/log_rank0_1775826820.txt` | None | 39 | 4.964791774749756 | 6.572065353393555 | 0 | 0 | 0 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/llama-2-13b-w4a16g128-zeroshot/log_rank0_1775877998.txt` | None | 39 | None | None | 0 | 0 | 0 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/llama-2-7b-w2a16/log_rank0_1775034704.txt` | None | 31 | 5129.58447265625 | 16626.1953125 | 0 | 0 | 6 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/llama-2-7b-w2a16g128/log_rank0_1775619126.txt` | None | 31 | 18.859512329101562 | 26.605562210083008 | 0 | 0 | 1 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/llama-2-7b-w2a16g128/log_rank0_1775627473.txt` | None | 0 | None | None | 0 | 0 | 0 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/llama-2-7b-w3a16g128/log_rank0_1775631519.txt` | None | 31 | 6.098163604736328 | 7.814173698425293 | 0 | 0 | 1 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/llama-2-7b-w4a16/log_rank0_1774923076.txt` | None | None | None | None | 0 | 0 | 0 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/llama-2-7b-w4a16/log_rank0_1774924517.txt` | None | 31 | 5.743441581726074 | 7.360231399536133 | 0 | 0 | 1 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/llama-2-7b-w4a16g128/log_rank0_1775638309.txt` | None | 31 | 5.593636512756348 | 7.121578693389893 | 0 | 0 | 1 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/llama-2-7b-w4a4/log_rank0_1774953543.txt` | None | None | None | None | 0 | 0 | 0 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/llama-2-7b-w4a4/log_rank0_1774953944.txt` | None | 31 | None | None | 0 | 0 | 11 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/llama-2-7b-w4a4/log_rank0_1774970720.txt` | None | 9 | None | None | 0 | 0 | 4 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/llama-2-7b-w4a4/log_rank0_1775020017.txt` | None | 31 | 20.388158798217773 | 26.501079559326172 | 0 | 0 | 11 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/w2_optimized/log_rank0_1776843256.txt` | None | 1 | None | None | 0 | 0 | 0 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/w2_optimized/log_rank0_1777003737.txt` | None | 4 | None | None | 0 | 0 | 5 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/w2_optimized_safe/log_rank0_1777007095.txt` | None | 39 | None | None | 0 | 0 | 19 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/w3_optimized/log_rank0_1776831237.txt` | None | 39 | None | None | 0 | 0 | 1 |
| `restore/scores/root/autodl-tmp/OmniQuant/log/w4_optimized/log_rank0_1776779629.txt` | None | 39 | None | None | 0 | 0 | 0 |
