# 基于 OmniQuant 的 LLaMA-2 文本基座量化复现与改进

课程项目：多模态大模型量化  
复现论文：**OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models** (ICLR 2024)  
项目成员：周昱航、季丽莎

本仓库整理了课程项目的可运行源码、复现实验脚本、实验日志、结果表、环境信息和 evidence 文件。项目以 OmniQuant 为基础，在 LLaMA-2-13B-HF 上复现低比特 weight-only 量化，并在复现基础上实现 `robust mode`，用于改善极低比特逐层校准过程的稳定性。

虽然课程方向是多模态大模型量化，本项目首先选择 LLaMA-2 backbone 作为实验对象。原因是 LLaVA 等主流多模态大模型通常将视觉编码器输出映射到语言模型 token embedding 空间，再由 LLaMA/LLaMA-2 文本基座完成跨模态推理与文本生成。因此，稳定复现并改进 LLaMA-2 backbone 的量化，是进一步扩展到图文、视频多模态模型量化的核心基础。

## 作业要求对照

| 作业要求 | 本仓库对应内容 |
|---|---|
| 选择 2024-2026 年顶会大模型量化论文复现 | 复现 ICLR 2024 OmniQuant |
| 完整复现核心算法与实验设置 | `main.py`、`quantize/omniquant.py`、`scripts/Llama-2/`、`final_artifacts/logs/` |
| 在公开模型上验证结果 | LLaMA-2-13B-HF 主复现，LLaMA-2-7B-HF 低成本消融验证 |
| Level-1 主复现 | 13B W4A16-g128、W3A16-g128、W2A16-g128 |
| Level-2 改进或扩展 | 实现 `robust mode`，补充 W2 模块消融、baseline/robust 对照和 group size 消融 |
| 结果对比与误差分析 | `final_artifacts/evidence/summary.csv`、`results/*.csv`、实验报告 |
| 可运行源码 | `main.py`、`quantize/`、`models/`、`lm_eval/`、`scripts/` |
| 代码、PPT、录制视频提交 | 本仓库用于代码与证据；PPT、视频和报告作为课程提交材料单独提交 |
| 双人分工 | README 末尾给出分工说明 |

## 一句话结论

Level-1 中，W4/W3 结果与 OmniQuant 原论文高度接近，说明复现流程可信；W2 对 LWC、group size 和逐层优化稳定性更敏感。Level-2 中，`robust mode` 将 NaN/Inf 检查、rollback、学习率衰减、FP32 fallback 和 progress file 加入逐层量化循环，使低比特校准过程更稳定、更容易复核。

## 第一部分：13B 主复现结果（Level-1）

结果来自 `final_artifacts/evidence/evidence_index.md` 与 `final_artifacts/evidence/summary.csv`，报告中也采用同一口径。

| 设置 | WikiText2 PPL ↓ | C4 PPL ↓ | 说明 |
|---|---:|---:|---|
| W4A16-g128 | 4.9648 | 6.5721 | 与原论文 W4 结果非常接近 |
| W3A16-g128 | 5.3232 | 7.0206 | 与原论文 W3 结果非常接近 |
| W2A16-g128 + LWC | 10.7503 | 13.2891 | 2-bit 重点分析设置 |

W4A16-g128 和 W3A16-g128 在 WikiText2 上与论文报告值的误差分别约为 `+0.30%` 和 `+0.82%`。这说明模型加载、校准配置、量化参数保存和 PPL 评估流程基本复现成功。

## 13B W2 模块消融与问题定位

W2 是本项目中最敏感的设置，因此我们进一步比较 LWC、LET+LWC 与 no-LWC。

| 13B W2 设置 | WikiText2 PPL ↓ | C4 PPL ↓ | 结论 |
|---|---:|---:|---|
| LWC | 10.7503 | 13.2891 | 保持 2-bit 可用性的关键模块 |
| LET+LWC | 14.1274 | 17.7756 | 用于分析 LET 与 LWC 组合影响 |
| no-LWC | 122.1143 | 139.6501 | 去掉 LWC 后 PPL 大幅上升 |

该消融说明，2-bit weight-only 量化不仅受位宽限制，还强烈依赖 learnable weight clipping、group-wise scale 以及逐层校准过程的数值稳定性。no-LWC 相比 LWC 的 WikiText2 PPL 约为 `11.36x`，因此 LWC 是 W2 量化中最关键的组件之一。

## 第二部分：Robust Mode 稳定性改进（Level-2）

`robust mode` 是本项目在复现基础上的主要代码改进。它保留 OmniQuant 原有 block-wise reconstruction 目标，在逐层优化循环中加入稳定性与可恢复机制。

主要实现位置：

- `main.py`: 新增 `--robust_mode`、`--grad_clip`、`--max_retries`、`--nan_lr_decay`、`--min_lr`、`--progress_file` 等命令行参数。
- `quantize/omniquant.py`: 实现参数快照、参数恢复、NaN/Inf 检查、rollback、学习率衰减、FP32 fallback 和进度记录。
- `scripts/level2_robust/`: 保存 robust mode 相关实验脚本和补丁脚本。

核心流程：

1. 每层优化前保存当前层量化参数快照。
2. 检查 loss 和 gradient 是否出现 NaN/Inf。
3. 若出现异常更新，则 rollback 到当前层快照。
4. 降低 LWC/LET 学习率后重新尝试。
5. 若混合精度路径仍不稳定，则切换 FP32 fallback。
6. 使用 `progress_file` 记录已完成层，便于中断恢复和结果核对。

## 第三部分：7B 消融验证（Level-2）

7B 实验用于低成本验证稳定性趋势和改进方向，不直接替代 13B 主复现结果。

| 设置 | 退出码 | 完整完成层数 | WikiText2 PPL ↓ | C4 PPL ↓ | 回滚次数 | FP32 fallback 次数 |
|---|---:|---:|---:|---:|---:|---:|
| W4A16-g128 baseline | 1 | 1 / 32 | - | - | 0 | 0 |
| W4A16-g128 robust | 0 | 32 / 32 | 5.7242 | 7.2450 | 2 | 2 |
| W3A16-g128 baseline | 1 | 1 / 32 | - | - | 0 | 0 |
| W3A16-g128 robust | 0 | 32 / 32 | 6.6159 | 8.3735 | 1 | 1 |
| W2A16-g128 naive, epochs=0 | 0 | 32 / 32 | 4272.6362 | 4907.0508 | 0 | 0 |
| W2A16-g128 robust | 0 | 32 / 32 | 1810.5115 | 4155.1655 | 3 | 3 |
| W2A16-g64 robust | 0 | 32 / 32 | 455.3946 | 667.1390 | 3 | 3 |

说明：

- baseline 在 layer 1 优化过程中中断，仅完整完成 layer 0，因此记为 `1 / 32`。
- naive 是 `epochs=0` 的参照，不进行 LWC 优化；robust-g128 是 `epochs=10`，同时包含 LWC 优化与 robust mode。
- 因此，naive 到 robust-g128 的 PPL 下降应理解为“经过 LWC 优化并配合 robust mode 后的整体改善”，不能全部归因于 rollback 或 FP32 fallback。
- robust-g64 相比 robust-g128 进一步改善 W2，说明更细粒度 group size 有助于降低极低比特量化误差。

## 仓库结构

```text
.
├── main.py                         # 模型加载、量化入口、PPL 评估和 robust 参数
├── quantize/
│   └── omniquant.py                # OmniQuant 逐层重构与 robust mode 主要实现
├── models/                         # LLaMA/OPT/Falcon 等模型封装
├── lm_eval/                        # 语言模型评估工具
├── scripts/
│   ├── Llama-2/                    # LLaMA-2 主复现实验脚本
│   └── level2_robust/              # Level-2 robust 实验与补丁脚本
├── final_artifacts/
│   ├── logs/                       # 13B 主复现评估日志
│   ├── recovered_logs/             # 13B 消融和历史实验日志
│   ├── level2_7b_ablation/         # 7B baseline/robust/g64 消融日志与命令
│   ├── evidence/                   # summary.csv、evidence.json、evidence_index.md
│   ├── environment/                # 环境信息和 Git diff
│   └── manifests/                  # 文件清单与大文件省略说明
├── results/                        # 轻量 CSV 结果表
├── figures/                        # 报告绘图素材
├── docs/                           # 报告撰写说明
└── requirements_note.md            # 环境与 PyTorch 版本注意事项
```

## 快速复核路径

老师或助教如果只想快速检查，可以按下面顺序看：

1. `README.md`: 项目主线、结果表和作业要求对照。
2. `main.py` 与 `quantize/omniquant.py`: 复现代码与 robust mode 改进。
3. `final_artifacts/evidence/evidence_index.md`: 主结果和自动解析日志索引。
4. `final_artifacts/evidence/summary.csv`: 所有日志解析出的 PPL、完成层数、rollback、FP32 fallback 等字段。
5. `final_artifacts/logs/`: 13B W4/W3/W2 主复现日志。
6. `final_artifacts/level2_7b_ablation/`: 7B robust mode 消融日志、命令和 progress 文件。
7. `final_artifacts/manifests/OMITTED_LARGE_FILES.txt`: 未随主仓库提交的大 checkpoint 和缓存文件说明。

## 运行环境

主要实验在 CUDA + PyTorch + Hugging Face 环境中完成。环境摘要见：

- `requirements_note.md`
- `final_artifacts/environment/environment_full.txt`
- `final_artifacts/environment/git_info.txt`

由于 LLaMA-2 权重需要单独授权下载，本仓库不包含模型权重和 Hugging Face token。复现实验前需要准备：

1. LLaMA-2-13B-HF 或 LLaMA-2-7B-HF 权重。
2. WikiText2 与 C4 数据集缓存，或可访问 Hugging Face datasets。
3. 支持 CUDA 的 PyTorch 环境。
4. 足够的 GPU 显存；13B 完整实验开销明显高于 7B 消融。

## 示例命令

下面是 7B robust W2 消融的命令形态。正式提交实验的完整脚本和日志保存在 `final_artifacts/level2_7b_ablation/`。

```bash
python -u main.py \
  --model /path/to/LLaMA-2-7B-HF \
  --cache_dir /path/to/cache \
  --output_dir ./experiments/7b_w2g64_robust \
  --calib_dataset wikitext2 \
  --nsamples 32 \
  --batch_size 1 \
  --seed 2 \
  --wbits 2 \
  --abits 16 \
  --group_size 64 \
  --epochs 10 \
  --lwc \
  --lwc_lr 0.001 \
  --eval_ppl \
  --robust_mode \
  --grad_clip 1.0 \
  --max_retries 3 \
  --nan_lr_decay 0.5 \
  --min_lr 1e-6 \
  --progress_file ./experiments/7b_w2g64_robust/quant_progress.json
```

如使用 PyTorch 2.6 或更高版本，加载缓存 testloader 时可能需要在 `torch.load(...)` 中显式设置 `weights_only=False`。具体说明见 `requirements_note.md`。

## 未包含的大文件

主仓库不包含以下文件：

- LLaMA/LLaMA-2 原始模型权重。
- Hugging Face token 或个人缓存。
- 数据集缓存文件。
- `omni_parameters.pth` 等大 checkpoint 文件。

这些大文件不影响检查源码、日志和结果证据。已省略文件清单见 `final_artifacts/manifests/OMITTED_LARGE_FILES.txt`。如果课程检查需要 checkpoint，可使用单独的 optional checkpoint 包。

## 分工说明

- 周昱航：OmniQuant 13B 主复现、W4/W3/W2 PPL 评估、部分日志整理与结果核对。
- 季丽莎：W2 模块消融、Level-2 robust mode 实现与 7B 消融验证、报告与展示材料整理。
- 共同完成：论文阅读、实验结果分析、误差分析、仓库 evidence 整理和最终报告撰写。

## 参考文献

```bibtex
@inproceedings{shao2024omniquant,
  title={OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models},
  author={Shao, Wenqi and Chen, Mengzhao and Zhang, Zhaoyang and Xu, Peng and Zhao, Lirui and Li, Zhiqian and Zhang, Kaipeng and Gao, Peng and Qiao, Yu and Luo, Ping},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
