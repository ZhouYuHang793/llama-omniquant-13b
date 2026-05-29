# 实验报告写作指南

## 1. 项目主题

本项目研究 LLaMA-2-13B 语言主干模型的低比特量化。虽然课程题目是多模态大模型量化，但许多多模态大模型的主要参数量和显存开销来自语言模型 backbone，因此本项目选择 LLaMA-2-13B 作为核心研究对象。

## 2. 实验方法

本项目使用 OmniQuant 进行 post-training quantization。主要实验设置包括：

- W4A16-g128
- W3A16-g128
- W2A16-g128

其中 W4/W3/W2 表示权重量化位宽，A16 表示 activation 保持 16-bit，g128 表示 group size 为 128。

## 3. 主要实验结果

主结果见：

    results/ppl_results.csv

核心结果：

- W4A16-g128: WikiText2 PPL = 4.9647, C4 PPL = 6.5721
- W3A16-g128: WikiText2 PPL = 5.3231, C4 PPL = 7.0206
- W2A16-g128: WikiText2 PPL = 10.7501, C4 PPL = 13.2894

结论：随着权重量化位宽从 4-bit 降到 2-bit，PPL 明显升高，说明模型语言建模能力出现退化。尤其是从 3-bit 到 2-bit 时，性能下降最明显。

## 4. 训练过程分析

训练过程摘要见：

    results/training_summary.csv

可写结论：

- W4 训练最稳定，最终 calibration loss 最低。
- W3 loss 高于 W4，但仍然稳定。
- W2 初始优化设置出现 NaN，说明 2-bit 量化存在明显数值不稳定问题。
- 降低学习率到 5e-3 并延长到 40 epochs 后，W2 safe 版本能够稳定训练。

## 5. 失败案例分析

失败诊断见：

    results/failure_diagnostics.csv

W2 optimized initial 的 checkpoint 只包含 layers 0-3，因此 resume 到 layer 4 时出现 KeyError: 4。这个失败案例可以写成 2-bit 极限量化不稳定性的证据，而不是简单写成实验失败。

## 6. 推荐报告结构

1. Introduction
   - 多模态大模型部署瓶颈
   - 为什么研究 LLM backbone 量化
   - 项目目标

2. Background
   - Post-training quantization
   - OmniQuant
   - LWC and LET
   - PPL 指标

3. Experimental Setup
   - Model: LLaMA-2-13B-HF
   - Datasets: WikiText2 and C4
   - Quantization settings
   - Hardware and environment

4. Results
   - PPL 主结果表
   - Bit-width sensitivity analysis
   - Calibration loss and runtime analysis

5. Diagnostic Analysis
   - W2 optimized initial failure
   - NaN and incomplete checkpoint
   - Safer optimization strategy

6. Discussion
   - 为什么 2-bit 量化困难
   - 对多模态大模型部署的意义
   - 局限性

7. Conclusion

## 7. 可用图表

图表位于：

    figures/

推荐放入报告：

- ppl_vs_bitwidth.png
- calibration_loss_comparison.png
- runtime_comparison.png
