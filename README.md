# LLaMA-2-13B Low-Bit Quantization with OmniQuant

This repository contains a course project on low-bit post-training quantization of the LLaMA-2-13B language backbone using OmniQuant.

Although the original course topic is multimodal large model quantization, this project focuses on the LLM backbone, which is the major parameter and memory bottleneck in many multimodal systems such as LLaVA-style architectures.

## Project Goal

The goal is to evaluate how different weight bit-widths affect the performance and stability of LLaMA-2-13B quantization.

The main research questions are:

1. How does perplexity change from 4-bit to 3-bit and 2-bit weight quantization?
2. Does extreme 2-bit quantization introduce numerical instability?
3. How can failed low-bit experiments be diagnosed and reported?

## Method

The project is based on OmniQuant, a post-training quantization method for large language models.

Main settings:

- W4A16-g128
- W3A16-g128
- W2A16-g128

Here, W4/W3/W2 indicate 4-bit, 3-bit, and 2-bit weight quantization. A16 indicates 16-bit activations. g128 indicates group size 128.

## Experimental Setup

- Model: LLaMA-2-13B-HF
- Quantization method: OmniQuant
- Evaluation datasets:
  - WikiText2
  - C4
- Calibration samples: 128
- Batch size: 1
- Group size: 128
- Metric: Perplexity, PPL

The original model weights and quantized checkpoints are not included due to license restrictions and file size.

## Main Results

| Setting | WikiText2 PPL | C4 PPL | Status |
|---|---:|---:|---|
| W4A16-g128 | 4.9647 | 6.5721 | Success |
| W3A16-g128 | 5.3231 | 7.0206 | Success |
| W2A16-g128 | 10.7501 | 13.2894 | Success |
| W2 optimized initial | N/A | N/A | Failed checkpoint |

The results show that perplexity increases as the weight bit-width decreases. The degradation is especially significant when moving from 3-bit to 2-bit quantization.

## Diagnostic Finding: 2-bit Instability

An additional W2 optimized experiment was attempted with more aggressive optimization settings. However, the saved checkpoint only contained layers 0 to 3, and loading failed at layer 4 with KeyError: 4.

This failed run is treated as a diagnostic case showing that extreme 2-bit quantization can suffer from numerical instability and incomplete optimization.

## Repository Structure

- results/ppl_results.csv: Main PPL results
- results/training_summary.csv: Training loss and runtime summary
- results/failure_diagnostics.csv: Failure and debugging records
- final_artifacts/logs/: Raw PPL evaluation logs
- final_artifacts/recovered_logs/: Recovered historical logs
- figures/: Generated figures for the report
- scripts/: Evaluation, extraction, and plotting scripts
- docs/report_guide_zh.md: Chinese report writing guide
- env/: Environment information

## Reproduction Notes

Before running the scripts, prepare:

1. The original LLaMA-2-13B-HF model directory.
2. OmniQuant checkpoints such as omni_parameters.pth.
3. Cached WikiText2 and C4 test loaders, or allow the code to regenerate them.

Example:

    bash scripts/run_eval_w4.sh

Please modify the model path and checkpoint path in the scripts before running.

## What Is Not Included

This repository does not include:

- LLaMA-2-13B model weights
- Quantized .pth checkpoint files
- HuggingFace tokens
- Dataset cache files
