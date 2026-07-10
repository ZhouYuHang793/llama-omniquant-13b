# LLaMA-2 Backbone Quantization with OmniQuant

This repository contains the source code, experiment logs, result summaries, and submission evidence for a course project on multimodal large-model quantization.

The project reproduces the ICLR 2024 paper **OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models** on the LLaMA-2 language backbone. Although the course topic is multimodal quantization, LLaMA/LLaMA-2 is the core text backbone used by many LLaVA-style multimodal models. Quantizing this backbone is therefore a necessary and high-impact step before extending the same compression pipeline to image/video-language systems.

## Project Scope

Level-1 focuses on reproducing the main low-bit weight-only quantization behavior on LLaMA-2-13B-HF:

- W4A16-g128
- W3A16-g128
- W2A16-g128

Level-2 extends the reproduction with a robust low-bit calibration mode and ablation experiments:

- 13B W2 ablations for LWC / LET+LWC / no-LWC analysis
- 7B baseline-vs-robust comparisons for W4/W3/W2
- W2 group-size comparison between g128 and g64

## Main 13B Results

| Setting | WikiText2 PPL | C4 PPL | Notes |
|---|---:|---:|---|
| W4A16-g128 | 4.9647 | 6.5721 | Close to the OmniQuant paper result |
| W3A16-g128 | 5.3231 | 7.0206 | Close to the OmniQuant paper result |
| W2A16-g128 + LWC | 10.7503 | 13.2891 | Shows stronger low-bit sensitivity |
| W2A16-g128 + LET+LWC | 14.1274 | 17.7756 | Used for W2 mechanism analysis |
| W2A16-g128 no-LWC | 122.1143 | 139.6501 | Confirms the importance of LWC |

The W4/W3 results match the paper trend very closely. The W2 ablations show that 2-bit quantization is much more sensitive to calibration details, group-wise scaling, and learnable clipping parameters.

## Level-2 Robust Mode

The main code extension is implemented in `quantize/omniquant.py`, with command-line options added in `main.py`.

Robust mode keeps the original OmniQuant block-wise reconstruction objective, and adds engineering safeguards inside the per-layer quantization loop:

- NaN/Inf checks on loss and gradients
- layer-level parameter snapshot before optimization
- rollback when a non-finite update is detected
- adaptive learning-rate decay during retry
- FP32 fallback for layers that remain unstable under mixed precision
- `progress_file` recording for completed layers and checkpoint status

This makes the low-bit calibration process easier to complete, resume, and audit.

## Level-2 7B Ablation Results

| Setting | Exit | Completed Layers | WikiText2 PPL | C4 PPL | Rollback | FP32 Fallback |
|---|---:|---:|---:|---:|---:|---:|
| W4A16-g128 baseline | 1 | 2 / 32 | - | - | 0 | 0 |
| W4A16-g128 robust | 0 | 32 / 32 | 5.7242 | 7.2450 | 2 | 2 |
| W3A16-g128 baseline | 1 | 2 / 32 | - | - | 0 | 0 |
| W3A16-g128 robust | 0 | 32 / 32 | 6.6159 | 8.3735 | 1 | 1 |
| W2A16-g128 naive | 0 | 32 / 32 | 4272.6362 | 4907.0508 | 0 | 0 |
| W2A16-g128 robust | 0 | 32 / 32 | 1810.5115 | 4155.1655 | 3 | 3 |
| W2A16-g64 robust | 0 | 32 / 32 | 455.3946 | 667.1390 | 3 | 3 |

These results support the Level-2 conclusion: robust calibration improves process stability, while smaller group size further reduces W2 quantization error.

## Repository Structure

- `main.py`: model loading, quantization entry point, evaluation arguments, and robust-mode options
- `quantize/omniquant.py`: OmniQuant block-wise reconstruction and robust-mode implementation
- `scripts/`: original reproduction scripts
- `scripts/level2_robust/`: robust-mode patching and ablation scripts
- `final_artifacts/logs/`: main 13B evaluation logs
- `final_artifacts/recovered_logs/`: recovered 13B and early 7B logs used in the report
- `final_artifacts/level2_7b_ablation/`: 7B robust-mode ablation logs, commands, progress files, and result extracts
- `final_artifacts/evidence/`: `summary.csv`, `evidence.json`, and `evidence_index.md`
- `final_artifacts/environment/`: environment and Git information
- `final_artifacts/manifests/`: file manifest and omitted-large-file list
- `results/`: compact CSV result tables used by earlier report drafts
- `figures/`: generated report figures
- `docs/`: Chinese report-writing notes

## Reproduction Notes

Prepare the following local resources before running full experiments:

1. LLaMA-2 model weights, such as `LLaMA-2-13B-HF` or `LLaMA-2-7B-HF`.
2. WikiText2 and C4 datasets, or network/cache access for the Hugging Face dataset loader.
3. A CUDA environment with enough GPU memory for the chosen model size.

Example command shape:

```bash
python main.py \
  --model /path/to/LLaMA-2-7B-HF \
  --epochs 5 \
  --nsamples 32 \
  --wbits 4 \
  --abits 16 \
  --group_size 128 \
  --lwc \
  --robust_mode \
  --progress_file ./progress.json
```

Exact commands and logs for the submitted runs are preserved under `final_artifacts/level2_7b_ablation/`.

## What Is Not Included

The repository intentionally does not include:

- LLaMA model weights
- `omni_parameters.pth` quantized checkpoint files
- Hugging Face access tokens
- dataset cache files

Large checkpoint and cache files are documented in `final_artifacts/manifests/OMITTED_LARGE_FILES.txt`. They belong to the optional checkpoint package and are not required for checking the logs, code, and report evidence in this repository.
