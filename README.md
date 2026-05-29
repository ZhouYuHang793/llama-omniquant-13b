# LLaMA-2-13B Low-Bit Quantization with OmniQuant

This repository contains a course project on low-bit post-training quantization of the LLaMA-2-13B language backbone using OmniQuant.

Although the original course topic is multimodal large model quantization, this project focuses on the LLM backbone, which is the major parameter and memory bottleneck in many multimodal systems such as LLaVA-style architectures.

## Method

The project is based on OmniQuant, a post-training quantization method for large language models. The experiments mainly evaluate weight-only low-bit quantization settings:

- W4A16-g128
- W3A16-g128
- W2A16-g128

Here, W4/W3/W2 indicate 4-bit, 3-bit, and 2-bit weight quantization, while A16 indicates 16-bit activations.

## Experimental Setup

- Model: LLaMA-2-13B-HF
- Quantization method: OmniQuant
- Evaluation datasets:
  - WikiText2
  - C4
- Calibration samples: 128
- Batch size: 1
- Group size: 128
- Evaluation metric: perplexity, PPL

The original model weights and quantized checkpoints are not included in this repository due to license restrictions and file size.

## Main Results

| Setting | WikiText2 PPL | C4 PPL | Status |
|---|---:|---:|---|
| W4A16-g128 | 4.9647 | 6.5721 | Success |
| W3A16-g128 | 5.3231 | 7.0206 | Success |
| W2A16-g128 | 10.7501 | 13.2894 | Success |
| W2 optimized initial | N/A | N/A | Failed checkpoint |

The results show that perplexity increases as the weight bit-width decreases. The performance degradation is especially significant when moving from 3-bit to 2-bit quantization.

## Diagnostic Finding: 2-bit Instability

An additional W2 optimized experiment was attempted with more aggressive optimization settings. However, the saved checkpoint only contained layers 0 to 3, and loading failed at layer 4 with `KeyError: 4`.

This failed run is treated as a diagnostic case showing that extreme 2-bit quantization can suffer from numerical instability and incomplete optimization.

## Repository Structure

```text
.
├── README.md
├── results/
│   └── ppl_results.csv
├── final_artifacts/
│   ├── logs/
│   └── results/
├── scripts/
│   ├── run_eval_w2.sh
│   ├── run_eval_w3.sh
│   └── run_eval_w4.sh
├── quantize/
├── main.py
└── datautils.py
cd /root/autodl-tmp/llama-omniquant-13b

cat > README.md <<'MD'
# LLaMA-2-13B Low-Bit Quantization with OmniQuant

This repository contains a course project on low-bit post-training quantization of the LLaMA-2-13B language backbone using OmniQuant.

Although the original course topic is multimodal large model quantization, this project focuses on the LLM backbone, which is the major parameter and memory bottleneck in many multimodal systems such as LLaVA-style architectures.

## Method

The project is based on OmniQuant, a post-training quantization method for large language models. The experiments mainly evaluate weight-only low-bit quantization settings:

- W4A16-g128
- W3A16-g128
- W2A16-g128

Here, W4/W3/W2 indicate 4-bit, 3-bit, and 2-bit weight quantization, while A16 indicates 16-bit activations.

## Experimental Setup

- Model: LLaMA-2-13B-HF
- Quantization method: OmniQuant
- Evaluation datasets:
  - WikiText2
  - C4
- Calibration samples: 128
- Batch size: 1
- Group size: 128
- Evaluation metric: perplexity, PPL

The original model weights and quantized checkpoints are not included in this repository due to license restrictions and file size.

## Main Results

| Setting | WikiText2 PPL | C4 PPL | Status |
|---|---:|---:|---|
| W4A16-g128 | 4.9647 | 6.5721 | Success |
| W3A16-g128 | 5.3231 | 7.0206 | Success |
| W2A16-g128 | 10.7501 | 13.2894 | Success |
| W2 optimized initial | N/A | N/A | Failed checkpoint |

The results show that perplexity increases as the weight bit-width decreases. The performance degradation is especially significant when moving from 3-bit to 2-bit quantization.

## Diagnostic Finding: 2-bit Instability

An additional W2 optimized experiment was attempted with more aggressive optimization settings. However, the saved checkpoint only contained layers 0 to 3, and loading failed at layer 4 with KeyError: 4.

This failed run is treated as a diagnostic case showing that extreme 2-bit quantization can suffer from numerical instability and incomplete optimization.

## Repository Structure

    .
    ├── README.md
    ├── results/
    │   └── ppl_results.csv
    ├── final_artifacts/
    │   ├── logs/
    │   └── results/
    ├── scripts/
    │   ├── run_eval_w2.sh
    │   ├── run_eval_w3.sh
    │   └── run_eval_w4.sh
    ├── quantize/
    ├── main.py
    └── datautils.py

## Reproduction Notes

Before running the scripts, prepare:

1. The original LLaMA-2-13B-HF model directory.
2. OmniQuant checkpoints such as omni_parameters.pth.
3. Cached WikiText2 and C4 test loaders, or allow the code to regenerate them.

Example command:

    bash scripts/run_eval_w4.sh

Please modify /path/to/llama-2-13b-hf and /path/to/.../omni_parameters.pth in the scripts before running.

## Notes

This repository does not include:

- LLaMA-2-13B model weights
- Quantized .pth checkpoint files
- HuggingFace tokens
- Dataset cache files
