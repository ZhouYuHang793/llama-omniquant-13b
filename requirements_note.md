# Environment Notes

The experiments were conducted in a Conda environment named omniquant.

Important packages included:

- Python 3.10
- PyTorch
- Transformers
- HuggingFace Hub
- datasets
- accelerate
- sentencepiece
- numpy
- tqdm

During reproduction, if torch.load() raises a weights_only error under PyTorch >= 2.6, the testloader loading line in main.py may need:

    torch.load(cache_testloader, weights_only=False)

This is required because cached testloaders may contain HuggingFace BatchEncoding objects rather than pure tensor weights.
