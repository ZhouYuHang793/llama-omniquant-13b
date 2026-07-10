from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="meta-llama/Llama-2-7b-hf",
    local_dir="/root/autodl-tmp/llama-2-7b-hf",
    local_dir_use_symlinks=False,
    resume_download=True,
    ignore_patterns=[
        "*.bin",
        "*.pth",
        "*.msgpack",
        "*.h5",
        "*.ot",
    ],
)

print(f"DOWNLOAD_FINISHED={path}")
