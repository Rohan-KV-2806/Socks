import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU:", props.name)
    print("VRAM:", round(props.total_memory / 1024**3, 2), "GB")
else:
    print(
        "CUDA is unavailable. Install a CUDA-enabled PyTorch build "
        "using the official PyTorch installation selector."
    )
