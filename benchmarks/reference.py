"""Optional PyTorch baselines; never a TensorSmith runtime dependency."""


def torch_backend(device):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "--torch needs PyTorch in the benchmark environment: pip install torch"
        ) from error
    selected = torch.device("mps" if device.type == "metal" else str(device))
    if selected.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is not available; cannot compare Metal to CPU fallback")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available")

    def complete(_device, _values):
        if selected.type == "mps":
            torch.mps.synchronize()
        elif selected.type == "cuda":
            torch.cuda.synchronize(selected)

    return torch, selected, complete
