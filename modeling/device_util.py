import torch


def get_device(device_arg: str | None = None) -> str:
    device = "cpu"
    if device_arg:
        device = device_arg
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    return device
