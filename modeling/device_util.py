import torch


def get_device(device_arg: str | None = None) -> tuple[str, str]:
    device = "cpu"
    if device_arg:
        device = device_arg
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return device, device_type
