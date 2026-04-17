import torch
import os

def get_device():
    """Returns the best available device: cuda, mps, or cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_compute_type(device=None):
    """Returns the recommended compute type for the given device."""
    if device is None:
        device = get_device()
    
    if device == "cuda":
        return "float16"
    else:
        # MPS and CPU often perform better or are more stable with float32/int8
        # WhisperX specifically often needs float32 on CPU/MPS for certain ops
        return "float32"
