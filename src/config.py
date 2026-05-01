from pathlib import Path

import torch

ROOT_DIR = Path(__file__).parent.parent

DATA_DIR = ROOT_DIR / "data"

MODEL_DIR = ROOT_DIR / "model"


def get_compute_device():
    """自动检测并返回最佳可用的计算设备：MPS (Apple Silicon) > CUDA (NVIDIA GPU) > CPU"""
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE: torch.device = get_compute_device()
# DEVICE: torch.device = torch.device("cpu")
