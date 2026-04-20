import sys
import types
from unittest.mock import patch

from src.utils import get_compute_type, get_device


def test_get_compute_type_cuda():
    assert get_compute_type("cuda") == "float16"


def test_get_compute_type_mps():
    assert get_compute_type("mps") == "float32"


def test_get_compute_type_cpu():
    assert get_compute_type("cpu") == "float32"


def test_get_compute_type_default_uses_get_device():
    with patch("src.utils.get_device", return_value="cuda"):
        assert get_compute_type() == "float16"


def _make_fake_torch(cuda_available, mps_available):
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    fake.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps_available)
    )
    return fake


def test_get_device_prefers_cuda_when_available():
    fake = _make_fake_torch(cuda_available=True, mps_available=False)
    with patch.dict(sys.modules, {"torch": fake}):
        assert get_device() == "cuda"


def test_get_device_falls_back_to_cpu():
    fake = _make_fake_torch(cuda_available=False, mps_available=False)
    with patch.dict(sys.modules, {"torch": fake}):
        assert get_device() == "cpu"
