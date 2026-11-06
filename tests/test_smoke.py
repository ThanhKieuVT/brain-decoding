"""Kiểm tra môi trường T00: các gói chính nạp được và tính toán chạy trên CPU."""

import importlib

import pytest
import torch

PACKAGES = ["common", "bench", "brain", "brain.data", "brain.decoders", "gaze", "gaze.data"]


@pytest.mark.parametrize("name", PACKAGES)
def test_repo_package_imports(name: str) -> None:
    assert importlib.import_module(name) is not None


def test_torch_cpu_matmul() -> None:
    a = torch.eye(3)
    b = torch.arange(9.0).reshape(3, 3)
    assert torch.equal(a @ b, b)
