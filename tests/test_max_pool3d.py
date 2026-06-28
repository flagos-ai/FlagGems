import torch
import torch.nn.functional as F

from flag_gems.ops.max_pool3d import (
    max_pool3d,
)


def test_max_pool3d_basic():

    x = torch.randn(
        (2, 3, 8, 16, 16),
        device="cuda",
        dtype=torch.float32
    )

    custom_output = max_pool3d(
        x,
        kernel_size=2,
        stride=2
    )

    torch_output = F.max_pool3d(
        x,
        kernel_size=2,
        stride=2
    )

    assert torch.allclose(
        custom_output,
        torch_output,
        atol=1e-6,
        rtol=1e-4
    )


def test_max_pool3d_padding():

    x = torch.randn(
        (1, 2, 10, 10, 10),
        device="cuda",
        dtype=torch.float32
    )

    custom_output = max_pool3d(
        x,
        kernel_size=3,
        stride=2,
        padding=1
    )

    torch_output = F.max_pool3d(
        x,
        kernel_size=3,
        stride=2,
        padding=1
    )

    assert torch.allclose(
        custom_output,
        torch_output,
        atol=1e-6,
        rtol=1e-4
    )


def test_max_pool3d_fp16():

    x = torch.randn(
        (1, 2, 8, 8, 8),
        device="cuda",
        dtype=torch.float16
    )

    custom_output = max_pool3d(
        x,
        kernel_size=2
    )

    torch_output = F.max_pool3d(
        x,
        kernel_size=2
    )

    assert torch.allclose(
        custom_output,
        torch_output,
        atol=1e-3,
        rtol=1e-3
    )
