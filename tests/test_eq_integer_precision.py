"""Regression tests for generic eq/eq_scalar/equal integer precision.

These tests require a supported accelerator and a working Triton backend.
"""

import pytest
import torch

import flag_gems
from flag_gems.ops.eq import eq, eq_scalar, equal


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_generic_eq_preserves_large_integer_ids(dtype, strided):
    values = [2**24, 2**24 + 1, 2**24 + 2]
    x_cpu = torch.tensor(values, dtype=dtype)
    y_cpu = torch.full_like(x_cpu, 2**24 + 1)
    x = x_cpu.to(flag_gems.device)
    y = y_cpu.to(flag_gems.device)
    if strided:
        storage = torch.empty(2 * x.numel(), dtype=dtype, device=flag_gems.device)
        storage[::2].copy_(x)
        x = storage[::2]

    torch.testing.assert_close(eq(x, y).cpu(), x_cpu.eq(y_cpu), rtol=0, atol=0)
    torch.testing.assert_close(
        eq_scalar(x, 2**24 + 1).cpu(), x_cpu.eq(2**24 + 1), rtol=0, atol=0
    )
    assert equal(x[:1], y[:1]) is False


@pytest.mark.parametrize("strided", [False, True])
def test_generic_eq_int64_around_float64_precision_limit(strided):
    base = 2**53
    x_cpu = torch.tensor([base, base + 1, base + 2], dtype=torch.int64)
    y_cpu = torch.full_like(x_cpu, base + 1)
    x = x_cpu.to(flag_gems.device)
    y = y_cpu.to(flag_gems.device)
    if strided:
        storage = torch.empty(2 * x.numel(), dtype=x.dtype, device=x.device)
        storage[::2].copy_(x)
        x = storage[::2]

    torch.testing.assert_close(eq(x, y).cpu(), x_cpu.eq(y_cpu), rtol=0, atol=0)
    torch.testing.assert_close(
        eq_scalar(x, base + 1).cpu(), x_cpu.eq(base + 1), rtol=0, atol=0
    )
    assert not equal(x[:1], y[:1])
