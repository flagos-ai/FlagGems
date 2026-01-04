# _RESHAPE_ALIAS operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops._reshape_alias import (
    _reshape_alias as gems__reshape_alias,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
try:
    from tests.accuracy_utils import gems_assert_close  # noqa: E402
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


@pytest.mark.reshape_alias
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (256, 256), (8, 16, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["identity", "flatten", "reshape2d", "permute"])
def test__reshape_alias_tensor(shape, dtype, case):
    def test_contiguous_strides_for(sz):
        if len(sz) == 0:
            return []
        st = [0] * len(sz)
        acc = 1
        for i in range(len(sz) - 1, -1, -1):
            st[i] = acc
            acc *= sz[i]
        return st

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    orig_size = list(x.size())
    orig_stride = list(x.stride())
    numel = x.numel()

    if case == "identity":
        new_size = orig_size
        new_stride = orig_stride
    elif case == "flatten":
        new_size = [numel]
        new_stride = [1]
    elif case == "reshape2d":
        first = orig_size[0] if len(orig_size) > 0 else 1
        first = max(first, 1)
        new_size = [first, numel // first]
        new_stride = test_contiguous_strides_for(new_size)
    elif case == "permute":
        if len(orig_size) <= 1:
            new_size = orig_size
            new_stride = orig_stride
        else:
            perm = list(range(len(orig_size)))[::-1]
            new_size = [orig_size[i] for i in perm]
            new_stride = [orig_stride[i] for i in perm]
    else:
        raise ValueError("unknown case")

    ref_out = torch.ops.aten._reshape_alias(ref_x, new_size, new_stride)
    with flag_gems.use_gems():
        act_out = gems__reshape_alias(x, new_size, new_stride)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.reshape_alias
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (256, 256), (8, 16, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["identity", "flatten", "reshape2d", "permute"])
def test__reshape_alias_benchmark_tensor(shape, dtype, case):
    quantiles = [0.5, 0.2, 0.8]

    def test_contiguous_strides_for(sz):
        if len(sz) == 0:
            return []
        st = [0] * len(sz)
        acc = 1
        for i in range(len(sz) - 1, -1, -1):
            st[i] = acc
            acc *= sz[i]
        return st

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    orig_size = list(x.size())
    orig_stride = list(x.stride())
    numel = x.numel()

    if case == "identity":
        new_size = orig_size
        new_stride = orig_stride
    elif case == "flatten":
        new_size = [numel]
        new_stride = [1]
    elif case == "reshape2d":
        first = orig_size[0] if len(orig_size) > 0 else 1
        first = max(first, 1)
        new_size = [first, numel // first]
        new_stride = test_contiguous_strides_for(new_size)
    elif case == "permute":
        if len(orig_size) <= 1:
            new_size = orig_size
            new_stride = orig_stride
        else:
            perm = list(range(len(orig_size)))[::-1]
            new_size = [orig_size[i] for i in perm]
            new_stride = [orig_stride[i] for i in perm]
    else:
        raise ValueError("unknown case")

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten._reshape_alias(ref_x, new_size, new_stride),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems__reshape_alias(x, new_size, new_stride),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"_reshape_alias {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
