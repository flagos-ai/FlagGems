# AS_STRIDED_ operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.as_strided_ import as_strided_ as gems_as_strided_

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import gems_assert_close, TO_CPU
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    ref_inp = inp
    if TO_CPU:
        ref_inp = ref_inp.to("cpu")
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.as_strided_
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "size_stride",
    [
        ((2, 3), (3, 1)),
        ((4, 4), (5, 1)),
        ((8, 16, 4), (64, 4, 1)),
        ((128, 256), (256, 1)),
    ],
)
def test_as_strided__no_storage_offset(dtype, size_stride):
    size, stride = size_stride
    storage_offset = 0
    base_len = storage_offset + sum((n - 1) * s for n, s in zip(size, stride)) + 1
    base = torch.randn(int(base_len), dtype=dtype, device=flag_gems.device)

    ref_self = to_reference(base)
    act_self = base.clone()

    ref_out = torch.ops.aten.as_strided_(ref_self, size, stride)

    with flag_gems.use_gems():
        act_out = gems_as_strided_(act_self, size, stride)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.as_strided_
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "case",
    [
        ((3, 3), (0, 1), 10),
        ((8, 16, 4), (64, 4, 1), 7),
        ((64, 64, 16), (1024, 16, 1), 0),
        ((2, 5), (6, 1), 3),
    ],
)
def test_as_strided__with_storage_offset(dtype, case):
    size, stride, storage_offset = case
    base_len = (
        storage_offset + sum((n - 1) * s for n, s in zip(size, stride) if s >= 0) + 1
    )
    base = torch.randn(int(base_len), dtype=dtype, device=flag_gems.device)

    ref_self = to_reference(base)
    act_self = base.clone()

    ref_out = torch.ops.aten.as_strided_(ref_self, size, stride, storage_offset)

    with flag_gems.use_gems():
        act_out = gems_as_strided_(act_self, size, stride, storage_offset)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.as_strided_
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "size_stride",
    [
        ((2, 3), (3, 1)),
        ((4, 4), (5, 1)),
        ((8, 16, 4), (64, 4, 1)),
        ((128, 256), (256, 1)),
    ],
)
def test_as_strided__no_storage_offset_performance(dtype, size_stride):
    quantiles = [0.5, 0.2, 0.8]

    size, stride = size_stride
    storage_offset = 0
    base_len = storage_offset + sum((n - 1) * s for n, s in zip(size, stride)) + 1
    base = torch.randn(int(base_len), dtype=dtype, device=flag_gems.device)

    ref_self = base.clone()
    act_self = base.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.as_strided_(ref_self, size, stride),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_as_strided_(act_self, size, stride),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"as_strided_ {size} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.as_strided_
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "case",
    [
        ((3, 3), (0, 1), 10),
        ((8, 16, 4), (64, 4, 1), 7),
        ((64, 64, 16), (1024, 16, 1), 0),
        ((2, 5), (6, 1), 3),
    ],
)
def test_as_strided__with_storage_offset_performance(dtype, case):
    quantiles = [0.5, 0.2, 0.8]

    size, stride, storage_offset = case
    base_len = (
        storage_offset + sum((n - 1) * s for n, s in zip(size, stride) if s >= 0) + 1
    )
    base = torch.randn(int(base_len), dtype=dtype, device=flag_gems.device)

    ref_self = base.clone()
    act_self = base.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.as_strided_(ref_self, size, stride, storage_offset),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_as_strided_(act_self, size, stride, storage_offset),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"as_strided_ {size} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
