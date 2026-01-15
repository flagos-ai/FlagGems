# AS_STRIDED_COPY operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.as_strided_copy import (
    as_strided_copy as gems_as_strided_copy,
)
from flag_gems.experimental_ops.as_strided_copy import (
    as_strided_copy_out as gems_as_strided_copy_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    if TO_CPU:
        ref_inp = inp.to("cpu")
    else:
        ref_inp = inp.clone()
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.as_strided_copy
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["identity", "transpose", "slice"])
def test_as_strided_copy_tensor(shape, dtype, kind):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    act_inp = inp.clone()

    strides = list(ref_inp.stride())
    if kind == "identity":
        size = list(shape)
        stride = strides
        storage_offset = 0
    elif kind == "transpose":
        size = [shape[1], shape[0]]
        stride = [strides[1], strides[0]]
        storage_offset = 0
    else:
        i = 1 if shape[0] > 1 else 0
        j = 1 if shape[1] > 1 else 0
        h = max(1, min(3, shape[0] - i))
        w = max(1, min(3, shape[1] - j))
        size = [h, w]
        stride = strides
        storage_offset = i * strides[0] + j * strides[1]

    ref_out = torch.ops.aten.as_strided_copy(ref_inp, size, stride, storage_offset)
    with flag_gems.use_gems():
        act_out = gems_as_strided_copy(act_inp, size, stride, storage_offset)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.as_strided_copy
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["identity", "transpose", "slice"])
def test_as_strided_copy_out(shape, dtype, kind):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    act_inp = inp.clone()

    strides = list(ref_inp.stride())
    if kind == "identity":
        size = list(shape)
        stride = strides
        storage_offset = 0
    elif kind == "transpose":
        size = [shape[1], shape[0]]
        stride = [strides[1], strides[0]]
        storage_offset = 0
    else:
        i = 1 if shape[0] > 1 else 0
        j = 1 if shape[1] > 1 else 0
        h = max(1, min(3, shape[0] - i))
        w = max(1, min(3, shape[1] - j))
        size = [h, w]
        stride = strides
        storage_offset = i * strides[0] + j * strides[1]

    ref_out_buf = torch.empty(size, dtype=dtype, device=ref_inp.device)
    ref_out = torch.ops.aten.as_strided_copy.out(
        ref_inp, size, stride, storage_offset, out=ref_out_buf
    )

    act_out_buf = torch.empty(size, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        act_out = gems_as_strided_copy_out(
            act_inp, size, stride, storage_offset, act_out_buf
        )

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.as_strided_copy
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["identity", "transpose", "slice"])
def test_as_strided_copy_benchmark_tensor(shape, dtype, kind):
    quantiles = [0.5, 0.2, 0.8]

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = inp.clone()
    act_inp = inp.clone()

    strides = list(ref_inp.stride())
    if kind == "identity":
        size = list(shape)
        stride = strides
        storage_offset = 0
    elif kind == "transpose":
        size = [shape[1], shape[0]]
        stride = [strides[1], strides[0]]
        storage_offset = 0
    else:
        i = 1 if shape[0] > 1 else 0
        j = 1 if shape[1] > 1 else 0
        h = max(1, min(3, shape[0] - i))
        w = max(1, min(3, shape[1] - j))
        size = [h, w]
        stride = strides
        storage_offset = i * strides[0] + j * strides[1]

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.as_strided_copy(ref_inp, size, stride, storage_offset),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_as_strided_copy(act_inp, size, stride, storage_offset),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"as_strided_copy {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.as_strided_copy
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["identity", "transpose", "slice"])
def test_as_strided_copy_benchmark_out(shape, dtype, kind):
    quantiles = [0.5, 0.2, 0.8]

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = inp.clone()
    act_inp = inp.clone()

    strides = list(ref_inp.stride())
    if kind == "identity":
        size = list(shape)
        stride = strides
        storage_offset = 0
    elif kind == "transpose":
        size = [shape[1], shape[0]]
        stride = [strides[1], strides[0]]
        storage_offset = 0
    else:
        i = 1 if shape[0] > 1 else 0
        j = 1 if shape[1] > 1 else 0
        h = max(1, min(3, shape[0] - i))
        w = max(1, min(3, shape[1] - j))
        size = [h, w]
        stride = strides
        storage_offset = i * strides[0] + j * strides[1]

    ref_out_buf = torch.empty(size, dtype=dtype, device=flag_gems.device)
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.as_strided_copy.out(
            ref_inp, size, stride, storage_offset, out=ref_out_buf
        ),
        rep=100,
        quantiles=quantiles,
    )

    act_out_buf = torch.empty(size, dtype=dtype, device=flag_gems.device)

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_as_strided_copy_out(
                act_inp, size, stride, storage_offset, act_out_buf
            ),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"as_strided_copy {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
