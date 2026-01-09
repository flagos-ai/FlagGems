# AS_STRIDED operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.as_strided import as_strided as gems_as_strided

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


@pytest.mark.as_strided
@pytest.mark.parametrize("base_shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "case", ["contig", "transpose", "subsample", "offset_cropped", "reshape_like"]
)
def test_as_strided_2d(base_shape, dtype, case):
    x = torch.randn(base_shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)

    s0, s1 = ref_x.stride()
    h, w = base_shape
    total = ref_x.numel()

    if case == "contig":
        size = [h, w]
        stride = [s0, s1]
        storage_offset = 0
    elif case == "transpose":
        size = [w, h]
        stride = [s1, s0]
        storage_offset = 0
    elif case == "subsample":
        step0 = 2 if h >= 2 else 1
        step1 = 2 if w >= 2 else 1
        size = [max(h // step0, 1), max(w // step1, 1)]
        stride = [s0 * step0, s1 * step1]
        storage_offset = 0
    elif case == "offset_cropped":
        off = (s0 if h > 1 else 0) + (s1 if w > 1 else 0)
        size = [max(h - 1, 1), max(w - 1, 1)]
        stride = [s0, s1]
        storage_offset = off if off > 0 else 0
    elif case == "reshape_like":
        if total % 4 == 0:
            k = 4
        elif total % 2 == 0:
            k = 2
        else:
            k = 1
        size = [total // k, k]
        stride = [k, 1]
        storage_offset = 0
    else:
        raise AssertionError("unknown case")

    ref_out = torch.ops.aten.as_strided(ref_x, size, stride, storage_offset)

    with flag_gems.use_gems():
        act_out = gems_as_strided(x, size, stride, storage_offset)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.as_strided
@pytest.mark.parametrize("base_shape", [(6,), (64,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["contig", "skip_step", "offset_start", "reshape2d"])
def test_as_strided_1d(base_shape, dtype, case):
    x = torch.randn(base_shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)

    (N,) = base_shape
    (s,) = ref_x.stride()
    total = N

    if case == "contig":
        size = [N]
        stride = [s]
        storage_offset = 0
    elif case == "skip_step":
        step = 2 if N >= 2 else 1
        size = [max(N // step, 1)]
        stride = [s * step]
        storage_offset = 0
    elif case == "offset_start":
        off = 1 if N > 1 else 0
        size = [max(N - off, 1)]
        stride = [s]
        storage_offset = off
    elif case == "reshape2d":
        if total % 4 == 0:
            k = 4
        elif total % 2 == 0:
            k = 2
        else:
            k = 1
        size = [total // k, k]
        stride = [k, 1]
        storage_offset = 0
    else:
        raise AssertionError("unknown case")

    ref_out = torch.ops.aten.as_strided(ref_x, size, stride, storage_offset)

    with flag_gems.use_gems():
        act_out = gems_as_strided(x, size, stride, storage_offset)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.as_strided
@pytest.mark.parametrize("base_shape", [(2, 3, 4), (8, 16, 8), (16, 32, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "case", ["contig", "swap_last_two", "subsample", "offset_plane"]
)
def test_as_strided_3d(base_shape, dtype, case):
    x = torch.randn(base_shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)

    d0, d1, d2 = base_shape
    s0, s1, s2 = ref_x.stride()

    if case == "contig":
        size = [d0, d1, d2]
        stride = [s0, s1, s2]
        storage_offset = 0
    elif case == "swap_last_two":
        size = [d0, d2, d1]
        stride = [s0, s2, s1]
        storage_offset = 0
    elif case == "subsample":
        step0 = 2 if d0 >= 2 else 1
        step1 = 2 if d1 >= 2 else 1
        step2 = 1
        size = [max(d0 // step0, 1), max(d1 // step1, 1), max(d2 // step2, 1)]
        stride = [s0 * step0, s1 * step1, s2 * step2]
        storage_offset = 0
    elif case == "offset_plane":
        off = (s0 if d0 > 1 else 0) + (s2 if d2 > 1 else 0)
        size = [max(d0 - 1, 1), d1, max(d2 - 1, 1)]
        stride = [s0, s1, s2]
        storage_offset = off if off > 0 else 0
    else:
        raise AssertionError("unknown case")

    ref_out = torch.ops.aten.as_strided(ref_x, size, stride, storage_offset)

    with flag_gems.use_gems():
        act_out = gems_as_strided(x, size, stride, storage_offset)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.as_strided
@pytest.mark.parametrize("base_shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "case", ["contig", "transpose", "subsample", "offset_cropped", "reshape_like"]
)
def test_as_strided_2d_performance(base_shape, dtype, case):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(base_shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    s0, s1 = ref_x.stride()
    h, w = base_shape
    total = ref_x.numel()

    if case == "contig":
        size = [h, w]
        stride = [s0, s1]
        storage_offset = 0
    elif case == "transpose":
        size = [w, h]
        stride = [s1, s0]
        storage_offset = 0
    elif case == "subsample":
        step0 = 2 if h >= 2 else 1
        step1 = 2 if w >= 2 else 1
        size = [max(h // step0, 1), max(w // step1, 1)]
        stride = [s0 * step0, s1 * step1]
        storage_offset = 0
    elif case == "offset_cropped":
        off = (s0 if h > 1 else 0) + (s1 if w > 1 else 0)
        size = [max(h - 1, 1), max(w - 1, 1)]
        stride = [s0, s1]
        storage_offset = off if off > 0 else 0
    elif case == "reshape_like":
        if total % 4 == 0:
            k = 4
        elif total % 2 == 0:
            k = 2
        else:
            k = 1
        size = [total // k, k]
        stride = [k, 1]
        storage_offset = 0
    else:
        raise AssertionError("unknown case")

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.as_strided(ref_x, size, stride, storage_offset),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_as_strided(x, size, stride, storage_offset),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"as_strided {base_shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.as_strided
@pytest.mark.parametrize("base_shape", [(6,), (64,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["contig", "skip_step", "offset_start", "reshape2d"])
def test_as_strided_1d_performance(base_shape, dtype, case):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(base_shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    (N,) = base_shape
    (s,) = ref_x.stride()
    total = N

    if case == "contig":
        size = [N]
        stride = [s]
        storage_offset = 0
    elif case == "skip_step":
        step = 2 if N >= 2 else 1
        size = [max(N // step, 1)]
        stride = [s * step]
        storage_offset = 0
    elif case == "offset_start":
        off = 1 if N > 1 else 0
        size = [max(N - off, 1)]
        stride = [s]
        storage_offset = off
    elif case == "reshape2d":
        if total % 4 == 0:
            k = 4
        elif total % 2 == 0:
            k = 2
        else:
            k = 1
        size = [total // k, k]
        stride = [k, 1]
        storage_offset = 0
    else:
        raise AssertionError("unknown case")

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.as_strided(ref_x, size, stride, storage_offset),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_as_strided(x, size, stride, storage_offset),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"as_strided {base_shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.as_strided
@pytest.mark.parametrize("base_shape", [(2, 3, 4), (8, 16, 8), (16, 32, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "case", ["contig", "swap_last_two", "subsample", "offset_plane"]
)
def test_as_strided_3d_performance(base_shape, dtype, case):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(base_shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    d0, d1, d2 = base_shape
    s0, s1, s2 = ref_x.stride()

    if case == "contig":
        size = [d0, d1, d2]
        stride = [s0, s1, s2]
        storage_offset = 0
    elif case == "swap_last_two":
        size = [d0, d2, d1]
        stride = [s0, s2, s1]
        storage_offset = 0
    elif case == "subsample":
        step0 = 2 if d0 >= 2 else 1
        step1 = 2 if d1 >= 2 else 1
        step2 = 1
        size = [max(d0 // step0, 1), max(d1 // step1, 1), max(d2 // step2, 1)]
        stride = [s0 * step0, s1 * step1, s2 * step2]
        storage_offset = 0
    elif case == "offset_plane":
        off = (s0 if d0 > 1 else 0) + (s2 if d2 > 1 else 0)
        size = [max(d0 - 1, 1), d1, max(d2 - 1, 1)]
        stride = [s0, s1, s2]
        storage_offset = off if off > 0 else 0
    else:
        raise AssertionError("unknown case")

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.as_strided(ref_x, size, stride, storage_offset),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_as_strided(x, size, stride, storage_offset),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"as_strided {base_shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
