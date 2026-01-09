# UPSAMPLE_NEAREST1D operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.upsample_nearest1d import (
    upsample_nearest1d as gems_upsample_nearest1d,
)
from flag_gems.experimental_ops.upsample_nearest1d import (
    upsample_nearest1d_out as gems_upsample_nearest1d_out,
)

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


@pytest.mark.upsample_nearest1d
@pytest.mark.parametrize("shape", [(1, 1, 8), (2, 3, 15), (4, 8, 64), (8, 16, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size_mode", ["same", "double", "half"])
def test_upsample_nearest1d_default(shape, dtype, size_mode):
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    ref_x = to_reference(x)

    N, C, L = shape
    if size_mode == "same":
        L_out = L
    elif size_mode == "double":
        L_out = L * 2
    elif size_mode == "half":
        L_out = max(1, L // 2)
    else:
        L_out = L

    output_size = [L_out]

    ref_out = torch.ops.aten.upsample_nearest1d(ref_x, output_size, None)

    with flag_gems.use_gems():
        act_out = gems_upsample_nearest1d(x, output_size, None)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.upsample_nearest1d
@pytest.mark.parametrize("shape", [(1, 2, 10), (2, 4, 32), (4, 8, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size_mode", ["same", "double", "half"])
def test_upsample_nearest1d_vec_output_size(shape, dtype, size_mode):
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    ref_x = to_reference(x)

    N, C, L = shape
    if size_mode == "same":
        L_out = L
    elif size_mode == "double":
        L_out = L * 2
    elif size_mode == "half":
        L_out = max(1, L // 2)
    else:
        L_out = L

    output_size = [L_out]
    scale_factors = None

    ref_out = torch.ops.aten.upsample_nearest1d.vec(ref_x, output_size, scale_factors)

    with flag_gems.use_gems():
        act_out = torch.ops.aten.upsample_nearest1d.vec(x, output_size, scale_factors)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.upsample_nearest1d
@pytest.mark.parametrize("shape", [(1, 2, 9), (2, 4, 20), (4, 8, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [0.5, 1.0, 1.5, 2.0])
def test_upsample_nearest1d_vec_scale(shape, dtype, scale):
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    ref_x = to_reference(x)

    output_size = None
    scale_factors = [float(scale)]

    ref_out = torch.ops.aten.upsample_nearest1d.vec(ref_x, output_size, scale_factors)

    with flag_gems.use_gems():
        act_out = torch.ops.aten.upsample_nearest1d.vec(x, output_size, scale_factors)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.upsample_nearest1d
@pytest.mark.parametrize("shape", [(1, 1, 8), (2, 3, 15), (4, 8, 64), (8, 16, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size_mode", ["same", "double", "half"])
def test_upsample_nearest1d_out(shape, dtype, size_mode):
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    ref_x = to_reference(x)

    N, C, L = shape
    if size_mode == "same":
        L_out = L
    elif size_mode == "double":
        L_out = L * 2
    elif size_mode == "half":
        L_out = max(1, L // 2)
    else:
        L_out = L

    output_size = [L_out]
    ref_out = torch.ops.aten.upsample_nearest1d(ref_x, output_size, None)

    out_tensor = torch.empty((N, C, L_out), device=flag_gems.device, dtype=dtype)
    with flag_gems.use_gems():
        act_out = gems_upsample_nearest1d_out(x, output_size, None, out=out_tensor)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.upsample_nearest1d
@pytest.mark.parametrize("shape", [(1, 1, 8), (2, 3, 15), (4, 8, 64), (8, 16, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size_mode", ["same", "double", "half"])
def test_upsample_nearest1d_default_performance(shape, dtype, size_mode):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    ref_x = x.clone()

    N, C, L = shape
    if size_mode == "same":
        L_out = L
    elif size_mode == "double":
        L_out = L * 2
    elif size_mode == "half":
        L_out = max(1, L // 2)
    else:
        L_out = L

    output_size = [L_out]

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.upsample_nearest1d(ref_x, output_size, None),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_upsample_nearest1d(x, output_size, None),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"upsample_nearest1d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.upsample_nearest1d
@pytest.mark.parametrize("shape", [(1, 2, 10), (2, 4, 32), (4, 8, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size_mode", ["same", "double", "half"])
def test_upsample_nearest1d_vec_output_size_performance(shape, dtype, size_mode):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    ref_x = x.clone()

    N, C, L = shape
    if size_mode == "same":
        L_out = L
    elif size_mode == "double":
        L_out = L * 2
    elif size_mode == "half":
        L_out = max(1, L // 2)
    else:
        L_out = L

    output_size = [L_out]
    scale_factors = None

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.upsample_nearest1d.vec(
            ref_x, output_size, scale_factors
        ),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: torch.ops.aten.upsample_nearest1d.vec(
                x, output_size, scale_factors
            ),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"upsample_nearest1d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.upsample_nearest1d
@pytest.mark.parametrize("shape", [(1, 2, 9), (2, 4, 20), (4, 8, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [0.5, 1.0, 1.5, 2.0])
def test_upsample_nearest1d_vec_scale_performance(shape, dtype, scale):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    ref_x = x.clone()

    output_size = None
    scale_factors = [float(scale)]

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.upsample_nearest1d.vec(
            ref_x, output_size, scale_factors
        ),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: torch.ops.aten.upsample_nearest1d.vec(
                x, output_size, scale_factors
            ),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"upsample_nearest1d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.upsample_nearest1d
@pytest.mark.parametrize("shape", [(1, 1, 8), (2, 3, 15), (4, 8, 64), (8, 16, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size_mode", ["same", "double", "half"])
def test_upsample_nearest1d_benchmark_out(shape, dtype, size_mode):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    ref_x = x.clone()

    N, C, L = shape
    if size_mode == "same":
        L_out = L
    elif size_mode == "double":
        L_out = L * 2
    elif size_mode == "half":
        L_out = max(1, L // 2)
    else:
        L_out = L

    output_size = [L_out]
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.upsample_nearest1d(ref_x, output_size, None),
        rep=100,
        quantiles=quantiles,
    )

    out_tensor = torch.empty((N, C, L_out), device=flag_gems.device, dtype=dtype)

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_upsample_nearest1d_out(x, output_size, None, out=out_tensor),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"upsample_nearest1d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
