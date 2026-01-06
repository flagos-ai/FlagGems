# _UPSAMPLE_NEAREST_EXACT1D operator test
import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops._upsample_nearest_exact1d import (
    _upsample_nearest_exact1d as gems__upsample_nearest_exact1d,
)
from flag_gems.experimental_ops._upsample_nearest_exact1d import (
    _upsample_nearest_exact1d_out as gems__upsample_nearest_exact1d_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import gems_assert_close  # noqa: E402
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


@pytest.mark.upsample_nearest_exact1d
@pytest.mark.parametrize("shape", [(2, 3, 16), (4, 8, 64), (8, 16, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("factor", [2, 3])
def test__upsample_nearest_exact1d_tensor(shape, dtype, factor):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    out_size = [shape[-1] * factor]

    ref_out = torch.ops.aten._upsample_nearest_exact1d(ref_x, out_size, None)

    with flag_gems.use_gems():
        act_out = gems__upsample_nearest_exact1d(x, out_size, None)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.upsample_nearest_exact1d
@pytest.mark.parametrize("shape", [(2, 3, 16), (4, 8, 64), (8, 16, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("factor", [2])
def test__upsample_nearest_exact1d_out(shape, dtype, factor):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    out_len = shape[-1] * factor
    out_shape = (shape[0], shape[1], out_len)

    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    ref_out = torch.ops.aten._upsample_nearest_exact1d.out(
        ref_x, [out_len], None, out=ref_out_buf
    )

    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        act_out = gems__upsample_nearest_exact1d_out(x, [out_len], None, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.upsample_nearest_exact1d
@pytest.mark.parametrize("shape", [(2, 3, 16), (4, 8, 64), (8, 16, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("factor", [2, 3])
@pytest.mark.parametrize("mode", ["output_size", "scale_factors"])
def test__upsample_nearest_exact1d_vec(shape, dtype, factor, mode):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    if mode == "output_size":
        output_size = [shape[-1] * factor]
        scale_factors = None
    else:
        output_size = None
        scale_factors = [float(factor)]

    ref_out = torch.ops.aten._upsample_nearest_exact1d.vec(
        ref_x, output_size, scale_factors
    )

    with flag_gems.use_gems():
        act_out = torch.ops.aten._upsample_nearest_exact1d.vec(
            x, output_size, scale_factors
        )

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.upsample_nearest_exact1d
@pytest.mark.parametrize("shape", [(2, 3, 16), (4, 8, 64), (8, 16, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("factor", [2, 3])
def test__upsample_nearest_exact1d_benchmark_tensor(shape, dtype, factor):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    out_size = [shape[-1] * factor]

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten._upsample_nearest_exact1d(ref_x, out_size, None),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems__upsample_nearest_exact1d(x, out_size, None),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"_upsample_nearest_exact1d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.upsample_nearest_exact1d
@pytest.mark.parametrize("shape", [(2, 3, 16), (4, 8, 64), (8, 16, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("factor", [2])
def test__upsample_nearest_exact1d_benchmark_out(shape, dtype, factor):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    out_len = shape[-1] * factor
    out_shape = (shape[0], shape[1], out_len)

    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten._upsample_nearest_exact1d.out(
            ref_x, [out_len], None, out=ref_out_buf
        ),
        rep=100,
        quantiles=quantiles,
    )

    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems__upsample_nearest_exact1d_out(x, [out_len], None, act_out_buf),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"_upsample_nearest_exact1d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.upsample_nearest_exact1d
@pytest.mark.parametrize("shape", [(2, 3, 16), (4, 8, 64), (8, 16, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("factor", [2, 3])
@pytest.mark.parametrize("mode", ["output_size", "scale_factors"])
def test__upsample_nearest_exact1d_vec_performance(shape, dtype, factor, mode):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    if mode == "output_size":
        output_size = [shape[-1] * factor]
        scale_factors = None
    else:
        output_size = None
        scale_factors = [float(factor)]

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten._upsample_nearest_exact1d.vec(
            ref_x, output_size, scale_factors
        ),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: torch.ops.aten._upsample_nearest_exact1d.vec(
                x, output_size, scale_factors
            ),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"_upsample_nearest_exact1d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
