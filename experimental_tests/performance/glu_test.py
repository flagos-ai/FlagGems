# GLU operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.glu import glu as gems_glu
from flag_gems.experimental_ops.glu import glu_out as gems_glu_out

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
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


@pytest.mark.glu
@pytest.mark.parametrize("shape", [(4, 6), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_glu_benchmark_tensor(shape, dtype, dim):
    quantiles = [0.5, 0.2, 0.8]

    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = input_tensor.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.glu(ref_input, dim), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_glu(input_tensor, dim), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"glu {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.glu
@pytest.mark.parametrize("shape", [(4, 6), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_glu_benchmark_out(shape, dtype, dim):
    quantiles = [0.5, 0.2, 0.8]

    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = input_tensor.clone()

    ndim = len(shape)
    dim_idx = dim if dim >= 0 else dim + ndim
    out_shape = list(shape)
    out_shape[dim_idx] = out_shape[dim_idx] // 2

    out_ref = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    out_act = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.glu.out(ref_input, dim, out=out_ref),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_glu_out(input_tensor, dim, out_act),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"glu {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"glu {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
