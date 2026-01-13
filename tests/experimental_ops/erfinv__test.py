# ERFINV_ operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.erfinv_ import erfinv_ as gems_erfinv_

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


@pytest.mark.erfinv_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_erfinv__tensor(shape, dtype):
    x = torch.empty(shape, device=flag_gems.device, dtype=dtype).uniform_(-0.95, 0.95)
    ref_in = to_reference(x)
    act_in = x.clone()

    ref_out = torch.ops.aten.erfinv_(ref_in)

    with flag_gems.use_gems():
        act_out = gems_erfinv_(act_in)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.erfinv_
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_erfinv__noncontiguous(dtype):
    base = torch.empty((128, 128), device=flag_gems.device, dtype=dtype).uniform_(
        -0.95, 0.95
    )
    base_ref = base.clone()
    base_act = base.clone()

    # Keep reference consistent with TO_CPU
    ref_in = to_reference(base_ref).t()[::2, ::2]
    act_in = base_act.t()[::2, ::2]

    ref_out = torch.ops.aten.erfinv_(ref_in)

    with flag_gems.use_gems():
        act_out = gems_erfinv_(act_in)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.erfinv_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_erfinv__benchmark_tensor(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.empty(shape, device=flag_gems.device, dtype=dtype).uniform_(-0.95, 0.95)
    ref_in = x.clone()
    act_in = x.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.erfinv_(ref_in), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_erfinv_(act_in), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"erfinv_ {ref_in.shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.erfinv_
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_erfinv__noncontiguous_performance(dtype):
    quantiles = [0.5, 0.2, 0.8]

    base = torch.empty((128, 128), device=flag_gems.device, dtype=dtype).uniform_(
        -0.95, 0.95
    )
    base_ref = base.clone()
    base_act = base.clone()

    ref_in = base_ref.t()[::2, ::2]
    act_in = base_act.t()[::2, ::2]

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.erfinv_(ref_in), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_erfinv_(act_in), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"erfinv_ {ref_in.shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
