# ARCCOSH_ operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.arccosh_ import arccosh_ as gems_arccosh_

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


@pytest.mark.arccosh_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [0.001, 9.0])
@pytest.mark.parametrize("layout", ["contig", "noncontig"])
def test_arccosh__tensor(shape, dtype, scale, layout):
    if layout == "contig":
        data = 1 + torch.rand(shape, dtype=dtype, device=flag_gems.device) * scale
        ref_input = to_reference(data)
        act_input = data.clone()
    else:
        m, n = shape
        base = 1 + torch.rand((n, m), dtype=dtype, device=flag_gems.device) * scale
        base2 = base.clone()
        # Keep reference on CPU when TO_CPU to avoid device mismatch
        ref_input = to_reference(base).transpose(0, 1)
        act_input = base2.transpose(0, 1)

    ref_out = torch.ops.aten.arccosh_(ref_input)
    with flag_gems.use_gems():
        act_out = gems_arccosh_(act_input)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.arccosh_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [0.001, 9.0])
@pytest.mark.parametrize("layout", ["contig", "noncontig"])
def test_arccosh__benchmark_tensor(shape, dtype, scale, layout):
    quantiles = [0.5, 0.2, 0.8]

    if layout == "contig":
        data = 1 + torch.rand(shape, dtype=dtype, device=flag_gems.device) * scale
        ref_input = data.clone()
        act_input = data.clone()
    else:
        m, n = shape
        base = 1 + torch.rand((n, m), dtype=dtype, device=flag_gems.device) * scale
        base2 = base.clone()
        ref_input = base.transpose(0, 1)
        act_input = base2.transpose(0, 1)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.arccosh_(ref_input), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_arccosh_(act_input), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"arccosh_ {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
