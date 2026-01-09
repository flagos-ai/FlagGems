# ARCCOSH operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.arccosh import arccosh as gems_arccosh
from flag_gems.experimental_ops.arccosh import arccosh_out as gems_arccosh_out

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


@pytest.mark.arccosh
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (4, 8, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_arccosh_tensor(shape, dtype):
    input_tensor = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0

    ref_input = to_reference(input_tensor)
    ref_out = torch.ops.aten.arccosh(ref_input)

    with flag_gems.use_gems():
        act_out = gems_arccosh(input_tensor)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.arccosh
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (4, 8, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "noncontiguous"])
def test_arccosh_out(shape, dtype, layout):
    input_tensor = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
    ref_input = to_reference(input_tensor)
    act_input = input_tensor.clone()

    if layout == "contiguous":
        ref_out = torch.empty(shape, dtype=dtype, device=ref_input.device)
        act_out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    else:
        dims = len(shape)
        perm = list(reversed(range(dims)))
        ref_base = torch.empty(
            tuple(reversed(shape)), dtype=dtype, device=ref_input.device
        )
        act_base = torch.empty(
            tuple(reversed(shape)), dtype=dtype, device=flag_gems.device
        )
        ref_out = ref_base.permute(perm)
        act_out = act_base.permute(perm)

    ref_ret = torch.ops.aten.arccosh.out(ref_input, out=ref_out)

    with flag_gems.use_gems():
        act_ret = gems_arccosh_out(act_input, act_out)

    gems_assert_close(act_out, ref_out, dtype=dtype)
    gems_assert_close(act_ret, ref_ret, dtype=dtype)


@pytest.mark.arccosh
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (4, 8, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_arccosh_benchmark_tensor(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    input_tensor = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0

    ref_input = input_tensor.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.arccosh(ref_input), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_arccosh(input_tensor), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"arccosh {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.arccosh
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (4, 8, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "noncontiguous"])
def test_arccosh_benchmark_out(shape, dtype, layout):
    quantiles = [0.5, 0.2, 0.8]

    input_tensor = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
    ref_input = input_tensor.clone()
    act_input = input_tensor.clone()

    if layout == "contiguous":
        ref_out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
        act_out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    else:
        dims = len(shape)
        perm = list(reversed(range(dims)))
        ref_base = torch.empty(
            tuple(reversed(shape)), dtype=dtype, device=flag_gems.device
        )
        act_base = torch.empty(
            tuple(reversed(shape)), dtype=dtype, device=flag_gems.device
        )
        ref_out = ref_base.permute(perm)
        act_out = act_base.permute(perm)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.arccosh.out(ref_input, out=ref_out),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_arccosh_out(act_input, act_out), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"arccosh {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"arccosh {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
