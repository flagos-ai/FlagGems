# SIGN_ operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.sign_ import sign_ as gems_sign_

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


@pytest.mark.sign_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sign__tensor(shape, dtype):
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if base.numel() >= 4:
        base.view(-1)[: base.numel() // 4] = 0
    ref_input = to_reference(base)
    act_input = base.clone()

    ref_out = torch.ops.aten.sign_(ref_input)
    with flag_gems.use_gems():
        act_out = gems_sign_(act_input)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.sign_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sign__tensor_noncontiguous(shape, dtype):
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if base.numel() >= 4:
        base.view(-1)[: base.numel() // 4] = 0

    base_ref = base.clone()
    base_act = base.clone()

    # Reference follows TO_CPU path for correct device during comparison
    ref_input = to_reference(base_ref).transpose(0, 1)
    act_input = base_act.transpose(0, 1)

    ref_out = torch.ops.aten.sign_(ref_input)
    with flag_gems.use_gems():
        act_out = gems_sign_(act_input)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.sign_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sign__benchmark_tensor(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if base.numel() >= 4:
        base.view(-1)[: base.numel() // 4] = 0
    ref_input = base.clone()
    act_input = base.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.sign_(ref_input), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_sign_(act_input), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"sign_ {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.sign_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sign__tensor_noncontiguous_performance(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if base.numel() >= 4:
        base.view(-1)[: base.numel() // 4] = 0

    base_ref = base.clone()
    base_act = base.clone()

    ref_input = base_ref.transpose(0, 1)
    act_input = base_act.transpose(0, 1)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.sign_(ref_input), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_sign_(act_input), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"sign_ {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
