# ALIAS_COPY operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.alias_copy import alias_copy as gems_alias_copy
from flag_gems.experimental_ops.alias_copy import alias_copy_out as gems_alias_copy_out

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


@pytest.mark.alias_copy
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_alias_copy_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x)
    ref_out = torch.ops.aten.alias_copy(ref_x)

    with flag_gems.use_gems():
        act_out = gems_alias_copy(x)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.alias_copy
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_alias_copy_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x)
    ref_out = torch.empty_like(ref_x)
    torch.ops.aten.alias_copy.out(ref_x, out=ref_out)

    act_out = torch.empty_like(x)
    with flag_gems.use_gems():
        gems_alias_copy_out(x, act_out)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.alias_copy
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_alias_copy_benchmark(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    # PyTorch reference
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.alias_copy(ref_x), rep=100, quantiles=quantiles
    )

    # FlagGems implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_alias_copy(x), rep=100, quantiles=quantiles
        )

    speedup = ms_torch / ms_triton
    print(f"alias_copy {shape} {dtype}:")
    print(f"  PyTorch: {ms_torch:.3f}ms")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.alias_copy
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_alias_copy_out_benchmark(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    ref_out = torch.empty_like(ref_x)
    act_out = torch.empty_like(x)

    # PyTorch reference
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.alias_copy.out(ref_x, out=ref_out),
        rep=100,
        quantiles=quantiles,
    )

    # FlagGems implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_alias_copy_out(x, act_out), rep=100, quantiles=quantiles
        )

    speedup = ms_torch / ms_triton
    print(f"alias_copy.out {shape} {dtype}:")
    print(f"  PyTorch: {ms_torch:.3f}ms")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
