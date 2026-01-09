# EXP2 operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.exp2 import exp2 as gems_exp2
from flag_gems.experimental_ops.exp2 import exp2_out as gems_exp2_out

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


# ============ Accuracy Tests ============


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


@pytest.mark.exp2
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_accuracy_exp2_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x)
    ref_out = torch.ops.aten.exp2(ref_x)

    with flag_gems.use_gems():
        act_out = gems_exp2(x)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.exp2
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_accuracy_exp2_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x)
    ref_out = torch.empty_like(ref_x)
    torch.ops.aten.exp2.out(ref_x, out=ref_out)

    with flag_gems.use_gems():
        act_out = torch.empty_like(x)
        gems_exp2_out(x, act_out)

    gems_assert_close(act_out, ref_out, dtype=dtype)


# ============ Performance Tests ============


@pytest.mark.exp2
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_perf_exp2_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x)

    # PyTorch reference benchmark
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.exp2(ref_x), rep=100, quantiles=[0.5, 0.2, 0.8]
    )

    # Triton implementation benchmark
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_exp2(x), rep=100, quantiles=[0.5, 0.2, 0.8]
        )

    speedup = ms_torch / ms_triton
    print(
        f"exp2_tensor: shape={shape}, dtype={dtype}, "
        f"torch={ms_torch:.4f}ms, triton={ms_triton:.4f}ms, speedup={speedup:.2f}x"
    )


@pytest.mark.exp2
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_perf_exp2_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = to_reference(x)
    ref_out = torch.empty_like(ref_x)

    # PyTorch reference benchmark
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.exp2.out(ref_x, out=ref_out),
        rep=100,
        quantiles=[0.5, 0.2, 0.8],
    )

    # Triton implementation benchmark
    act_out = torch.empty_like(x)
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_exp2_out(x, act_out), rep=100, quantiles=[0.5, 0.2, 0.8]
        )

    speedup = ms_torch / ms_triton
    print(
        f"exp2_out: shape={shape}, dtype={dtype}, "
        f"torch={ms_torch:.4f}ms, triton={ms_triton:.4f}ms, speedup={speedup:.2f}x"
    )
