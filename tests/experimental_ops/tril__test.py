# TRIL_ operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.tril_ import tril_ as gems_tril_  # noqa: E402

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


@pytest.mark.tril_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (4, 64, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("diagonal", [0, 1, -1, 3])
def test_tril__tensor(shape, dtype, diagonal):
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = to_reference(base)
    act_input = base.clone()

    ref_out = torch.ops.aten.tril_(ref_input, diagonal)

    with flag_gems.use_gems():
        act_out = gems_tril_(act_input, diagonal)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.tril_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (4, 64, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("diagonal", [0, 1, -1, 3])
def test_tril__benchmark_tensor(shape, dtype, diagonal):
    quantiles = [0.5, 0.2, 0.8]

    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = base.clone()
    act_input = base.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.tril_(ref_input, diagonal), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_tril_(act_input, diagonal), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"tril_ {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
