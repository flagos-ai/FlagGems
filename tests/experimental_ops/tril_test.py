# TRIL operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
try:
    from tests.accuracy_utils import gems_assert_close, TO_CPU  # noqa: E402
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.tril import tril as gems_tril  # noqa: E402
from flag_gems.experimental_ops.tril import tril_out as gems_tril_out  # noqa: E402


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


@pytest.mark.tril
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("diagonal", [-2, -1, 0, 1, 3])
def test_tril_tensor(shape, dtype, diagonal):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = to_reference(input_tensor)
    ref_out = torch.ops.aten.tril(ref_input, diagonal)
    with flag_gems.use_gems():
        act_out = gems_tril(input_tensor, diagonal)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.tril
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("diagonal", [-2, -1, 0, 1, 3])
def test_tril_out(shape, dtype, diagonal):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = to_reference(input_tensor)
    out_ref = torch.empty_like(ref_input)
    ref_out = torch.ops.aten.tril.out(ref_input, diagonal, out=out_ref)
    out_act = torch.empty_like(input_tensor)
    with flag_gems.use_gems():
        act_out = gems_tril_out(input_tensor, diagonal, out_act)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.tril
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("diagonal", [-2, -1, 0, 1, 3])
def test_tril_benchmark_tensor(shape, dtype, diagonal):
    import torch.utils.benchmark as benchmark  # noqa: E402, F401, F401

    quantiles = [0.5, 0.2, 0.8]

    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = input_tensor.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.tril(ref_input, diagonal), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_tril(input_tensor, diagonal), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"tril {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.tril
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("diagonal", [-2, -1, 0, 1, 3])
def test_tril_benchmark_out(shape, dtype, diagonal):
    quantiles = [0.5, 0.2, 0.8]

    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = input_tensor.clone()
    out_ref = torch.empty_like(ref_input)
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.tril.out(ref_input, diagonal, out=out_ref),
        rep=100,
        quantiles=quantiles,
    )
    out_act = torch.empty_like(input_tensor)

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_tril_out(input_tensor, diagonal, out_act),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"tril {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
