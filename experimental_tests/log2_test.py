# LOG2 operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.log2 import log2 as gems_log2  # noqa: E402
from flag_gems.experimental_ops.log2 import log2_out as gems_log2_out  # noqa: E402

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


@pytest.mark.log2
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_log2_tensor(shape, dtype):
    input_tensor = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    ref_input = to_reference(input_tensor)

    ref_out = torch.ops.aten.log2(ref_input)

    with flag_gems.use_gems():
        act_out = gems_log2(input_tensor)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.log2
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_log2_out(shape, dtype):
    input_tensor = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    ref_input = to_reference(input_tensor)

    ref_out_buf = torch.empty_like(ref_input)
    ref_out = torch.ops.aten.log2.out(ref_input, out=ref_out_buf)

    with flag_gems.use_gems():
        act_out_buf = torch.empty_like(input_tensor)
        act_out = gems_log2_out(input_tensor, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.log2
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_log2_benchmark_tensor(shape, dtype):
    import torch.utils.benchmark as benchmark  # noqa: E402, F401

    quantiles = [0.5, 0.2, 0.8]

    input_tensor = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    ref_input = input_tensor.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.log2(ref_input), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_log2(input_tensor), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"log2 {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.log2
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_log2_benchmark_out(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    input_tensor = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    ref_input = input_tensor.clone()

    ref_out_buf = torch.empty_like(ref_input)
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.log2.out(ref_input, out=ref_out_buf),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        act_out_buf = torch.empty_like(input_tensor)
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_log2_out(input_tensor, act_out_buf),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"log2 {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
