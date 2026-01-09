# TAKE operator test

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
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.take import take as gems_take  # noqa: E402
from flag_gems.experimental_ops.take import take_out as gems_take_out  # noqa: E402


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


@pytest.mark.take
@pytest.mark.parametrize("in_shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("idx_shape", [(6,), (32, 32), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_take_tensor(in_shape, idx_shape, dtype):
    x = torch.randn(in_shape, device=flag_gems.device, dtype=dtype)
    idx = torch.randint(
        0, x.numel(), idx_shape, device=flag_gems.device, dtype=torch.int64
    )

    ref_x = to_reference(x)
    ref_idx = to_reference(idx)
    ref_out = torch.ops.aten.take(ref_x, ref_idx)

    with flag_gems.use_gems():
        act_out = gems_take(x, idx)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.take
@pytest.mark.parametrize("in_shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("idx_shape", [(6,), (32, 32), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_take_out(in_shape, idx_shape, dtype):
    x = torch.randn(in_shape, device=flag_gems.device, dtype=dtype)
    idx = torch.randint(
        0, x.numel(), idx_shape, device=flag_gems.device, dtype=torch.int64
    )

    out_ref = torch.empty(idx_shape, device=flag_gems.device, dtype=dtype)
    out_act = torch.empty(idx_shape, device=flag_gems.device, dtype=dtype)

    ref_x = to_reference(x)
    ref_idx = to_reference(idx)
    ref_out = torch.ops.aten.take.out(ref_x, ref_idx, out=out_ref)

    with flag_gems.use_gems():
        act_out = gems_take_out(x, idx, out_act)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.take
@pytest.mark.parametrize("in_shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("idx_shape", [(6,), (32, 32), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_take_benchmark_tensor(in_shape, idx_shape, dtype):
    import torch.utils.benchmark as benchmark  # noqa: E402, F401, F401

    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(in_shape, device=flag_gems.device, dtype=dtype)
    idx = torch.randint(
        0, x.numel(), idx_shape, device=flag_gems.device, dtype=torch.int64
    )

    ref_x = x.clone()
    ref_idx = idx.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.take(ref_x, ref_idx), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_take(x, idx), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"take {idx_shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.take
@pytest.mark.parametrize("in_shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("idx_shape", [(6,), (32, 32), (1024,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_take_benchmark_out(in_shape, idx_shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(in_shape, device=flag_gems.device, dtype=dtype)
    idx = torch.randint(
        0, x.numel(), idx_shape, device=flag_gems.device, dtype=torch.int64
    )

    out_ref = torch.empty(idx_shape, device=flag_gems.device, dtype=dtype)
    out_act = torch.empty(idx_shape, device=flag_gems.device, dtype=dtype)

    ref_x = x.clone()
    ref_idx = idx.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.take.out(ref_x, ref_idx, out=out_ref),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_take_out(x, idx, out_act), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"take {idx_shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
