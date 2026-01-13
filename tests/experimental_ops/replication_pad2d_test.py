# REPLICATION_PAD2D operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.replication_pad2d import (  # noqa: E402
    replication_pad2d as gems_replication_pad2d,
)
from flag_gems.experimental_ops.replication_pad2d import (  # noqa: E402
    replication_pad2d_out as gems_replication_pad2d_out,
)

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


@pytest.mark.replication_pad2d
@pytest.mark.parametrize("shape", [(2, 3, 8, 8), (4, 8, 128, 256), (2, 4, 512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding", [(0, 0, 0, 0), (1, 1, 2, 2), (3, 0, 0, 3)])
def test_replication_pad2d_tensor(shape, dtype, padding):
    input_tensor = torch.randn(shape, device=flag_gems.device, dtype=dtype)

    ref_input = to_reference(input_tensor)
    ref_out = torch.ops.aten.replication_pad2d(ref_input, padding)

    with flag_gems.use_gems():
        act_out = gems_replication_pad2d(input_tensor, padding)

    gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.replication_pad2d
@pytest.mark.parametrize("shape", [(2, 3, 8, 8), (4, 8, 128, 256), (2, 4, 512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding", [(0, 0, 0, 0), (1, 1, 2, 2), (3, 0, 0, 3)])
def test_replication_pad2d_out(shape, dtype, padding):
    input_tensor = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    left, right, top, bottom = padding
    n, c, h, w = shape
    out_shape = (n, c, h + top + bottom, w + left + right)

    ref_input = to_reference(input_tensor)
    ref_out_buf = torch.empty(out_shape, device=ref_input.device, dtype=dtype)
    ref_out = torch.ops.aten.replication_pad2d.out(ref_input, padding, out=ref_out_buf)

    act_out_buf = torch.empty(out_shape, device=flag_gems.device, dtype=dtype)
    with flag_gems.use_gems():
        act_out = gems_replication_pad2d_out(input_tensor, padding, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.replication_pad2d
@pytest.mark.parametrize("shape", [(2, 3, 8, 8), (4, 8, 128, 256), (2, 4, 512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding", [(0, 0, 0, 0), (1, 1, 2, 2), (3, 0, 0, 3)])
def test_replication_pad2d_benchmark_tensor(shape, dtype, padding):
    import torch.utils.benchmark as benchmark  # noqa: E402, F401

    quantiles = [0.5, 0.2, 0.8]

    input_tensor = torch.randn(shape, device=flag_gems.device, dtype=dtype)

    ref_input = input_tensor.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.replication_pad2d(ref_input, padding),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_replication_pad2d(input_tensor, padding),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"replication_pad2d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.replication_pad2d
@pytest.mark.parametrize("shape", [(2, 3, 8, 8), (4, 8, 128, 256), (2, 4, 512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding", [(0, 0, 0, 0), (1, 1, 2, 2), (3, 0, 0, 3)])
def test_replication_pad2d_benchmark_out(shape, dtype, padding):
    quantiles = [0.5, 0.2, 0.8]

    input_tensor = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    left, right, top, bottom = padding
    n, c, h, w = shape
    out_shape = (n, c, h + top + bottom, w + left + right)

    ref_input = input_tensor.clone()
    ref_out_buf = torch.empty(out_shape, device=flag_gems.device, dtype=dtype)
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.replication_pad2d.out(
            ref_input, padding, out=ref_out_buf
        ),
        rep=100,
        quantiles=quantiles,
    )

    act_out_buf = torch.empty(out_shape, device=flag_gems.device, dtype=dtype)

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_replication_pad2d_out(input_tensor, padding, act_out_buf),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"replication_pad2d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
