# REFLECTION_PAD2D operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.reflection_pad2d import (  # noqa: E402
    reflection_pad2d as gems_reflection_pad2d,
)
from flag_gems.experimental_ops.reflection_pad2d import (  # noqa: E402
    reflection_pad2d_out as gems_reflection_pad2d_out,
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


@pytest.mark.reflection_pad2d
@pytest.mark.parametrize(
    "case",
    [
        ((1, 1, 4, 4), (1, 1, 1, 1)),
        ((2, 3, 8, 8), (1, 1, 2, 2)),
        ((4, 8, 64, 32), (3, 3, 2, 2)),
        ((2, 16, 256, 256), (4, 4, 4, 4)),
        ((3, 16, 16), (2, 2, 2, 2)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reflection_pad2d_tensor(case, dtype):
    shape, padding = case
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = to_reference(input_tensor)

    ref_out = torch.ops.aten.reflection_pad2d(ref_input, padding)

    with flag_gems.use_gems():
        act_out = gems_reflection_pad2d(input_tensor, padding)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.reflection_pad2d
@pytest.mark.parametrize(
    "case",
    [
        ((1, 1, 4, 4), (1, 1, 1, 1)),
        ((2, 3, 8, 8), (1, 1, 2, 2)),
        ((4, 8, 64, 32), (3, 3, 2, 2)),
        ((2, 16, 256, 256), (4, 4, 4, 4)),
        ((3, 16, 16), (2, 2, 2, 2)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reflection_pad2d_out(case, dtype):
    shape, padding = case
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = to_reference(input_tensor)
    pl, pr, pt, pb = padding

    if len(shape) == 3:
        C, H, W = shape
        out_shape = (C, H + pt + pb, W + pl + pr)
    else:
        N, C, H, W = shape
        out_shape = (N, C, H + pt + pb, W + pl + pr)

    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=ref_input.device)
    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten.reflection_pad2d.out(ref_input, padding, out=ref_out_buf)

    with flag_gems.use_gems():
        act_out = gems_reflection_pad2d_out(input_tensor, padding, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.reflection_pad2d
@pytest.mark.parametrize(
    "case",
    [
        ((1, 1, 4, 4), (1, 1, 1, 1)),
        ((2, 3, 8, 8), (1, 1, 2, 2)),
        ((4, 8, 64, 32), (3, 3, 2, 2)),
        ((2, 16, 256, 256), (4, 4, 4, 4)),
        ((3, 16, 16), (2, 2, 2, 2)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reflection_pad2d_benchmark_tensor(case, dtype):
    import torch.utils.benchmark as benchmark  # noqa: E402, F401

    quantiles = [0.5, 0.2, 0.8]

    shape, padding = case
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = input_tensor.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.reflection_pad2d(ref_input, padding),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_reflection_pad2d(input_tensor, padding),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"reflection_pad2d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.reflection_pad2d
@pytest.mark.parametrize(
    "case",
    [
        ((1, 1, 4, 4), (1, 1, 1, 1)),
        ((2, 3, 8, 8), (1, 1, 2, 2)),
        ((4, 8, 64, 32), (3, 3, 2, 2)),
        ((2, 16, 256, 256), (4, 4, 4, 4)),
        ((3, 16, 16), (2, 2, 2, 2)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reflection_pad2d_benchmark_out(case, dtype):
    quantiles = [0.5, 0.2, 0.8]

    shape, padding = case
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = input_tensor.clone()
    pl, pr, pt, pb = padding

    if len(shape) == 3:
        C, H, W = shape
        out_shape = (C, H + pt + pb, W + pl + pr)
    else:
        N, C, H, W = shape
        out_shape = (N, C, H + pt + pb, W + pl + pr)

    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.reflection_pad2d.out(
            ref_input, padding, out=ref_out_buf
        ),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_reflection_pad2d_out(input_tensor, padding, act_out_buf),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"reflection_pad2d {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
