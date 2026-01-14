# IM2COL operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.im2col import im2col as gems_im2col
from flag_gems.experimental_ops.im2col import im2col_out as gems_im2col_out

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


@pytest.mark.im2col
@pytest.mark.parametrize("shape", [(3, 8, 8), (16, 64, 64), (32, 128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "kernel_size, dilation, padding, stride",
    [
        ((3, 3), (1, 1), (1, 1), (1, 1)),
        ((3, 3), (1, 1), (0, 0), (2, 2)),
        ((5, 4), (2, 2), (2, 1), (1, 2)),
        ((1, 1), (1, 1), (0, 0), (1, 1)),
    ],
)
def test_im2col_tensor(shape, dtype, kernel_size, dilation, padding, stride):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)

    ref_out = torch.ops.aten.im2col(ref_x, kernel_size, dilation, padding, stride)

    with flag_gems.use_gems():
        act_out = gems_im2col(x, kernel_size, dilation, padding, stride)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.im2col
@pytest.mark.parametrize("shape", [(3, 8, 8), (16, 64, 64), (32, 128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "kernel_size, dilation, padding, stride",
    [
        ((3, 3), (1, 1), (1, 1), (1, 1)),
        ((3, 3), (1, 1), (0, 0), (2, 2)),
        ((5, 4), (2, 2), (2, 1), (1, 2)),
        ((1, 1), (1, 1), (0, 0), (1, 1)),
    ],
)
def test_im2col_out(shape, dtype, kernel_size, dilation, padding, stride):
    def compute_out_shape(c, h, w, k, d, p, s):
        kH, kW = k
        dH, dW = d
        pH, pW = p
        sH, sW = s
        out_h = (h + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        out_w = (w + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        return (c * kH * kW, out_h * out_w)

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)

    C, H, W = shape
    out_shape = compute_out_shape(C, H, W, kernel_size, dilation, padding, stride)

    out_ref = torch.empty(out_shape, dtype=dtype, device=ref_x.device)
    out_act = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten.im2col.out(
        ref_x, kernel_size, dilation, padding, stride, out=out_ref
    )

    with flag_gems.use_gems():
        act_out = gems_im2col_out(x, kernel_size, dilation, padding, stride, out_act)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.im2col
@pytest.mark.parametrize("shape", [(3, 8, 8), (16, 64, 64), (32, 128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "kernel_size, dilation, padding, stride",
    [
        ((3, 3), (1, 1), (1, 1), (1, 1)),
        ((3, 3), (1, 1), (0, 0), (2, 2)),
        ((5, 4), (2, 2), (2, 1), (1, 2)),
        ((1, 1), (1, 1), (0, 0), (1, 1)),
    ],
)
def test_im2col_benchmark_tensor(shape, dtype, kernel_size, dilation, padding, stride):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.im2col(ref_x, kernel_size, dilation, padding, stride),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_im2col(x, kernel_size, dilation, padding, stride),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"im2col {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.im2col
@pytest.mark.parametrize("shape", [(3, 8, 8), (16, 64, 64), (32, 128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "kernel_size, dilation, padding, stride",
    [
        ((3, 3), (1, 1), (1, 1), (1, 1)),
        ((3, 3), (1, 1), (0, 0), (2, 2)),
        ((5, 4), (2, 2), (2, 1), (1, 2)),
        ((1, 1), (1, 1), (0, 0), (1, 1)),
    ],
)
def test_im2col_benchmark_out(shape, dtype, kernel_size, dilation, padding, stride):
    quantiles = [0.5, 0.2, 0.8]

    def compute_out_shape(c, h, w, k, d, p, s):
        kH, kW = k
        dH, dW = d
        pH, pW = p
        sH, sW = s
        out_h = (h + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        out_w = (w + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        return (c * kH * kW, out_h * out_w)

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()

    C, H, W = shape
    out_shape = compute_out_shape(C, H, W, kernel_size, dilation, padding, stride)

    out_ref = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    out_act = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.im2col.out(
            ref_x, kernel_size, dilation, padding, stride, out=out_ref
        ),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_im2col_out(x, kernel_size, dilation, padding, stride, out_act),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"im2col {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
