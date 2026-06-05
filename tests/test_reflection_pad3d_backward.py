"""Accuracy tests for ``aten::reflection_pad3d_backward``."""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.reflection_pad3d_backward
@pytest.mark.parametrize(
    "shape",
    [
        # (C, D, H, W) unbatched
        (3, 3, 4, 5),
        # (N, C, D, H, W) batched
        (1, 1, 2, 2, 2),
        (1, 1, 3, 4, 5),
        (1, 3, 4, 5, 6),
        (2, 4, 5, 6, 7),
        (2, 8, 4, 8, 8),
    ],
)
@pytest.mark.parametrize(
    "padding",
    [
        (0, 0, 0, 0, 0, 0),
        (1, 1, 0, 0, 0, 0),
        (0, 0, 1, 1, 0, 0),
        (0, 0, 0, 0, 1, 1),
        (1, 1, 1, 1, 1, 1),
        (2, 1, 1, 2, 1, 1),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reflection_pad3d_backward(shape, padding, dtype):
    # Reflection requires pad < input_dim along each axis.  Skip combos that
    # violate the constraint -- they exercise the wrapper's error path
    # separately.
    pl, pr, pt, pb, pf, pbk = padding
    D_in, H_in, W_in = shape[-3], shape[-2], shape[-1]
    if max(pl, pr) >= W_in or max(pt, pb) >= H_in or max(pf, pbk) >= D_in:
        pytest.skip("padding >= input dim -- not allowed for reflection")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out_shape = (
        inp.shape[:-3] + (D_in + pf + pbk,) + (H_in + pt + pb,) + (W_in + pl + pr,)
    )
    grad_output = torch.randn(out_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_grad_output = utils.to_reference(grad_output).to(torch.float32)
    ref_out = torch.ops.aten.reflection_pad3d_backward(
        ref_grad_output, ref_inp, padding
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.reflection_pad3d_backward(grad_output, inp, padding)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.reflection_pad3d_backward
def test_reflection_pad3d_backward_grad_input_variant():
    inp = torch.randn((2, 3, 4, 5, 6), dtype=torch.float32, device=flag_gems.device)
    # padding=(1, 2, 1, 1, 1, 1): W_out=6+1+2=9, H_out=5+2=7, D_out=4+2=6
    grad_output = torch.randn(
        (2, 3, 6, 7, 9), dtype=torch.float32, device=flag_gems.device
    )
    out_buf = torch.empty_like(inp)
    ref_buf = torch.empty_like(inp)

    torch.ops.aten.reflection_pad3d_backward.grad_input(
        grad_output, inp, (1, 2, 1, 1, 1, 1), grad_input=ref_buf
    )
    with flag_gems.use_gems():
        res = torch.ops.aten.reflection_pad3d_backward.grad_input(
            grad_output, inp, (1, 2, 1, 1, 1, 1), grad_input=out_buf
        )

    assert res is out_buf
    utils.gems_assert_close(out_buf, ref_buf, torch.float32)


@pytest.mark.reflection_pad3d_backward
def test_reflection_pad3d_backward_no_padding():
    inp = torch.randn((2, 4, 5, 6, 7), dtype=torch.float32, device=flag_gems.device)
    grad_output = torch.randn_like(inp)

    ref = torch.ops.aten.reflection_pad3d_backward(grad_output, inp, (0, 0, 0, 0, 0, 0))
    with flag_gems.use_gems():
        res = torch.ops.aten.reflection_pad3d_backward(
            grad_output, inp, (0, 0, 0, 0, 0, 0)
        )

    utils.gems_assert_close(res, ref, torch.float32)


@pytest.mark.reflection_pad3d_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reflection_pad3d_backward_max_padding(dtype):
    """pad = input_dim - 1 on each axis (the largest allowed value)."""
    inp = torch.randn((1, 2, 3, 4, 5), dtype=dtype, device=flag_gems.device)
    # max pads: D=2, H=3, W=4
    padding = (4, 4, 3, 3, 2, 2)
    out_shape = (1, 2, 3 + 4, 4 + 6, 5 + 8)
    grad_output = torch.randn(out_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_grad_output = utils.to_reference(grad_output).to(torch.float32)
    ref_out = torch.ops.aten.reflection_pad3d_backward(
        ref_grad_output, ref_inp, padding
    ).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.reflection_pad3d_backward(grad_output, inp, padding)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.reflection_pad3d_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reflection_pad3d_backward_asymmetric(dtype):
    """Independently asymmetric padding on D, H, W within the reflection
    constraint (pad < dim)."""
    inp = torch.randn((1, 2, 4, 5, 6), dtype=dtype, device=flag_gems.device)
    padding = (2, 4, 3, 1, 1, 2)  # all < dim
    out_shape = (
        1,
        2,
        4 + 1 + 2,
        5 + 3 + 1,
        6 + 2 + 4,
    )
    grad_output = torch.randn(out_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_grad_output = utils.to_reference(grad_output).to(torch.float32)
    ref_out = torch.ops.aten.reflection_pad3d_backward(
        ref_grad_output, ref_inp, padding
    ).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.reflection_pad3d_backward(grad_output, inp, padding)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.reflection_pad3d_backward
def test_reflection_pad3d_backward_invalid_padding_length():
    inp = torch.randn((1, 2, 4, 4, 4), device=flag_gems.device)
    grad_output = torch.randn((1, 2, 5, 5, 5), device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(ValueError, match="length 6"):
        flag_gems.reflection_pad3d_backward(grad_output, inp, (1, 1, 1, 1))


@pytest.mark.reflection_pad3d_backward
def test_reflection_pad3d_backward_negative_padding():
    inp = torch.randn((1, 2, 4, 4, 4), device=flag_gems.device)
    grad_output = torch.randn((1, 2, 5, 5, 5), device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(ValueError, match=">= 0"):
        flag_gems.reflection_pad3d_backward(grad_output, inp, (-1, 1, 0, 0, 0, 0))


@pytest.mark.reflection_pad3d_backward
def test_reflection_pad3d_backward_pad_too_large():
    """pad >= input_dim must raise for reflection."""
    inp = torch.randn((1, 2, 4, 4, 4), device=flag_gems.device)
    grad_output = torch.randn((1, 2, 4, 4, 8), device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(ValueError, match="strictly less than"):
        # pad_left = 4 == W_in = 4 -> error.
        flag_gems.reflection_pad3d_backward(grad_output, inp, (4, 0, 0, 0, 0, 0))
