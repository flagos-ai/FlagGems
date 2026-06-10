"""Accuracy tests for ``aten::replication_pad1d_backward``."""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.replication_pad1d_backward
@pytest.mark.parametrize(
    "shape",
    [
        # (C, L) -- 2D input (unbatched)
        (4, 8),
        # (N, C, L) -- 3D input (batched)
        (1, 1, 1),
        (1, 1, 4),
        (1, 3, 5),
        (2, 4, 7),
        (8, 16, 32),
        (4, 8, 64),
    ],
)
@pytest.mark.parametrize(
    "padding",
    [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (2, 3),
        (3, 2),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_replication_pad1d_backward(shape, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    L_out = inp.shape[-1] + padding[0] + padding[1]
    out_shape = inp.shape[:-1] + (L_out,)
    grad_output = torch.randn(out_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_grad_output = utils.to_reference(grad_output).to(torch.float32)
    ref_out = torch.ops.aten.replication_pad1d_backward(
        ref_grad_output, ref_inp, padding
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.replication_pad1d_backward(grad_output, inp, padding)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.replication_pad1d_backward
def test_replication_pad1d_backward_grad_input_variant():
    inp = torch.randn((2, 3, 5), dtype=torch.float32, device=flag_gems.device)
    grad_output = torch.randn((2, 3, 8), dtype=torch.float32, device=flag_gems.device)
    out_buf = torch.empty_like(inp)
    ref_buf = torch.empty_like(inp)

    torch.ops.aten.replication_pad1d_backward.grad_input(
        grad_output, inp, (1, 2), grad_input=ref_buf
    )
    with flag_gems.use_gems():
        res = torch.ops.aten.replication_pad1d_backward.grad_input(
            grad_output, inp, (1, 2), grad_input=out_buf
        )

    assert res is out_buf
    utils.gems_assert_close(out_buf, ref_buf, torch.float32)


@pytest.mark.replication_pad1d_backward
def test_replication_pad1d_backward_no_padding():
    """`(0, 0)` padding -> backward is just an identity copy."""
    inp = torch.randn((2, 4, 9), dtype=torch.float32, device=flag_gems.device)
    grad_output = torch.randn_like(inp)

    ref = torch.ops.aten.replication_pad1d_backward(grad_output, inp, (0, 0))
    with flag_gems.use_gems():
        res = torch.ops.aten.replication_pad1d_backward(grad_output, inp, (0, 0))

    utils.gems_assert_close(res, ref, torch.float32)


@pytest.mark.replication_pad1d_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_replication_pad1d_backward_single_input(dtype):
    """L_in == 1 collapses both boundaries to the same input element."""
    inp = torch.randn((2, 3, 1), dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn((2, 3, 6), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_grad_output = utils.to_reference(grad_output).to(torch.float32)
    ref_out = torch.ops.aten.replication_pad1d_backward(
        ref_grad_output, ref_inp, (2, 3)
    ).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.replication_pad1d_backward(grad_output, inp, (2, 3))
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.replication_pad1d_backward
def test_replication_pad1d_backward_invalid_padding_length():
    inp = torch.randn((1, 2, 4), device=flag_gems.device)
    grad_output = torch.randn((1, 2, 6), device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(ValueError, match="length 2"):
        # padding must be length-2 sequence
        flag_gems.replication_pad1d_backward(grad_output, inp, (1, 1, 1))


@pytest.mark.replication_pad1d_backward
def test_replication_pad1d_backward_negative_padding():
    inp = torch.randn((1, 2, 4), device=flag_gems.device)
    grad_output = torch.randn((1, 2, 5), device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(ValueError, match=">= 0"):
        flag_gems.replication_pad1d_backward(grad_output, inp, (-1, 2))
