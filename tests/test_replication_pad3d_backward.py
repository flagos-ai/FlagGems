"""Accuracy tests for ``aten::replication_pad3d_backward``."""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.replication_pad3d_backward
@pytest.mark.parametrize(
    "shape",
    [
        # (C, D, H, W) unbatched
        (3, 2, 4, 5),
        # (N, C, D, H, W) batched
        (1, 1, 1, 1, 1),
        (1, 1, 2, 3, 4),
        (1, 3, 4, 5, 6),
        (2, 4, 3, 6, 5),
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
        (2, 1, 0, 2, 3, 1),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_replication_pad3d_backward(shape, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    pl, pr, pt, pb, pf, pbk = padding
    D_in, H_in, W_in = inp.shape[-3], inp.shape[-2], inp.shape[-1]
    out_shape = (
        inp.shape[:-3] + (D_in + pf + pbk,) + (H_in + pt + pb,) + (W_in + pl + pr,)
    )
    grad_output = torch.randn(out_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_grad_output = utils.to_reference(grad_output).to(torch.float32)
    ref_out = torch.ops.aten.replication_pad3d_backward(
        ref_grad_output, ref_inp, padding
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.replication_pad3d_backward(grad_output, inp, padding)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.replication_pad3d_backward
def test_replication_pad3d_backward_grad_input_variant():
    inp = torch.randn((2, 3, 3, 4, 5), dtype=torch.float32, device=flag_gems.device)
    grad_output = torch.randn(
        (2, 3, 5, 7, 8), dtype=torch.float32, device=flag_gems.device
    )
    out_buf = torch.empty_like(inp)
    ref_buf = torch.empty_like(inp)

    torch.ops.aten.replication_pad3d_backward.grad_input(
        grad_output, inp, (1, 2, 1, 2, 1, 1), grad_input=ref_buf
    )
    with flag_gems.use_gems():
        res = torch.ops.aten.replication_pad3d_backward.grad_input(
            grad_output, inp, (1, 2, 1, 2, 1, 1), grad_input=out_buf
        )

    assert res is out_buf
    utils.gems_assert_close(out_buf, ref_buf, torch.float32)


@pytest.mark.replication_pad3d_backward
def test_replication_pad3d_backward_no_padding():
    inp = torch.randn((2, 4, 5, 6, 7), dtype=torch.float32, device=flag_gems.device)
    grad_output = torch.randn_like(inp)

    ref = torch.ops.aten.replication_pad3d_backward(
        grad_output, inp, (0, 0, 0, 0, 0, 0)
    )
    with flag_gems.use_gems():
        res = torch.ops.aten.replication_pad3d_backward(
            grad_output, inp, (0, 0, 0, 0, 0, 0)
        )

    utils.gems_assert_close(res, ref, torch.float32)


@pytest.mark.replication_pad3d_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_replication_pad3d_backward_single_voxel(dtype):
    """A 1x1x1 input collapses every padded output into the same voxel."""
    inp = torch.randn((2, 3, 1, 1, 1), dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn((2, 3, 4, 5, 6), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_grad_output = utils.to_reference(grad_output).to(torch.float32)
    ref_out = torch.ops.aten.replication_pad3d_backward(
        ref_grad_output, ref_inp, (2, 3, 1, 3, 1, 2)
    ).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.replication_pad3d_backward(
            grad_output, inp, (2, 3, 1, 3, 1, 2)
        )
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.replication_pad3d_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_replication_pad3d_backward_asymmetric_all_axes(dtype):
    """Independently asymmetric padding on D, H, and W."""
    inp = torch.randn((1, 2, 3, 4, 5), dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(
        (1, 2, 3 + 1 + 2, 4 + 2 + 3, 5 + 0 + 4),
        dtype=dtype,
        device=flag_gems.device,
    )

    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_grad_output = utils.to_reference(grad_output).to(torch.float32)
    ref_out = torch.ops.aten.replication_pad3d_backward(
        ref_grad_output, ref_inp, (0, 4, 2, 3, 1, 2)
    ).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.replication_pad3d_backward(
            grad_output, inp, (0, 4, 2, 3, 1, 2)
        )
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.replication_pad3d_backward
def test_replication_pad3d_backward_invalid_padding_length():
    inp = torch.randn((1, 2, 4, 4, 4), device=flag_gems.device)
    grad_output = torch.randn((1, 2, 5, 5, 5), device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(ValueError, match="length 6"):
        flag_gems.replication_pad3d_backward(grad_output, inp, (1, 1, 1, 1))


@pytest.mark.replication_pad3d_backward
def test_replication_pad3d_backward_negative_padding():
    inp = torch.randn((1, 2, 4, 4, 4), device=flag_gems.device)
    grad_output = torch.randn((1, 2, 5, 5, 5), device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(ValueError, match=">= 0"):
        flag_gems.replication_pad3d_backward(grad_output, inp, (-1, 1, 0, 0, 0, 0))
