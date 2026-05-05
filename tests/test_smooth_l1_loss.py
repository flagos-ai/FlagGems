import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    SMOOTH_L1_SHAPES = [(0,), (2, 3)]
    BETA_VALUES = [0.0, 0.5, 1.0]
else:
    SMOOTH_L1_SHAPES = [(0,), (1,), (2, 3), (33, 17), (4, 8, 16)]
    BETA_VALUES = [0.0, 0.5, 1.0, 2.0]


def _reference_smooth_l1(input, target, reduction, beta, dtype):
    ref_input = utils.to_reference(input).to(torch.float32)
    ref_target = utils.to_reference(target).to(torch.float32)
    ref = torch.ops.aten.smooth_l1_loss(ref_input, ref_target, reduction, beta)
    return ref.to(dtype)


def _reference_smooth_l1_backward(  # noqa: E501
    grad_output, input, target, reduction, beta, dtype
):
    ref_grad = utils.to_reference(grad_output).to(torch.float32)
    ref_input = utils.to_reference(input).to(torch.float32)
    ref_target = utils.to_reference(target).to(torch.float32)
    return torch.ops.aten.smooth_l1_loss_backward(
        ref_grad, ref_input, ref_target, reduction, beta
    ).to(dtype)


def _reference_smooth_l1_backward_grad_input(
    grad_output, input, target, reduction, beta, dtype
):
    ref_grad = utils.to_reference(grad_output).to(torch.float32)
    ref_input = utils.to_reference(input).to(torch.float32)
    ref_target = utils.to_reference(target).to(torch.float32)
    ref_out = torch.empty((0,), dtype=torch.float32, device=ref_input.device)
    return torch.ops.aten.smooth_l1_loss_backward.grad_input(
        ref_grad,
        ref_input,
        ref_target,
        reduction,
        beta,
        grad_input=ref_out,
    ).to(dtype)


def _randn(shape, dtype):
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("shape", SMOOTH_L1_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("beta", BETA_VALUES)
def test_smooth_l1_loss(shape, dtype, reduction, beta):
    input = _randn(shape, dtype)
    target = _randn(shape, dtype)
    if len(shape) >= 2 and shape[0] > 0:
        input = input.transpose(0, 1).contiguous().transpose(0, 1)

    ref_out = _reference_smooth_l1(input, target, reduction, beta, dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss(input, target, reduction, beta)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_smooth_l1_loss_broadcast(dtype, reduction):
    input = _randn((2, 3, 4), dtype)
    target = _randn((4,), dtype)

    ref_out = _reference_smooth_l1(input, target, reduction, 1.0, dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss(input, target, reduction, 1.0)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.smooth_l1_loss
def test_smooth_l1_loss_out_none_and_reduced():
    input = _randn((8, 16), torch.float32)
    target = _randn((8, 16), torch.float32)

    for reduction, beta in [(0, 0.5), (1, 1.0), (2, 1.0)]:
        out = torch.empty((0,), dtype=torch.float16, device=flag_gems.device)
        ref_out = torch.empty_like(utils.to_reference(out))
        torch.ops.aten.smooth_l1_loss.out(
            utils.to_reference(input),
            utils.to_reference(target),
            reduction,
            beta,
            out=ref_out,
        )
        with flag_gems.use_gems():
            res_out = torch.ops.aten.smooth_l1_loss.out(
                input, target, reduction, beta, out=out
            )

        assert res_out is out
        utils.gems_assert_close(
            out, ref_out.to(out.dtype), out.dtype, equal_nan=True, atol=2e-2
        )


@pytest.mark.smooth_l1_loss
def test_smooth_l1_loss_functional_and_negative_beta():
    input = _randn((8, 16), torch.float32)
    target = _randn((8, 16), torch.float32)

    ref_out = torch.nn.functional.smooth_l1_loss(
        utils.to_reference(input),
        utils.to_reference(target),
        reduction="mean",
        beta=0.5,
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.smooth_l1_loss(
            input, target, reduction="mean", beta=0.5
        )
    utils.gems_assert_close(res_out, ref_out, torch.float32)

    with flag_gems.use_gems(), pytest.raises(RuntimeError, match="negative"):
        torch.ops.aten.smooth_l1_loss(input, target, 1, -1.0)


@pytest.mark.smooth_l1_loss_backward
@pytest.mark.parametrize(
    "shape,target_shape",
    [
        ((0,), (0,)),
        ((2, 3), (2, 3)),
        ((33, 17), (33, 17)),
        ((2, 3, 4), (4,)),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("beta", BETA_VALUES)
def test_smooth_l1_loss_backward(shape, target_shape, dtype, reduction, beta):
    if cfg.TO_CPU and beta == 0.0:
        pytest.skip("PyTorch CPU and CUDA differ for beta=0 backward.")

    input = _randn(shape, dtype)
    target = _randn(target_shape, dtype)
    if reduction == 0:
        grad_shape = torch.broadcast_shapes(shape, target_shape)
        grad_output = _randn(grad_shape, dtype)
    else:
        grad_output = _randn((), dtype)

    ref_out = _reference_smooth_l1_backward(
        grad_output, input, target, reduction, beta, dtype
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss_backward(
            grad_output, input, target, reduction, beta
        )

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.smooth_l1_loss_backward
def test_smooth_l1_loss_backward_grad_input_and_beta_zero_equal():
    input = torch.zeros((8,), dtype=torch.float32, device=flag_gems.device)
    target = torch.zeros_like(input)
    grad_output = torch.ones_like(input)
    grad_input = torch.empty_like(input)

    if cfg.TO_CPU:
        pytest.skip(  # noqa: E501
            "PyTorch CPU and CUDA differ for beta=0 equal-input backward."
        )
    ref_out = torch.ops.aten.smooth_l1_loss_backward.grad_input(
        grad_output, input, target, 0, 0.0, grad_input=torch.empty_like(input)
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss_backward.grad_input(
            grad_output, input, target, 0, 0.0, grad_input=grad_input
        )

    assert res_out is grad_input
    utils.gems_assert_close(res_out, ref_out, torch.float32, equal_nan=True)


@pytest.mark.smooth_l1_loss_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_smooth_l1_loss_backward_expands_input(dtype, reduction):
    input = _randn((2, 1, 4), dtype)
    target = _randn((3, 4), dtype)
    if reduction == 0:
        grad_shape = (2, 3, 4)
        grad_output = _randn(grad_shape, dtype)
    else:
        grad_output = _randn((), dtype)

    ref_out = _reference_smooth_l1_backward_grad_input(
        grad_output, input, target, reduction, 1.0, dtype
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss_backward(
            grad_output, input, target, reduction, 1.0
        )

    assert res_out.shape == ref_out.shape == torch.Size((2, 3, 4))
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.smooth_l1_loss_backward
def test_smooth_l1_loss_backward_grad_input_expands_input():
    dtype = torch.float32
    input = _randn((2, 1, 4), dtype)
    target = _randn((3, 4), dtype)
    grad_output = _randn((), dtype)
    grad_input = torch.empty((0,), dtype=dtype, device=flag_gems.device)
    ref_grad_output = utils.to_reference(grad_output)
    ref_input = utils.to_reference(input)
    ref_target = utils.to_reference(target)
    ref_grad_input = torch.empty((0,), dtype=dtype, device=ref_input.device)

    ref_out = torch.ops.aten.smooth_l1_loss_backward.grad_input(
        ref_grad_output,
        ref_input,
        ref_target,
        1,
        1.0,
        grad_input=ref_grad_input,
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss_backward.grad_input(
            grad_output, input, target, 1, 1.0, grad_input=grad_input
        )

    assert res_out is grad_input
    assert res_out.shape == ref_out.shape == torch.Size((2, 3, 4))
    utils.gems_assert_close(res_out, ref_out, torch.float32, equal_nan=True)
