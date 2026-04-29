import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize(
    "shape",
    [
        (0,),
        (1,),
        (2, 3),
        (32, 17),
        (4, 8, 16),
        (2, 3, 16, 16),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0, 2.0])
def test_smooth_l1_loss(shape, dtype, reduction, beta):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    if len(shape) >= 2 and shape[0] > 0:
        inp = inp.transpose(0, 1).contiguous().transpose(0, 1)

    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_target = utils.to_reference(target).to(torch.float32)
    ref_out = torch.ops.aten.smooth_l1_loss(ref_inp, ref_target, reduction, beta).to(
        dtype
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss(inp, target, reduction, beta)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.smooth_l1_loss_backward
@pytest.mark.parametrize(
    "shape,target_shape",
    [
        ((0,), (0,)),
        ((1,), (1,)),
        ((2, 3), (2, 3)),
        ((32, 17), (32, 17)),
        ((4, 8, 16), (4, 8, 16)),
        ((2, 3, 16, 16), (2, 3, 16, 16)),
        ((2, 3, 4), (4,)),
        ((2, 1, 4), (3, 4)),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0, 2.0])
def test_smooth_l1_loss_backward(shape, target_shape, dtype, reduction, beta):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(target_shape, dtype=dtype, device=flag_gems.device)
    if beta == 0.0:
        inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
        target = torch.ones(target_shape, dtype=dtype, device=flag_gems.device)
    out_shape = torch.broadcast_shapes(shape, target_shape)

    if len(shape) >= 2 and shape[0] > 0:
        inp = inp.transpose(0, 1).contiguous().transpose(0, 1)

    if reduction == 0:
        grad_output = torch.randn(out_shape, dtype=dtype, device=flag_gems.device)
        if len(out_shape) >= 2 and out_shape[0] > 0:
            grad_output = grad_output.transpose(0, 1).contiguous().transpose(0, 1)
    else:
        grad_output = torch.randn((), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_target = utils.to_reference(target).to(torch.float32)
    ref_grad_output = utils.to_reference(grad_output).to(torch.float32)
    ref_out = torch.ops.aten.smooth_l1_loss_backward(
        ref_grad_output, ref_inp, ref_target, reduction, beta
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss_backward(
            grad_output, inp, target, reduction, beta
        )

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.smooth_l1_loss_backward
def test_smooth_l1_loss_backward_scalar_grad_output():
    inp = torch.tensor([-1.0, -0.5, 1.0], device=flag_gems.device)
    target = torch.zeros_like(inp)
    grad_output = torch.tensor(2.0, device=flag_gems.device)
    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_target = utils.to_reference(target).to(torch.float32)
    ref_grad_output = utils.to_reference(grad_output).to(torch.float32)

    ref_out = torch.ops.aten.smooth_l1_loss_backward(
        ref_grad_output, ref_inp, ref_target, 0, 0.0
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss_backward(
            grad_output, inp, target, 0, 0.0
        )

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.smooth_l1_loss_backward
def test_smooth_l1_loss_backward_beta_zero_equal_inputs_cuda_behavior():
    if cfg.TO_CPU:
        pytest.skip("PyTorch CPU and CUDA differ for beta=0 with equal inputs.")

    inp = torch.zeros((4,), dtype=torch.float32, device=flag_gems.device)
    target = torch.zeros_like(inp)
    grad_output = torch.ones_like(inp)
    ref_out = torch.ops.aten.smooth_l1_loss_backward(grad_output, inp, target, 0, 0.0)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss_backward(
            grad_output, inp, target, 0, 0.0
        )

    utils.gems_assert_close(res_out, ref_out, torch.float32, equal_nan=True)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_smooth_l1_loss_broadcast(dtype, reduction):
    inp = torch.randn((2, 3, 4), dtype=dtype, device=flag_gems.device)
    target = torch.randn((4,), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_target = utils.to_reference(target).to(torch.float32)
    ref_out = torch.ops.aten.smooth_l1_loss(ref_inp, ref_target, reduction, 1.0).to(
        dtype
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss(inp, target, reduction, 1.0)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_special_values(dtype):
    inp = torch.tensor(
        [0.0, -0.0, 1.0, -2.0, float("inf"), float("-inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    target = torch.tensor(
        [0.0, 1.0, -1.0, -2.0, 1.0, float("-inf"), 0.0],
        dtype=dtype,
        device=flag_gems.device,
    )

    ref_out = torch.ops.aten.smooth_l1_loss(
        utils.to_reference(inp).to(torch.float32),
        utils.to_reference(target).to(torch.float32),
        0,
        1.0,
    ).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss(inp, target, 0, 1.0)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.smooth_l1_loss
def test_smooth_l1_loss_out():
    inp = torch.randn((8, 16), dtype=torch.float32, device=flag_gems.device)
    target = torch.randn((8, 16), dtype=torch.float32, device=flag_gems.device)
    out = torch.empty_like(inp)
    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_target = utils.to_reference(target).to(torch.float32)
    ref_out = torch.empty_like(ref_inp)

    torch.ops.aten.smooth_l1_loss.out(ref_inp, ref_target, 0, 0.5, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss.out(inp, target, 0, 0.5, out=out)

    assert res_out is out
    utils.gems_assert_close(out, ref_out, torch.float32)


@pytest.mark.smooth_l1_loss
def test_smooth_l1_loss_out_reduced():
    inp = torch.randn((8, 16), dtype=torch.float32, device=flag_gems.device)
    target = torch.randn((8, 16), dtype=torch.float32, device=flag_gems.device)
    out = torch.empty_like(inp)
    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_target = utils.to_reference(target).to(torch.float32)
    ref_out = torch.empty_like(ref_inp)

    torch.ops.aten.smooth_l1_loss.out(ref_inp, ref_target, 1, 1.0, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss.out(inp, target, 1, 1.0, out=out)

    assert res_out is out
    utils.gems_assert_close(out, ref_out, torch.float32)


@pytest.mark.smooth_l1_loss
def test_smooth_l1_loss_functional():
    inp = torch.randn((8, 16), dtype=torch.float32, device=flag_gems.device)
    target = torch.randn((8, 16), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref_target = utils.to_reference(target).to(torch.float32)

    ref_out = torch.nn.functional.smooth_l1_loss(
        ref_inp, ref_target, reduction="mean", beta=0.5
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.smooth_l1_loss(
            inp, target, reduction="mean", beta=0.5
        )

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.smooth_l1_loss
def test_smooth_l1_loss_negative_beta():
    inp = torch.randn((8,), dtype=torch.float32, device=flag_gems.device)
    target = torch.randn((8,), dtype=torch.float32, device=flag_gems.device)

    with flag_gems.use_gems(), pytest.raises(RuntimeError, match="negative"):
        torch.ops.aten.smooth_l1_loss(inp, target, 1, -1.0)


@pytest.mark.smooth_l1_loss_backward
def test_smooth_l1_loss_backward_negative_beta():
    grad_output = torch.randn((), dtype=torch.float32, device=flag_gems.device)
    inp = torch.randn((8,), dtype=torch.float32, device=flag_gems.device)
    target = torch.randn((8,), dtype=torch.float32, device=flag_gems.device)

    with flag_gems.use_gems(), pytest.raises(RuntimeError, match="negative"):
        torch.ops.aten.smooth_l1_loss_backward(grad_output, inp, target, 1, -1.0)
