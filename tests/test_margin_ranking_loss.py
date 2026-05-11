import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.margin_ranking_loss
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 256)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("margin", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_margin_ranking_loss(shape, dtype, margin, reduction):
    input1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    input2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    target = (
        torch.randint(0, 2, shape, device=flag_gems.device, dtype=torch.int8) * 2 - 1
    ).to(dtype)

    ref_input1 = utils.to_reference(input1)
    ref_input2 = utils.to_reference(input2)
    ref_target = utils.to_reference(target)
    ref_out = torch.ops.aten.margin_ranking_loss(
        ref_input1, ref_input2, ref_target, margin, reduction
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.margin_ranking_loss(
            input1, input2, target, margin, reduction
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


REDUCTION_MAP = {0: "none", 1: "mean", 2: "sum"}


@pytest.mark.margin_ranking_loss
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 256)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("margin", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_margin_ranking_loss_backward(shape, dtype, margin, reduction):
    input1 = torch.randn(
        shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    input2 = torch.randn(
        shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    )

    target = (
        torch.randint(0, 2, shape, device=flag_gems.device, dtype=torch.int8) * 2 - 1
    ).to(dtype)

    # Avoid boundary elements where val = -y*(x1-x2)+margin ≈ 0, since the
    # discontinuous gradient at the boundary can differ between Triton (float32
    # internal compute) and PyTorch native (input dtype compute).
    with torch.no_grad():
        val = -target * (input1 - input2) + margin
        boundary_mask = val.abs() < 0.01
        if boundary_mask.any():
            # Push boundary elements away from zero by adjusting x1
            input1.data[boundary_mask] += 0.1 * target.data[boundary_mask]

    ref_input1 = utils.to_reference(input1)
    ref_input2 = utils.to_reference(input2)
    ref_target = utils.to_reference(target)

    ref_out = torch.nn.functional.margin_ranking_loss(
        ref_input1,
        ref_input2,
        ref_target,
        margin=margin,
        reduction=REDUCTION_MAP[reduction],
    )

    with flag_gems.use_gems():
        res_out = torch.nn.functional.margin_ranking_loss(
            input1,
            input2,
            target,
            margin=margin,
            reduction=REDUCTION_MAP[reduction],
        )

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad)

    ref_in_grad1, ref_in_grad2 = torch.autograd.grad(
        ref_out,
        (ref_input1, ref_input2),
        ref_grad,
    )
    res_in_grad1, res_in_grad2 = torch.autograd.grad(
        res_out,
        (input1, input2),
        out_grad,
    )

    utils.gems_assert_close(res_in_grad1, ref_in_grad1, dtype)
    utils.gems_assert_close(res_in_grad2, ref_in_grad2, dtype)
