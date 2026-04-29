import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


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
    ref_out = torch.empty_like(inp)

    torch.ops.aten.smooth_l1_loss.out(inp, target, 0, 0.5, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss.out(inp, target, 0, 0.5, out=out)

    assert res_out is out
    utils.gems_assert_close(out, ref_out, torch.float32)


@pytest.mark.smooth_l1_loss
def test_smooth_l1_loss_out_reduced():
    inp = torch.randn((8, 16), dtype=torch.float32, device=flag_gems.device)
    target = torch.randn((8, 16), dtype=torch.float32, device=flag_gems.device)
    out = torch.empty((), dtype=torch.float32, device=flag_gems.device)
    ref_out = torch.empty((), dtype=torch.float32, device=flag_gems.device)

    torch.ops.aten.smooth_l1_loss.out(inp, target, 1, 1.0, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss.out(inp, target, 1, 1.0, out=out)

    assert res_out is out
    utils.gems_assert_close(out, ref_out, torch.float32)


@pytest.mark.smooth_l1_loss
def test_smooth_l1_loss_functional():
    inp = torch.randn((8, 16), dtype=torch.float32, device=flag_gems.device)
    target = torch.randn((8, 16), dtype=torch.float32, device=flag_gems.device)

    ref_out = torch.nn.functional.smooth_l1_loss(
        inp, target, reduction="mean", beta=0.5
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
