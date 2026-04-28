import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference
from .conftest import QUICK_MODE

FLOAT_DTYPES = [torch.float32] if QUICK_MODE else FLOAT_DTYPES

SMOOTH_L1_SHAPES = (
    [(1, 2), (64, 64), (4096, 256)]
    if QUICK_MODE
    else [
        (1, 2),
        (64, 64),
        (4096, 256),
        (200, 40999, 3),
        (1024, 1024),
        (7,),
        (33, 257),
        (1, 1024),
        (1024, 1),
        (17, 31, 53),
    ]
)

SMOOTH_L1_BETAS = [0.0, 0.5, 1.0, 2.0] if not QUICK_MODE else [0.0, 1.0]


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("shape", SMOOTH_L1_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
@pytest.mark.parametrize("beta", SMOOTH_L1_BETAS)
def test_accuracy_smooth_l1_loss(shape, dtype, reduction, beta):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target, True)

    ref_out = torch.nn.functional.smooth_l1_loss(
        ref_inp, ref_target, reduction=reduction, beta=beta
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.smooth_l1_loss(
            inp, target, reduction=reduction, beta=beta
        )

    if reduction == "none":
        gems_assert_close(res_out, ref_out, dtype)
    else:
        reduce_dim = shape[-1] if len(shape) > 0 else 1
        gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_smooth_l1_loss_special_values(dtype):
    shape = (64, 64)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    inp[0, 0] = float("nan")
    inp[0, 1] = float("inf")
    inp[0, 2] = float("-inf")
    target[1, 0] = float("nan")

    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target, True)

    ref_out = torch.nn.functional.smooth_l1_loss(
        ref_inp, ref_target, reduction="none", beta=1.0
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.smooth_l1_loss(
            inp, target, reduction="none", beta=1.0
        )

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_smooth_l1_loss_non_contiguous(dtype):
    shape = (64, 128)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device).t()
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device).t()

    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target, True)

    for reduction in ["mean", "sum", "none"]:
        ref_out = torch.nn.functional.smooth_l1_loss(
            ref_inp, ref_target, reduction=reduction, beta=1.0
        )
        with flag_gems.use_gems():
            res_out = torch.nn.functional.smooth_l1_loss(
                inp, target, reduction=reduction, beta=1.0
            )
        if reduction == "none":
            gems_assert_close(res_out, ref_out, dtype)
        else:
            gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[0])


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_smooth_l1_loss_broadcast(dtype):
    inp = torch.randn((4, 1, 64), dtype=dtype, device=flag_gems.device)
    target = torch.randn((1, 8, 64), dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target, True)

    ref_out = torch.nn.functional.smooth_l1_loss(
        ref_inp, ref_target, reduction="mean", beta=1.0
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.smooth_l1_loss(
            inp, target, reduction="mean", beta=1.0
        )
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=64)


@pytest.mark.smooth_l1_loss
def test_accuracy_smooth_l1_loss_zero_size():
    inp = torch.randn((0,), dtype=torch.float32, device=flag_gems.device)
    target = torch.randn((0,), dtype=torch.float32, device=flag_gems.device)

    ref_inp = to_reference(inp)
    ref_target = to_reference(target)

    ref_out = torch.nn.functional.smooth_l1_loss(
        ref_inp, ref_target, reduction="mean", beta=1.0
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.smooth_l1_loss(
            inp, target, reduction="mean", beta=1.0
        )
    gems_assert_close(res_out, ref_out, torch.float32, equal_nan=True)
