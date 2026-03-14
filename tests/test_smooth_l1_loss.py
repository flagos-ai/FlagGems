import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference
from .conftest import QUICK_MODE

SMOOTH_L1_LOSS_CONFIGS = [
    ((8,), (8,), "none", 1.0),
    ((8,), (8,), "mean", 1.0),
    ((8,), (8,), "sum", 1.0),
    ((64, 64), (64, 64), "none", 1.0),
    ((64, 64), (64, 64), "mean", 1.0),
    ((64, 64), (64, 64), "sum", 1.0),
    ((256, 256), (256, 256), "mean", 1.0),
    ((1024, 1024), (1024, 1024), "mean", 1.0),
    ((64, 64), (64, 64), "mean", 0.5),
    ((64, 64), (64, 64), "mean", 2.0),
    ((64, 64), (64, 64), "mean", 0.1),
    ((16, 32, 32), (16, 32, 32), "mean", 1.0),
    ((8, 16, 16, 16), (8, 16, 16, 16), "mean", 1.0),
    ((4, 8, 8, 8, 8), (4, 8, 8, 8, 8), "mean", 1.0),
]

if QUICK_MODE:
    SMOOTH_L1_LOSS_CONFIGS = [
        SMOOTH_L1_LOSS_CONFIGS[0],
        SMOOTH_L1_LOSS_CONFIGS[4],
    ]
    FLOAT_DTYPES_TEST = [torch.float32]
else:
    FLOAT_DTYPES_TEST = FLOAT_DTYPES


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize(
    "input_shape, target_shape, reduction, beta", SMOOTH_L1_LOSS_CONFIGS
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_smooth_l1_loss_forward(
    input_shape, target_shape, reduction, beta, dtype
):
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(target_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target, True)
    ref_out = torch.nn.functional.smooth_l1_loss(
        ref_inp, ref_target, reduction=reduction, beta=beta
    )
    res_out = flag_gems.smooth_l1_loss(inp, target, reduction=reduction, beta=beta)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.smooth_l1_loss_backward
@pytest.mark.parametrize(
    "input_shape, target_shape, reduction, beta", SMOOTH_L1_LOSS_CONFIGS
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_smooth_l1_loss_backward(
    input_shape, target_shape, reduction, beta, dtype
):
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    target = torch.randn(target_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, upcast=True)
    ref_target = to_reference(target, upcast=True)
    ref_out = torch.nn.functional.smooth_l1_loss(
        ref_inp, ref_target, reduction=reduction, beta=beta
    )
    res_out = flag_gems.smooth_l1_loss(inp, target, reduction=reduction, beta=beta)
    if reduction == "none":
        out_grad = torch.randn_like(res_out, device=flag_gems.device)
        ref_grad = to_reference(out_grad, upcast=True)
    else:
        out_grad = torch.randn((), dtype=dtype, device=flag_gems.device)
        ref_grad = to_reference(out_grad, upcast=True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    res_in_grad = flag_gems.smooth_l1_loss_backward(
        out_grad, inp, target, reduction=reduction, beta=beta
    )
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_smooth_l1_loss_edge_cases(dtype):
    inp = torch.zeros((64, 64), dtype=dtype, device=flag_gems.device)
    target = torch.zeros((64, 64), dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target, True)
    ref_out = torch.nn.functional.smooth_l1_loss(ref_inp, ref_target, reduction="mean")
    res_out = flag_gems.smooth_l1_loss(inp, target, reduction="mean")
    gems_assert_close(res_out, ref_out, dtype)

    inp = torch.randn((64, 64), dtype=dtype, device=flag_gems.device) - 5.0
    target = torch.randn((64, 64), dtype=dtype, device=flag_gems.device) - 5.0
    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target, True)
    ref_out = torch.nn.functional.smooth_l1_loss(ref_inp, ref_target, reduction="mean")
    res_out = flag_gems.smooth_l1_loss(inp, target, reduction="mean")
    gems_assert_close(res_out, ref_out, dtype)

    inp = torch.randn((64, 64), dtype=dtype, device=flag_gems.device) * 10
    target = torch.randn((64, 64), dtype=dtype, device=flag_gems.device) * 10
    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target, True)
    ref_out = torch.nn.functional.smooth_l1_loss(ref_inp, ref_target, reduction="mean")
    res_out = flag_gems.smooth_l1_loss(inp, target, reduction="mean")
    gems_assert_close(res_out, ref_out, dtype)

    inp = torch.randn((64, 64), dtype=dtype, device=flag_gems.device) * 0.1
    target = torch.randn((64, 64), dtype=dtype, device=flag_gems.device) * 0.1
    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target, True)
    ref_out = torch.nn.functional.smooth_l1_loss(ref_inp, ref_target, reduction="mean")
    res_out = flag_gems.smooth_l1_loss(inp, target, reduction="mean")
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_smooth_l1_loss_different_reductions(dtype):
    shape = (64, 64)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    for reduction in ["none", "mean", "sum"]:
        ref_inp = to_reference(inp, True)
        ref_target = to_reference(target, True)
        ref_out = torch.nn.functional.smooth_l1_loss(
            ref_inp, ref_target, reduction=reduction
        )
        res_out = flag_gems.smooth_l1_loss(inp, target, reduction=reduction)
        gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_smooth_l1_loss_different_betas(dtype):
    shape = (64, 64)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    for beta in [0.1, 0.5, 1.0, 2.0, 5.0]:
        ref_inp = to_reference(inp, True)
        ref_target = to_reference(target, True)
        ref_out = torch.nn.functional.smooth_l1_loss(
            ref_inp, ref_target, reduction="mean", beta=beta
        )
        res_out = flag_gems.smooth_l1_loss(inp, target, reduction="mean", beta=beta)
        gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_smooth_l1_loss_different_shapes(dtype):
    shapes = [
        (8,),
        (8, 8),
        (8, 8, 8),
        (4, 8, 8, 8),
        (2, 4, 8, 8, 8),
    ]
    for shape in shapes:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        target = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = to_reference(inp, True)
        ref_target = to_reference(target, True)
        ref_out = torch.nn.functional.smooth_l1_loss(
            ref_inp, ref_target, reduction="mean"
        )
        res_out = flag_gems.smooth_l1_loss(inp, target, reduction="mean")
        gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_smooth_l1_loss_large_input(dtype):
    shape = (4096, 4096)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target, True)
    ref_out = torch.nn.functional.smooth_l1_loss(ref_inp, ref_target, reduction="mean")
    res_out = flag_gems.smooth_l1_loss(inp, target, reduction="mean")
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_smooth_l1_loss_beta_zero(dtype):
    shape = (64, 64)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target, True)
    ref_out = torch.nn.functional.smooth_l1_loss(
        ref_inp, ref_target, reduction="mean", beta=0.0
    )
    res_out = flag_gems.smooth_l1_loss(inp, target, reduction="mean", beta=0.0)
    gems_assert_close(res_out, ref_out, dtype)
