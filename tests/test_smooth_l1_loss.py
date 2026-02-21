"""Test cases for smooth_l1_loss operator."""
import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference
from .conftest import QUICK_MODE

# Test configurations for smooth_l1_loss
# Format: (input_shape, target_shape, reduction, beta)
SMOOTH_L1_LOSS_CONFIGS = [
    # ===== Small sizes =====
    ((8,), (8,), "none", 1.0),
    ((8,), (8,), "mean", 1.0),
    ((8,), (8,), "sum", 1.0),
    # ===== Medium sizes =====
    ((64, 64), (64, 64), "none", 1.0),
    ((64, 64), (64, 64), "mean", 1.0),
    ((64, 64), (64, 64), "sum", 1.0),
    ((256, 256), (256, 256), "mean", 1.0),
    # ===== Large sizes =====
    ((1024, 1024), (1024, 1024), "mean", 1.0),
    # ===== Different beta values =====
    ((64, 64), (64, 64), "mean", 0.5),
    ((64, 64), (64, 64), "mean", 2.0),
    ((64, 64), (64, 64), "mean", 0.1),
    # ===== Multi-dimensional =====
    ((16, 32, 32), (16, 32, 32), "mean", 1.0),
    ((8, 16, 16, 16), (8, 16, 16, 16), "mean", 1.0),
    ((4, 8, 8, 8, 8), (4, 8, 8, 8, 8), "mean", 1.0),
]

# Quick mode: use fewer test cases
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
    """Test forward pass of smooth_l1_loss."""
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(target_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target, True)

    # PyTorch reference
    ref_out = torch.nn.functional.smooth_l1_loss(
        ref_inp,
        ref_target,
        reduction=reduction,
        beta=beta,
    )

    # FlagGems implementation
    res_out = flag_gems.smooth_l1_loss(
        inp,
        target,
        reduction=reduction,
        beta=beta,
    )

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.smooth_l1_loss_backward
@pytest.mark.parametrize(
    "input_shape, target_shape, reduction, beta", SMOOTH_L1_LOSS_CONFIGS
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_smooth_l1_loss_backward(
    input_shape, target_shape, reduction, beta, dtype
):
    """Test backward pass of smooth_l1_loss."""
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    target = torch.randn(target_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(inp, upcast=True)
    ref_target = to_reference(target, upcast=True)

    # PyTorch reference forward
    ref_out = torch.nn.functional.smooth_l1_loss(
        ref_inp,
        ref_target,
        reduction=reduction,
        beta=beta,
    )

    # FlagGems forward
    res_out = flag_gems.smooth_l1_loss(
        inp,
        target,
        reduction=reduction,
        beta=beta,
    )

    # Create gradient
    if reduction == "none":
        out_grad = torch.randn_like(res_out, device=flag_gems.device)
        ref_grad = to_reference(out_grad, upcast=True)
    else:
        # For mean/sum reduction, gradient is scalar
        out_grad = torch.randn((), dtype=dtype, device=flag_gems.device)
        ref_grad = to_reference(out_grad, upcast=True)

    # PyTorch reference backward
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)

    # FlagGems backward
    res_in_grad = flag_gems.smooth_l1_loss_backward(
        out_grad,
        inp,
        target,
        reduction=reduction,
        beta=beta,
    )

    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_smooth_l1_loss_edge_cases(dtype):
    """Test edge cases for smooth_l1_loss."""
    # Test with zeros
    inp = torch.zeros((64, 64), dtype=dtype, device=flag_gems.device)
    target = torch.zeros((64, 64), dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target, True)

    ref_out = torch.nn.functional.smooth_l1_loss(ref_inp, ref_target, reduction="mean")
    res_out = flag_gems.smooth_l1_loss(inp, target, reduction="mean")
    gems_assert_close(res_out, ref_out, dtype)

    # Test with negative values
    inp = torch.randn((64, 64), dtype=dtype, device=flag_gems.device) - 5.0
    target = torch.randn((64, 64), dtype=dtype, device=flag_gems.device) - 5.0
    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target, True)

    ref_out = torch.nn.functional.smooth_l1_loss(ref_inp, ref_target, reduction="mean")
    res_out = flag_gems.smooth_l1_loss(inp, target, reduction="mean")
    gems_assert_close(res_out, ref_out, dtype)

    # Test with large differences (L1 region)
    inp = torch.randn((64, 64), dtype=dtype, device=flag_gems.device) * 10
    target = torch.randn((64, 64), dtype=dtype, device=flag_gems.device) * 10
    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target, True)

    ref_out = torch.nn.functional.smooth_l1_loss(ref_inp, ref_target, reduction="mean")
    res_out = flag_gems.smooth_l1_loss(inp, target, reduction="mean")
    gems_assert_close(res_out, ref_out, dtype)

    # Test with small differences (L2 region)
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
    """Test smooth_l1_loss with different reduction modes."""
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
    """Test smooth_l1_loss with different beta values."""
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
    """Test smooth_l1_loss with different input shapes."""
    shapes = [
        (8,),  # 1D
        (8, 8),  # 2D
        (8, 8, 8),  # 3D
        (4, 8, 8, 8),  # 4D
        (2, 4, 8, 8, 8),  # 5D
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
    """Test smooth_l1_loss with large input sizes."""
    # Large 2D
    shape = (4096, 4096)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target, True)

    ref_out = torch.nn.functional.smooth_l1_loss(ref_inp, ref_target, reduction="mean")
    res_out = flag_gems.smooth_l1_loss(inp, target, reduction="mean")

    gems_assert_close(res_out, ref_out, dtype)
