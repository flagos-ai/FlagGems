import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    # QUICK_MODE uses float32 only for faster CI testing
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

# Make sure every thread has same seed.
random.seed(time.time() // 100)


@pytest.mark.binary_cross_entropy
@pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_binary_cross_entropy(shape, dtype, reduction):
    # Generate input in (0, 1) range using sigmoid
    inp = torch.sigmoid(torch.randn(shape, dtype=dtype, device=flag_gems.device))
    # Generate binary targets (0 or 1)
    target = torch.randint(0, 2, shape, device=flag_gems.device).to(dtype)

    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)

    ref_out = torch.nn.functional.binary_cross_entropy(
        ref_inp, ref_target, reduction=reduction
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.binary_cross_entropy(
            inp, target, reduction=reduction
        )

    if reduction == "none":
        # Elementwise comparison, no reduction error to account for
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    elif reduction == "sum":
        # Sum reduction over all elements; tolerance scales with numel
        # due to floating-point accumulation in atomic_add
        utils.gems_assert_close(
            res_out,
            ref_out,
            dtype,
            equal_nan=True,
            reduce_dim=inp.numel(),
        )
    else:
        # Mean reduction normalizes per-element error
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.binary_cross_entropy
@pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_binary_cross_entropy_weight(shape, dtype, reduction):
    # Generate input in (0, 1) range using sigmoid
    inp = torch.sigmoid(torch.randn(shape, dtype=dtype, device=flag_gems.device))
    # Generate binary targets (0 or 1)
    target = torch.randint(0, 2, shape, device=flag_gems.device).to(dtype)
    # Generate positive weights
    weight = torch.rand(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.nn.functional.binary_cross_entropy(
        ref_inp, ref_target, weight=ref_weight, reduction=reduction
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.binary_cross_entropy(
            inp, target, weight=weight, reduction=reduction
        )

    if reduction == "none":
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    elif reduction == "sum":
        utils.gems_assert_close(
            res_out,
            ref_out,
            dtype,
            equal_nan=True,
            reduce_dim=inp.numel(),
        )
    else:
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# Map reduction string to aten enum: 0=none, 1=mean, 2=sum
_REDUCTION_ENUM = {"none": 0, "mean": 1, "sum": 2}


@pytest.mark.binary_cross_entropy
@pytest.mark.parametrize("with_weight", [False, True])
@pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_binary_cross_entropy_out(shape, dtype, reduction, with_weight):
    # Generate input in (0, 1) range using sigmoid
    inp = torch.sigmoid(torch.randn(shape, dtype=dtype, device=flag_gems.device))
    # Generate binary targets (0 or 1)
    target = torch.randint(0, 2, shape, device=flag_gems.device).to(dtype)
    weight = (
        torch.rand(shape, dtype=dtype, device=flag_gems.device) if with_weight else None
    )

    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    ref_weight = utils.to_reference(weight, True) if with_weight else None

    ref_out = torch.nn.functional.binary_cross_entropy(
        ref_inp, ref_target, weight=ref_weight, reduction=reduction
    )

    # Pre-fill the out buffer with garbage so we can confirm it is written into.
    if reduction == "none":
        out = torch.full_like(inp, -123.0)
    else:
        out = torch.full((), -123.0, dtype=inp.dtype, device=flag_gems.device)

    red_enum = _REDUCTION_ENUM[reduction]
    with flag_gems.use_gems():
        returned = torch.ops.aten.binary_cross_entropy.out(
            inp, target, weight, red_enum, out=out
        )

    # out-variant semantics: the returned tensor must be the same object as `out`,
    # and the result must actually live in the caller-provided buffer.
    assert returned is out, "out variant must return the provided out tensor"

    if reduction == "none":
        utils.gems_assert_close(out, ref_out, dtype, equal_nan=True)
    elif reduction == "sum":
        utils.gems_assert_close(
            out,
            ref_out,
            dtype,
            equal_nan=True,
            reduce_dim=inp.numel(),
        )
    else:
        utils.gems_assert_close(out, ref_out, dtype, equal_nan=True)


@pytest.mark.binary_cross_entropy
@pytest.mark.parametrize("with_weight", [False, True])
@pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_binary_cross_entropy_boundary(dtype, reduction, with_weight):
    # PyTorch clamps the internal log to a lower bound of -100, so input==0 with
    # target==1 (and input==1 with target==0) yields exactly 100 rather than inf.
    inp = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.5], dtype=dtype, device=flag_gems.device)
    target = torch.tensor(
        [1.0, 0.0, 0.0, 1.0, 1.0], dtype=dtype, device=flag_gems.device
    )
    weight = (
        torch.rand(inp.shape, dtype=dtype, device=flag_gems.device)
        if with_weight
        else None
    )

    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    ref_weight = utils.to_reference(weight, True) if with_weight else None

    ref_out = torch.nn.functional.binary_cross_entropy(
        ref_inp, ref_target, weight=ref_weight, reduction=reduction
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.binary_cross_entropy(
            inp, target, weight=weight, reduction=reduction
        )

    if reduction == "sum":
        utils.gems_assert_close(
            res_out, ref_out, dtype, equal_nan=True, reduce_dim=inp.numel()
        )
    else:
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
