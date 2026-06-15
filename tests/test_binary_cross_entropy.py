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


@pytest.mark.binary_cross_entropy
@pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_binary_cross_entropy_out(shape, dtype, reduction):
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
