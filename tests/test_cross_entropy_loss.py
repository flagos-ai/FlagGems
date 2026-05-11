import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    SMOOTH_IGNORE_SHAPE = [(0.1, 1, utils.REDUCTION_SHAPES[0])]
    CROSS_ENTROPY_LOSS_REDUCTION = ["mean"]
    SMOOTH_SHAPE = [(0.1, utils.REDUCTION_SHAPES[0])]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    SMOOTH_IGNORE_SHAPE = list(zip([0, 0.1, 1], [1, 200, -100], utils.REDUCTION_SHAPES))
    CROSS_ENTROPY_LOSS_REDUCTION = ["mean", "none", "sum"]
    SMOOTH_SHAPE = list(zip([1, 0.1, 0], utils.REDUCTION_SHAPES))

random.seed(time.time() // 100)


@pytest.mark.cross_entropy_loss
@pytest.mark.parametrize("label_smoothing, ignore_index, shape", SMOOTH_IGNORE_SHAPE)
@pytest.mark.parametrize("reduction", CROSS_ENTROPY_LOSS_REDUCTION)
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def testcross_entropy_loss_indices(
    shape, dtype, weight, ignore_index, reduction, label_smoothing
):
    dim = 1
    up_limit = shape[dim] - 1
    target_shape = list(shape)
    del target_shape[dim]

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    target = torch.randint(0, up_limit, target_shape, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target)

    if weight:
        wgt = torch.randn(shape[dim], dtype=dtype, device=flag_gems.device)
        ref_wgt = utils.to_reference(wgt, True)
    else:
        wgt = None
        ref_wgt = None

    ref_out = torch.nn.functional.cross_entropy(
        ref_inp,
        ref_target,
        weight=ref_wgt,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )

    res_out = flag_gems.cross_entropy_loss(
        inp,
        target,
        weight=wgt,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])


@pytest.mark.cross_entropy_loss
@pytest.mark.parametrize("label_smoothing, shape", SMOOTH_SHAPE)
@pytest.mark.parametrize("reduction", CROSS_ENTROPY_LOSS_REDUCTION)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def testcross_entropy_loss_probabilities(shape, dtype, reduction, label_smoothing):
    dim = 1
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(shape[dim], dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    ref_weight = utils.to_reference(weight, True)
    ref_out = torch.nn.functional.cross_entropy(
        ref_inp,
        ref_target,
        weight=ref_weight,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
    res_out = flag_gems.cross_entropy_loss(
        inp, target, weight=weight, reduction=reduction, label_smoothing=label_smoothing
    )

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])
