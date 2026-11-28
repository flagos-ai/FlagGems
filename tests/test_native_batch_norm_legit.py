# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import pytest
import torch

import flag_gems
from flag_gems.ops._native_batch_norm_legit import _bn_legit_bwd_run

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    TRAININGS = [True]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    TRAININGS = [True, False]

SHAPES = [
    (16, 3),
    (32, 32, 32),
    (8, 32, 224, 224),
    (2050, 16, 32, 32),
    (8, 16, 3, 224, 224),
]


@pytest.mark.native_batch_norm_legit
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("training", TRAININGS)
def test_native_batch_norm_legit(shape, dtype, affine, training):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(23)
        torch.mlu.manual_seed_all(23)
    C = shape[1]

    inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    weight = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device) if affine else None
    )
    bias = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device) if affine else None
    )
    running_mean = torch.zeros(size=(C,), dtype=dtype, device=flag_gems.device)
    running_var = torch.ones(size=(C,), dtype=dtype, device=flag_gems.device)

    eps = 1e-5
    momentum = 0.1

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)
    ref_running_mean = utils.to_reference(running_mean, True)
    ref_running_var = utils.to_reference(running_var, True)

    (
        ref_out,
        ref_save_mean,
        ref_save_var,
    ) = torch.ops.aten._native_batch_norm_legit.default(
        ref_inp,
        ref_weight,
        ref_bias,
        ref_running_mean,
        ref_running_var,
        training,
        momentum,
        eps,
    )

    (
        res_out,
        res_save_mean,
        res_save_var,
    ) = flag_gems._native_batch_norm_legit(
        inp,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(running_mean, ref_running_mean, dtype)
    utils.gems_assert_close(running_var, ref_running_var, dtype)
    if training:
        utils.gems_assert_close(res_save_mean, ref_save_mean, dtype)
        utils.gems_assert_close(res_save_var, ref_save_var, dtype)


@pytest.mark.native_batch_norm_legit_no_stats
@pytest.mark.parametrize("shape", [(16, 3), (8, 16, 32), (4, 8, 16, 16)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("affine", [True, False])
def test_native_batch_norm_legit_no_stats(shape, dtype, affine):
    channels = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    original = utils.to_reference(inp)
    weight = (
        torch.randn(channels, dtype=dtype, device=flag_gems.device) if affine else None
    )
    bias = (
        torch.randn(channels, dtype=dtype, device=flag_gems.device) if affine else None
    )

    ref = torch.ops.aten._native_batch_norm_legit.no_stats(
        utils.to_reference(inp, True),
        utils.to_reference(weight, True),
        utils.to_reference(bias, True),
        True,
        0.1,
        1e-5,
    )
    result = flag_gems._native_batch_norm_legit_no_stats(
        inp, weight, bias, True, 0.1, 1e-5
    )

    for actual, expected in zip(result, ref):
        utils.gems_assert_close(actual, expected, dtype)
    utils.gems_assert_equal(inp, original)


@pytest.mark.native_batch_norm_legit_out
@pytest.mark.native_batch_norm_legit_no_stats_out
@pytest.mark.parametrize(
    "overload",
    [
        pytest.param("out", marks=pytest.mark.native_batch_norm_legit_out),
        pytest.param(
            "no_stats_out", marks=pytest.mark.native_batch_norm_legit_no_stats_out
        ),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_native_batch_norm_legit_out(overload, dtype):
    shape = (4, 8, 16, 16)
    channels = shape[1]
    stats_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(channels, dtype=dtype, device=flag_gems.device)
    bias = torch.randn(channels, dtype=dtype, device=flag_gems.device)

    # Upcast the reference path so the (optionally CPU) reference is computed
    # in high precision, matching how the other tests build their baseline.
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    # The reference out tensors follow the (upcast) reference dtype/device; the
    # gems-side out tensors stay in the native dtype on flag_gems.device.
    ref_out = torch.empty_like(ref_inp)
    ref_mean = torch.empty(channels, dtype=ref_inp.dtype, device=ref_inp.device)
    ref_invstd = torch.empty_like(ref_mean)
    out = torch.empty_like(inp)
    save_mean = torch.empty(channels, dtype=stats_dtype, device=flag_gems.device)
    save_invstd = torch.empty_like(save_mean)

    if overload == "out":
        running_mean = torch.zeros(channels, dtype=dtype, device=flag_gems.device)
        running_var = torch.ones(channels, dtype=dtype, device=flag_gems.device)
        ref_running_mean = utils.to_reference(running_mean.clone(), True)
        ref_running_var = utils.to_reference(running_var.clone(), True)
        ref = torch.ops.aten._native_batch_norm_legit.out(
            ref_inp,
            ref_weight,
            ref_bias,
            ref_running_mean,
            ref_running_var,
            True,
            0.1,
            1e-5,
            out=ref_out,
            save_mean=ref_mean,
            save_invstd=ref_invstd,
        )
        result = flag_gems._native_batch_norm_legit_out(
            inp,
            weight,
            bias,
            running_mean,
            running_var,
            True,
            0.1,
            1e-5,
            out=out,
            save_mean=save_mean,
            save_invstd=save_invstd,
        )
        utils.gems_assert_close(running_mean, ref_running_mean, dtype)
        utils.gems_assert_close(running_var, ref_running_var, dtype)
    else:
        ref = torch.ops.aten._native_batch_norm_legit.no_stats_out(
            ref_inp,
            ref_weight,
            ref_bias,
            True,
            0.1,
            1e-5,
            out=ref_out,
            save_mean=ref_mean,
            save_invstd=ref_invstd,
        )
        result = flag_gems._native_batch_norm_legit_no_stats_out(
            inp,
            weight,
            bias,
            True,
            0.1,
            1e-5,
            out=out,
            save_mean=save_mean,
            save_invstd=save_invstd,
        )

    assert result[0].data_ptr() == out.data_ptr()
    assert result[1].data_ptr() == save_mean.data_ptr()
    assert result[2].data_ptr() == save_invstd.data_ptr()
    # gems-side output dtypes: out follows the input dtype, stats stay float32.
    # The reference may be upcast, so compare dtypes against the gems-side
    # expectation rather than the (possibly upcast) reference dtype.
    expected_dtypes = (dtype, stats_dtype, stats_dtype)
    for actual, expected, expected_dtype in zip(result, ref, expected_dtypes):
        assert actual.dtype == expected_dtype
        utils.gems_assert_close(actual.to(dtype), expected.to(dtype), dtype)


def _backward_reference(shape, dtype, training, affine, device):
    """Build the forward graph and return its native gradients.

    The reference reuses the same saved statistics the FlagGems forward hands to
    its backward, so the comparison isolates the backward kernel rather than the
    forward reduction order.
    """
    channels = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(channels, dtype=dtype, device=device) if affine else None
    bias = torch.randn(channels, dtype=dtype, device=device) if affine else None
    running_mean = torch.zeros(channels, dtype=dtype, device=device)
    running_var = torch.ones(channels, dtype=dtype, device=device)
    grad = torch.randn(shape, dtype=dtype, device=device)
    return inp, weight, bias, running_mean, running_var, grad


@pytest.mark.native_batch_norm_legit_backward
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("training", TRAININGS)
def test_native_batch_norm_legit_backward(shape, dtype, affine, training):
    channels = shape[1]
    inp, weight, bias, running_mean, running_var, grad = _backward_reference(
        shape, dtype, training, affine, flag_gems.device
    )
    reduction = math.prod(shape) // channels

    res_inp = inp.detach().clone().requires_grad_(True)
    res_weight = (
        None if weight is None else weight.detach().clone().requires_grad_(True)
    )
    res_bias = None if bias is None else bias.detach().clone().requires_grad_(True)
    out, save_mean, save_invstd = flag_gems._native_batch_norm_legit(
        res_inp,
        res_weight,
        res_bias,
        running_mean.clone(),
        running_var.clone(),
        training,
        0.1,
        1e-5,
    )
    # The ATen node keeps the saved statistics out of the graph.
    assert save_mean.requires_grad is False
    assert save_invstd.requires_grad is False
    assert type(out.grad_fn).__name__ == "_NativeBatchNormLegitBackwardBackward"

    targets = [t for t in (res_inp, res_weight, res_bias) if t is not None]
    res_grads = torch.autograd.grad(out, targets, grad)

    # The reference runs on the CPU under ``--ref cpu``, so its tensors are
    # built from the reference copies and moved back for the comparison.
    ref_inp = utils.to_reference(inp, True).detach().requires_grad_(True)
    ref_weight = (
        None
        if weight is None
        else utils.to_reference(weight, True).detach().requires_grad_(True)
    )
    ref_bias = (
        None
        if bias is None
        else utils.to_reference(bias, True).detach().requires_grad_(True)
    )
    ref_out, _, _ = torch.ops.aten._native_batch_norm_legit.default(
        ref_inp,
        ref_weight,
        ref_bias,
        utils.to_reference(running_mean, True).clone(),
        utils.to_reference(running_var, True).clone(),
        training,
        0.1,
        1e-5,
    )
    ref_grad = utils.to_reference(grad, True)
    ref_grads = torch.autograd.grad(
        ref_out,
        [t for t in (ref_inp, ref_weight, ref_bias) if t is not None],
        ref_grad,
    )

    # ``--ref cpu`` upcasts the reference to float64; compare in the operator's
    # own dtype as the other backward tests in this repository do.
    utils.gems_assert_close(out, ref_out.to(out.dtype), dtype)
    for res_grad, ref_grad in zip(res_grads, ref_grads):
        ref_grad = ref_grad.to(res_grad.dtype)
        assert res_grad.shape == ref_grad.shape
        assert res_grad.dtype == ref_grad.dtype
        utils.gems_assert_close(
            res_grad, ref_grad, dtype, reduce_dim=reduction, atol=1e-2
        )


@pytest.mark.native_batch_norm_legit_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_native_batch_norm_legit_backward_noncontiguous(dtype):
    shape = (8, 8, 6, 10)
    channels = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Both layouts are dense but not row-major, so they exercise the strided
    # gradient path; ``channels_last`` must also keep ATen's memory format.
    channels_last = inp.to(memory_format=torch.channels_last)
    assert channels_last.is_contiguous(memory_format=torch.channels_last)
    weight = torch.randn(channels, dtype=dtype, device=flag_gems.device)
    reduction = math.prod(shape) // channels
    layouts = {
        "channels_last": channels_last,
        "permuted": inp.transpose(2, 3),
    }
    for name, value in layouts.items():
        grad = torch.randn_like(value)
        res_inp = value.detach().clone().requires_grad_(True)
        # The reference leaf lives wherever ``to_reference`` puts it (CPU under
        # ``--ref cpu``) and keeps the layout being exercised.
        ref_leaf = utils.to_reference(value, True)
        ref_inp = ref_leaf.detach().clone().requires_grad_(True)
        res_out = flag_gems._native_batch_norm_legit(
            res_inp,
            weight,
            None,
            torch.zeros(channels, dtype=dtype, device=flag_gems.device),
            torch.ones(channels, dtype=dtype, device=flag_gems.device),
            True,
            0.1,
            1e-5,
        )[0]
        ref_stats = (
            torch.zeros(channels, dtype=ref_inp.dtype, device=ref_inp.device),
            torch.ones(channels, dtype=ref_inp.dtype, device=ref_inp.device),
        )
        ref_out = torch.ops.aten._native_batch_norm_legit.default(
            ref_inp,
            utils.to_reference(weight, True),
            None,
            ref_stats[0],
            ref_stats[1],
            True,
            0.1,
            1e-5,
        )[0]
        (res_grad,) = torch.autograd.grad(res_out, res_inp, grad)
        (ref_grad,) = torch.autograd.grad(
            ref_out, ref_inp, utils.to_reference(grad, True)
        )
        ref_grad = ref_grad.to(res_grad.dtype)
        # The gradient buffer must follow ATen's memory-format choice.
        assert res_grad.stride() == ref_grad.stride(), name
        utils.gems_assert_close(
            res_grad, ref_grad, dtype, reduce_dim=reduction, atol=1e-2
        )


@pytest.mark.native_batch_norm_legit_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    "mask", [[True, True, True], [True, False, False], [False, True, False]]
)
def test_native_batch_norm_legit_backward_output_mask(dtype, mask):
    shape = (4, 8, 5, 7)
    channels = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(channels, dtype=dtype, device=flag_gems.device)
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    save_mean = torch.zeros(channels, dtype=torch.float32, device=flag_gems.device)
    save_invstd = torch.ones(channels, dtype=torch.float32, device=flag_gems.device)
    # grad_weight/grad_bias are undivided channel sums, so the tolerance follows
    # the number of reduced elements (same rule as test_batch_norm_backward).
    reduction = math.prod(shape) // channels
    res = _bn_legit_bwd_run(
        grad,
        inp,
        weight,
        None,
        None,
        save_mean,
        save_invstd,
        True,
        1e-5,
        mask,
    )
    ref = torch.ops.aten.native_batch_norm_backward.default(
        utils.to_reference(grad, True),
        utils.to_reference(inp, True),
        utils.to_reference(weight, True),
        None,
        None,
        utils.to_reference(save_mean, True),
        utils.to_reference(save_invstd, True),
        True,
        1e-5,
        mask,
    )
    for actual, expected in zip(res, ref):
        assert (actual is None) == (expected is None)
        if actual is not None:
            utils.gems_assert_close(
                actual,
                expected.to(actual.dtype),
                dtype,
                reduce_dim=reduction,
                atol=1e-2,
            )
