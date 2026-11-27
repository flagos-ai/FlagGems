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

from . import accuracy_utils as utils


@pytest.mark.batch_norm_with_update_backward
@pytest.mark.parametrize(
    "shape",
    [
        (16, 3),
        (32, 32, 32),
        (8, 32, 224, 224),
        (2050, 16, 32, 32),
        (8, 16, 3, 224, 224),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("affine", [True, False])
def test_batch_norm_with_update_backward(shape, dtype, affine):
    # The op is dispatched through the ATen registration, so the test calls
    # flag_gems.batch_norm_with_update_backward directly; flag_gems.use_gems()
    # would reroute the reference comparison instead of the kernel under
    # test, so the test stays in the upstream no-context style.
    C = shape[1]
    inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    weight = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device) if affine else None
    )
    bias = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device) if affine else None
    )
    grad_out = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)

    running_mean = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
    running_var = (
        torch.abs(torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)) + 0.1
    )

    momentum = 0.1
    eps = 1e-5
    if affine:
        output_mask = [True, True, True]
    else:
        output_mask = [True, False, False]

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)
    ref_grad_out = utils.to_reference(grad_out, True)
    ref_running_mean = utils.to_reference(running_mean.clone(), True)
    ref_running_var = utils.to_reference(running_var.clone(), True)

    (
        _,
        ref_save_mean,
        ref_save_invstd,
        _,
    ) = torch.ops.aten._batch_norm_with_update(
        ref_inp,
        ref_weight,
        ref_bias,
        ref_running_mean,
        ref_running_var,
        momentum,
        eps,
    )

    (
        ref_in_grad,
        ref_weight_grad,
        ref_bias_grad,
    ) = torch.ops.aten.native_batch_norm_backward(
        ref_grad_out,
        ref_inp,
        ref_weight,
        ref_running_mean,
        ref_running_var,
        ref_save_mean,
        ref_save_invstd,
        True,
        eps,
        output_mask,
    )

    res_running_mean = running_mean.clone()
    res_running_var = running_var.clone()
    (
        _,
        res_save_mean,
        res_save_invstd,
        _,
    ) = flag_gems._batch_norm_with_update(
        inp,
        weight,
        bias,
        res_running_mean,
        res_running_var,
        momentum,
        eps,
    )

    (
        res_in_grad,
        res_weight_grad,
        res_bias_grad,
    ) = flag_gems.batch_norm_with_update_backward(
        grad_out,
        inp,
        weight,
        res_running_mean,
        res_running_var,
        res_save_mean,
        res_save_invstd,
        True,
        eps,
        output_mask,
    )

    reduce_dim = math.prod(shape) // C
    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=reduce_dim)
    if affine:
        utils.gems_assert_close(
            res_weight_grad, ref_weight_grad, dtype, reduce_dim=reduce_dim
        )
        utils.gems_assert_close(
            res_bias_grad, ref_bias_grad, dtype, reduce_dim=reduce_dim
        )


@pytest.mark.batch_norm_with_update_backward
def test_batch_norm_with_update_backward_2d_layout():
    # 2D inputs use the (N, C) layout; the 3D flatten maps spatial dim to 1.
    shape = (64, 4)
    dtype = torch.float32
    inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    grad_out = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)

    running_mean = torch.randn(size=(4,), dtype=dtype, device=flag_gems.device)
    running_var = (
        torch.abs(torch.randn(size=(4,), dtype=dtype, device=flag_gems.device)) + 0.1
    )

    momentum = 0.1
    eps = 1e-5

    ref_inp = utils.to_reference(inp, True)
    ref_grad_out = utils.to_reference(grad_out, True)
    ref_running_mean = utils.to_reference(running_mean.clone(), True)
    ref_running_var = utils.to_reference(running_var.clone(), True)

    (
        _,
        ref_save_mean,
        ref_save_invstd,
        _,
    ) = torch.ops.aten._batch_norm_with_update(
        ref_inp,
        None,
        None,
        ref_running_mean,
        ref_running_var,
        momentum,
        eps,
    )

    ref_in_grad, _, _ = torch.ops.aten.native_batch_norm_backward(
        ref_grad_out,
        ref_inp,
        None,
        ref_running_mean,
        ref_running_var,
        ref_save_mean,
        ref_save_invstd,
        True,
        eps,
        [True, False, False],
    )

    res_running_mean = running_mean.clone()
    res_running_var = running_var.clone()
    (
        _,
        res_save_mean,
        res_save_invstd,
        _,
    ) = flag_gems._batch_norm_with_update(
        inp,
        None,
        None,
        res_running_mean,
        res_running_var,
        momentum,
        eps,
    )

    res_in_grad, _, _ = flag_gems.batch_norm_with_update_backward(
        grad_out,
        inp,
        None,
        res_running_mean,
        res_running_var,
        res_save_mean,
        res_save_invstd,
        True,
        eps,
        [True, False, False],
    )

    reduce_dim = math.prod(shape) // 4
    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=reduce_dim)


@pytest.mark.batch_norm_with_update_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_batch_norm_with_update_backward_output_mask(dtype):
    # output_mask entries select which gradients are computed, like ATen.
    C = 8
    shape = (16, C, 64)
    inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
    bias = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
    grad_out = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)

    running_mean = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
    running_var = (
        torch.abs(torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)) + 0.1
    )

    momentum = 0.1
    eps = 1e-5

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)
    ref_grad_out = utils.to_reference(grad_out, True)
    ref_running_mean = utils.to_reference(running_mean.clone(), True)
    ref_running_var = utils.to_reference(running_var.clone(), True)

    (
        _,
        ref_save_mean,
        ref_save_invstd,
        _,
    ) = torch.ops.aten._batch_norm_with_update(
        ref_inp,
        ref_weight,
        ref_bias,
        ref_running_mean,
        ref_running_var,
        momentum,
        eps,
    )

    res_running_mean = running_mean.clone()
    res_running_var = running_var.clone()
    (
        _,
        res_save_mean,
        res_save_invstd,
        _,
    ) = flag_gems._batch_norm_with_update(
        inp,
        weight,
        bias,
        res_running_mean,
        res_running_var,
        momentum,
        eps,
    )

    ref_gi, _, _ = torch.ops.aten.native_batch_norm_backward(
        ref_grad_out,
        ref_inp,
        ref_weight,
        ref_running_mean,
        ref_running_var,
        ref_save_mean,
        ref_save_invstd,
        True,
        eps,
        [True, False, False],
    )

    res_gi, res_gw, res_gb = flag_gems.batch_norm_with_update_backward(
        grad_out,
        inp,
        weight,
        res_running_mean,
        res_running_var,
        res_save_mean,
        res_save_invstd,
        True,
        eps,
        [True, False, False],
    )

    reduce_dim = math.prod(shape) // C
    utils.gems_assert_close(res_gi, ref_gi, dtype, reduce_dim=reduce_dim)
    assert res_gw is None
    assert res_gb is None


@pytest.mark.batch_norm_with_update_backward
def test_batch_norm_with_update_backward_5d():
    # 5D input: spatial dims fold into one axis, validating the flatten logic.
    shape = (8, 16, 3, 224, 224)
    dtype = torch.float32
    inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(size=(16,), dtype=dtype, device=flag_gems.device)
    bias = torch.randn(size=(16,), dtype=dtype, device=flag_gems.device)
    grad_out = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)

    running_mean = torch.randn(size=(16,), dtype=dtype, device=flag_gems.device)
    running_var = (
        torch.abs(torch.randn(size=(16,), dtype=dtype, device=flag_gems.device)) + 0.1
    )

    momentum = 0.1
    eps = 1e-5

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)
    ref_grad_out = utils.to_reference(grad_out, True)
    ref_running_mean = utils.to_reference(running_mean.clone(), True)
    ref_running_var = utils.to_reference(running_var.clone(), True)

    (
        _,
        ref_save_mean,
        ref_save_invstd,
        _,
    ) = torch.ops.aten._batch_norm_with_update(
        ref_inp,
        ref_weight,
        ref_bias,
        ref_running_mean,
        ref_running_var,
        momentum,
        eps,
    )

    ref_in_grad, ref_weight_grad, ref_bias_grad = (
        torch.ops.aten.native_batch_norm_backward(
            ref_grad_out,
            ref_inp,
            ref_weight,
            ref_running_mean,
            ref_running_var,
            ref_save_mean,
            ref_save_invstd,
            True,
            eps,
            [True, True, True],
        )
    )

    res_running_mean = running_mean.clone()
    res_running_var = running_var.clone()
    (
        _,
        res_save_mean,
        res_save_invstd,
        _,
    ) = flag_gems._batch_norm_with_update(
        inp,
        weight,
        bias,
        res_running_mean,
        res_running_var,
        momentum,
        eps,
    )

    res_in_grad, res_weight_grad, res_bias_grad = (
        flag_gems.batch_norm_with_update_backward(
            grad_out,
            inp,
            weight,
            res_running_mean,
            res_running_var,
            res_save_mean,
            res_save_invstd,
            True,
            eps,
            [True, True, True],
        )
    )

    reduce_dim = math.prod(shape) // 16
    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=reduce_dim)
    utils.gems_assert_close(
        res_weight_grad, ref_weight_grad, dtype, reduce_dim=reduce_dim
    )
    utils.gems_assert_close(res_bias_grad, ref_bias_grad, dtype, reduce_dim=reduce_dim)
