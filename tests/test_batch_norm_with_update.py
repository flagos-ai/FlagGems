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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.batch_norm_with_update
@pytest.mark.parametrize(
    "shape",
    [
        (16, 3),
        (32, 32, 32),
        (8, 32, 224, 224),
        (2050, 16, 32, 32),
        (8, 16, 3, 224, 224),
        # Both the batch and spatial dimensions require multiple tiles with a
        # masked spatial tail: (2049, C, 9) -> BLOCK_M=2048, BLOCK_N=8, 2x2 tiles.
        (2049, 16, 9),
    ],
)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("affine", [True, False])
def test_batch_norm_with_update(shape, dtype, affine):
    C = shape[1]
    inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    weight = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device) if affine else None
    )
    bias = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device) if affine else None
    )

    running_mean = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
    running_var = (
        torch.abs(torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)) + 0.1
    )

    momentum = 0.1
    eps = 1e-5

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)
    ref_running_mean = utils.to_reference(running_mean.clone(), True)
    ref_running_var = utils.to_reference(running_var.clone(), True)

    # Reference: training-mode batch norm updates running stats in place.
    (
        ref_out,
        ref_save_mean,
        ref_save_invstd,
        ref_reserve,
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
        res_out,
        res_save_mean,
        res_save_invstd,
        res_reserve,
    ) = flag_gems._batch_norm_with_update(
        inp,
        weight,
        bias,
        res_running_mean,
        res_running_var,
        momentum,
        eps,
    )

    # save_mean / save_invstd follow the accumulation dtype: FP64 for FP64
    # inputs, FP32 otherwise, matching ATen.
    acc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert res_save_mean.dtype == acc_dtype
    assert res_save_invstd.dtype == acc_dtype
    utils.gems_assert_close(res_save_mean, ref_save_mean, acc_dtype)
    utils.gems_assert_close(res_save_invstd, ref_save_invstd, acc_dtype)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_running_mean, ref_running_mean, dtype)
    utils.gems_assert_close(res_running_var, ref_running_var, dtype)


@pytest.mark.batch_norm_with_update
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_batch_norm_with_update_non_contiguous_params(dtype):
    # weight, bias and running stats are strided views; the kernel must
    # honor their strides when reading and updating them in place.
    C = 8
    shape = (32, C, 16)
    inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)

    weight_base = torch.randn((2 * C,), dtype=dtype, device=flag_gems.device)
    bias_base = torch.randn((2 * C,), dtype=dtype, device=flag_gems.device)
    weight = weight_base[::2]
    bias = bias_base[::2]
    assert not weight.is_contiguous()
    assert not bias.is_contiguous()

    mean_base = torch.randn((2 * C,), dtype=dtype, device=flag_gems.device)
    var_base = (
        torch.abs(torch.randn((2 * C,), dtype=dtype, device=flag_gems.device)) + 0.1
    )
    running_mean = mean_base[::2]
    running_var = var_base[::2]
    assert not running_mean.is_contiguous()
    assert not running_var.is_contiguous()
    odd_mean_before = mean_base[1::2].clone()
    odd_var_before = var_base[1::2].clone()

    momentum = 0.1
    eps = 1e-5

    # The reference runs through cuDNN, which requires contiguous parameters;
    # compare against the same values the strided views alias.
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight_base[::2].contiguous(), True)
    ref_bias = utils.to_reference(bias_base[::2].contiguous(), True)
    ref_running_mean = utils.to_reference(mean_base[::2].contiguous(), True)
    ref_running_var = utils.to_reference(var_base[::2].contiguous(), True)

    (
        ref_out,
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

    # Pass the strided views directly: running_mean / running_var are updated
    # in place through their strides.
    (
        res_out,
        res_save_mean,
        res_save_invstd,
        _,
    ) = flag_gems._batch_norm_with_update(
        inp,
        weight,
        bias,
        running_mean,
        running_var,
        momentum,
        eps,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    acc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    utils.gems_assert_close(res_save_mean, ref_save_mean, acc_dtype)
    utils.gems_assert_close(res_save_invstd, ref_save_invstd, acc_dtype)

    # The views are updated in place, and the elements they skip must keep
    # their original values.
    assert running_mean.stride() == (2,)
    assert running_var.stride() == (2,)
    utils.gems_assert_close(running_mean, ref_running_mean, dtype)
    utils.gems_assert_close(running_var, ref_running_var, dtype)
    # Untouched elements must be bit-identical to their original values.
    assert torch.equal(mean_base[1::2], odd_mean_before)
    assert torch.equal(var_base[1::2], odd_var_before)


@pytest.mark.batch_norm_with_update
def test_batch_norm_with_update_requires_running_stats():
    # The aten schema requires running_mean and running_var; missing
    # arguments must fail instead of silently corrupting the input.
    inp = torch.randn((8, 3, 4), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems._batch_norm_with_update(inp, None, None, None, None)


@pytest.mark.batch_norm_with_update
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("affine", [True, False])
def test_batch_norm_with_update_backward(dtype, affine):
    # The forward statistics feed native_batch_norm_backward; input, weight
    # and bias gradients must match the ATen reference.
    C = 8
    shape = (32, C, 16)
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
    ) = flag_gems.batch_norm_backward(
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

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
    if affine:
        utils.gems_assert_close(res_weight_grad, ref_weight_grad, dtype)
        utils.gems_assert_close(res_bias_grad, ref_bias_grad, dtype)


@pytest.mark.batch_norm_with_update
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_batch_norm_with_update_param_size_mismatch(dtype):
    # The kernel indexes the 1-D parameters by feature id, so a feature-count
    # mismatch would read or write outside their storage.
    C = 4
    inp = torch.randn((8, C, 5), dtype=dtype, device=flag_gems.device)
    running_mean = torch.zeros(C, dtype=dtype, device=flag_gems.device)
    running_var = torch.ones(C, dtype=dtype, device=flag_gems.device)

    for kwargs in (
        {"weight": torch.randn(C + 1, dtype=dtype, device=flag_gems.device)},
        {"weight": torch.randn(C - 1, dtype=dtype, device=flag_gems.device)},
        {"bias": torch.randn(C + 1, dtype=dtype, device=flag_gems.device)},
        {"running_mean": torch.zeros(C + 1, dtype=dtype, device=flag_gems.device)},
        {"running_var": torch.ones(C - 1, dtype=dtype, device=flag_gems.device)},
    ):
        args = {
            "weight": None,
            "bias": None,
            "running_mean": running_mean,
            "running_var": running_var,
        }
        args.update(kwargs)
        with pytest.raises(RuntimeError):
            flag_gems._batch_norm_with_update(
                inp,
                args["weight"],
                args["bias"],
                args["running_mean"],
                args["running_var"],
                0.1,
                1e-5,
            )


@pytest.mark.batch_norm_with_update
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_batch_norm_with_update_empty_dims(dtype):
    # Empty batch or spatial dimensions hold no elements to normalize and
    # must return empty results instead of crashing.
    C = 4
    running_mean = torch.zeros(C, dtype=dtype, device=flag_gems.device)
    running_var = torch.ones(C, dtype=dtype, device=flag_gems.device)

    for shape in ((0, C, 5), (8, C, 0)):
        inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
        out, save_mean, save_invstd, reserve = flag_gems._batch_norm_with_update(
            inp, None, None, running_mean, running_var, 0.1, 1e-5
        )
        assert out.shape == shape
        assert save_mean.shape == (C,)
        assert save_invstd.shape == (C,)
