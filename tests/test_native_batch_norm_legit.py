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

    with flag_gems.use_gems():
        (
            res_out,
            res_save_mean,
            res_save_var,
        ) = torch.ops.aten._native_batch_norm_legit.default(
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
    with flag_gems.use_gems():
        result = torch.ops.aten._native_batch_norm_legit.no_stats(
            inp, weight, bias, True, 0.1, 1e-5
        )

    for actual, expected in zip(result, ref):
        utils.gems_assert_close(actual, expected, dtype)
    utils.gems_assert_equal(inp, original)


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
        with flag_gems.use_gems():
            result = torch.ops.aten._native_batch_norm_legit.out(
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
        with flag_gems.use_gems():
            result = torch.ops.aten._native_batch_norm_legit.no_stats_out(
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
