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

FWD_SHAPES = [
    (16, 128, 64),
    (1, 128, 64),
    (2, 100, 70),
    (5, 512, 129),
    (6, 512, 129),
    (4, 1024, 512),
    (2, 4096, 64),
    (0, 128, 64),
    (7, 256, 192),
    (48, 1024, 512),
    (192, 1024, 1024),
]


@pytest.mark.linear
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("batch_size, in_features, out_features", FWD_SHAPES)
def test_linear_2d_with_bias(dtype, batch_size, in_features, out_features):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 linear test on tsingmicro platform")

    input_tensor = torch.randn(
        (batch_size, in_features), dtype=dtype, device=flag_gems.device
    )
    weight = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )
    bias = torch.randn((out_features,), dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input_tensor, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.nn.functional.linear(ref_input, ref_weight, ref_bias)
    res_out = flag_gems.linear(input_tensor, weight, bias)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=in_features)


@pytest.mark.linear
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("batch_size, in_features, out_features", FWD_SHAPES)
def test_linear_2d_without_bias(dtype, batch_size, in_features, out_features):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 linear test on tsingmicro platform")

    input_tensor = torch.randn(
        (batch_size, in_features), dtype=dtype, device=flag_gems.device
    )
    weight = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )

    ref_input = utils.to_reference(input_tensor, True)
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.nn.functional.linear(ref_input, ref_weight)
    res_out = flag_gems.linear(input_tensor, weight)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=in_features)


FWD_3D_SHAPES = [
    (4, 8, 128, 64),
    (1, 1, 128, 64),
    (2, 2, 100, 70),
    (2, 4, 256, 192),
    (8, 24, 1024, 1024),
]


@pytest.mark.linear
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("batch1, batch2, in_features, out_features", FWD_3D_SHAPES)
def test_linear_3d_with_bias(dtype, batch1, batch2, in_features, out_features):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 linear test on tsingmicro platform")

    input_tensor = torch.randn(
        (batch1, batch2, in_features), dtype=dtype, device=flag_gems.device
    )
    weight = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )
    bias = torch.randn((out_features,), dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input_tensor, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.nn.functional.linear(ref_input, ref_weight, ref_bias)
    res_out = flag_gems.linear(input_tensor, weight, bias)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=in_features)


FWD_1D_SHAPES = [
    (128, 64),
    (16, 16),
    (129, 70),
]


@pytest.mark.linear
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("in_features, out_features", FWD_1D_SHAPES)
def test_linear_1d_with_bias(dtype, in_features, out_features):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 linear test on tsingmicro platform")

    input_tensor = torch.randn((in_features,), dtype=dtype, device=flag_gems.device)
    weight = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )
    bias = torch.randn((out_features,), dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input_tensor, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.nn.functional.linear(ref_input, ref_weight, ref_bias)
    res_out = flag_gems.linear(input_tensor, weight, bias)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=in_features)
