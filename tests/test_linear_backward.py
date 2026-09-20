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

# Shapes for linear_backward: (batch, in_features), weight is (out_features, in_features)
LINEAR_SHAPES = [
    (4, 16),
    (8, 32),
    (16, 64),
    (32, 128),
]
LINEAR_OUT_FEATURES = [32, 64, 128, 256]


@pytest.mark.linear_backward
@pytest.mark.parametrize("batch, in_features", LINEAR_SHAPES)
@pytest.mark.parametrize("out_features", LINEAR_OUT_FEATURES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linear_backward(batch, in_features, out_features, dtype):
    # Create inputs
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    weight = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )

    ref_input = utils.to_reference(input_tensor, True)
    ref_weight = utils.to_reference(weight, True)
    ref_grad_output = utils.to_reference(grad_output, True)

    # Reference computation
    ref_grad_input = ref_grad_output @ ref_weight
    ref_grad_weight = ref_grad_output.t() @ ref_input
    ref_grad_bias = ref_grad_output.sum(dim=0)

    # GEMS computation
    with flag_gems.use_gems():
        res_grad_input, res_grad_weight, res_grad_bias = torch.ops.aten.linear_backward(
            input_tensor, grad_output, weight, (True, True, True)
        )

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)
    utils.gems_assert_close(res_grad_weight, ref_grad_weight, dtype)
    utils.gems_assert_close(res_grad_bias, ref_grad_bias, dtype)


@pytest.mark.linear_backward
@pytest.mark.parametrize("batch, in_features", LINEAR_SHAPES)
@pytest.mark.parametrize("out_features", LINEAR_OUT_FEATURES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linear_backward_grad_input_only(batch, in_features, out_features, dtype):
    """Test with output_mask=(True, False, False)"""
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    weight = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    ref_weight = utils.to_reference(weight, True)
    ref_grad_output = utils.to_reference(grad_output, True)

    # Reference computation
    ref_grad_input = ref_grad_output @ ref_weight

    # GEMS computation
    with flag_gems.use_gems():
        res = torch.ops.aten.linear_backward(
            input_tensor, grad_output, weight, (True, False, False)
        )
        res_grad_input = res[0]

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)


@pytest.mark.linear_backward
@pytest.mark.parametrize("batch, in_features", LINEAR_SHAPES)
@pytest.mark.parametrize("out_features", LINEAR_OUT_FEATURES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linear_backward_grad_weight_only(batch, in_features, out_features, dtype):
    """Test with output_mask=(False, True, False)"""
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    weight = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )

    ref_input = utils.to_reference(input_tensor, True)
    ref_grad_output = utils.to_reference(grad_output, True)

    # Reference computation
    ref_grad_weight = ref_grad_output.t() @ ref_input

    # GEMS computation
    with flag_gems.use_gems():
        res = torch.ops.aten.linear_backward(
            input_tensor, grad_output, weight, (False, True, False)
        )
        res_grad_weight = res[1]

    utils.gems_assert_close(res_grad_weight, ref_grad_weight, dtype)


@pytest.mark.linear_backward
@pytest.mark.parametrize("batch, in_features", LINEAR_SHAPES)
@pytest.mark.parametrize("out_features", LINEAR_OUT_FEATURES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linear_backward_grad_bias_only(batch, in_features, out_features, dtype):
    """Test with output_mask=(False, False, True)"""
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    weight = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )

    ref_grad_output = utils.to_reference(grad_output, True)

    # Reference computation
    ref_grad_bias = ref_grad_output.sum(dim=0)

    # GEMS computation
    with flag_gems.use_gems():
        res = torch.ops.aten.linear_backward(
            input_tensor, grad_output, weight, (False, False, True)
        )
        res_grad_bias = res[2]

    utils.gems_assert_close(res_grad_bias, ref_grad_bias, dtype)
