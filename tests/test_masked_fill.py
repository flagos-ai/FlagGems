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


@pytest.mark.masked_fill
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("value_type", ["tensor", "scalar_tensor", "scalar"])
def test_masked_fill(shape, dtype, threshold, value_type):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randn(shape, dtype=dtype, device=flag_gems.device) < threshold
    if value_type == "tensor":
        value = torch.tensor(1024, device=flag_gems.device)
    elif value_type == "scalar_tensor":
        value = torch.scalar_tensor(1024, device=flag_gems.device)
    else:
        value = 1024

    ref_inp = utils.to_reference(inp)
    ref_mask = utils.to_reference(mask)
    if torch.is_tensor(value):
        ref_out = torch.masked_fill(ref_inp, ref_mask, utils.to_reference(value))
    else:
        ref_out = torch.masked_fill(ref_inp, ref_mask, value)
    with flag_gems.use_gems():
        res_out = torch.masked_fill(inp, mask, value)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.masked_fill_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("value_type", ["tensor", "scalar_tensor", "scalar"])
def test_masked_fill_(shape, dtype, threshold, value_type):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randn(shape, dtype=dtype, device=flag_gems.device) < threshold
    if value_type == "tensor":
        value = torch.tensor(1024, device=flag_gems.device)
    elif value_type == "scalar_tensor":
        value = torch.scalar_tensor(1024, device=flag_gems.device)
    else:
        value = 1024

    ref_inp = utils.to_reference(inp)
    ref_mask = utils.to_reference(mask)
    if torch.is_tensor(value):
        ref_inp.masked_fill_(ref_mask, utils.to_reference(value))
    else:
        ref_inp.masked_fill_(ref_mask, value)
    with flag_gems.use_gems():
        inp.masked_fill_(mask, value)

    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.masked_fill_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_masked_fill_scalar(shape, dtype, threshold):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randn(shape, dtype=dtype, device=flag_gems.device) < threshold
    value = 1024

    ref_inp = utils.to_reference(inp)
    ref_mask = utils.to_reference(mask)
    ref_out = torch.masked_fill(ref_inp, ref_mask, value)
    with flag_gems.use_gems():
        res_out = torch.masked_fill(inp, mask, value)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.masked_fill_scalar_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("value", [1024, -512, 0.5, -0.25])
def test_masked_fill_scalar_(shape, dtype, threshold, value):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randn(shape, dtype=dtype, device=flag_gems.device) < threshold

    ref_inp = utils.to_reference(inp)
    ref_mask = utils.to_reference(mask)
    ref_inp.masked_fill_(ref_mask, value)
    with flag_gems.use_gems():
        inp.masked_fill_(mask, value)

    utils.gems_assert_equal(inp, ref_inp)
