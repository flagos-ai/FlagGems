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


@pytest.mark.tanh
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tanh(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.tanh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.tanh(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.tanh_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tanh_(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp.clone(), True)

    ref_out = torch.tanh_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.tanh_(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.tanh_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tanh_backward(shape, dtype):
    res_out = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_out)

    ref_out = utils.to_reference(res_out, True)
    ref_grad = utils.to_reference(res_grad, True)

    ref_in_grad = torch.ops.aten.tanh_backward(ref_grad, ref_out)
    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten.tanh_backward(res_grad, res_out)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
