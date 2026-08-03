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


@pytest.mark.mish
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mish(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.mish(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.mish(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mish_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mish_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.mish_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.mish_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mish_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mish_backward(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_out = torch.randn_like(inp)

    ref_inp = utils.to_reference(inp, True)
    ref_grad_out = utils.to_reference(grad_out, True)

    ref_in_grad = torch.ops.aten.mish_backward(ref_grad_out, ref_inp)
    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten.mish_backward(grad_out, inp)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
