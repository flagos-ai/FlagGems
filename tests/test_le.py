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


@pytest.mark.le
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_le(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.le(ref_inp1, ref_inp2)
    res_out = flag_gems.le(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.le_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_le_scalar(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = 0
    ref_inp1 = utils.to_reference(inp1)

    ref_out = torch.le(ref_inp1, inp2)
    res_out = flag_gems.le_scalar(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.le_backward
def test_le_backward_non_differentiable():
    """``le`` outputs a bool tensor, so it has no gradient.

    The backward form is undefined; this pins that the output matches the
    native op and is non-differentiable, rather than fabricating a gradient
    for a boolean result.
    """
    A = torch.randn(1024, dtype=torch.float32, device=flag_gems.device)
    B = torch.randn(1024, dtype=torch.float32, device=flag_gems.device)
    ref_A = utils.to_reference(A, False)
    ref_B = utils.to_reference(B, False)

    res_out = flag_gems.le(A, B)
    ref_out = torch.le(ref_A, ref_B)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.dtype == torch.bool
    assert not res_out.requires_grad
    assert res_out.grad_fn is None
