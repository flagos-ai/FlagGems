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

import random

import pytest

import torch

import flag_gems

from . import accuracy_utils as utils

def _snapshot_or_input(inp):
    return (
        inp.device,
        inp.dtype,
        tuple(inp.shape),
        tuple(inp.stride()),
        inp.storage_offset(),
        inp.detach().clone(),
    )


def _assert_or_input_unchanged(inp, before):
    device, dtype, shape, stride, offset, values = before
    assert inp.device == device, "or_tensor changed input device"
    assert inp.dtype == dtype, "or_tensor changed input dtype"
    assert tuple(inp.shape) == shape, "or_tensor changed input shape"
    assert tuple(inp.stride()) == stride, "or_tensor changed input stride"
    assert inp.storage_offset() == offset, "or_tensor changed input storage offset"
    assert torch.equal(inp.cpu(), values.cpu()), "or_tensor changed input values"


def _assert_or_tensor_contract(output, lhs, rhs, lhs_before, rhs_before):
    assert isinstance(output, torch.Tensor), "or_tensor output must be a Tensor"
    assert output.device == lhs_before[0], "or_tensor returned the wrong device"
    assert output.dtype == torch.result_type(lhs_before[-1], rhs_before[-1]), (
        "or_tensor returned the wrong dtype"
    )
    assert tuple(output.shape) == torch.broadcast_shapes(lhs_before[2], rhs_before[2]), (
        "or_tensor returned the wrong shape"
    )
    assert not torch._C._is_alias_of(output, lhs), "or_tensor output aliases self"
    assert not torch._C._is_alias_of(output, rhs), "or_tensor output aliases other"
    _assert_or_input_unchanged(lhs, lhs_before)
    _assert_or_input_unchanged(rhs, rhs_before)


@pytest.mark.dunder_or_tensor
@pytest.mark.or_tensor
@pytest.mark.parametrize('shape', utils.POINTWISE_SHAPES)
@pytest.mark.parametrize('dtype', utils.INT_DTYPES + utils.BOOL_TYPES)
def test_dunder_or(shape, dtype):
    if dtype in utils.BOOL_TYPES:
        inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device='cpu').to(flag_gems.device)
        inp2 = torch.randint(0, 2, size=shape, dtype=dtype, device='cpu').to(flag_gems.device)
    else:
        inp1 = torch.randint(low=-32767, high=32767, size=shape, dtype=dtype, device='cpu').to(flag_gems.device)
        inp2 = torch.randint(low=-32767, high=32767, size=shape, dtype=dtype, device='cpu').to(flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)
    ref_out = ref_inp1 | ref_inp2
    inp1_before = _snapshot_or_input(inp1)
    inp2_before = _snapshot_or_input(inp2)
    candidate = getattr(flag_gems, "or_tensor", flag_gems.bitwise_or_tensor)
    res_out = candidate(inp1, inp2)
    _assert_or_tensor_contract(res_out, inp1, inp2, inp1_before, inp2_before)
    utils.gems_assert_equal(res_out, ref_out)
