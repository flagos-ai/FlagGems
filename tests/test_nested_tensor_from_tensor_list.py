# Copyright 2026, The FlagOS Contributors.
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

pytestmark = pytest.mark.nested_tensor_from_tensor_list

# Each case is a list of component shapes. Nested tensors constructed from a
# tensor list require every component to share the same trailing dimensions and
# dtype while allowing the leading dimension to vary.
COMPONENT_SHAPES = [
    [(2, 32), (4, 32)],
    [(16, 128), (32, 128), (8, 128)],
    [(3, 64, 16), (5, 64, 16), (7, 64, 16), (2, 64, 16)],
]


@pytest.mark.nested_tensor_from_tensor_list
@pytest.mark.parametrize("shapes", COMPONENT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_tensor_from_tensor_list(shapes, dtype):
    tensor_list = [
        torch.randn(shape, dtype=dtype, device=flag_gems.device) for shape in shapes
    ]
    ref_list = [utils.to_reference(t) for t in tensor_list]

    ref_out = torch.ops.aten._nested_tensor_from_tensor_list(ref_list)

    res_out = flag_gems._nested_tensor_from_tensor_list(tensor_list)

    assert res_out.is_nested
    assert ref_out.is_nested

    res_unbind = torch.unbind(res_out)
    ref_unbind = torch.unbind(ref_out)

    assert len(res_unbind) == len(ref_unbind)
    for res_t, ref_t in zip(res_unbind, ref_unbind):
        assert res_t.shape == ref_t.shape
        ref_t_matched = ref_t if utils.TO_CPU else ref_t.to(res_t.device)
        utils.gems_assert_close(res_t, ref_t_matched, dtype)
