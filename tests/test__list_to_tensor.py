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

# NOTE: aten::_list_to_tensor is a JIT-only prim operator without a c10
# dispatcher kernel. Keep the standard use_gems call for registration-path
# coverage, and call the GEMS implementation directly to verify the Triton
# implementation itself.


@pytest.mark.list_to_tensor
@pytest.mark.parametrize(
    "self_list",
    [
        [1, 2, 3, 4],
        [5],
        [0, -1, 7, 100, -42],
        [2**31 - 1, -(2**31), 0],
        list(range(2000)),
        [],
    ],
)
def test_accuracy__list_to_tensor(self_list):
    ref_out = torch.ops.aten._list_to_tensor(self_list)

    with flag_gems.use_gems():
        res_out = torch.ops.aten._list_to_tensor(self_list)

    gems_out = flag_gems._list_to_tensor(self_list)

    utils.gems_assert_equal(res_out.cpu(), ref_out)
    utils.gems_assert_equal(gems_out.cpu(), ref_out)
    assert res_out.dtype == ref_out.dtype
    assert gems_out.dtype == ref_out.dtype
