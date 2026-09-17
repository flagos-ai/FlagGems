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

# broadcast_to test cases - define source shapes and target shapes for broadcasting
BROADCAST_TEST_CASES = [
    # (source_shape, target_shape)
    ((1,), (3,)),  # broadcast 1D
    ((3,), (3, 3)),  # broadcast 1D to 2D
    ((1, 3), (2, 3)),  # broadcast 2D
    ((3, 1), (3, 3)),  # broadcast 2D
    ((1, 1), (4, 5)),  # broadcast small to larger
    ((2, 1), (2, 3)),  # broadcast 2D
    ((1, 2, 3), (4, 2, 3)),  # broadcast 3D
    ((4, 1, 1), (4, 5, 6)),  # broadcast 3D
]


@pytest.mark.broadcast_to
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_broadcast_to(dtype):
    # Test with a single broadcast case for each dtype
    src_shape = (1,)
    target_shape = (4,)
    inp = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.broadcast_to(ref_inp, target_shape)
    with flag_gems.use_gems():
        res_out = torch.broadcast_to(inp, target_shape)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.broadcast_to
@pytest.mark.parametrize("src_shape,target_shape", BROADCAST_TEST_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_broadcast_to_shapes(src_shape, target_shape, dtype):
    inp = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.broadcast_to(ref_inp, target_shape)
    with flag_gems.use_gems():
        res_out = torch.broadcast_to(inp, target_shape)

    utils.gems_assert_close(res_out, ref_out, dtype)
