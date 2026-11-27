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


@pytest.mark.gcd_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_gcd_(shape, dtype):
    # GCD is only defined for integer types
    inp1 = torch.randint(low=1, high=100, size=shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    inp2 = torch.randint(low=1, high=100, size=shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    ref_inp1 = utils.to_reference(inp1.clone())
    ref_inp2 = utils.to_reference(inp2)

    ref_out = ref_inp1.gcd_(ref_inp2)
    self_metadata = (inp1.device, inp1.dtype, inp1.shape, inp1.stride(), inp1.storage_offset(), inp1.data_ptr())
    other_metadata = (inp2.device, inp2.dtype, inp2.shape, inp2.stride(), inp2.storage_offset(), inp2.data_ptr())
    other_snapshot = inp2.detach().cpu().clone()
    assert other_snapshot.dtype == inp2.dtype and other_snapshot.shape == inp2.shape
    res_out = flag_gems.gcd_(inp1, inp2)

    assert res_out is inp1
    assert (inp1.device, inp1.dtype, inp1.shape, inp1.stride(), inp1.storage_offset(), inp1.data_ptr()) == self_metadata
    assert (inp2.device, inp2.dtype, inp2.shape, inp2.stride(), inp2.storage_offset(), inp2.data_ptr()) == other_metadata
    assert res_out.device == self_metadata[0]
    assert res_out.dtype == dtype
    assert res_out.shape == shape
    other_after = inp2.detach().cpu()
    assert other_after.dtype == other_snapshot.dtype and other_after.shape == other_snapshot.shape
    assert torch.equal(other_after, other_snapshot)

    utils.gems_assert_equal(res_out, ref_out)
