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
from . import conftest as cfg

# Test shapes for inplace bitwise right shift, covering broadcasting dimensions
if cfg.QUICK_MODE:
    # Minimal set for quick mode
    INPLACE_BITWISE_SHAPES = [
        ((512, 1024), (512, 1024)),
    ]
else:
    # Cover broadcasting and scalar edge cases for inplace
    INPLACE_BITWISE_SHAPES = [
        ((512, 1024), (512, 1024)),
        ((256, 512), (1, 512)),
        ((256, 512), (256, 1)),
        ((1024,), ()),
    ]


@pytest.mark.bitwise_right_shift_
@pytest.mark.parametrize("shapes", INPLACE_BITWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES + [torch.uint8])
def test_bitwise_right_shift_(shapes, dtype):
    shape_a, shape_b = shapes
    res_a = torch.randint(0, 100, shape_a, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    res_b = torch.randint(0, 8, shape_b, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_a = utils.to_reference(res_a.clone())
    ref_b = utils.to_reference(res_b)

    ref_a.bitwise_right_shift_(ref_b)
    before = (
        res_a.device,
        res_a.dtype,
        tuple(res_a.shape),
        res_a.stride(),
        res_a.data_ptr(),
    )
    other_before = (
        res_b.device,
        res_b.dtype,
        tuple(res_b.shape),
        res_b.stride(),
        res_b.data_ptr(),
    )
    other_bytes = res_b.cpu().contiguous().reshape(-1).view(torch.uint8).clone()
    result = flag_gems.bitwise_right_shift_(res_a, res_b)
    assert isinstance(result, torch.Tensor) and result is res_a
    assert (
        result.device,
        result.dtype,
        tuple(result.shape),
        result.stride(),
        result.data_ptr(),
    ) == before
    assert (
        res_b.device,
        res_b.dtype,
        tuple(res_b.shape),
        res_b.stride(),
        res_b.data_ptr(),
    ) == other_before
    assert torch.equal(
        res_b.cpu().contiguous().reshape(-1).view(torch.uint8), other_bytes
    )

    utils.gems_assert_close(res_a, ref_a, dtype)
