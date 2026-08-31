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


@pytest.mark.masked_softmax
@pytest.mark.parametrize("shape", [(128, 256), (8, 16, 64), (64, 128)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_masked_softmax_mask_type_2(shape, dtype, dim):
    # mask_type 2: elementwise mask with the same shape as the input.
    if dim >= len(shape):
        pytest.skip("dim out of range for this shape")
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_mask = utils.to_reference(mask)
    ref_out = torch.ops.aten._masked_softmax(ref_x, ref_mask, dim, 2)

    res_out = flag_gems._masked_softmax(x, mask, dim, 2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.masked_softmax
@pytest.mark.parametrize("shape", [(4, 8, 32, 32), (2, 4, 64, 64)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_masked_softmax_mask_type_1(shape, dtype):
    # mask_type 1: (B, H, L, L) attention scores with a (B, L) padding mask.
    B, _, _, L = shape
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randint(0, 2, (B, L), dtype=torch.bool, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_mask = utils.to_reference(mask)
    ref_out = torch.ops.aten._masked_softmax(ref_x, ref_mask, 3, 1)

    res_out = flag_gems._masked_softmax(x, mask, 3, 1)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.masked_softmax
@pytest.mark.parametrize("shape", [(4, 8, 32, 32), (2, 4, 48, 48)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_masked_softmax_mask_type_0(shape, dtype):
    # mask_type 0: (B, H, L, L) attention scores with an (L, L) source mask.
    L = shape[-1]
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randint(0, 2, (L, L), dtype=torch.bool, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_mask = utils.to_reference(mask)
    ref_out = torch.ops.aten._masked_softmax(ref_x, ref_mask, 3, 0)

    res_out = flag_gems._masked_softmax(x, mask, 3, 0)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.masked_softmax
def test_masked_softmax_dispatch():
    # Guard against a false pass: confirm the op is wired into the FlagGems
    # registration table and that its implementation is callable end to end.
    registered = {entry[0] for entry in flag_gems._FULL_CONFIG}
    assert "_masked_softmax" in registered

    x = torch.randn((32, 64), dtype=torch.float32, device=flag_gems.device)
    mask = torch.randint(0, 2, (32, 64), dtype=torch.bool, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_mask = utils.to_reference(mask)
    ref_out = torch.ops.aten._masked_softmax(ref_x, ref_mask, -1, 2)

    res_out = flag_gems._masked_softmax(x, mask, -1, 2)
    utils.gems_assert_close(res_out, ref_out, torch.float32, equal_nan=True)
