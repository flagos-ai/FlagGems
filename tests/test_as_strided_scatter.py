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


@pytest.mark.as_strided_scatter
# Covers contiguous, sparse-stride, overlapping, scalar, and default-offset views.
@pytest.mark.parametrize(
    "size,stride,storage_offset,self_len",
    [
        ((2, 3), (3, 1), 0, 6),
        ((4, 4), (10, 2), 3, 40),
        ((2, 2), (1, 1), 0, 8),
        ((), (), 0, 8),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_as_strided_scatter(dtype, size, stride, storage_offset, self_len):
    inp = torch.randn(self_len, device=flag_gems.device, dtype=dtype)
    src = torch.randn(size, device=flag_gems.device, dtype=dtype)
    expected = torch.ops.aten.as_strided_scatter(
        utils.to_reference(inp),
        utils.to_reference(src),
        size,
        stride,
        storage_offset,
    )

    with flag_gems.use_gems():
        actual = torch.ops.aten.as_strided_scatter(
            inp, src, size, stride, storage_offset
        )

    utils.gems_assert_close(actual, expected, dtype)


@pytest.mark.as_strided_scatter
def test_as_strided_scatter_preserves_storage_geometry():
    base = torch.arange(20, device=flag_gems.device, dtype=torch.float32)
    inp = base[3:15]
    src = torch.tensor([91.0, 92.0, 93.0], device=flag_gems.device)
    expected = torch.ops.aten.as_strided_scatter(inp, src, (3,), (2,), None)

    with flag_gems.use_gems():
        actual = torch.ops.aten.as_strided_scatter(inp, src, (3,), (2,), None)

    assert actual.stride() == expected.stride()
    assert actual.storage_offset() == expected.storage_offset()
    assert actual.untyped_storage().nbytes() == expected.untyped_storage().nbytes()
    utils.gems_assert_close(actual, expected, torch.float32)


@pytest.mark.as_strided_scatter
def test_as_strided_scatter_noncontiguous_self():
    inp = torch.randn((5, 7), device=flag_gems.device).mT
    src = torch.randn((2, 2), device=flag_gems.device)
    expected = torch.ops.aten.as_strided_scatter(inp, src, (2, 2), (1, 5), None)
    with flag_gems.use_gems():
        actual = torch.ops.aten.as_strided_scatter(inp, src, (2, 2), (1, 5), None)
    assert actual.stride() == expected.stride()
    utils.gems_assert_close(actual, expected, torch.float32)
