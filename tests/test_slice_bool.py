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


@pytest.mark.slice
@pytest.mark.parametrize(
    "layout", ["contiguous", "transpose", "offset", "stepped", "empty"]
)
@pytest.mark.parametrize(
    "dim,start,end,step",
    [
        (0, 0, 2, 1),
        (1, 1, 5, 2),
        (0, -3, -1, 1),
        (1, 3, 1, 1),
        (0, 0, 0, 1),
        (1, None, None, 1),
    ],
)
def test_bool_slice_view(layout, dim, start, end, step):
    base = (torch.arange(48, device=flag_gems.device).reshape(6, 8) % 2).bool()
    if layout == "contiguous":
        inp = base
    elif layout == "transpose":
        inp = base.t()
    elif layout == "offset":
        inp = base[1:, 1:]
    elif layout == "stepped":
        inp = base[:, ::2]
    else:
        inp = base[:0]

    expected = torch.ops.aten.slice.Tensor(inp, dim, start, end, step)
    with flag_gems.use_gems():
        actual = torch.ops.aten.slice.Tensor(inp, dim, start, end, step)

    assert actual.dtype == torch.bool
    assert torch.equal(actual, expected)
    assert actual.shape == expected.shape
    assert actual.stride() == expected.stride()
    assert actual.storage_offset() == expected.storage_offset()
    assert actual.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
    if actual.numel():
        actual.logical_not_()
        assert torch.equal(actual, expected)
