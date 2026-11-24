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

pytestmark = [
    pytest.mark.index,
    pytest.mark.skipif(
        flag_gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _reference():
    return torch.ops.aten.index.Tensor


def _inputs(shape, dtype, device):
    if len(shape) == 3:
        rows, columns, pitch = shape
        x = torch.randn((rows, pitch), device=device, dtype=dtype)[:, pitch - columns :]
    else:
        x = torch.randn(shape, device=device, dtype=dtype)
    count = 1 if shape[-1] == 6144 else 1024
    idx = torch.randint(-shape[0], shape[0], (count,), device=device, dtype=torch.int64)
    return (x, [idx]), {}


@pytest.mark.parametrize("shape", [(8,), (1024, 6144), (1048576, 64, 128)])
def test_index_native(shape):
    args, kwargs = _inputs(shape, torch.bfloat16, flag_gems.device)
    reference = _reference()
    expected = reference(*args, **kwargs)
    actual = flag_gems.index(*args, **kwargs)
    torch.testing.assert_close(actual, expected)


def test_index_padded_rows_small_index_list():
    x = torch.randn((13, 160), dtype=torch.bfloat16, device=flag_gems.device)[:, 3:132]
    idx = torch.tensor([-1, 0, 7], dtype=torch.int32, device=x.device)
    actual = flag_gems.index(x, [idx])
    torch.testing.assert_close(
        actual, torch.ops.aten.index.Tensor(x, [idx]), rtol=0, atol=0
    )
