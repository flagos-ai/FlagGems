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


def make_inputs(shape, device):
    m, order = shape
    ids = torch.randint(0, 100000, (m, order), dtype=torch.int32, device=device)
    multipliers = torch.randint(1, 100000, (2, order), dtype=torch.int64, device=device)
    sizes = torch.randint(
        1, 100000, (2, order - 1, 8), dtype=torch.int32, device=device
    )
    flat = sizes.flatten(1)
    offsets = flat.cumsum(1, dtype=torch.int32) - flat
    return (ids, multipliers, sizes, offsets), {}


@pytest.mark.engram_hash
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend"
    or not torch.npu.get_device_name().startswith("Ascend950"),
    reason="requires Atlas A5",
)
@pytest.mark.parametrize("shape", [(512, 3), (8001, 3)])
def test_engram_hash(shape):
    reference = pytest.importorskip(
        "tile_kernels.engram.engram_hash_kernel"
    ).engram_hash
    args, kwargs = make_inputs(shape, flag_gems.device)
    expected = reference(*args, **kwargs)
    actual = flag_gems.engram_hash(*args, **kwargs)
    torch.testing.assert_close(actual, expected)
