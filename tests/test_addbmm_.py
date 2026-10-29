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
from .conftest import QUICK_MODE

if QUICK_MODE:
    MNK_SHAPES = [(1, 1, 32), (16, 16, 64)]
else:
    MNK_SHAPES = [
        (1, 1, 32),
        (16, 16, 64),
        (15, 160, 1024),
        (128, 256, 512),
        (256, 256, 256),
        (495, 5333, 71),
        (1024, 1024, 128),
    ]


@pytest.mark.addbmm_
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addbmm_(M, N, K, scalar, dtype):
    torch.manual_seed(0)
    batch1 = torch.randn((4, M, K), dtype=dtype, device=flag_gems.device)
    batch2 = torch.randn((4, K, N), dtype=dtype, device=flag_gems.device)
    self = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_self = utils.to_reference(self, True).clone()
    pointer = self.data_ptr()
    ref_result = ref_self.addbmm_(
        utils.to_reference(batch1, True),
        utils.to_reference(batch2, True),
        alpha=scalar,
        beta=scalar,
    )
    result = flag_gems.addbmm_(self, batch1, batch2, alpha=scalar, beta=scalar)
    assert result is self
    assert self.data_ptr() == pointer
    utils.gems_assert_close(result, ref_result, dtype, reduce_dim=K)
    utils.gems_assert_close(self, ref_self, dtype, reduce_dim=K)


@pytest.mark.addbmm_
def test_addbmm_noncontiguous_self_preserves_storage():
    torch.manual_seed(0)
    dtype = torch.float32
    self = torch.randn((5, 3), dtype=dtype, device=flag_gems.device).t()
    batch1 = torch.randn((2, 3, 4), dtype=dtype, device=flag_gems.device)
    batch2 = torch.randn((2, 4, 5), dtype=dtype, device=flag_gems.device)
    ref_self = utils.to_reference(self, True).clone()
    pointer = self.data_ptr()
    stride = self.stride()
    ref_result = ref_self.addbmm_(utils.to_reference(batch1, True), utils.to_reference(batch2, True))
    result = flag_gems.addbmm_(self, batch1, batch2)
    assert result is self
    assert self.data_ptr() == pointer
    assert self.stride() == stride
    utils.gems_assert_close(result, ref_result, dtype, reduce_dim=4)


@pytest.mark.addbmm_
def test_addbmm_rejects_incompatible_shapes():
    torch.manual_seed(0)
    self = torch.randn((3, 5), device=flag_gems.device)
    batch1 = torch.randn((2, 3, 4), device=flag_gems.device)
    batch2 = torch.randn((2, 6, 5), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.addbmm_(self, batch1, batch2)
