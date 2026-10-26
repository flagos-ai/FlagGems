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

def _to_reference(value, upcast=False):
    if flag_gems.vendor_name == "thead":
        reference = value.cpu()
        return reference.double() if upcast else reference
    return utils.to_reference(value, upcast)


def _assert_reference_close(actual, expected, dtype, *, reduce_dim, device):
    if flag_gems.vendor_name == "thead":
        assert actual.device == device
        assert actual.dtype == dtype
        assert actual.shape == expected.shape
        return utils.gems_assert_close(actual.cpu(), expected.cpu(), dtype, reduce_dim=reduce_dim)
    return utils.gems_assert_close(actual, expected, dtype, reduce_dim=reduce_dim)


if QUICK_MODE:
    MNK_SHAPES = [(1, 1, 32)]
    FLOAT_DTYPES = [torch.float32]
else:
    MNK_SHAPES = [(1, 1, 32), (15, 160, 1024), (495, 5333, 71)]
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.baddbmm_
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_baddbmm_(M, N, K, scalar, dtype):
    torch.manual_seed(0)
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #3794: not working")
    batch = 4
    batch1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    batch2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    self = torch.randn((batch, M, N), dtype=dtype, device=flag_gems.device)
    ref_batch1 = _to_reference(batch1, True)
    ref_batch2 = _to_reference(batch2, True)
    ref_self = _to_reference(self, True).clone()
    pointer = self.data_ptr()

    ref_result = ref_self.baddbmm_(ref_batch1, ref_batch2, alpha=scalar, beta=scalar)
    with flag_gems.use_gems():
        result = self.baddbmm_(batch1, batch2, alpha=scalar, beta=scalar)

    assert result is self
    assert self.data_ptr() == pointer
    _assert_reference_close(result, ref_result, dtype, reduce_dim=K, device=self.device)
    _assert_reference_close(self, ref_self, dtype, reduce_dim=K, device=self.device)


@pytest.mark.baddbmm_
def test_baddbmm_noncontiguous_self_preserves_storage():
    if flag_gems.vendor_name == "mthreads" and not utils.TO_CPU:
        pytest.skip("MUSA native baddbmm_ does not write noncontiguous self")
    torch.manual_seed(0)
    dtype = torch.float32
    self = torch.randn((2, 5, 3), dtype=dtype, device=flag_gems.device).transpose(1, 2)
    batch1 = torch.randn((2, 3, 4), dtype=dtype, device=flag_gems.device)
    batch2 = torch.randn((2, 4, 5), dtype=dtype, device=flag_gems.device)
    ref_self = _to_reference(self, True).clone()
    pointer = self.data_ptr()
    stride = self.stride()
    ref_result = ref_self.baddbmm_(
        _to_reference(batch1, True), _to_reference(batch2, True)
    )
    with flag_gems.use_gems():
        result = self.baddbmm_(batch1, batch2)
    assert result is self
    assert self.data_ptr() == pointer
    assert self.stride() == stride
    _assert_reference_close(result, ref_result, dtype, reduce_dim=4, device=self.device)


@pytest.mark.baddbmm_
def test_baddbmm_rejects_batch_mismatch():
    torch.manual_seed(0)
    self = torch.randn((2, 3, 5), device=flag_gems.device)
    batch1 = torch.randn((2, 3, 4), device=flag_gems.device)
    batch2 = torch.randn((3, 4, 5), device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        self.baddbmm_(batch1, batch2)
