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
    MNK_SHAPES = [(1, 1, 32)]
    FLOAT_DTYPES = [torch.float32]
else:
    MNK_SHAPES = [(1, 1, 32), (15, 160, 1024), (495, 5333, 71)]
    FLOAT_DTYPES = utils.FLOAT_DTYPES


def _reference_input(value):
    # PPU lacks native inplace BLAS, and Enflame native baddbmm_ can return
    # incorrect values. CPU ATen supplies an independent numerical oracle;
    # cloning keeps reference mutation separate from candidate inputs.
    if flag_gems.vendor_name in {"thead", "enflame"}:
        return value.cpu().double().clone()
    reference = utils.to_reference(value, True)
    # This is an independent value reference, so its output need not share
    # the candidate's strided layout. muDNN requires a contiguous output.
    if flag_gems.vendor_name == "mthreads":
        return reference.clone(memory_format=torch.contiguous_format)
    return reference.clone()


def _assert_values(result, reference, expected_device, dtype, reduce_dim):
    assert result.device == expected_device
    assert result.dtype == dtype
    assert result.shape == reference.shape
    utils.gems_assert_close(result.cpu(), reference.cpu(), dtype, reduce_dim=reduce_dim)


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
    ref_batch1 = _reference_input(batch1)
    ref_batch2 = _reference_input(batch2)
    ref_self = _reference_input(self)
    expected_device = self.device
    pointer = self.data_ptr()

    ref_result = ref_self.baddbmm_(ref_batch1, ref_batch2, alpha=scalar, beta=scalar)
    result = flag_gems.baddbmm_(self, batch1, batch2, alpha=scalar, beta=scalar)

    assert result is self
    assert self.data_ptr() == pointer
    _assert_values(result, ref_result, expected_device, dtype, reduce_dim=K)
    _assert_values(self, ref_self, expected_device, dtype, reduce_dim=K)


@pytest.mark.baddbmm_
def test_baddbmm_noncontiguous_self_preserves_storage():
    torch.manual_seed(0)
    dtype = torch.float32
    self = torch.randn((2, 5, 3), dtype=dtype, device=flag_gems.device).transpose(1, 2)
    batch1 = torch.randn((2, 3, 4), dtype=dtype, device=flag_gems.device)
    batch2 = torch.randn((2, 4, 5), dtype=dtype, device=flag_gems.device)
    ref_self = _reference_input(self)
    expected_device = self.device
    pointer = self.data_ptr()
    stride = self.stride()
    ref_result = ref_self.baddbmm_(_reference_input(batch1), _reference_input(batch2))
    result = flag_gems.baddbmm_(self, batch1, batch2)
    assert result is self
    assert self.data_ptr() == pointer
    assert self.stride() == stride
    _assert_values(result, ref_result, expected_device, dtype, reduce_dim=4)


@pytest.mark.baddbmm_
def test_baddbmm_rejects_batch_mismatch():
    torch.manual_seed(0)
    self = torch.randn((2, 3, 5), device=flag_gems.device)
    batch1 = torch.randn((2, 3, 4), device=flag_gems.device)
    batch2 = torch.randn((3, 4, 5), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.baddbmm_(self, batch1, batch2)
