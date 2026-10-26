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

import os

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


@pytest.mark.addmm_
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_addmm_(M, N, K, scalar, dtype):
    torch.manual_seed(0)
    if flag_gems.vendor_name == "mthreads":
        os.environ["MUSA_ENABLE_SQMMA"] = "1"
    try:
        mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
        mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
        self = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
        ref_mat1 = utils.to_reference(mat1, True)
        ref_mat2 = utils.to_reference(mat2, True)
        ref_self = utils.to_reference(self, True).clone()
        original_pointer = self.data_ptr()

        ref_result = ref_self.addmm_(ref_mat1, ref_mat2, alpha=scalar, beta=scalar)
        with flag_gems.use_gems():
            result = self.addmm_(mat1, mat2, alpha=scalar, beta=scalar)

        assert result is self
        assert result.data_ptr() == original_pointer
        utils.gems_assert_close(result, ref_result, dtype, reduce_dim=K)
        utils.gems_assert_close(self, ref_self, dtype, reduce_dim=K)
    finally:
        if flag_gems.vendor_name == "mthreads":
            os.environ.pop("MUSA_ENABLE_SQMMA", None)


@pytest.mark.addmm_
def test_addmm_noncontiguous_self_preserves_storage():
    torch.manual_seed(0)
    dtype = torch.float32
    backing = torch.randn((5, 3), dtype=dtype, device=flag_gems.device)
    self = backing.t()
    mat1 = torch.randn((3, 4), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((4, 5), dtype=dtype, device=flag_gems.device)
    ref_self = utils.to_reference(self, True).clone()
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    pointer = self.data_ptr()
    stride = self.stride()

    ref_result = ref_self.addmm_(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        result = self.addmm_(mat1, mat2)

    assert result is self
    assert self.data_ptr() == pointer
    assert self.stride() == stride
    utils.gems_assert_close(result, ref_result, dtype, reduce_dim=4)


@pytest.mark.addmm_
def test_addmm_rejects_incompatible_shapes():
    torch.manual_seed(0)
    self = torch.randn((2, 3), device=flag_gems.device)
    mat1 = torch.randn((2, 4), device=flag_gems.device)
    mat2 = torch.randn((5, 3), device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        self.addmm_(mat1, mat2)
