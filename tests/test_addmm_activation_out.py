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
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES


@pytest.mark.addmm_activation_out
@pytest.mark.underscore_addmm_activation_out
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("use_gelu", [False, True])
def test_addmm_activation_out(M, N, K, scalar, dtype, use_gelu):
    torch.manual_seed(0)
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    self = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    ref_out = torch.empty((M, N), dtype=utils.to_reference(self, True).dtype, device=utils.to_reference(self, True).device)
    pointer = out.data_ptr()
    ref_result = torch.ops.aten._addmm_activation.out(
        utils.to_reference(self, True),
        utils.to_reference(mat1, True),
        utils.to_reference(mat2, True),
        alpha=scalar,
        beta=scalar,
        use_gelu=use_gelu,
        out=ref_out,
    )
    with flag_gems.use_gems():
        result = torch.ops.aten._addmm_activation.out(
            self, mat1, mat2, alpha=scalar, beta=scalar, use_gelu=use_gelu, out=out
        )
    assert result is out
    assert out.data_ptr() == pointer
    utils.gems_assert_close(out, ref_result, dtype, reduce_dim=K)
    utils.gems_assert_close(result, ref_result, dtype, reduce_dim=K)


@pytest.mark.addmm_activation_out
@pytest.mark.underscore_addmm_activation_out
def test_addmm_activation_out_resizes_empty_buffer():
    torch.manual_seed(0)
    self = torch.randn((5,), device=flag_gems.device)
    mat1 = torch.randn((3, 4), device=flag_gems.device)
    mat2 = torch.randn((4, 5), device=flag_gems.device)
    out = torch.empty((0,), device=flag_gems.device)
    ref_out = torch.empty((0,), device="cpu")
    ref_result = torch.ops.aten._addmm_activation.out(
        utils.to_reference(self, True),
        utils.to_reference(mat1, True),
        utils.to_reference(mat2, True),
        out=ref_out,
    )
    with flag_gems.use_gems():
        result = torch.ops.aten._addmm_activation.out(self, mat1, mat2, out=out)
    assert result is out
    assert out.shape == (3, 5)
    utils.gems_assert_close(out, ref_result, torch.float32, reduce_dim=4)


@pytest.mark.addmm_activation_out
@pytest.mark.underscore_addmm_activation_out
def test_addmm_activation_out_noncontiguous_preserves_storage_and_stride():
    torch.manual_seed(0)
    self = torch.randn((5,), device=flag_gems.device)
    mat1 = torch.randn((3, 4), device=flag_gems.device)
    mat2 = torch.randn((4, 5), device=flag_gems.device)
    out = torch.empty((5, 3), device=flag_gems.device).t()
    ref_out = torch.empty((5, 3), device="cpu").t()
    pointer = out.data_ptr()
    stride = out.stride()
    ref_result = torch.ops.aten._addmm_activation.out(
        utils.to_reference(self, True),
        utils.to_reference(mat1, True),
        utils.to_reference(mat2, True),
        out=ref_out,
    )
    with flag_gems.use_gems():
        result = torch.ops.aten._addmm_activation.out(self, mat1, mat2, out=out)
    assert result is out
    assert out.data_ptr() == pointer
    assert out.stride() == stride
    utils.gems_assert_close(out, ref_result, torch.float32, reduce_dim=4)


@pytest.mark.addmm_activation_out
@pytest.mark.underscore_addmm_activation_out
def test_addmm_activation_out_rejects_wrong_dtype():
    torch.manual_seed(0)
    self = torch.randn((5,), device=flag_gems.device)
    mat1 = torch.randn((3, 4), device=flag_gems.device)
    mat2 = torch.randn((4, 5), device=flag_gems.device)
    out = torch.empty((3, 5), dtype=torch.int32, device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        torch.ops.aten._addmm_activation.out(self, mat1, mat2, out=out)
