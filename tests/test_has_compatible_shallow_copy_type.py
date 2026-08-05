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


@pytest.mark.has_compatible_shallow_copy_type
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_has_compatible_shallow_copy_type_dense(shape, dtype):
    # Two dense tensors are always shallow-copy compatible, regardless of
    # dtype, shape or device.
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch._has_compatible_shallow_copy_type(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch._has_compatible_shallow_copy_type(inp1, inp2)

    assert res_out == ref_out


@pytest.mark.has_compatible_shallow_copy_type
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_has_compatible_shallow_copy_type_dense_vs_sparse(dtype):
    # Dense and sparse COO tensors are NOT compatible.
    dense = torch.randn((4, 4), dtype=dtype, device=flag_gems.device)
    sparse = torch.randn((4, 4), dtype=dtype, device=flag_gems.device).to_sparse()

    ref_out = torch._has_compatible_shallow_copy_type(dense, sparse)
    with flag_gems.use_gems():
        res_out = torch._has_compatible_shallow_copy_type(dense, sparse)

    assert res_out == ref_out
    assert res_out is False


@pytest.mark.has_compatible_shallow_copy_type
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_has_compatible_shallow_copy_type_sparse(dtype):
    # Two sparse COO tensors are compatible.
    sp1 = torch.randn((4, 4), dtype=dtype, device=flag_gems.device).to_sparse()
    sp2 = torch.randn((2, 3), dtype=dtype, device=flag_gems.device).to_sparse()

    ref_out = torch._has_compatible_shallow_copy_type(sp1, sp2)
    with flag_gems.use_gems():
        res_out = torch._has_compatible_shallow_copy_type(sp1, sp2)

    assert res_out == ref_out
    assert res_out is True


@pytest.mark.has_compatible_shallow_copy_type
def test_accuracy_has_compatible_shallow_copy_type_cross_dtype():
    # Compatibility is independent of dtype for dense tensors.
    fp = torch.randn((3, 4), device=flag_gems.device)
    it = torch.randint(0, 5, (3, 4), device=flag_gems.device)

    ref_out = torch._has_compatible_shallow_copy_type(fp, it)
    with flag_gems.use_gems():
        res_out = torch._has_compatible_shallow_copy_type(fp, it)

    assert res_out == ref_out
    assert res_out is True
