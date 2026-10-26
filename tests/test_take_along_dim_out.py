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

REAL_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + [torch.int8, torch.uint8]
    + utils.ALL_INT_DTYPES
    + utils.BOOL_TYPES
)
CASES = [
    ((32, 32), (32, 32), 0),
    ((32, 32), (32, 32), -1),
    ((128, 64), (128, 8), 1),
    ((8, 128), (4, 128), 0),
    ((16, 8, 32), (16, 3, 32), 1),
    ((1, 8, 32), (16, 8, 4), 2),
    ((1024, 1024), (1024, 1024), 1),
]


def _make_input(shape, dtype):
    torch.manual_seed(0)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=flag_gems.device).bool()
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(
        0 if dtype == torch.uint8 else -50,
        50,
        shape,
        dtype=dtype,
        device=flag_gems.device,
    )


def _make_index(inp_shape, idx_shape, dim):
    torch.manual_seed(0)
    dim = dim % len(inp_shape)
    return torch.randint(0, inp_shape[dim], idx_shape, device=flag_gems.device)


@pytest.mark.take_along_dim_out
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("inp_shape,idx_shape,dim", CASES)
def test_take_along_dim_out_preserves_storage(dtype, inp_shape, idx_shape, dim):
    if (
        flag_gems.vendor_name == "kunlunxin"
        and not utils.TO_CPU
        and dtype in (torch.uint8, torch.int16)
    ):
        pytest.skip(
            "Kunlunxin native take_along_dim reference lacks uint8/int16 support"
        )
    inp = _make_input(inp_shape, dtype)
    index = _make_index(inp_shape, idx_shape, dim)
    ref = torch.take_along_dim(
        utils.to_reference(inp, False), utils.to_reference(index, False), dim=dim
    )
    out = torch.empty(idx_shape, dtype=dtype, device=flag_gems.device)
    pointer, stride, storage_offset = out.data_ptr(), out.stride(), out.storage_offset()
    result = flag_gems.take_along_dim_out(inp, index, dim=dim, out=out)
    assert result is out
    assert out.data_ptr() == pointer
    assert out.stride() == stride
    assert out.storage_offset() == storage_offset
    utils.gems_assert_equal(out, ref)


@pytest.mark.take_along_dim_out
def test_take_along_dim_out_none_and_resize():
    torch.manual_seed(0)
    inp = torch.randn((8, 8), device=flag_gems.device)
    index = torch.randint(0, inp.numel(), (3, 5), device=flag_gems.device)
    ref = torch.take_along_dim(
        utils.to_reference(inp, False), utils.to_reference(index, False)
    )
    out = torch.empty((0,), dtype=inp.dtype, device=flag_gems.device)
    result = flag_gems.take_along_dim_out(inp, index, dim=None, out=out)
    assert result is out
    assert out.shape == ref.shape
    utils.gems_assert_equal(out, ref)


@pytest.mark.take_along_dim_out
def test_take_along_dim_out_noncontiguous():
    torch.manual_seed(0)
    inp = torch.randn((7, 11), device=flag_gems.device).T
    index = torch.randint(0, inp.shape[1], (11, 4), device=flag_gems.device)
    ref = torch.take_along_dim(
        utils.to_reference(inp, False), utils.to_reference(index, False), dim=1
    )
    out = torch.empty((4, 11), device=flag_gems.device).T
    pointer, stride, storage_offset = out.data_ptr(), out.stride(), out.storage_offset()
    result = flag_gems.take_along_dim_out(inp, index, dim=1, out=out)
    assert result is out
    assert out.data_ptr() == pointer
    assert out.stride() == stride
    assert out.storage_offset() == storage_offset
    utils.gems_assert_equal(out, ref)


@pytest.mark.take_along_dim_out
def test_take_along_dim_out_rejects_dtype_mismatch():
    torch.manual_seed(0)
    inp = torch.randn((4, 6), device=flag_gems.device)
    index = torch.randint(0, 6, (4, 3), device=flag_gems.device)
    out = torch.empty((4, 3), dtype=torch.int32, device=flag_gems.device)
    snapshot = out.clone()
    with pytest.raises(RuntimeError):
        flag_gems.take_along_dim_out(inp, index, dim=1, out=out)
    assert out.device == snapshot.device
    assert out.dtype == snapshot.dtype
    assert out.shape == snapshot.shape
    utils.gems_assert_equal(out.cpu(), snapshot.cpu())
