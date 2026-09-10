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

# (input shape, index shape, dim) triples that exercise same-shape gather,
# broadcasting on non-dim axes, and different dim positions.
CASES = [
    ((32, 32), (32, 32), 0),
    ((32, 32), (32, 32), 1),
    ((32, 32), (32, 32), -1),
    ((128, 64), (128, 8), 1),
    ((8, 128), (4, 128), 0),
    ((16, 8, 32), (16, 8, 32), 2),
    ((16, 8, 32), (16, 3, 32), 1),
    ((1, 8, 32), (16, 8, 4), 2),  # broadcast on dim 0 (input) and dim 2 sizes
    ((1024, 1024), (1024, 1024), 1),
]


def _make_input(shape, dtype):
    if dtype == torch.bool:
        return torch.randint(
            0, 2, shape, dtype=torch.int32, device=flag_gems.device
        ).bool()
    if dtype == torch.uint8:
        return torch.randint(0, 100, shape, dtype=dtype, device=flag_gems.device)
    if dtype in utils.ALL_INT_DTYPES or dtype == torch.int8:
        return torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


def _make_index(inp_shape, idx_shape, dim):
    dim = dim % len(inp_shape)
    hi = inp_shape[dim]
    return torch.randint(0, hi, idx_shape, dtype=torch.int64, device=flag_gems.device)


@pytest.mark.take_along_dim
# Pure gather copies values exactly, so representative float precisions suffice
# here; full dtype coverage (int/bool) is exercised in test_take_along_dim_dtypes.
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inp_shape,idx_shape,dim", CASES)
def test_take_along_dim(dtype, inp_shape, idx_shape, dim):
    inp = _make_input(inp_shape, dtype)
    index = _make_index(inp_shape, idx_shape, dim)
    ref_inp = utils.to_reference(inp)
    ref_idx = utils.to_reference(index)

    ref_out = torch.take_along_dim(ref_inp, ref_idx, dim=dim)
    res_out = flag_gems.take_along_dim(inp, index, dim=dim)

    # Pure gather: values must match exactly.
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.take_along_dim
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_take_along_dim_dtypes(dtype):
    inp = _make_input((16, 24), dtype)
    index = _make_index((16, 24), (16, 8), 1)
    ref_inp = utils.to_reference(inp)
    ref_idx = utils.to_reference(index)

    ref_out = torch.take_along_dim(ref_inp, ref_idx, dim=1)
    res_out = flag_gems.take_along_dim(inp, index, dim=1)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.take_along_dim
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_take_along_dim_none(dtype):
    # dim=None flattens the input; result takes the shape of indices.
    inp = _make_input((8, 8), dtype)
    index = torch.randint(
        0, inp.numel(), (3, 5), dtype=torch.int64, device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)
    ref_idx = utils.to_reference(index)

    ref_out = torch.take_along_dim(ref_inp, ref_idx)
    res_out = flag_gems.take_along_dim(inp, index)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.take_along_dim
# Representative float precisions; full dtype coverage is in test_take_along_dim_dtypes.
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_take_along_dim_noncontiguous(dtype):
    inp = _make_input((7, 11), dtype).transpose(0, 1)
    index = _make_index(tuple(inp.shape), (11, 4), 1)
    ref_inp = utils.to_reference(inp)
    ref_idx = utils.to_reference(index)

    ref_out = torch.take_along_dim(ref_inp, ref_idx, dim=1)
    res_out = flag_gems.take_along_dim(inp, index, dim=1)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.take_along_dim_out
# Pure gather copies values exactly, so representative float precisions suffice.
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inp_shape,idx_shape,dim", CASES)
def test_take_along_dim_out(dtype, inp_shape, idx_shape, dim):
    inp = _make_input(inp_shape, dtype)
    index = _make_index(inp_shape, idx_shape, dim)
    ref_inp = utils.to_reference(inp)
    ref_idx = utils.to_reference(index)

    ref_out = torch.take_along_dim(ref_inp, ref_idx, dim=dim)
    out = torch.empty_like(ref_out, device=flag_gems.device)
    res_out = flag_gems.take_along_dim_out(inp, index, dim=dim, out=out)

    assert res_out is out
    utils.gems_assert_equal(res_out, ref_out)
