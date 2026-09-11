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
from . import conftest as cfg

REDUCE_MODES = ["prod", "mean", "amax", "amin"]
INDEX_DTYPES = [torch.int64, torch.int32]

if cfg.QUICK_MODE:
    DTYPE_LIST = [torch.float32]
    CASES = [((4, 5), 1)]
else:
    DTYPE_LIST = utils.ALL_FLOAT_DTYPES
    CASES = [
        ((8,), 0),
        ((4, 7), 1),
        ((5, 3, 4), 0),
        ((3, 5, 2), -2),
    ]


def _make_values(shape, dtype):
    if dtype in (torch.float16, torch.bfloat16):
        values = torch.empty(shape, dtype=dtype, device=flag_gems.device)
        return values.uniform_(0.5, 1.5)
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


def _make_index(index_len, out_len, index_dtype):
    base = torch.arange(index_len, dtype=index_dtype, device=flag_gems.device)
    return (base * 3 + 1) % out_len


@pytest.mark.index_reduce_
@pytest.mark.parametrize(("shape", "dim"), CASES)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("reduce", REDUCE_MODES)
@pytest.mark.parametrize("include_self", [True, False])
def test_index_reduce_(shape, dim, dtype, index_dtype, reduce, include_self):
    inp = _make_values(shape, dtype)
    dim = dim % inp.ndim
    source_shape = list(shape)
    source_shape[dim] = max(1, shape[dim] + 2)
    source = _make_values(source_shape, dtype)
    index = _make_index(source_shape[dim], shape[dim], index_dtype)

    ref_inp = utils.to_reference(inp.clone(), upcast=True)
    ref_source = utils.to_reference(source, upcast=True)
    ref_index = utils.to_reference(index)
    ref_inp.index_reduce_(dim, ref_index, ref_source, reduce, include_self=include_self)

    with flag_gems.use_gems():
        res = inp.index_reduce_(dim, index, source, reduce, include_self=include_self)

    assert res is inp
    utils.gems_assert_close(inp, ref_inp, dtype=dtype, reduce_dim=source_shape[dim])


@pytest.mark.index_reduce_
@pytest.mark.parametrize("reduce", REDUCE_MODES)
def test_index_reduce_noncontiguous(reduce):
    dtype = torch.float32
    inp = torch.randn((6, 4), dtype=dtype, device=flag_gems.device).t()
    source = torch.randn((8, 4), dtype=dtype, device=flag_gems.device).t()
    index = torch.tensor([0, 2, 1, 0, 3, 1, 2, 0], device=flag_gems.device)
    dim = 1

    ref_inp = utils.to_reference(inp.clone(), upcast=True)
    ref_source = utils.to_reference(source, upcast=True)
    ref_index = utils.to_reference(index)
    ref_inp.index_reduce_(dim, ref_index, ref_source, reduce, include_self=False)

    with flag_gems.use_gems():
        inp.index_reduce_(dim, index, source, reduce, include_self=False)

    utils.gems_assert_close(inp, ref_inp, dtype=dtype, reduce_dim=source.size(dim))


@pytest.mark.index_reduce_
@pytest.mark.parametrize("reduce", REDUCE_MODES)
@pytest.mark.parametrize("include_self", [True, False])
def test_index_reduce_duplicate_index_short_source(reduce, include_self):
    dtype = torch.float32
    inp = torch.randn((4, 6), dtype=dtype, device=flag_gems.device)
    source = torch.randn((4, 4), dtype=dtype, device=flag_gems.device)
    index = torch.tensor([0, 2, 2, 4], dtype=torch.int64, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.clone(), upcast=True)
    ref_source = utils.to_reference(source, upcast=True)
    ref_index = utils.to_reference(index)
    ref_inp.index_reduce_(1, ref_index, ref_source, reduce, include_self=include_self)

    with flag_gems.use_gems():
        inp.index_reduce_(1, index, source, reduce, include_self=include_self)

    utils.gems_assert_close(inp, ref_inp, dtype=dtype, reduce_dim=source.size(1))


@pytest.mark.index_reduce_
@pytest.mark.parametrize("include_self", [True, False])
def test_index_reduce_empty_index(include_self):
    dtype = torch.float32
    inp = torch.randn((3, 4), dtype=dtype, device=flag_gems.device)
    source = torch.empty((3, 0), dtype=dtype, device=flag_gems.device)
    index = torch.empty((0,), dtype=torch.int64, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.clone(), upcast=True)
    ref_source = utils.to_reference(source, upcast=True)
    ref_index = utils.to_reference(index)
    ref_inp.index_reduce_(1, ref_index, ref_source, "mean", include_self=include_self)

    with flag_gems.use_gems():
        inp.index_reduce_(1, index, source, "mean", include_self=include_self)

    utils.gems_assert_close(inp, ref_inp, dtype=dtype, reduce_dim=1)
