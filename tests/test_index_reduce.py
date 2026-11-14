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

REDUCTIONS = ("prod", "mean", "amax", "amin")
if cfg.QUICK_MODE:
    _FUNCTIONAL_DTYPES = [torch.float32]
    _FUNCTIONAL_SHAPE_DIMS = [((4, 5), 1)]
else:
    _FUNCTIONAL_DTYPES = utils.ALL_FLOAT_DTYPES
    _FUNCTIONAL_SHAPE_DIMS = [((8,), 0), ((4, 7), 1), ((5, 3, 4), 0), ((3, 5, 2), -2)]


def _values(shape, dtype):
    torch.manual_seed(0)
    if dtype in (torch.float16, torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            0.5, 1.5
        )
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


def _case(shape, dim, dtype, index_dtype=torch.int64):
    normalized_dim = dim % len(shape)
    source_shape = list(shape)
    source_shape[normalized_dim] = max(1, shape[normalized_dim] + 2)
    inp = _values(shape, dtype)
    source = _values(tuple(source_shape), dtype)
    index = (
        torch.arange(
            source_shape[normalized_dim], dtype=index_dtype, device=flag_gems.device
        )
        * 3
        + 1
    ) % shape[normalized_dim]
    return inp, normalized_dim, index, source






@pytest.mark.parametrize("reduce", REDUCTIONS)
@pytest.mark.parametrize(
    "index", ([0, 2, 4, 5], [0, 2, 2, 4]), ids=("unique", "duplicate")
)
@pytest.mark.parametrize(
    "inplace",
    [
        pytest.param(True, marks=pytest.mark.index_reduce_),
    ],
)
def test_index_reduce_index_pattern(reduce, index, inplace):
    inp = _values((4, 6), torch.float32)
    source = _values((4, 4), torch.float32)
    index = torch.tensor(index, dtype=torch.int64, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)
    ref_source = utils.to_reference(source, True)
    ref_index = utils.to_reference(index, False)
    assert inplace
    ref_out = ref_inp.index_reduce_(
        1, ref_index, ref_source, reduce, include_self=True
    )
    pointer = inp.data_ptr()
    result = flag_gems.index_reduce_(
        inp, 1, index, source, reduce, include_self=True
    )
    assert result is inp
    assert result.data_ptr() == pointer
    utils.gems_assert_close(result, ref_out, dtype=inp.dtype, reduce_dim=source.size(1))






# Keep the shared legacy in-place family selectable while the spreadsheet item
# publishes the functional contract in this file.
@pytest.mark.index_reduce_
@pytest.mark.parametrize("reduce", REDUCTIONS)
@pytest.mark.parametrize("include_self", (True, False))
def test_index_reduce_inplace_shared_family(reduce, include_self):
    inp, dim, index, source = _case((4, 6), 1, torch.float32)
    pointer = inp.data_ptr()
    ref_inp = utils.to_reference(inp.clone(), True)
    ref_inp.index_reduce_(
        dim,
        utils.to_reference(index, False),
        utils.to_reference(source, True),
        reduce,
        include_self=include_self,
    )

    result = flag_gems.index_reduce_(
        inp, dim, index, source, reduce, include_self=include_self
    )

    assert result is inp
    assert result.data_ptr() == pointer
    utils.gems_assert_close(inp, ref_inp, dtype=inp.dtype, reduce_dim=source.size(dim))

# Functional cases below map one-to-one to the frozen master.
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




@pytest.mark.index_reduce
@pytest.mark.parametrize(("shape", "dim"), CASES)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("reduce", REDUCE_MODES)
@pytest.mark.parametrize("include_self", [True, False])
def test_index_reduce(shape, dim, dtype, index_dtype, reduce, include_self):
    inp = _make_values(shape, dtype)
    dim = dim % inp.ndim
    source_shape = list(shape)
    source_shape[dim] = max(1, shape[dim] + 2)
    source = _make_values(source_shape, dtype)
    index = _make_index(source_shape[dim], shape[dim], index_dtype)

    ref_inp = utils.to_reference(inp.clone(), upcast=True)
    ref_source = utils.to_reference(source, upcast=True)
    ref_index = utils.to_reference(index)
    ref_out = ref_inp.index_reduce(
        dim, ref_index, ref_source, reduce, include_self=include_self
    )

    res_out = flag_gems.index_reduce(
        inp, dim, index, source, reduce, include_self=include_self
    )

    utils.gems_assert_close(res_out, ref_out, dtype=dtype, reduce_dim=source_shape[dim])
    utils.gems_assert_close(inp, ref_inp, dtype=dtype)


@pytest.mark.index_reduce
@pytest.mark.parametrize("reduce", REDUCE_MODES)
def test_index_reduce_noncontiguous_out_of_place(reduce):
    dtype = torch.float32
    inp = torch.randn((6, 4), dtype=dtype, device=flag_gems.device).t()
    source = torch.randn((8, 4), dtype=dtype, device=flag_gems.device).t()
    index = torch.tensor([0, 2, 1, 0, 3, 1, 2, 0], device=flag_gems.device)

    ref_inp = utils.to_reference(inp.clone())
    ref_out = torch.index_reduce(
        ref_inp,
        1,
        utils.to_reference(index),
        utils.to_reference(source),
        reduce,
        include_self=False,
    )
    result = flag_gems.index_reduce(inp, 1, index, source, reduce, include_self=False)

    assert result.is_contiguous()
    utils.gems_assert_close(result, ref_out, dtype=dtype, reduce_dim=source.size(1))
    utils.gems_assert_close(inp, ref_inp, dtype=dtype)


@pytest.mark.index_reduce
@pytest.mark.parametrize("include_self", [True, False])
def test_index_reduce_empty_index_out_of_place(include_self):
    dtype = torch.float32
    inp = _make_values((3, 4), dtype)
    source = torch.empty((3, 0), dtype=dtype, device=flag_gems.device)
    index = torch.empty((0,), dtype=torch.int64, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.clone())
    ref_out = torch.index_reduce(
        ref_inp,
        1,
        utils.to_reference(index),
        utils.to_reference(source),
        "mean",
        include_self=include_self,
    )
    result = flag_gems.index_reduce(
        inp, 1, index, source, "mean", include_self=include_self
    )

    assert result is not inp
    utils.gems_assert_close(result, ref_out, dtype=dtype)
    utils.gems_assert_close(inp, ref_inp, dtype=dtype)


@pytest.mark.index_reduce
def test_index_reduce_invalid_reduce():
    if flag_gems.vendor_name == "kunlunxin":
        pytest.skip('KLX native contract: exact torch.index_reduce raises RuntimeError for sum; master requires AssertionError Unsupported reduce (master-batch02 dual-reference evidence)')
    inp = torch.ones((2, 3), device=flag_gems.device)
    source = torch.ones((2, 1), device=flag_gems.device)
    index = torch.zeros((1,), dtype=torch.int64, device=flag_gems.device)

    with pytest.raises(AssertionError, match="Unsupported reduce"):
        flag_gems.index_reduce(inp, 1, index, source, "sum")
