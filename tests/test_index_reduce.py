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

REDUCTIONS = ("prod", "mean", "amax", "amin")
SHAPE_DIMS = (((8,), 0), ((4, 7), 1), ((3, 5, 2), -2))


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


@pytest.mark.index_reduce
@pytest.mark.parametrize("shape,dim", SHAPE_DIMS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("index_dtype", (torch.int64, torch.int32))
@pytest.mark.parametrize("reduce", REDUCTIONS)
@pytest.mark.parametrize("include_self", (True, False))
def test_index_reduce(shape, dim, dtype, index_dtype, reduce, include_self):
    inp, dim, index, source = _case(shape, dim, dtype, index_dtype)
    snapshot = inp.clone()
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.index_reduce(
        ref_inp,
        dim,
        utils.to_reference(index, False),
        utils.to_reference(source, True),
        reduce,
        include_self=include_self,
    )

    result = flag_gems.index_reduce(
        inp, dim, index, source, reduce, include_self=include_self
    )

    assert result is not inp
    assert result.shape == inp.shape
    assert result.dtype == dtype
    utils.gems_assert_close(result, ref_out, dtype=dtype, reduce_dim=source.size(dim))
    utils.gems_assert_equal(inp, utils.to_reference(snapshot))


@pytest.mark.index_reduce
@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_index_reduce_noncontiguous(reduce):
    inp = _values((6, 4), torch.float32).t()
    source = _values((8, 4), torch.float32).t()
    index = torch.tensor([0, 2, 1, 0, 3, 1, 2, 0], device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.index_reduce(
        ref_inp,
        1,
        utils.to_reference(index, False),
        utils.to_reference(source, True),
        reduce,
        include_self=False,
    )

    result = flag_gems.index_reduce(inp, 1, index, source, reduce, include_self=False)

    assert result.is_contiguous()
    utils.gems_assert_close(result, ref_out, dtype=torch.float32, reduce_dim=8)


@pytest.mark.parametrize("reduce", REDUCTIONS)
@pytest.mark.parametrize(
    "index", ([0, 2, 4, 5], [0, 2, 2, 4]), ids=("unique", "duplicate")
)
@pytest.mark.parametrize(
    "inplace",
    [
        pytest.param(False, marks=pytest.mark.index_reduce),
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
    if inplace:
        ref_out = ref_inp.index_reduce_(
            1, ref_index, ref_source, reduce, include_self=True
        )
        pointer = inp.data_ptr()
        result = flag_gems.index_reduce_(
            inp, 1, index, source, reduce, include_self=True
        )
        assert result is inp
        assert result.data_ptr() == pointer
    else:
        ref_out = torch.index_reduce(
            ref_inp, 1, ref_index, ref_source, reduce, include_self=True
        )
        result = flag_gems.index_reduce(
            inp, 1, index, source, reduce, include_self=True
        )
        assert result is not inp
    utils.gems_assert_close(result, ref_out, dtype=inp.dtype, reduce_dim=source.size(1))


@pytest.mark.index_reduce
@pytest.mark.parametrize("include_self", (True, False))
def test_index_reduce_empty_index(include_self):
    inp = _values((3, 4), torch.float32)
    source = torch.empty((3, 0), dtype=inp.dtype, device=flag_gems.device)
    index = torch.empty((0,), dtype=torch.int64, device=flag_gems.device)
    ref_out = torch.index_reduce(
        utils.to_reference(inp, True),
        1,
        utils.to_reference(index, False),
        utils.to_reference(source, True),
        "mean",
        include_self=include_self,
    )

    result = flag_gems.index_reduce(
        inp, 1, index, source, "mean", include_self=include_self
    )

    utils.gems_assert_close(result, ref_out, dtype=inp.dtype)


@pytest.mark.index_reduce
def test_index_reduce_rejects_invalid_reduce():
    inp = torch.ones((2, 3), device=flag_gems.device)
    source = torch.ones((2, 1), device=flag_gems.device)
    index = torch.zeros((1,), dtype=torch.int64, device=flag_gems.device)

    with pytest.raises((AssertionError, RuntimeError)):
        flag_gems.index_reduce(inp, 1, index, source, "sum")


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
