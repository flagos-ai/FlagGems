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

from . import test_utils as tu

# aten::size reports the operand's shape as int[] (default call) or int (dim
# call) and reads no element, so its result is dtype- and payload-independent.
# One tensor operand leaves nothing to broadcast and a Python int has no
# autograd graph, so the covered dimensions are the dtype/shape/value-range
# grid, the two call forms, the view/layout cases and the negative rows.
SIZE_DTYPES = tu.REQUIRED_DTYPES + [torch.float64, torch.bool, torch.complex64]


def _assert_extents(res_out, ref_out):
    # int[] schema: a list of Python ints, compared through int64 buffers.
    assert isinstance(res_out, list)
    assert all(isinstance(e, int) and not isinstance(e, bool) for e in res_out)
    tu.assert_result_equal(
        torch.tensor(list(res_out), dtype=torch.int64),
        torch.tensor(list(ref_out), dtype=torch.int64),
    )


def _assert_dim(res_out, ref_out):
    # int schema: a Python int, compared through an int64 buffer.
    assert isinstance(res_out, int) and not isinstance(res_out, bool)
    tu.assert_result_equal(
        torch.tensor(res_out, dtype=torch.int64),
        torch.tensor(ref_out, dtype=torch.int64),
    )


@pytest.mark.size
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SIZE_DTYPES)
def test_size_default(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.size(ref_inp)
    res_out = flag_gems.size(inp)

    _assert_extents(res_out, ref_out)


# One row per (shape, dim): dim 0, a middle rank, the last dim and the matching
# negative positions for every rank the `int` call accepts.
SIZE_DIM_ROWS = [
    ((6,), 0),
    ((6,), -1),
    ((3, 5), 0),
    ((3, 5), 1),
    ((3, 5), -1),
    ((3, 5), -2),
    ((4, 6, 8), 0),
    ((4, 6, 8), 1),
    ((4, 6, 8), 2),
    ((4, 6, 8), -1),
    ((4, 6, 8), -3),
    ((2, 3, 5, 7), 0),
    ((2, 3, 5, 7), 2),
    ((2, 3, 5, 7), 3),
    ((2, 3, 5, 7), -1),
    ((2, 3, 5, 7), -4),
    ((2, 3, 4, 5, 6), 0),
    ((2, 3, 4, 5, 6), 3),
    ((2, 3, 4, 5, 6), 4),
    ((2, 3, 4, 5, 6), -1),
    ((2, 3, 4, 5, 6), -5),
]

SIZE_DIM_DTYPES = [torch.float32, torch.int64, torch.bool]

SIZE_DIM_RANGES = tu.selected_cases([["-1", "1"], ["min", "max"]], quick=[["-1", "1"]])


@pytest.mark.size
@pytest.mark.parametrize("shape,dim", SIZE_DIM_ROWS)
@pytest.mark.parametrize("value_range", SIZE_DIM_RANGES)
@pytest.mark.parametrize("dtype", SIZE_DIM_DTYPES)
def test_size_int_dim(shape, dim, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.size.int(ref_inp, dim)
    res_out = flag_gems.size(inp, dim)

    _assert_dim(res_out, ref_out)


@pytest.mark.size
def test_size_int_dim_keyword():
    # The same entry point accepts the dim argument by keyword.
    inp = tu.make_input(torch.float32, (4, 6), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.size(ref_inp, dim=1)
    res_out = flag_gems.size(inp, dim=1)

    _assert_dim(res_out, ref_out)


def _view(base, layout):
    # Operand layouts whose shape differs from the backing storage: a transpose,
    # a strided slice and a window starting at a nonzero storage offset.
    if layout == "asis":
        return base
    if layout == "transposed":
        return base.t()
    if layout == "column_step":
        return base[::2, ::3]
    if layout == "offset_window":
        return base[2:6, 1:5]
    raise ValueError("unsupported layout " + repr(layout))


# A metadata query must report the operand's own shape, not its base's, so these
# rows carry a non-contiguous view, a nonzero-offset view, an empty dimension and
# the 0-dim scalar.
SIZE_VIEW_ROWS = [
    ((16, 32), "transposed"),
    ((12, 24), "column_step"),
    ((10, 8), "offset_window"),
    ((), "asis"),
    ((5, 0, 7), "asis"),
    ((0, 3), "asis"),
]

# The same operands in the `int` call form, at an index valid for the view's own
# rank (rank 0 has no index and stays out of this list).
SIZE_VIEW_DIMS = [
    ((16, 32), "transposed", 0),
    ((16, 32), "transposed", -1),
    ((12, 24), "column_step", 1),
    ((10, 8), "offset_window", -2),
    ((5, 0, 7), "asis", 1),
    ((0, 3), "asis", 0),
]


@pytest.mark.size
@pytest.mark.parametrize("storage_shape,layout", SIZE_VIEW_ROWS)
@pytest.mark.parametrize("dtype", SIZE_DTYPES)
def test_size_view(storage_shape, layout, dtype):
    base = tu.make_input(dtype, storage_shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = _view(base, layout)
    ref_inp = _view(ref_base, layout)

    ref_out = torch.ops.aten.size(ref_inp)
    res_out = flag_gems.size(inp)

    _assert_extents(res_out, ref_out)


@pytest.mark.size
@pytest.mark.parametrize("storage_shape,layout,dim", SIZE_VIEW_DIMS)
@pytest.mark.parametrize("dtype", SIZE_DIM_DTYPES)
def test_size_view_dim(storage_shape, layout, dim, dtype):
    base = tu.make_input(dtype, storage_shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = _view(base, layout)
    ref_inp = _view(ref_base, layout)

    ref_out = torch.ops.aten.size.int(ref_inp, dim)
    res_out = flag_gems.size(inp, dim)

    _assert_dim(res_out, ref_out)


@pytest.mark.size
def test_size_extent_above_int32():
    # A zero-numel tensor whose first extent exceeds 2**31: the reported int must
    # stay exact rather than wrap through a 32-bit accumulator.
    inp = torch.zeros((2**33, 0), device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.size(ref_inp)
    res_out = flag_gems.size(inp)
    _assert_extents(res_out, ref_out)

    ref_dim = torch.ops.aten.size.int(ref_inp, 0)
    res_dim = flag_gems.size(inp, 0)
    _assert_dim(res_dim, ref_dim)


def _sparse_input(layout):
    if layout == "coo":
        return torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [2, 3]], device=flag_gems.device),
            torch.ones(2, device=flag_gems.device),
            (5, 6),
            device=flag_gems.device,
        )
    return torch.tensor(
        [[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]], device=flag_gems.device
    ).to_sparse_csr()


@pytest.mark.size
@pytest.mark.parametrize("layout", ["coo", "csr"])
def test_size_sparse_layout(layout):
    # Non-strided operands report their logical shape too (COO -> [5, 6],
    # CSR -> [2, 3]).
    inp = _sparse_input(layout)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.size(ref_inp)
    res_out = flag_gems.size(inp)

    _assert_extents(res_out, ref_out)


# Default-only, as for every positive special-value workload: the payload must
# leave the reported metadata unchanged. e4m3fn contributes the nan scenario
# (it cannot represent inf) while e5m2 and the non-FP8 floats contribute nan,
# inf and mixed.
SIZE_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases(
        [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float64,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]
    ),
    quick=[],
)


@pytest.mark.size
@pytest.mark.parametrize("dtype,scenario", SIZE_SPECIAL_CASES)
def test_size_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.size(ref_inp)
    res_out = flag_gems.size(inp)

    _assert_extents(res_out, ref_out)


# Native rejects an out-of-rank dim and a 0-dim operand with IndexError and a
# non-integer dim with RuntimeError. The str rows are the string-dim form, whose
# positive named-tensor path is unreachable (TorchScript refuses named tensors),
# so a string dim is only ever a rejected argument here.
SIZE_INVALID_DIMS = [
    ((4, 6), 2, IndexError),
    ((4, 6), -3, IndexError),
    ((4, 6), 1.5, (RuntimeError, TypeError)),
    ((4, 6), "1", (RuntimeError, TypeError)),
    ((4, 6), "dim_0", (RuntimeError, TypeError)),
    ((4, 6), None, (RuntimeError, TypeError)),
    ((), 0, IndexError),
]


@pytest.mark.size
@pytest.mark.parametrize("shape,dim,error", SIZE_INVALID_DIMS)
def test_size_invalid_dim(shape, dim, error):
    inp = torch.zeros(shape, device=flag_gems.device)

    with pytest.raises(error):
        flag_gems.size(inp, dim)
