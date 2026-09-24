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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import test_utils as tu

# The operator name starts with an underscore, so pytest cannot resolve the
# marker by attribute lookup; register it on the MarkGenerator.
setattr(
    pytest.mark,
    "_sparse_csr_prod",
    MarkDecorator(Mark("_sparse_csr_prod", (), {}, _ispytest=True), _ispytest=True),
)

_DTYPE_CAPS = {
    torch.bfloat16: flag_gems.runtime.device.support_bf16,
    torch.float64: flag_gems.runtime.device.support_fp64,
    torch.int64: flag_gems.runtime.device.support_int64,
}


def _supported(dtype):
    """Collection-time capability gate for input and requested output dtypes."""
    if dtype is None:
        return True
    return _DTYPE_CAPS.get(dtype, True)


# The CSR kernel asserts input_dim == 2, so the shared rank grid is expressed as
# 2-D workloads (its 2-D entry plus single row/column, single element, and empty
# row/column boundaries).
_GRID_SHAPES = tu.selected_cases(
    [
        (1024, 1024),
        (20, 320),
        (16, 128),
        (1, 256),
        (256, 1),
        (1, 1),
        (0, 8),
        (8, 0),
    ],
    quick=[(2, 19)],
)

# bool and both float8 dtypes have no kernel on this backend; the rejection test
# covers them.
_DTYPES = [
    dtype
    for dtype in (
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
    )
    if _supported(dtype)
]

# keepdim must be True natively; the grid exercises that call form and the
# rejection test covers False.
_GRID_DIMS = tu.selected_cases([[0], [1]], quick=[[0]])

_GRID_ROWS = [
    (shape, dtype, value_range, dim)
    for dtype in _DTYPES
    for shape in _GRID_SHAPES
    for value_range in tu.selected_ranges()
    for dim in _GRID_DIMS
]

# dim is an int[1] list: negative (normalized) values, multi-axis lists and the
# empty list select distinct reduction paths and output shapes.
_DIM_FORMS = ([0], [1], [-1], [-2], [0, 1], [1, 0], [])

_DIM_ROWS = [
    (shape, dtype, ["-1", "1"], dim)
    for shape, dtype in (
        ((1024, 1024), torch.float16),
        ((20, 320), torch.int32),
        ((16, 128), torch.float32),
    )
    if _supported(dtype)
    for dim in _DIM_FORMS
]

_PROD_ROWS = _GRID_ROWS + tu.selected_cases(_DIM_ROWS, quick=[])


@pytest.mark._sparse_csr_prod
@pytest.mark.parametrize("shape,dtype,value_range,dim", _PROD_ROWS)
def test__sparse_csr_prod(shape, dtype, value_range, dim):
    inp = tu.make_input(dtype, shape, value_range).to_sparse_csr()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_csr_prod.dim_dtype(ref_inp, dim, True)
    res_out = flag_gems._sparse_csr_prod(inp, dim, True)

    tu.assert_result_close(res_out, ref_out)


# Dense -> CSR drops explicit zeros, so stored zeros, empty rows/columns (an
# empty reduction group reduces to 0, not 1) and non-trivial column orders are
# built directly. int32 index tensors are valid natively as well.
# (crow, col, size, values)
_CSR_PATTERNS = [
    ([0, 2, 2, 3], [0, 3, 2], (3, 4), [2, 3, 4]),
    ([0, 1, 3, 4, 5], [2, 0, 2, 4, 1], (4, 5), [0, -2, -3, 1, -4]),
    ([0, 2, 4, 5, 8], [0, 4, 1, 3, 2, 0, 2, 4], (4, 5), [3, -2, 1, -3, 2, 5, -4, 2]),
    ([0, 0, 0, 0], [], (3, 3), []),
]

_PATTERN_DTYPES = [
    dtype
    for dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float64,
        torch.int32,
        torch.int64,
    )
    if _supported(dtype)
]

_PATTERN_ROWS = tu.selected_cases(
    [
        (pattern, dim, dtype, index_dtype)
        for pattern in range(len(_CSR_PATTERNS))
        for dim in ([0], [1])
        for dtype in _PATTERN_DTYPES
        for index_dtype in (torch.int64, torch.int32)
    ],
    quick=[],
)


@pytest.mark._sparse_csr_prod
@pytest.mark.parametrize("pattern,dim,dtype,index_dtype", _PATTERN_ROWS)
def test__sparse_csr_prod_sparse_pattern(pattern, dim, dtype, index_dtype):
    crow, col, size, values = _CSR_PATTERNS[pattern]
    inp = torch.sparse_csr_tensor(
        torch.tensor(crow, dtype=index_dtype, device=flag_gems.device),
        torch.tensor(col, dtype=index_dtype, device=flag_gems.device),
        torch.tensor(values, dtype=dtype, device=flag_gems.device),
        size=size,
        device=flag_gems.device,
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_csr_prod.dim_dtype(ref_inp, dim, True)
    res_out = flag_gems._sparse_csr_prod(inp, dim, True)

    tu.assert_result_close(res_out, ref_out)


# The optional dtype argument is omitted by the grid above and covered here,
# including the explicit None default; native integer requests yield int64.
_DTYPE_CONVERSION_ROWS = tu.selected_cases(
    [
        row
        for row in (
            ((20, 320), torch.int32, torch.int64, [0]),
            ((20, 320), torch.int32, torch.float32, [1]),
            ((20, 320), torch.float32, torch.int32, [0]),
            ((16, 128), torch.float32, torch.float64, [1]),
            ((16, 128), torch.float16, torch.float32, [0]),
            ((16, 128), torch.float32, torch.float16, [1]),
            ((16, 128), torch.float32, torch.complex64, [0]),
            ((2, 19), torch.float32, torch.float32, [1]),
            ((2, 19), torch.float32, None, [0]),
        )
        if _supported(row[1]) and _supported(row[2])
    ],
    quick=[],
)


@pytest.mark._sparse_csr_prod
@pytest.mark.parametrize("shape,in_dtype,out_dtype,dim", _DTYPE_CONVERSION_ROWS)
def test__sparse_csr_prod_result_dtype(shape, in_dtype, out_dtype, dim):
    inp = tu.make_input(in_dtype, shape, ["-1", "1"]).to_sparse_csr()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_csr_prod.dim_dtype(
        ref_inp, dim, True, dtype=out_dtype
    )
    res_out = flag_gems._sparse_csr_prod(inp, dim, True, dtype=out_dtype)

    tu.assert_result_close(res_out, ref_out)


# Dense -> CSR keeps the special values while dropping explicit zeros, so nan,
# inf and mixed remain distinct scenarios.
_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases(
        [
            dtype
            for dtype in (
                torch.float32,
                torch.float16,
                torch.bfloat16,
                torch.float64,
            )
            if _supported(dtype)
        ]
    ),
    quick=[],
)

_SPECIAL_DIMS = tu.selected_cases([[0], [1]], quick=[])


@pytest.mark._sparse_csr_prod
@pytest.mark.parametrize("dim", _SPECIAL_DIMS)
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test__sparse_csr_prod_special_values(dtype, scenario, dim):
    inp = tu.make_special_input(dtype, scenario).reshape(1, 5).to_sparse_csr()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_csr_prod.dim_dtype(ref_inp, dim, True)
    res_out = flag_gems._sparse_csr_prod(inp, dim, True)

    tu.assert_result_close(res_out, ref_out)


def _stale_csr_like(structure):
    """Same-nnz, same-dtype CSR buffer holding stale values.

    CSR requires the column indices of each row to be sorted and distinct, so
    the buffer keeps the result's own support and only replaces the stored
    values. An N x 1 result with fewer stored entries than rows admits other
    valid row distributions, and the buffer then holds them in the tail rows to
    exercise a genuinely different support.
    """
    crow = structure.crow_indices()
    col = structure.col_indices()
    device = structure.device
    values = torch.full_like(structure.values(), 7.0)
    rows, cols = structure.shape
    nnz = values.numel()
    if cols == 1 and nnz < rows:
        crow = torch.cat(
            [
                torch.zeros(rows - nnz + 1, dtype=crow.dtype, device=device),
                torch.arange(1, nnz + 1, dtype=crow.dtype, device=device),
            ]
        )
        col = torch.zeros(nnz, dtype=col.dtype, device=device)
    else:
        crow, col = crow.clone(), col.clone()
    return torch.sparse_csr_tensor(
        crow, col, values, size=structure.shape, device=device
    )


_OUT_ROWS = tu.selected_cases(
    [
        row
        for row in (
            ((1024, 1024), torch.float32, [0]),
            ((1024, 1024), torch.float32, [1]),
            ((16, 128), torch.float32, [0]),
            ((20, 320), torch.float16, [1]),
            ((20, 320), torch.bfloat16, [0]),
            ((2, 19), torch.bfloat16, [1]),
            ((1, 256), torch.float32, [0]),
            ((1, 256), torch.float16, [1]),
            ((1, 1), torch.float32, [0]),
            ((256, 1), torch.float64, [0]),
            ((0, 8), torch.float32, [0]),
        )
        if _supported(row[1])
    ],
    quick=[],
)


@pytest.mark._sparse_csr_prod
@pytest.mark.parametrize("shape,dtype,dim", _OUT_ROWS)
def test__sparse_csr_prod_out(shape, dtype, dim):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).to_sparse_csr()
    ref_inp = tu.to_reference(inp)

    ref_structure = torch.ops.aten._sparse_csr_prod.dim_dtype(ref_inp, dim, True)
    ref_out = torch.ops.aten._sparse_csr_prod.dim_dtype_out(
        ref_inp, dim, True, out=_stale_csr_like(ref_structure)
    )

    res_buf = _stale_csr_like(ref_structure).to(inp.device)
    res_out = flag_gems._sparse_csr_prod(inp, dim, True, out=res_buf)

    assert res_out is res_buf
    tu.assert_result_close(res_out, ref_out)


# The bool/float8 kernels are missing on the measured nvidia backend; other
# vendors keep their own capability and collect no cases here.
_REJECTED_DTYPES = (
    [torch.bool, torch.float8_e4m3fn, torch.float8_e5m2]
    if flag_gems.runtime.device.vendor_name == "nvidia"
    else []
)


@pytest.mark._sparse_csr_prod
@pytest.mark.parametrize(
    "dtype", tu.selected_cases(_REJECTED_DTYPES, quick=_REJECTED_DTYPES)
)
def test__sparse_csr_prod_rejects_unsupported_dtype(dtype):
    inp = tu.make_input(torch.float32, (4, 5), ["-1", "1"]).to(dtype).to_sparse_csr()
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._sparse_csr_prod(inp, [0], True)


@pytest.mark._sparse_csr_prod
def test__sparse_csr_prod_rejects_keepdim_false():
    inp = tu.make_input(torch.float32, (4, 5), ["-1", "1"]).to_sparse_csr()
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._sparse_csr_prod(inp, [0], False)


@pytest.mark._sparse_csr_prod
@pytest.mark.parametrize("dim", ([0, 0], [1, 1]))
def test__sparse_csr_prod_rejects_duplicate_dim(dim):
    inp = tu.make_input(torch.float32, (4, 5), ["-1", "1"]).to_sparse_csr()
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        flag_gems._sparse_csr_prod(inp, dim, True)


@pytest.mark._sparse_csr_prod
@pytest.mark.parametrize("dim", ([2], [-3]))
def test__sparse_csr_prod_rejects_dim_out_of_range(dim):
    inp = tu.make_input(torch.float32, (4, 5), ["-1", "1"]).to_sparse_csr()
    with pytest.raises((IndexError, RuntimeError, ValueError)):
        flag_gems._sparse_csr_prod(inp, dim, True)


@pytest.mark._sparse_csr_prod
def test__sparse_csr_prod_rejects_batched_csr_input():
    inp = tu.make_input(torch.float32, (2, 3, 4), ["-1", "1"]).to_sparse_csr()
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        flag_gems._sparse_csr_prod(inp, [0], True)


@pytest.mark._sparse_csr_prod
def test__sparse_csr_prod_rejects_strided_input():
    inp = tu.make_input(torch.float32, (4, 5), ["-1", "1"])
    with pytest.raises((RuntimeError, NotImplementedError, TypeError)):
        flag_gems._sparse_csr_prod(inp, [0], True)


@pytest.mark._sparse_csr_prod
def test__sparse_csr_prod_rejects_strided_out_buffer():
    inp = tu.make_input(torch.float32, (4, 5), ["-1", "1"]).to_sparse_csr()
    out = torch.empty(4, 1, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._sparse_csr_prod(inp, [1], True, out=out)


@pytest.mark._sparse_csr_prod
def test__sparse_csr_prod_rejects_mismatched_out_dtype():
    inp = tu.make_input(torch.float32, (4, 5), ["-1", "1"]).to_sparse_csr()
    # Placed on the input device so the tested rejection is the dtype, not a
    # reference-device placement.
    structure = torch.ops.aten._sparse_csr_prod.dim_dtype(
        tu.to_reference(inp), [0], True
    )
    out = structure.clone().to(inp.device).to(torch.float16)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._sparse_csr_prod(inp, [0], True, out=out)


@pytest.mark._sparse_csr_prod
def test__sparse_csr_prod_rejects_out_buffer_with_other_nnz():
    # A constant input stores one entry per column for dim=0 and one per row for
    # dim=1, so the dim=1 result has a different nnz than the dim=0 out call
    # requires.
    inp = tu.make_input(torch.float32, (4, 5), ["1", "1"]).to_sparse_csr()
    ref_inp = tu.to_reference(inp)
    out = torch.ops.aten._sparse_csr_prod.dim_dtype(ref_inp, [1], True).to(inp.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._sparse_csr_prod(inp, [0], True, out=out)


# Default-only, like the rest of the .out family, because it needs the same
# alternative-support buffer construction those rows exercise.
_OTHER_SUPPORT_DIMS = tu.selected_cases([[1]], quick=[])


@pytest.mark._sparse_csr_prod
@pytest.mark.parametrize("dim", _OTHER_SUPPORT_DIMS)
def test__sparse_csr_prod_out_overwrites_other_valid_support(dim):
    # Row 1 is empty, so this dim=1 result stores two entries for three rows and
    # a valid buffer may hold them in different rows than the result uses.
    inp = torch.sparse_csr_tensor(
        torch.tensor([0, 2, 2, 5], dtype=torch.int64, device=flag_gems.device),
        torch.tensor([3, 4, 1, 2, 3], dtype=torch.int64, device=flag_gems.device),
        torch.tensor([1.0, 2.0, -1.0, 3.0, -2.0], device=flag_gems.device),
        size=(3, 5),
        device=flag_gems.device,
    )
    ref_inp = tu.to_reference(inp)

    ref_structure = torch.ops.aten._sparse_csr_prod.dim_dtype(ref_inp, dim, True)
    ref_out = torch.ops.aten._sparse_csr_prod.dim_dtype_out(
        ref_inp, dim, True, out=_stale_csr_like(ref_structure)
    )

    res_buf = _stale_csr_like(ref_structure).to(inp.device)
    res_out = flag_gems._sparse_csr_prod(inp, dim, True, out=res_buf)

    assert res_out is res_buf
    tu.assert_result_close(res_out, ref_out)
