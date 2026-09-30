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
from . import test_utils as tu

# aten::stride is a host-side metadata query: it has no broadcast form and no
# autograd formula, so those spec dimensions do not apply. Coverage uses both
# callable overloads, layouts with non-unit or zero strides and non-zero storage
# offsets, empty and 0-dim tensors, and invalid dimensions.
# aten::stride.Dimname needs a named tensor, and every torch.ops.aten call
# written with names raises "NYI: Named tensors are currently unsupported in
# TorchScript", so that overload has no native reference.
_STRIDE_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + [torch.int16, torch.bool]
)

_EXPANDED_SHAPE = (4, 8, 16)

# Each row is cheap and reports a distinct stride pattern, so all rows stay in
# the quick subset too.
_LAYOUT_ROWS = [
    ((4, 6), "contiguous"),
    ((4, 6), "transposed"),
    ((4, 6), "step"),
    ((4, 6), "narrow"),
    ((4, 6), "sliced"),
    ((4, 6), "permuted"),
    ((1, 8, 16), "expanded"),
    ((256,), "as_strided"),
    ((2, 3, 4, 5), "channels_last"),
]

# Layouts only vary how storage is addressed, so one float, one integer and one
# fp8 type suffice; the full dtype set runs in the value grid.
_LAYOUT_DTYPES = [torch.float32, torch.int64, torch.float8_e4m3fn]

_EMPTY_SHAPES = [(0,), (0, 3), (2, 0, 3), (3, 0, 0)]

_SPARSE_CASES = [((4, 4), 8), ((8, 8, 8), 64), ((3, 5, 7), 17)]

_SPECIAL_SHAPES = [(5,), (1, 5)]

# Out-of-range dimensions: one dim past either end of the valid range.
_OOB_SHAPES = [(4, 6), (2, 3, 4)]


def _apply_layout(base, layout):
    if layout == "contiguous":
        return base
    if layout == "transposed":
        return base.transpose(0, -1)
    if layout == "step":
        return base[:, ::3]
    if layout == "narrow":
        return base.narrow(0, 1, 2)
    if layout == "sliced":
        return base[1:, ::2]
    if layout == "permuted":
        return base.permute(1, 0)
    if layout == "expanded":
        return base.expand(_EXPANDED_SHAPE)
    if layout == "as_strided":
        return torch.as_strided(base, (3, 4), (7, 2), 5)
    if layout == "channels_last":
        return base.to(memory_format=torch.channels_last)
    raise AssertionError("unknown layout: " + str(layout))


def _dim_cases_for(shapes):
    cases = []
    for shape in shapes:
        rank = len(shape)
        if rank == 0:
            continue  # a 0-dim tensor has no dimension to query
        dims = {0, -1, rank - 1, -rank}
        if rank > 2:
            dims |= {1, -2}
        cases.extend((shape, dim) for dim in sorted(dims))
    return cases


# `dim` parameter coverage: zero, positive, negative and both dimension bounds.
_DIM_CASES = tu.selected_cases(
    _dim_cases_for(tu.REQUIRED_SHAPES), quick=_dim_cases_for(tu.QUICK_SHAPES)
)


def _assert_stride_list(res_out, ref_out):
    # schema: stride(Tensor self) -> int[] (a sequence of Python ints).
    assert isinstance(res_out, (list, tuple)), type(res_out)
    assert all(type(item) is int for item in res_out), res_out
    assert list(res_out) == list(ref_out)


def _assert_stride_int(res_out, ref_out):
    # schema: stride.int(Tensor self, int dim) -> int
    assert type(res_out) is int, type(res_out)
    assert res_out == ref_out


def _layout_facts(tensor):
    # The query must not disturb the operand it reads.
    return (
        tuple(tensor.size()),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.data_ptr(),
    )


def _make_coo(shape, nnz, dtype):
    # Seeded CPU indices; repeats keep the tensor uncoalesced.
    generator = torch.Generator("cpu").manual_seed(0)
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=generator)
            for dim in shape
        ]
    )
    values = tu.make_input(dtype, (nnz,), ["-1", "1"])
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


@pytest.mark.stride
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _STRIDE_DTYPES)
def test_stride_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    before = _layout_facts(inp)

    ref_out = torch.ops.aten.stride(ref_inp)
    res_out = flag_gems.stride(inp)

    _assert_stride_list(res_out, ref_out)
    assert _layout_facts(inp) == before


@pytest.mark.stride
@pytest.mark.parametrize("shape,dim", _DIM_CASES)
@pytest.mark.parametrize("dtype", _STRIDE_DTYPES)
def test_stride_dim_over_shapes(shape, dim, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    before = _layout_facts(inp)

    ref_out = torch.ops.aten.stride(ref_inp, dim)
    res_out = flag_gems.stride(inp, dim)

    _assert_stride_int(res_out, ref_out)
    assert _layout_facts(inp) == before


@pytest.mark.stride
@pytest.mark.parametrize("storage_shape,layout", _LAYOUT_ROWS)
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test_stride_layouts(storage_shape, layout, dtype):
    base = tu.make_input(dtype, storage_shape, ["-1", "1"])
    inp = _apply_layout(base, layout)
    ref_inp = _apply_layout(tu.to_reference(base), layout)
    before = _layout_facts(inp)

    ref_out = torch.ops.aten.stride(ref_inp)
    res_out = flag_gems.stride(inp)

    _assert_stride_list(res_out, ref_out)
    assert _layout_facts(inp) == before


@pytest.mark.stride
@pytest.mark.parametrize("storage_shape,layout", _LAYOUT_ROWS)
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test_stride_layout_dim(storage_shape, layout, dim, dtype):
    base = tu.make_input(dtype, storage_shape, ["-1", "1"])
    inp = _apply_layout(base, layout)
    ref_inp = _apply_layout(tu.to_reference(base), layout)

    ref_out = torch.ops.aten.stride(ref_inp, dim)
    res_out = flag_gems.stride(inp, dim)

    _assert_stride_int(res_out, ref_out)


@pytest.mark.stride
@pytest.mark.parametrize("shape", _EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test_stride_empty_shapes(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.stride(ref_inp)
    res_out = flag_gems.stride(inp)
    _assert_stride_list(res_out, ref_out)

    ref_dim = torch.ops.aten.stride(ref_inp, 0)
    res_dim = flag_gems.stride(inp, 0)
    _assert_stride_int(res_dim, ref_dim)


@pytest.mark.stride
@pytest.mark.parametrize("shape,nnz", _SPARSE_CASES)
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test_stride_sparse_coo(shape, nnz, dtype):
    inp = _make_coo(shape, nnz, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.stride(ref_inp)
    res_out = flag_gems.stride(inp)
    _assert_stride_list(res_out, ref_out)

    # COO tensors report a zero stride for every sparse dimension.
    ref_dim = torch.ops.aten.stride(ref_inp, 0)
    res_dim = flag_gems.stride(inp, 0)
    _assert_stride_int(res_dim, ref_dim)


@pytest.mark.stride
@pytest.mark.parametrize("shape", _SPECIAL_SHAPES)
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_STRIDE_DTYPES))
)
def test_stride_special_values(dtype, scenario, shape):
    inp = tu.make_special_input(dtype, scenario).reshape(shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.stride(ref_inp)
    res_out = flag_gems.stride(inp)
    _assert_stride_list(res_out, ref_out)

    ref_dim = torch.ops.aten.stride(ref_inp, 0)
    res_dim = flag_gems.stride(inp, 0)
    _assert_stride_int(res_dim, ref_dim)


@pytest.mark.stride
@pytest.mark.parametrize("shape", _OOB_SHAPES)
def test_stride_rejects_out_of_range_dim(shape):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])

    with pytest.raises((IndexError, RuntimeError, ValueError, TypeError)):
        flag_gems.stride(inp, len(shape))
    with pytest.raises((IndexError, RuntimeError, ValueError, TypeError)):
        flag_gems.stride(inp, -len(shape) - 1)


@pytest.mark.stride
def test_stride_rejects_dim_on_scalar_tensor():
    inp = tu.make_input(torch.float32, (), ["-1", "1"])

    with pytest.raises((IndexError, RuntimeError, ValueError, TypeError)):
        flag_gems.stride(inp, 0)


@pytest.mark.stride
def test_stride_rejects_non_tensor():
    with pytest.raises((TypeError, ValueError, RuntimeError, NotImplementedError)):
        flag_gems.stride(3.14)


@pytest.mark.stride
def test_stride_rejects_non_integer_dim():
    inp = tu.make_input(torch.float32, (4, 6), ["-1", "1"])

    with pytest.raises((TypeError, ValueError, RuntimeError, NotImplementedError)):
        flag_gems.stride(inp, 1.0)


@pytest.mark.stride
def test_stride_rejects_sparse_csr():
    inp = torch.sparse_csr_tensor(
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
        torch.randn(2, device=flag_gems.device),
        (2, 2),
        device=flag_gems.device,
    )

    # Native raises RuntimeError("Sparse CSR tensors do not have strides").
    with pytest.raises((RuntimeError, NotImplementedError)):
        flag_gems.stride(inp)
