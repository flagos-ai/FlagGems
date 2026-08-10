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

import importlib

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

FLOAT_DTYPES = utils.FLOAT_DTYPES
REDUCE_MODES = ("sum", "prod", "mean", "amax", "amin")
# None exercises the schema default by omitting the include_self keyword.
INCLUDE_SELF_CASES = (None, False)

SHAPE_DIM_CASES = (
    pytest.param(utils.UT_SHAPES_1D[1], 0, id="1d_dim0"),
    pytest.param(utils.UT_SHAPES_2D[0], -1, id="2d_dim_last"),
    pytest.param((4, 8, 4), 1, id="3d_dim1"),
    pytest.param((2, 4, 3, 4), 2, id="4d_dim2"),
    pytest.param((2, 4, 3, 4, 5), -2, id="5d_dim_neg2"),
)
HIGH_DIM_SHAPE_DIM_CASES = (
    pytest.param((2, 3, 2, 2, 2, 2), 0, id="6d_dim0"),
    pytest.param((2, 2, 2, 3, 2, 2), 3, id="6d_dim3"),
    pytest.param((2, 2, 2, 2, 2, 2, 2, 3), 7, id="8d_dim_last"),
    pytest.param((2, 2, 2, 3, 2, 2, 2, 2), -5, id="8d_dim_neg5"),
)
ACTIVE_PREFIX_CASES = (
    pytest.param(
        3,
        (2, 2, 2, 3, 2, 2, 2, 2),
        (1, 2, 1, 4, 1, 2, 1, 2),
        (3, 2, 2, 4, 2, 3, 2, 2),
        id="8d_active_prefix",
    ),
)

if not cfg.QUICK_MODE:
    # Full accuracy runs retain the quick structural cases and add roughly
    # 1/4-million-element shapes that exercise long index ranges and reduction axes.
    SHAPE_DIM_CASES += (
        pytest.param((1 << 18,), 0, id="large_1d_dim0"),
        pytest.param((256, 1000), -1, id="large_2d_dim_last"),
        pytest.param((5, 320, 128), 1, id="large_3d_dim1"),
        pytest.param((4, 64, 64, 16), 2, id="large_4d_dim2"),
        pytest.param((2, 16, 16, 16, 32), -2, id="large_5d_dim_neg2"),
    )
    HIGH_DIM_SHAPE_DIM_CASES += (
        pytest.param((8, 8, 8, 8, 8, 1), 0, id="large_6d_dim0"),
        pytest.param((1, 8, 8, 16, 16, 16), 3, id="large_6d_dim3"),
        pytest.param((2, 2, 2, 4, 4, 4, 4, 128), 7, id="large_8d_dim_last"),
        pytest.param((2, 2, 4, 16, 2, 8, 8, 8), -5, id="large_8d_dim_neg5"),
    )
    ACTIVE_PREFIX_CASES += (
        pytest.param(
            3,
            (2, 2, 4, 16, 4, 4, 8, 8),
            (1, 2, 4, 32, 4, 4, 4, 8),
            (2, 2, 4, 32, 4, 4, 8, 8),
            id="large_8d_active_prefix",
        ),
    )


def _make_test_data(shape, dim, dtype, reduce):
    """Create valid tensors for the aten scatter_reduce overload family."""
    torch.manual_seed(0)
    normalized_dim = dim % len(shape)
    src_shape = list(shape)
    src_shape[normalized_dim] *= 2

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    if reduce == "prod":
        inp.mul_(0.1).add_(1.0)
        src.mul_(0.1).add_(1.0)
    index = torch.randint(
        0,
        shape[normalized_dim],
        src_shape,
        dtype=torch.long,
        device=flag_gems.device,
    )
    return inp, index, src


def _make_high_dim_active_prefix_data(
    dtype, reduce, dim, self_shape, index_shape, src_shape
):
    """Create unequal, noncontiguous 8-D tensors with a valid active prefix."""
    torch.manual_seed(0)
    reverse_dims = tuple(reversed(range(len(self_shape))))
    inp = torch.randn(
        tuple(reversed(self_shape)),
        dtype=dtype,
        device=flag_gems.device,
    ).permute(reverse_dims)
    src = torch.randn(
        tuple(reversed(src_shape)),
        dtype=dtype,
        device=flag_gems.device,
    ).permute(reverse_dims)
    index = torch.randint(
        0,
        self_shape[dim],
        tuple(reversed(index_shape)),
        dtype=torch.long,
        device=flag_gems.device,
    ).permute(reverse_dims)
    if reduce == "prod":
        inp.mul_(0.1).add_(1.0)
        src.mul_(0.1).add_(1.0)

    assert not inp.is_contiguous()
    assert not index.is_contiguous()
    assert not src.is_contiguous()
    return inp, index, src


def _include_self_kwargs(include_self):
    """Translate the None test sentinel into an omitted ATen keyword."""
    return {} if include_self is None else {"include_self": include_self}


def _reference_inputs(inp, index, src):
    """Move reference tensors to --ref cpu when requested and upcast values."""
    return (
        utils.to_reference(inp, upcast=True),
        utils.to_reference(index),
        utils.to_reference(src, upcast=True),
    )


def _assert_scatter_reduce_close(result, reference, dtype, dim, src, reduce):
    """Compare scatter reductions with accumulation-aware tolerances."""
    normalized_dim = dim % src.ndim
    reduce_dim = src.shape[normalized_dim] if reduce in ("sum", "prod", "mean") else 1
    utils.gems_assert_close(result, reference, dtype, reduce_dim=reduce_dim)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("shape,dim", SHAPE_DIM_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduce", REDUCE_MODES)
@pytest.mark.parametrize("include_self", INCLUDE_SELF_CASES)
def test_scatter_reduce(shape, dim, dtype, reduce, include_self):
    """Validate ordinary accuracy for aten::scatter_reduce.two."""
    inp, index, src = _make_test_data(shape, dim, dtype, reduce)
    ref_inp, ref_index, ref_src = _reference_inputs(inp, index, src)
    kwargs = _include_self_kwargs(include_self)

    ref_out = torch.ops.aten.scatter_reduce.two(
        ref_inp, dim, ref_index, ref_src, reduce, **kwargs
    )
    with flag_gems.use_gems():
        result = torch.ops.aten.scatter_reduce.two(
            inp, dim, index, src, reduce, **kwargs
        )

    assert result.data_ptr() != inp.data_ptr()
    _assert_scatter_reduce_close(result, ref_out, dtype, dim, src, reduce)


@pytest.mark.scatter_reduce_two_
@pytest.mark.parametrize("shape,dim", SHAPE_DIM_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduce", REDUCE_MODES)
@pytest.mark.parametrize("include_self", INCLUDE_SELF_CASES)
def test_scatter_reduce_(shape, dim, dtype, reduce, include_self):
    """Validate ordinary accuracy and aliasing for aten::scatter_reduce_.two."""
    inp, index, src = _make_test_data(shape, dim, dtype, reduce)
    ref_inp, ref_index, ref_src = _reference_inputs(inp.clone(), index, src)
    kwargs = _include_self_kwargs(include_self)

    ref_out = torch.ops.aten.scatter_reduce_.two(
        ref_inp, dim, ref_index, ref_src, reduce, **kwargs
    )
    with flag_gems.use_gems():
        result = torch.ops.aten.scatter_reduce_.two(
            inp, dim, index, src, reduce, **kwargs
        )

    assert result.data_ptr() == inp.data_ptr()
    _assert_scatter_reduce_close(result, ref_out, dtype, dim, src, reduce)


@pytest.mark.scatter_reduce_two_out
@pytest.mark.parametrize("shape,dim", SHAPE_DIM_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduce", REDUCE_MODES)
@pytest.mark.parametrize("include_self", INCLUDE_SELF_CASES)
def test_scatter_reduce_out(shape, dim, dtype, reduce, include_self):
    """Validate ordinary accuracy and storage for aten::scatter_reduce.two_out."""
    inp, index, src = _make_test_data(shape, dim, dtype, reduce)
    ref_inp, ref_index, ref_src = _reference_inputs(inp, index, src)
    result_out = torch.empty_like(inp)
    ref_out = torch.empty_like(ref_inp)
    kwargs = _include_self_kwargs(include_self)

    ref_result = torch.ops.aten.scatter_reduce.two_out(
        ref_inp, dim, ref_index, ref_src, reduce, out=ref_out, **kwargs
    )
    with flag_gems.use_gems():
        result = torch.ops.aten.scatter_reduce.two_out(
            inp, dim, index, src, reduce, out=result_out, **kwargs
        )

    assert result.data_ptr() == result_out.data_ptr()
    assert ref_result.data_ptr() == ref_out.data_ptr()
    _assert_scatter_reduce_close(result, ref_result, dtype, dim, src, reduce)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("shape,dim", HIGH_DIM_SHAPE_DIM_CASES)
@pytest.mark.parametrize("reduce", REDUCE_MODES)
@pytest.mark.parametrize("include_self", INCLUDE_SELF_CASES)
def test_scatter_reduce_high_dim(shape, dim, reduce, include_self):
    """Validate the functional overload for 6-D and 8-D tensors."""
    dtype = torch.float32
    inp, index, src = _make_test_data(shape, dim, dtype, reduce)
    ref_inp, ref_index, ref_src = _reference_inputs(inp, index, src)
    kwargs = _include_self_kwargs(include_self)

    ref_out = torch.ops.aten.scatter_reduce.two(
        ref_inp, dim, ref_index, ref_src, reduce, **kwargs
    )
    with flag_gems.use_gems():
        result = torch.ops.aten.scatter_reduce.two(
            inp, dim, index, src, reduce, **kwargs
        )

    assert result.data_ptr() != inp.data_ptr()
    _assert_scatter_reduce_close(result, ref_out, dtype, dim, src, reduce)


@pytest.mark.scatter_reduce_two_
@pytest.mark.parametrize("shape,dim", HIGH_DIM_SHAPE_DIM_CASES)
@pytest.mark.parametrize("reduce", REDUCE_MODES)
@pytest.mark.parametrize("include_self", INCLUDE_SELF_CASES)
def test_scatter_reduce__high_dim(shape, dim, reduce, include_self):
    """Validate the in-place overload for 6-D and 8-D tensors."""
    dtype = torch.float32
    inp, index, src = _make_test_data(shape, dim, dtype, reduce)
    ref_inp, ref_index, ref_src = _reference_inputs(inp.clone(), index, src)
    kwargs = _include_self_kwargs(include_self)

    ref_out = torch.ops.aten.scatter_reduce_.two(
        ref_inp, dim, ref_index, ref_src, reduce, **kwargs
    )
    with flag_gems.use_gems():
        result = torch.ops.aten.scatter_reduce_.two(
            inp, dim, index, src, reduce, **kwargs
        )

    assert result.data_ptr() == inp.data_ptr()
    _assert_scatter_reduce_close(result, ref_out, dtype, dim, src, reduce)


@pytest.mark.scatter_reduce_two_out
@pytest.mark.parametrize("shape,dim", HIGH_DIM_SHAPE_DIM_CASES)
@pytest.mark.parametrize("reduce", REDUCE_MODES)
@pytest.mark.parametrize("include_self", INCLUDE_SELF_CASES)
def test_scatter_reduce_out_high_dim(shape, dim, reduce, include_self):
    """Validate the out overload for 6-D and 8-D tensors."""
    dtype = torch.float32
    inp, index, src = _make_test_data(shape, dim, dtype, reduce)
    ref_inp, ref_index, ref_src = _reference_inputs(inp, index, src)
    result_out = torch.empty_like(inp)
    ref_out = torch.empty_like(ref_inp)
    kwargs = _include_self_kwargs(include_self)

    ref_result = torch.ops.aten.scatter_reduce.two_out(
        ref_inp, dim, ref_index, ref_src, reduce, out=ref_out, **kwargs
    )
    with flag_gems.use_gems():
        result = torch.ops.aten.scatter_reduce.two_out(
            inp, dim, index, src, reduce, out=result_out, **kwargs
        )

    assert result.data_ptr() == result_out.data_ptr()
    assert ref_result.data_ptr() == ref_out.data_ptr()
    _assert_scatter_reduce_close(result, ref_result, dtype, dim, src, reduce)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("dim,self_shape,index_shape,src_shape", ACTIVE_PREFIX_CASES)
@pytest.mark.parametrize("reduce", REDUCE_MODES)
@pytest.mark.parametrize("include_self", INCLUDE_SELF_CASES)
def test_scatter_reduce_high_dim_active_prefix(
    dim, self_shape, index_shape, src_shape, reduce, include_self
):
    """Validate functional 8-D canonicalization with an active prefix."""
    dtype = torch.float32
    inp, index, src = _make_high_dim_active_prefix_data(
        dtype, reduce, dim, self_shape, index_shape, src_shape
    )
    ref_inp, ref_index, ref_src = _reference_inputs(inp, index, src)
    kwargs = _include_self_kwargs(include_self)

    ref_out = torch.ops.aten.scatter_reduce.two(
        ref_inp,
        dim,
        ref_index,
        ref_src,
        reduce,
        **kwargs,
    )
    with flag_gems.use_gems():
        result = torch.ops.aten.scatter_reduce.two(
            inp,
            dim,
            index,
            src,
            reduce,
            **kwargs,
        )

    assert result.data_ptr() != inp.data_ptr()
    _assert_scatter_reduce_close(result, ref_out, dtype, dim, src, reduce)


@pytest.mark.scatter_reduce_two_
@pytest.mark.parametrize("dim,self_shape,index_shape,src_shape", ACTIVE_PREFIX_CASES)
@pytest.mark.parametrize("reduce", REDUCE_MODES)
@pytest.mark.parametrize("include_self", INCLUDE_SELF_CASES)
def test_scatter_reduce__high_dim_active_prefix(
    dim, self_shape, index_shape, src_shape, reduce, include_self
):
    """Validate in-place 8-D canonicalization with an active prefix."""
    dtype = torch.float32
    inp, index, src = _make_high_dim_active_prefix_data(
        dtype, reduce, dim, self_shape, index_shape, src_shape
    )
    ref_inp, ref_index, ref_src = _reference_inputs(inp.clone(), index, src)
    kwargs = _include_self_kwargs(include_self)

    ref_out = torch.ops.aten.scatter_reduce_.two(
        ref_inp,
        dim,
        ref_index,
        ref_src,
        reduce,
        **kwargs,
    )
    with flag_gems.use_gems():
        result = torch.ops.aten.scatter_reduce_.two(
            inp,
            dim,
            index,
            src,
            reduce,
            **kwargs,
        )

    assert result.data_ptr() == inp.data_ptr()
    _assert_scatter_reduce_close(result, ref_out, dtype, dim, src, reduce)


@pytest.mark.scatter_reduce_two_out
@pytest.mark.parametrize("dim,self_shape,index_shape,src_shape", ACTIVE_PREFIX_CASES)
@pytest.mark.parametrize("reduce", REDUCE_MODES)
@pytest.mark.parametrize("include_self", INCLUDE_SELF_CASES)
def test_scatter_reduce_out_high_dim_active_prefix(
    dim, self_shape, index_shape, src_shape, reduce, include_self
):
    """Validate out 8-D canonicalization with an active prefix."""
    dtype = torch.float32
    inp, index, src = _make_high_dim_active_prefix_data(
        dtype, reduce, dim, self_shape, index_shape, src_shape
    )
    ref_inp, ref_index, ref_src = _reference_inputs(inp, index, src)
    result_out = torch.empty_like(inp)
    ref_out = torch.empty_like(ref_inp)
    kwargs = _include_self_kwargs(include_self)

    ref_result = torch.ops.aten.scatter_reduce.two_out(
        ref_inp,
        dim,
        ref_index,
        ref_src,
        reduce,
        out=ref_out,
        **kwargs,
    )
    with flag_gems.use_gems():
        result = torch.ops.aten.scatter_reduce.two_out(
            inp,
            dim,
            index,
            src,
            reduce,
            out=result_out,
            **kwargs,
        )

    assert result.data_ptr() == result_out.data_ptr()
    assert ref_result.data_ptr() == ref_out.data_ptr()
    _assert_scatter_reduce_close(result, ref_result, dtype, dim, src, reduce)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("reduce", ("amax", "amin"))
def test_scatter_reduce_5d_canonical_gate(monkeypatch, reduce):
    """Exercise the large-5D extrema gate without allocating a benchmark shape."""
    scatter_reduce_module = importlib.import_module("flag_gems.ops.scatter_reduce")
    monkeypatch.setattr(scatter_reduce_module, "_CANONICALIZE_5D_MIN_ELEMENTS", 0)

    shape, dim, dtype = (2, 2, 3, 2, 2), 2, torch.float32
    inp, index, src = _make_test_data(shape, dim, dtype, reduce)
    ref_inp, ref_index, ref_src = _reference_inputs(inp, index, src)

    ref_out = torch.ops.aten.scatter_reduce.two(
        ref_inp, dim, ref_index, ref_src, reduce
    )
    with flag_gems.use_gems():
        result = torch.ops.aten.scatter_reduce.two(inp, dim, index, src, reduce)

    _assert_scatter_reduce_close(result, ref_out, dtype, dim, src, reduce)


def _make_empty_test_data(dtype=torch.float32):
    """Create the empty index/source special case for all three overloads."""
    inp = torch.randn(8, dtype=dtype, device=flag_gems.device)
    index = torch.empty(0, dtype=torch.long, device=flag_gems.device)
    src = torch.empty(0, dtype=dtype, device=flag_gems.device)
    return inp, index, src


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("include_self", (True, False))
@pytest.mark.parametrize("reduce", REDUCE_MODES)
def test_scatter_reduce_empty(include_self, reduce):
    """Validate aten::scatter_reduce.two for an empty index and source."""
    dtype = torch.float32
    inp, index, src = _make_empty_test_data(dtype)
    ref_inp, ref_index, ref_src = _reference_inputs(inp, index, src)
    ref_out = torch.ops.aten.scatter_reduce.two(
        ref_inp, 0, ref_index, ref_src, reduce, include_self=include_self
    )

    with flag_gems.use_gems():
        result = torch.ops.aten.scatter_reduce.two(
            inp, 0, index, src, reduce, include_self=include_self
        )

    assert result.data_ptr() != inp.data_ptr()
    utils.gems_assert_close(result, ref_out, dtype)


@pytest.mark.scatter_reduce_two_
@pytest.mark.parametrize("include_self", (True, False))
@pytest.mark.parametrize("reduce", REDUCE_MODES)
def test_scatter_reduce__empty(include_self, reduce):
    """Validate aten::scatter_reduce_.two for an empty index and source."""
    dtype = torch.float32
    inp, index, src = _make_empty_test_data(dtype)
    ref_inp, ref_index, ref_src = _reference_inputs(inp.clone(), index, src)
    ref_out = torch.ops.aten.scatter_reduce_.two(
        ref_inp, 0, ref_index, ref_src, reduce, include_self=include_self
    )

    with flag_gems.use_gems():
        result = torch.ops.aten.scatter_reduce_.two(
            inp, 0, index, src, reduce, include_self=include_self
        )

    assert result.data_ptr() == inp.data_ptr()
    utils.gems_assert_close(result, ref_out, dtype)


@pytest.mark.scatter_reduce_two_out
@pytest.mark.parametrize("include_self", (True, False))
@pytest.mark.parametrize("reduce", REDUCE_MODES)
def test_scatter_reduce_out_empty(include_self, reduce):
    """Validate aten::scatter_reduce.two_out for an empty index and source."""
    dtype = torch.float32
    inp, index, src = _make_empty_test_data(dtype)
    ref_inp, ref_index, ref_src = _reference_inputs(inp, index, src)
    result_out = torch.empty_like(inp)
    ref_out = torch.empty_like(ref_inp)
    ref_result = torch.ops.aten.scatter_reduce.two_out(
        ref_inp,
        0,
        ref_index,
        ref_src,
        reduce,
        include_self=include_self,
        out=ref_out,
    )

    with flag_gems.use_gems():
        result = torch.ops.aten.scatter_reduce.two_out(
            inp,
            0,
            index,
            src,
            reduce,
            include_self=include_self,
            out=result_out,
        )

    assert result.data_ptr() == result_out.data_ptr()
    utils.gems_assert_close(result, ref_result, dtype)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("reduce", REDUCE_MODES)
def test_scatter_reduce_noncontiguous(reduce):
    """Validate aten::scatter_reduce.two with noncontiguous tensors."""
    dtype = torch.float32
    inp = torch.randn(16, 8, dtype=dtype, device=flag_gems.device).transpose(0, 1)
    src = torch.randn(32, 8, dtype=dtype, device=flag_gems.device).transpose(0, 1)
    index = torch.randint(
        0, 16, (32, 8), dtype=torch.long, device=flag_gems.device
    ).transpose(0, 1)
    assert not inp.is_contiguous()
    assert not src.is_contiguous()
    assert not index.is_contiguous()
    ref_inp, ref_index, ref_src = _reference_inputs(inp, index, src)
    ref_out = torch.ops.aten.scatter_reduce.two(ref_inp, -1, ref_index, ref_src, reduce)

    with flag_gems.use_gems():
        result = torch.ops.aten.scatter_reduce.two(inp, -1, index, src, reduce)

    _assert_scatter_reduce_close(result, ref_out, dtype, -1, src, reduce)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("reduce", REDUCE_MODES)
def test_scatter_reduce_high_contention(reduce):
    """Validate aten::scatter_reduce.two when all source values share one index."""
    dtype = torch.float32
    inp = torch.ones(1, dtype=dtype, device=flag_gems.device)
    src = torch.randn(256, dtype=dtype, device=flag_gems.device)
    if reduce == "prod":
        src.mul_(0.01).add_(1.0)
    index = torch.zeros(256, dtype=torch.long, device=flag_gems.device)
    ref_inp, ref_index, ref_src = _reference_inputs(inp, index, src)
    ref_out = torch.ops.aten.scatter_reduce.two(ref_inp, 0, ref_index, ref_src, reduce)

    with flag_gems.use_gems():
        result = torch.ops.aten.scatter_reduce.two(inp, 0, index, src, reduce)

    _assert_scatter_reduce_close(result, ref_out, dtype, 0, src, reduce)


@pytest.mark.scatter_reduce_two
def test_scatter_reduce_nan():
    """Validate NaN propagation for aten::scatter_reduce.two with sum reduction."""
    dtype = torch.float32
    inp = torch.ones(4, dtype=dtype, device=flag_gems.device)
    src = torch.tensor([float("nan"), 2.0], dtype=dtype, device=flag_gems.device)
    index = torch.tensor([0, 1], dtype=torch.long, device=flag_gems.device)
    ref_inp, ref_index, ref_src = _reference_inputs(inp, index, src)
    ref_out = torch.ops.aten.scatter_reduce.two(ref_inp, 0, ref_index, ref_src, "sum")

    with flag_gems.use_gems():
        result = torch.ops.aten.scatter_reduce.two(inp, 0, index, src, "sum")

    utils.gems_assert_close(result, ref_out, dtype, equal_nan=True)


@pytest.mark.scatter_reduce_two
def test_scatter_reduce_prod_nan():
    """Validate NaN propagation when product reduction multiplies Inf by zero."""
    dtype = torch.float32
    inp = torch.ones(1, dtype=dtype, device=flag_gems.device)
    src = torch.tensor([float("inf"), 0.0], dtype=dtype, device=flag_gems.device)
    index = torch.zeros(2, dtype=torch.long, device=flag_gems.device)
    ref_inp, ref_index, ref_src = _reference_inputs(inp, index, src)
    ref_out = torch.ops.aten.scatter_reduce.two(ref_inp, 0, ref_index, ref_src, "prod")

    with flag_gems.use_gems():
        result = torch.ops.aten.scatter_reduce.two(inp, 0, index, src, "prod")

    utils.gems_assert_close(result, ref_out, dtype, equal_nan=True)


@pytest.mark.scatter_reduce_two
def test_scatter_reduce_invalid_reduce():
    """Validate the error contract of aten::scatter_reduce.two for an invalid reduction."""
    inp, index, src = _make_test_data((8,), 0, torch.float32, "sum")

    with flag_gems.use_gems():
        with pytest.raises(
            (AssertionError, RuntimeError), match="[Uu]nsupported|reduce"
        ):
            torch.ops.aten.scatter_reduce.two(inp, 0, index, src, "invalid")


@pytest.mark.scatter_reduce_two
def test_scatter_reduce_invalid_dim():
    """Validate the error contract of aten::scatter_reduce.two for an invalid dimension."""
    inp, index, src = _make_test_data((8,), 0, torch.float32, "sum")

    with flag_gems.use_gems():
        with pytest.raises(IndexError, match="Dimension out of range"):
            torch.ops.aten.scatter_reduce.two(inp, 1, index, src, "sum")


@pytest.mark.scatter_reduce_two_out
def test_scatter_reduce_out_dtype_mismatch():
    """Validate the error contract of aten::scatter_reduce.two_out for a wrong out dtype."""
    inp, index, src = _make_test_data((8,), 0, torch.float32, "sum")
    out = torch.empty_like(inp, dtype=torch.float16)

    with flag_gems.use_gems():
        with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
            torch.ops.aten.scatter_reduce.two_out(inp, 0, index, src, "sum", out=out)
