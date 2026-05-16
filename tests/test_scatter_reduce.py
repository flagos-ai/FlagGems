import itertools

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


REDUCE_OPS = ["sum", "prod", "mean", "amax", "amin"]
OVERLOADS = ["functional", "inplace", "out"]
SHAPE_DIM_CASES = [
    ((4, 5), (6, 5), 0),
    ((4, 5), (4, 7), 1),
    ((3, 4, 5), (5, 4, 6), 0),
    ((3, 4, 5), (3, 6, 5), 1),
    ((3, 4, 5), (3, 4, 7), -1),
]


def _make_index(index_shape, inp_shape, dim, device):
    dim = dim % len(inp_shape)
    index = torch.empty(index_shape, dtype=torch.long, device=device)
    dim_values = torch.arange(index_shape[dim], dtype=torch.long, device=device)
    dim_values = dim_values % inp_shape[dim]
    outer_ranges = [
        range(size) for axis, size in enumerate(index_shape) if axis != dim
    ]

    for outer_idx in itertools.product(*outer_ranges):
        slices = []
        outer_pos = 0
        for axis in range(len(index_shape)):
            if axis == dim:
                slices.append(slice(None))
            else:
                slices.append(outer_idx[outer_pos])
                outer_pos += 1
        index[tuple(slices)] = dim_values
    return index


def _make_inputs(src_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    index_shape = tuple(
        min(src_extent, inp_extent)
        for src_extent, inp_extent in zip(src_shape, inp_shape)
    )
    index = _make_index(index_shape, inp_shape, dim, flag_gems.device)
    return inp, index, src


def _reference(inp, dim, index, src, reduce, include_self, overload, dtype):
    ref_inp = utils.to_reference(inp.clone(), upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    if overload == "functional":
        return torch.scatter_reduce(
            ref_inp,
            dim,
            ref_index,
            ref_src,
            reduce=reduce,
            include_self=include_self,
        ).to(dtype)
    if overload == "inplace":
        return ref_inp.scatter_reduce_(
            dim,
            ref_index,
            ref_src,
            reduce=reduce,
            include_self=include_self,
        ).to(dtype)

    ref_out = torch.empty_like(ref_inp)
    torch.scatter_reduce(
        ref_inp,
        dim,
        ref_index,
        ref_src,
        reduce=reduce,
        include_self=include_self,
        out=ref_out,
    )
    return ref_out.to(dtype)


def _gems_result(inp, dim, index, src, reduce, include_self, overload):
    res_inp = inp.clone()
    with flag_gems.use_gems():
        if overload == "functional":
            return torch.scatter_reduce(
                res_inp,
                dim,
                index,
                src,
                reduce=reduce,
                include_self=include_self,
            )
        if overload == "inplace":
            return res_inp.scatter_reduce_(
                dim,
                index,
                src,
                reduce=reduce,
                include_self=include_self,
            )

        out = torch.empty_like(res_inp)
        res = torch.scatter_reduce(
            res_inp,
            dim,
            index,
            src,
            reduce=reduce,
            include_self=include_self,
            out=out,
        )
        assert res is out
        return out


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduce", REDUCE_OPS)
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("overload", OVERLOADS)
def test_scatter_reduce_overloads(dtype, reduce, include_self, overload):
    src_shape, inp_shape, dim = SHAPE_DIM_CASES[0]
    inp, index, src = _make_inputs(src_shape, inp_shape, dim, dtype)

    ref_out = _reference(inp, dim, index, src, reduce, include_self, overload, dtype)
    res_out = _gems_result(inp, dim, index, src, reduce, include_self, overload)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=max(1, index.numel()))


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("src_shape, inp_shape, dim", SHAPE_DIM_CASES)
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("reduce", ["sum", "amax"])
def test_scatter_reduce_shapes_and_dims(src_shape, inp_shape, dim, include_self, reduce):
    dtype = torch.float32
    inp, index, src = _make_inputs(src_shape, inp_shape, dim, dtype)

    ref_out = _reference(
        inp, dim, index, src, reduce, include_self, "functional", dtype
    )
    res_out = _gems_result(inp, dim, index, src, reduce, include_self, "functional")

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=max(1, index.numel()))
