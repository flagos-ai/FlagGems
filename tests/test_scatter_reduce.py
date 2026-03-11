import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    SCATTER_REDUCE_SHAPES = [((16, 8, 4), (8, 4, 4))]
else:
    SCATTER_REDUCE_SHAPES = [
        ((64, 16, 8), (32, 8, 4)),
        ((48, 24, 12), (24, 12, 6)),
    ]

SCATTER_REDUCE_DTYPES = utils.FLOAT_DTYPES + utils.INT_DTYPES
SCATTER_REDUCE_REDUCTIONS = ["sum", "prod", "mean", "amax", "amin"]


def _scatter_reduce_upcast(reduce, dtype):
    return dtype in utils.FLOAT_DTYPES and reduce in {"sum", "mean"}


def _assert_scatter_reduce_result(res_out, ref_out, dtype, *, equal_nan=False):
    if dtype in utils.FLOAT_DTYPES:
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=equal_nan)
    else:
        utils.gems_assert_equal(res_out, ref_out, equal_nan=equal_nan)


def _make_scatter_reduce_tensors(inp_shape, src_shape, dim, dtype):
    utils.init_seed(0)
    if dtype in utils.INT_DTYPES:
        inp = torch.randint(-8, 8, inp_shape, device=flag_gems.device).to(dtype)
        src = torch.randint(-8, 8, src_shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
        src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    index = torch.randint(
        0, inp_shape[dim], src_shape, dtype=torch.long, device=flag_gems.device
    )
    return inp, index, src


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("inp_shape, src_shape", SCATTER_REDUCE_SHAPES)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("reduce", SCATTER_REDUCE_REDUCTIONS)
@pytest.mark.parametrize("dtype", SCATTER_REDUCE_DTYPES)
def test_scatter_reduce(inp_shape, src_shape, dim, include_self, reduce, dtype):
    inp, index, src = _make_scatter_reduce_tensors(inp_shape, src_shape, dim, dtype)
    ref_inp = utils.to_reference(inp, upcast=_scatter_reduce_upcast(reduce, dtype))
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=_scatter_reduce_upcast(reduce, dtype))

    ref_out = torch.scatter_reduce(
        ref_inp, dim, ref_index, ref_src, reduce, include_self=include_self
    )
    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(
            inp, dim, index, src, reduce, include_self=include_self
        )

    _assert_scatter_reduce_result(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("inp_shape, src_shape", SCATTER_REDUCE_SHAPES[:1])
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("reduce", SCATTER_REDUCE_REDUCTIONS)
@pytest.mark.parametrize("dtype", SCATTER_REDUCE_DTYPES)
def test_scatter_reduce_(inp_shape, src_shape, dim, include_self, reduce, dtype):
    inp, index, src = _make_scatter_reduce_tensors(inp_shape, src_shape, dim, dtype)
    ref_inp = utils.to_reference(inp, upcast=_scatter_reduce_upcast(reduce, dtype))
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=_scatter_reduce_upcast(reduce, dtype))

    ref_out = ref_inp.scatter_reduce_(
        dim, ref_index, ref_src, reduce, include_self=include_self
    )
    with flag_gems.use_gems():
        res_out = inp.scatter_reduce_(
            dim, index, src, reduce, include_self=include_self
        )

    _assert_scatter_reduce_result(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("inp_shape, src_shape", SCATTER_REDUCE_SHAPES[:1])
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("reduce", SCATTER_REDUCE_REDUCTIONS)
@pytest.mark.parametrize("dtype", SCATTER_REDUCE_DTYPES)
def test_scatter_reduce_out(inp_shape, src_shape, dim, include_self, reduce, dtype):
    inp, index, src = _make_scatter_reduce_tensors(inp_shape, src_shape, dim, dtype)
    ref_inp = utils.to_reference(inp, upcast=_scatter_reduce_upcast(reduce, dtype))
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=_scatter_reduce_upcast(reduce, dtype))
    ref_out_tensor = torch.empty_like(ref_inp)
    out_tensor = torch.empty_like(inp)

    ref_out = torch.scatter_reduce(
        ref_inp,
        dim,
        ref_index,
        ref_src,
        reduce,
        include_self=include_self,
        out=ref_out_tensor,
    )
    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(
            inp,
            dim,
            index,
            src,
            reduce,
            include_self=include_self,
            out=out_tensor,
        )

    _assert_scatter_reduce_result(res_out, ref_out, dtype)
    _assert_scatter_reduce_result(out_tensor, ref_out_tensor, dtype)


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("dim", [-1, -2, -3])
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("reduce", SCATTER_REDUCE_REDUCTIONS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_scatter_reduce_negative_dim(dim, include_self, reduce, dtype):
    inp, index, src = _make_scatter_reduce_tensors((16, 8, 4), (8, 4, 4), dim, dtype)
    ref_inp = utils.to_reference(inp, upcast=_scatter_reduce_upcast(reduce, dtype))
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=_scatter_reduce_upcast(reduce, dtype))

    ref_out = torch.scatter_reduce(
        ref_inp, dim, ref_index, ref_src, reduce, include_self=include_self
    )
    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(
            inp, dim, index, src, reduce, include_self=include_self
        )

    _assert_scatter_reduce_result(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("dim", [3, -4])
def test_scatter_reduce_invalid_dim(dim):
    inp, index, src = _make_scatter_reduce_tensors(
        (16, 8, 4), (8, 4, 4), 2, torch.float32
    )
    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src)

    with pytest.raises(IndexError, match="Dimension out of range"):
        torch.scatter_reduce(ref_inp, dim, ref_index, ref_src, "sum")
    with flag_gems.use_gems():
        with pytest.raises(IndexError, match="Dimension out of range"):
            torch.scatter_reduce(inp, dim, index, src, "sum")


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("reduce", SCATTER_REDUCE_REDUCTIONS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_scatter_reduce_scalar(dim, include_self, reduce, dtype):
    if dtype == torch.int32:
        inp = torch.tensor(3, dtype=dtype, device=flag_gems.device)
        src = torch.tensor(2, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.tensor(3.0, dtype=dtype, device=flag_gems.device)
        src = torch.tensor(2.0, dtype=dtype, device=flag_gems.device)
    index = torch.tensor(0, dtype=torch.long, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, upcast=_scatter_reduce_upcast(reduce, dtype))
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=_scatter_reduce_upcast(reduce, dtype))

    ref_out = torch.scatter_reduce(
        ref_inp, dim, ref_index, ref_src, reduce, include_self=include_self
    )
    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(
            inp, dim, index, src, reduce, include_self=include_self
        )

    _assert_scatter_reduce_result(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("reduce", ["sum", "amax"])
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_scatter_reduce_noncontiguous(reduce, include_self, dtype):
    dim = 2
    if dtype in utils.INT_DTYPES:
        inp = torch.randint(-8, 8, (6, 8, 10), device=flag_gems.device).to(
            dtype
        ).transpose(0, 1)
        src = torch.randint(-8, 8, (6, 8, 5), device=flag_gems.device).to(
            dtype
        ).transpose(0, 1)
    else:
        inp = torch.randn((6, 8, 10), dtype=dtype, device=flag_gems.device).transpose(
            0, 1
        )
        src = torch.randn((6, 8, 5), dtype=dtype, device=flag_gems.device).transpose(
            0, 1
        )
    index = torch.randint(
        0, inp.shape[dim], src.shape, dtype=torch.long, device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp, upcast=_scatter_reduce_upcast(reduce, dtype))
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=_scatter_reduce_upcast(reduce, dtype))
    ref_out_tensor = torch.empty_strided(
        ref_inp.shape, ref_inp.stride(), dtype=ref_inp.dtype, device=ref_inp.device
    )
    out_tensor = torch.empty_strided(
        inp.shape, inp.stride(), dtype=inp.dtype, device=inp.device
    )

    ref_out = torch.scatter_reduce(
        ref_inp,
        dim,
        ref_index,
        ref_src,
        reduce,
        include_self=include_self,
        out=ref_out_tensor,
    )
    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(
            inp,
            dim,
            index,
            src,
            reduce,
            include_self=include_self,
            out=out_tensor,
        )

    _assert_scatter_reduce_result(res_out, ref_out, dtype)
    _assert_scatter_reduce_result(out_tensor, ref_out_tensor, dtype)


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("reduce", SCATTER_REDUCE_REDUCTIONS)
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_scatter_reduce_empty_src(reduce, include_self, dtype):
    dim = 1
    if dtype in utils.INT_DTYPES:
        inp = torch.randint(-8, 8, (4, 5, 3), device=flag_gems.device).to(dtype)
        src = torch.empty((4, 0, 3), dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randn((4, 5, 3), dtype=dtype, device=flag_gems.device)
        src = torch.empty((4, 0, 3), dtype=dtype, device=flag_gems.device)
    index = torch.empty((4, 0, 3), dtype=torch.long, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, upcast=_scatter_reduce_upcast(reduce, dtype))
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=_scatter_reduce_upcast(reduce, dtype))

    ref_out = torch.scatter_reduce(
        ref_inp, dim, ref_index, ref_src, reduce, include_self=include_self
    )
    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(
            inp, dim, index, src, reduce, include_self=include_self
        )

    _assert_scatter_reduce_result(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("reduce", SCATTER_REDUCE_REDUCTIONS)
@pytest.mark.parametrize("include_self", [True, False])
def test_scatter_reduce_special_values(reduce, include_self):
    dim = 1
    dtype = torch.float32
    inp = torch.tensor(
        [[1.0, float("nan"), 3.0, float("-inf")], [float("inf"), -2.0, 0.0, 4.0]],
        dtype=dtype,
        device=flag_gems.device,
    )
    src = torch.tensor(
        [[float("inf"), -5.0, float("nan")], [2.0, float("-inf"), 7.0]],
        dtype=dtype,
        device=flag_gems.device,
    )
    index = torch.tensor(
        [[0, 1, 1], [3, 0, 2]], dtype=torch.long, device=flag_gems.device
    )

    ref_inp = utils.to_reference(inp, upcast=_scatter_reduce_upcast(reduce, dtype))
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=_scatter_reduce_upcast(reduce, dtype))
    ref_out_tensor = torch.empty_like(ref_inp)
    out_tensor = torch.empty_like(inp)

    ref_out = torch.scatter_reduce(
        ref_inp,
        dim,
        ref_index,
        ref_src,
        reduce,
        include_self=include_self,
        out=ref_out_tensor,
    )
    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(
            inp,
            dim,
            index,
            src,
            reduce,
            include_self=include_self,
            out=out_tensor,
        )

    _assert_scatter_reduce_result(res_out, ref_out, dtype, equal_nan=True)
    _assert_scatter_reduce_result(out_tensor, ref_out_tensor, dtype, equal_nan=True)


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("reduce", SCATTER_REDUCE_REDUCTIONS)
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_scatter_reduce_empty_self_out(reduce, include_self, dtype):
    inp = torch.empty((0, 4), dtype=dtype, device=flag_gems.device)
    src = torch.empty((0, 2), dtype=dtype, device=flag_gems.device)
    index = torch.empty((0, 2), dtype=torch.long, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, upcast=_scatter_reduce_upcast(reduce, dtype))
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=_scatter_reduce_upcast(reduce, dtype))
    ref_out_tensor = torch.empty_like(ref_inp)
    out_tensor = torch.empty_like(inp)

    ref_out = torch.scatter_reduce(
        ref_inp,
        0,
        ref_index,
        ref_src,
        reduce,
        include_self=include_self,
        out=ref_out_tensor,
    )
    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(
            inp,
            0,
            index,
            src,
            reduce,
            include_self=include_self,
            out=out_tensor,
        )

    _assert_scatter_reduce_result(res_out, ref_out, dtype)
    _assert_scatter_reduce_result(out_tensor, ref_out_tensor, dtype)


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("reduce", SCATTER_REDUCE_REDUCTIONS)
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_scatter_reduce_out_alias_input(reduce, include_self, dtype):
    dim = 2
    inp, index, src = _make_scatter_reduce_tensors((16, 8, 4), (8, 4, 4), dim, dtype)
    ref_inp = utils.to_reference(inp, upcast=_scatter_reduce_upcast(reduce, dtype))
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=_scatter_reduce_upcast(reduce, dtype))

    ref_out = torch.scatter_reduce(
        ref_inp,
        dim,
        ref_index,
        ref_src,
        reduce,
        include_self=include_self,
        out=ref_inp,
    )
    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(
            inp,
            dim,
            index,
            src,
            reduce,
            include_self=include_self,
            out=inp,
        )

    _assert_scatter_reduce_result(res_out, ref_out, dtype)
    _assert_scatter_reduce_result(inp, ref_inp, dtype)


@pytest.mark.scatter_reduce
@pytest.mark.parametrize(
    "case",
    [
        "index_dtype",
        "invalid_reduce",
        "rank_mismatch",
        "shape_mismatch",
    ],
)
def test_scatter_reduce_invalid_inputs(case):
    inp = torch.randn((4, 5), dtype=torch.float32, device=flag_gems.device)
    src = torch.randn((4, 5), dtype=torch.float32, device=flag_gems.device)
    if case == "index_dtype":
        index = torch.zeros((4, 5), dtype=torch.int32, device=flag_gems.device)
        args = (inp, 1, index, src, "sum")
    elif case == "invalid_reduce":
        index = torch.zeros((4, 5), dtype=torch.long, device=flag_gems.device)
        args = (inp, 1, index, src, "foo")
    elif case == "rank_mismatch":
        index = torch.zeros((4,), dtype=torch.long, device=flag_gems.device)
        args = (inp, 1, index, src, "sum")
    else:
        index = torch.zeros((4, 6), dtype=torch.long, device=flag_gems.device)
        args = (inp, 1, index, src, "sum")

    ref_args = tuple(utils.to_reference(arg) if torch.is_tensor(arg) else arg for arg in args)
    with pytest.raises(RuntimeError):
        torch.scatter_reduce(*ref_args)
    with flag_gems.use_gems():
        with pytest.raises(RuntimeError):
            torch.scatter_reduce(*args)
