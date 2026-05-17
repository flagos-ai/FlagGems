import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    SOURCE_SHAPES = [(32, 8, 4)]
    INPUT_SHAPES = [(64, 16, 8)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    SOURCE_SHAPES = [(128, 16, 4), (256, 32, 8)]
    INPUT_SHAPES = [(512, 128, 32), (1024, 64, 16)]

random.seed(time.time() // 100)


PUBLIC_SCATTER_REDUCTIONS = ("sum", "prod", "mean", "amax", "amin")
PUBLIC_SCATTER_REDUCE_FLOAT_DTYPES = (
    [torch.float32] if cfg.QUICK_MODE else [torch.float16, torch.float32]
)
PUBLIC_SCATTER_REDUCE_SHAPE_CASES = (
    ((6, 4), (5, 4), 0),
    ((3, 6, 4), (3, 5, 4), 1),
    ((2, 3, 7), (2, 3, 5), -1),
)
PUBLIC_SCATTER_REDUCE_OVERLOADS = ("functional", "inplace", "out")


def _scatter_reduce_name(reduce):
    if reduce == "add":
        return "sum"
    if reduce == "multiply":
        return "prod"
    return reduce


def _reference_scatter_reduce(inp, dim, index, src, reduce):
    return torch.scatter_reduce(
        inp,
        dim,
        index,
        src,
        reduce=_scatter_reduce_name(reduce),
        include_self=True,
    )


def _reference_scatter_reduce_(inp, dim, index, src, reduce):
    return inp.scatter_reduce_(
        dim,
        index,
        src,
        reduce=_scatter_reduce_name(reduce),
        include_self=True,
    )


def _public_scatter_reduce_inputs():
    inp = torch.tensor(
        [[10.0, -4.0, 3.0], [2.0, 5.0, -6.0]],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    index = torch.tensor(
        [[0, 0, 1], [2, 1, 1]],
        dtype=torch.long,
        device=flag_gems.device,
    )
    src = torch.tensor(
        [[1.0, 2.0, -7.0], [3.0, -8.0, 4.0]],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    return inp, index, src


def _public_scatter_reduce_shape_inputs(inp_shape, src_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    dim = dim % len(inp_shape)
    dim_index = torch.arange(src_shape[dim], dtype=torch.long, device=flag_gems.device)
    dim_index = dim_index % inp_shape[dim]
    view_shape = [1] * len(src_shape)
    view_shape[dim] = src_shape[dim]
    index = dim_index.reshape(view_shape).expand(src_shape).clone()
    return inp, index, src


def _reference_public_scatter_reduce(
    inp, dim, index, src, reduce, include_self, overload, dtype
):
    ref_inp = utils.to_reference(inp.clone(), upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)

    if overload == "functional":
        return torch.scatter_reduce(
            ref_inp,
            dim,
            ref_index,
            ref_src,
            reduce,
            include_self=include_self,
        ).to(dtype)

    if overload == "inplace":
        return ref_inp.scatter_reduce_(
            dim,
            ref_index,
            ref_src,
            reduce,
            include_self=include_self,
        ).to(dtype)

    ref_out = torch.empty_like(ref_inp)
    torch.scatter_reduce(
        ref_inp,
        dim,
        ref_index,
        ref_src,
        reduce,
        include_self=include_self,
        out=ref_out,
    )
    return ref_out.to(dtype)


def _result_public_scatter_reduce(inp, dim, index, src, reduce, include_self, overload):
    if overload == "functional":
        with flag_gems.use_gems(include=["scatter_reduce"]):
            return torch.scatter_reduce(
                inp, dim, index, src, reduce, include_self=include_self
            )

    if overload == "inplace":
        with flag_gems.use_gems(include=["scatter_reduce_"]):
            return inp.clone().scatter_reduce_(
                dim, index, src, reduce, include_self=include_self
            )

    out = torch.empty_like(inp)
    with flag_gems.use_gems(include=["scatter_reduce"]):
        result = torch.scatter_reduce(
            inp, dim, index, src, reduce, include_self=include_self, out=out
        )
    assert result.data_ptr() == out.data_ptr()
    return out


def _assert_registered_keys(expected_keys):
    keys = set(flag_gems.all_registered_keys())
    missing = set(expected_keys) - keys
    assert not missing, f"missing registered keys: {sorted(missing)}"


@pytest.mark.scatter_reduce
def test_scatter_reduce_public_api_registered():
    with flag_gems.use_gems(include=["scatter_reduce", "scatter_reduce_"]):
        _assert_registered_keys(
            {
                "scatter_reduce.two",
                "scatter_reduce.two_out",
                "scatter_reduce_.two",
            }
        )


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("reduce", PUBLIC_SCATTER_REDUCTIONS)
@pytest.mark.parametrize("include_self", [True, False])
def test_scatter_reduce_public_api_variants(reduce, include_self):
    inp, index, src = _public_scatter_reduce_inputs()
    dim = 1

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src)
    ref_out = torch.scatter_reduce(
        ref_inp, dim, ref_index, ref_src, reduce, include_self=include_self
    )

    with flag_gems.use_gems(include=["scatter_reduce"]):
        _assert_registered_keys({"scatter_reduce.two", "scatter_reduce.two_out"})
        res_out = torch.scatter_reduce(
            inp, dim, index, src, reduce, include_self=include_self
        )
    utils.gems_assert_close(res_out, ref_out, torch.float32)

    ref_inplace = ref_inp.clone().scatter_reduce_(
        dim, ref_index, ref_src, reduce, include_self=include_self
    )
    with flag_gems.use_gems(include=["scatter_reduce_"]):
        _assert_registered_keys({"scatter_reduce_.two"})
        res_inplace = inp.clone().scatter_reduce_(
            dim, index, src, reduce, include_self=include_self
        )
    utils.gems_assert_close(res_inplace, ref_inplace, torch.float32)

    ref_out_tensor = torch.empty_like(ref_inp)
    torch.scatter_reduce(
        ref_inp,
        dim,
        ref_index,
        ref_src,
        reduce,
        include_self=include_self,
        out=ref_out_tensor,
    )
    res_out_tensor = torch.empty_like(inp)
    with flag_gems.use_gems(include=["scatter_reduce"]):
        _assert_registered_keys({"scatter_reduce.two", "scatter_reduce.two_out"})
        res_return = torch.scatter_reduce(
            inp,
            dim,
            index,
            src,
            reduce,
            include_self=include_self,
            out=res_out_tensor,
        )
    assert res_return.data_ptr() == res_out_tensor.data_ptr()
    utils.gems_assert_close(res_out_tensor, ref_out_tensor, torch.float32)


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("reduce", ["prod", "amax", "amin"])
@pytest.mark.parametrize("include_self", [True, False])
def test_scatter_reduce_nan_matches_pytorch(reduce, include_self):
    inp, index, src = _public_scatter_reduce_inputs()
    inp[0, 0] = float("nan")
    src[1, 2] = float("nan")
    dim = 1

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src)

    ref_out = torch.scatter_reduce(
        ref_inp, dim, ref_index, ref_src, reduce, include_self=include_self
    )
    with flag_gems.use_gems(include=["scatter_reduce"]):
        res_out = torch.scatter_reduce(
            inp, dim, index, src, reduce, include_self=include_self
        )
    utils.gems_assert_close(res_out, ref_out, torch.float32, equal_nan=True)

    ref_inplace = ref_inp.clone().scatter_reduce_(
        dim, ref_index, ref_src, reduce, include_self=include_self
    )
    with flag_gems.use_gems(include=["scatter_reduce_"]):
        res_inplace = inp.clone().scatter_reduce_(
            dim, index, src, reduce, include_self=include_self
        )
    utils.gems_assert_close(res_inplace, ref_inplace, torch.float32, equal_nan=True)

    ref_out_tensor = torch.empty_like(ref_inp)
    torch.scatter_reduce(
        ref_inp,
        dim,
        ref_index,
        ref_src,
        reduce,
        include_self=include_self,
        out=ref_out_tensor,
    )
    res_out_tensor = torch.empty_like(inp)
    with flag_gems.use_gems(include=["scatter_reduce"]):
        res_return = torch.scatter_reduce(
            inp,
            dim,
            index,
            src,
            reduce,
            include_self=include_self,
            out=res_out_tensor,
        )
    assert res_return.data_ptr() == res_out_tensor.data_ptr()
    utils.gems_assert_close(
        res_out_tensor, ref_out_tensor, torch.float32, equal_nan=True
    )


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("bad_index_dtype", [torch.float32, torch.bool, torch.int16])
def test_scatter_reduce_invalid_index_dtype_falls_back_to_pytorch_error(
    bad_index_dtype,
):
    inp, index, src = _public_scatter_reduce_inputs()
    index = index.to(bad_index_dtype)

    with pytest.raises(RuntimeError):
        with flag_gems.use_gems(include=["scatter_reduce"]):
            torch.scatter_reduce(inp, 1, index, src, "sum", include_self=True)


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("inp_shape, src_shape, dim", PUBLIC_SCATTER_REDUCE_SHAPE_CASES)
@pytest.mark.parametrize("dtype", PUBLIC_SCATTER_REDUCE_FLOAT_DTYPES)
@pytest.mark.parametrize("reduce", ["sum", "mean"])
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("overload", PUBLIC_SCATTER_REDUCE_OVERLOADS)
def test_scatter_reduce_public_api_shapes_dims_dtypes(
    inp_shape, src_shape, dim, dtype, reduce, include_self, overload
):
    inp, index, src = _public_scatter_reduce_shape_inputs(
        inp_shape, src_shape, dim, dtype
    )

    ref_out = _reference_public_scatter_reduce(
        inp, dim, index, src, reduce, include_self, overload, dtype
    )
    res_out = _result_public_scatter_reduce(
        inp, dim, index, src, reduce, include_self, overload
    )

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=max(1, index.numel()))


@pytest.mark.scatter_reduce
def test_scatter_reduce_invalid_index_rank_falls_back_to_pytorch_error():
    inp, index, src = _public_scatter_reduce_inputs()
    index = index.unsqueeze(-1)
    src = src.unsqueeze(-1)

    with pytest.raises(RuntimeError):
        with flag_gems.use_gems(include=["scatter_reduce"]):
            torch.scatter_reduce(inp, 1, index, src, "sum", include_self=True)


@pytest.mark.scatter_src
@pytest.mark.parametrize("src_shape", SOURCE_SHAPES)
@pytest.mark.parametrize("inp_shape", INPUT_SHAPES)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scatter_src(src_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
        random.randint(1, min(src_shape[2], inp_shape[2])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n, o = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src)
    ref_out = torch.scatter(ref_inp, dim, ref_index, ref_src)
    with flag_gems.use_gems():
        res_out = torch.scatter(inp, dim, index, src)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("src_shape", SOURCE_SHAPES)
@pytest.mark.parametrize("inp_shape", INPUT_SHAPES)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_scatter_reduce_add(src_shape, inp_shape, dim, dtype):
    utils.init_seed(0)

    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
        random.randint(1, min(src_shape[2], inp_shape[2])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n, o = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = _reference_scatter_reduce(ref_inp, dim, ref_index, ref_src, "add")
    with flag_gems.use_gems():
        res_out = torch.scatter(inp, dim, index, src, reduce="add")

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_add_
@pytest.mark.parametrize("src_shape", SOURCE_SHAPES)
@pytest.mark.parametrize("inp_shape", INPUT_SHAPES)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_scatter_add_(src_shape, inp_shape, dim, dtype):
    utils.init_seed(0)

    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
        random.randint(1, min(src_shape[2], inp_shape[2])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n, o = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = ref_inp.scatter_add_(dim, ref_index, ref_src)
    with flag_gems.use_gems():
        res_out = inp.scatter_add_(dim, index, src)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("src_shape", SOURCE_SHAPES)
@pytest.mark.parametrize("inp_shape", INPUT_SHAPES)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_scatter_reduce_multiply(src_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
        random.randint(1, min(src_shape[2], inp_shape[2])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n, o = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src)
    ref_out = _reference_scatter_reduce(ref_inp, dim, ref_index, ref_src, "multiply")
    with flag_gems.use_gems():
        res_out = torch.scatter(inp, dim, index, src, reduce="multiply")

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_src_
@pytest.mark.parametrize("src_shape", SOURCE_SHAPES)
@pytest.mark.parametrize("inp_shape", INPUT_SHAPES)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scatter_src_(src_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
        random.randint(1, min(src_shape[2], inp_shape[2])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n, o = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src)
    ref_out = ref_inp.clone().scatter_(dim, ref_index, ref_src)
    with flag_gems.use_gems():
        res_out = inp.clone().scatter_(dim, index, src)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.scatter_reduce_
@pytest.mark.parametrize("src_shape", SOURCE_SHAPES)
@pytest.mark.parametrize("inp_shape", INPUT_SHAPES)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_scatter_reduce_add_(src_shape, inp_shape, dim, dtype):
    utils.init_seed(0)

    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
        random.randint(1, min(src_shape[2], inp_shape[2])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n, o = index_shape

    index_size_dim = index_shape[dim]

    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = _reference_scatter_reduce_(
        ref_inp.clone(), dim, ref_index, ref_src, "add"
    )
    with flag_gems.use_gems():
        res_out = inp.clone().scatter_(dim, index, src, reduce="add")

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce_
@pytest.mark.parametrize("src_shape", SOURCE_SHAPES)
@pytest.mark.parametrize("inp_shape", INPUT_SHAPES)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_scatter_reduce_multiply_(src_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
        random.randint(1, min(src_shape[2], inp_shape[2])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n, o = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src)
    ref_out = _reference_scatter_reduce_(
        ref_inp.clone(), dim, ref_index, ref_src, "multiply"
    )
    with flag_gems.use_gems():
        res_out = inp.clone().scatter_(dim, index, src, reduce="multiply")

    utils.gems_assert_close(res_out, ref_out, dtype)
