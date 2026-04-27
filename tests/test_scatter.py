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
    ref_out = torch.scatter(ref_inp, dim, ref_index, ref_src, reduce="add")
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
    ref_out = torch.scatter(ref_inp, dim, ref_index, ref_src, reduce="multiply")
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
    ref_out = ref_inp.clone().scatter_(dim, ref_index, ref_src, reduce="add")
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
    ref_out = ref_inp.clone().scatter_(dim, ref_index, ref_src, reduce="multiply")
    with flag_gems.use_gems():
        res_out = inp.clone().scatter_(dim, index, src, reduce="multiply")

    utils.gems_assert_close(res_out, ref_out, dtype)
