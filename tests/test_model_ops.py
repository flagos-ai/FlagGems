#TODO: enflame locally entire file
import random
import time

import numpy as np
import pytest
import torch

import flag_gems

from .accuracy_utils import (
    gems_assert_close,
    gems_assert_equal,
    init_seed,
    to_reference,
    SCALARS,
)
from .conftest import QUICK_MODE
from typing import Optional

# Make sure every thread has same seed.
random.seed(time.time() // 100)


@pytest.mark.log_softmax
@pytest.mark.parametrize("dtype, shape, dim", [
    (dtype, shape, dim)
    for dtype in [torch.float32]
    for shape in [(4,151936), (3,151936), (256,151936), (1,151936)]
    for dim in [0, 1]
])
def test_accuracy_log_softmax(shape, dtype, dim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.log_softmax(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.log_softmax(inp, dim=dim)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.softmax
@pytest.mark.parametrize("dtype, shape, dim", [
    (dtype, shape, dim)
    for dtype in [torch.float32]
    for shape in [(4,151936), (3,151936), (256,151936), (1,151936)]
    for dim in [0, 1]
])
@pytest.mark.parametrize("neg_inf", [True, False])
def test_accuracy_softmax(shape, dtype, dim, neg_inf):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if neg_inf:
        inp = torch.where(inp < 0.0, float("-inf"), inp)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.softmax(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.softmax(inp, dim=dim)
    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.arange
@pytest.mark.parametrize("dtype, start, step, end", [
    (dtype, start, step, end)
    for dtype in [torch.float32]
    for start, step, end in [(0, 1, 64), (1, 1, 40961)]
])
# @pytest.mark.parametrize("device", [device, None])
@pytest.mark.parametrize(
    "pin_memory", [False, None]
)  # Since triton only target to GPU, pin_memory only used in CPU tensors.
def test_arange(start, step, end, dtype, pin_memory):
    ref_out = torch.arange(
        start, end, step, dtype=dtype, device=flag_gems.device, pin_memory=pin_memory
    )
    with flag_gems.use_gems():
        res_out = torch.arange(
            start, end, step, dtype=dtype, device=flag_gems.device, pin_memory=pin_memory
        )

    gems_assert_equal(res_out, ref_out)


@pytest.mark.argmax
@pytest.mark.parametrize("shape", [(1, 151936), (3,151936), (4,151936), (256, 151936)])
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("keepdim", [False])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_argmax(shape, dim, keepdim, dtype): #output stype is i32
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.argmax(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.argmax(inp, dim=dim, keepdim=keepdim)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.cat
@pytest.mark.parametrize("shape", [((40960, 64), (40960, 64))])
@pytest.mark.parametrize("dim", [1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_cat(shape, dim, dtype):
    inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    ref_inp = [to_reference(_) for _ in inp]
    ref_out = torch.cat(ref_inp, dim)

    with flag_gems.use_gems():
        res_out = torch.cat(inp, dim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.cos
@pytest.mark.parametrize("shape", [(40960,64)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_cos(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.cos(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.cos(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.cos_
@pytest.mark.parametrize("shape", [(40960,64)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_cos_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.clone(), True)

    ref_out = torch.cos_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.cos_(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cumsum
@pytest.mark.parametrize("shape", [(256,151936)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_cumsum(shape, dtype):
    dim = -1
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.cumsum(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.cumsum(inp, dim=dim)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])


DIV_SHAPES = (
    ((4,151936), (4, 1), torch.float32, torch.bfloat16),
    ((3,151936), (3, 1), torch.float32, torch.bfloat16),
    ((64), (), torch.float32, ''),
    ((256,151936), (256,1), torch.float32, torch.bfloat16),
    ((256,151936), (256,151936), torch.float32, torch.float32),
    ((1,151936), (1,1), torch.float32, torch.bfloat16),
)
@pytest.mark.div
@pytest.mark.parametrize("lhs_shape, rhs_shape, lhs_dtype, rhs_dtype", DIV_SHAPES)
def test_accuracy_div_tensor_tensor(lhs_shape, rhs_shape, lhs_dtype, rhs_dtype):
    if rhs_shape:
        inp1 = torch.randn(lhs_shape, dtype=lhs_dtype, device=flag_gems.device)
        inp2 = torch.randn(rhs_shape, dtype=rhs_dtype, device=flag_gems.device)
    else:
        inp1 = torch.randn(lhs_shape, dtype=lhs_dtype, device=flag_gems.device)
        inp2 = torch.randn(rhs_shape, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, False)
    ref_inp2 = to_reference(inp2, False)

    ref_out = torch.div(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    gems_assert_close(res_out, ref_out, lhs_dtype, equal_nan=True)


@pytest.mark.inplace
@pytest.mark.div_
@pytest.mark.parametrize("lhs_shape, rhs_shape, lhs_dtype, rhs_dtype", DIV_SHAPES)
def test_accuracy_div_tensor_tensor_(lhs_shape, rhs_shape, lhs_dtype, rhs_dtype):
    if rhs_shape:
        inp1 = torch.randn(lhs_shape, dtype=lhs_dtype, device=flag_gems.device)
        inp2 = torch.randn(rhs_shape, dtype=rhs_dtype, device=flag_gems.device)
    else:
        inp1 = torch.randn(lhs_shape, dtype=lhs_dtype, device=flag_gems.device)
        inp2 = torch.randn(rhs_shape, device=flag_gems.device)
    ref_inp1 = to_reference(inp1.clone(), False)
    ref_inp2 = to_reference(inp2, False)

    ref_out = ref_inp1.div_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.div_(inp2)

    gems_assert_close(res_out, ref_out, lhs_dtype, equal_nan=True)


@pytest.mark.exponential_
@pytest.mark.parametrize("shape", [(256,151936)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_exponential_(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        x.exponential_()
    assert x.min() > 0


FILL_INPUTS = (
    ((128), torch.bfloat16),
    ((4096), torch.bfloat16),
    ((2,2313,65536), torch.bfloat16),
    ((1), torch.bool),
)
@pytest.mark.fill
@pytest.mark.parametrize("value", [1])
@pytest.mark.parametrize("shape, dtype", FILL_INPUTS)
def test_fill(value, shape, dtype):
    # Test fill.Scalar
    x = torch.ones(shape, device=flag_gems.device, dtype=dtype)
    ref_x = to_reference(x, False)

    ref_out = torch.fill(ref_x, value)
    with flag_gems.use_gems():
        res_out = torch.fill(x, value)

    gems_assert_equal(res_out, ref_out)

    # Test fill.Tensor
    value_tensor = torch.tensor(value, device=flag_gems.device, dtype=dtype)
    ref_value_tensor = to_reference(value_tensor, False)
    ref_out_tensor = torch.fill(ref_x, ref_value_tensor)
    with flag_gems.use_gems():
        res_out_tensor = torch.fill(x, value_tensor)

    gems_assert_equal(res_out_tensor, ref_out_tensor)


@pytest.mark.gather
@pytest.mark.parametrize("inp_shape", [(256,151936)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_gather(inp_shape, dim, dtype):
    inp = torch.randn(
        inp_shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    size_dim = inp_shape[dim]

    import random

    index_shape = [
        random.randint(1, inp_shape[0]),
        random.randint(1, inp_shape[1]),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
                ii = [i, j]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_out = torch.gather(ref_inp, dim, ref_index)

    with flag_gems.use_gems():
        res_out = torch.gather(inp, dim, index)

    gems_assert_equal(res_out, ref_out)

    if dtype in (torch.bfloat16,):
        return

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    with flag_gems.use_gems():
        (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    res_in_grad = to_reference(res_in_grad)
    gems_assert_equal(res_in_grad, ref_in_grad)


def gen_indices(input_shape, indices_shape, accumulate):
    indices = []
    for i, shape in enumerate(indices_shape):
        index = np.random.choice(
            np.arange(input_shape[i]), size=shape, replace=accumulate
        )
        indices.append(torch.tensor(index, device=flag_gems.device))
    return indices

INDEX_SHAPES = (
    ((4,151936), (4,)),
    ((3,151936), (3,)),
    ((256,151936), (256,)),
    ((1,151936), (1,)),
)
@pytest.mark.index
@pytest.mark.parametrize("input_shape, indices_shape", INDEX_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_index(input_shape, indices_shape, dtype):
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    indices = gen_indices(input_shape, indices_shape, True)

    ref_inp = to_reference(inp)
    ref_indices = [to_reference(index) for index in indices]
    ref_out = torch.ops.aten.index(ref_inp, ref_indices)
    out = flag_gems.index(inp, indices)
    gems_assert_close(out, ref_out, dtype)


INDEX_SELECT_SHAPES = (
    ((4,4096), (4,)),
    ((151936,4096), (4,)),
    ((151936,4096), (3,)),
    ((3,4096), (3,)),
    ((151936,4096), (2048,)),
    ((2048,4096), (256,)),
    ((151936,4096), (5,)),
    ((5,4096), (1,)),
    ((151936,4096), (22,)),
    ((22,4096), (4,)),
)
@pytest.mark.index_select
@pytest.mark.parametrize("shape, index", INDEX_SELECT_SHAPES)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_accuracy_index_select(shape, index, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    index_size = inp.size(dim)
    from math import floor

    index = torch.randint(
        0, index_size, index, device=flag_gems.device
    )
    # don't support int64, the default type for torch.randint.
    index = index.to(torch.int32)

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_out = torch.index_select(ref_inp, dim, ref_index)
    with flag_gems.use_gems():
        res_out = torch.index_select(inp, dim, index)

    gems_assert_equal(res_out, ref_out)


LE_LT_SHAPES = (
    ((256,151936), (256,1)),
)
@pytest.mark.le
@pytest.mark.parametrize("lhs_shape, rhs_shape", LE_LT_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_le(lhs_shape, rhs_shape, dtype):
    inp1 = torch.randn(lhs_shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(rhs_shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.le(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.le(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.lt
@pytest.mark.parametrize("lhs_shape, rhs_shape", LE_LT_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_lt(lhs_shape, rhs_shape, dtype):
    inp1 = torch.randn(lhs_shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(rhs_shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.lt(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.lt(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.masked_fill_
@pytest.mark.parametrize("shape", [(256,151936)])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
@pytest.mark.parametrize(
    "value",
    [
        torch.tensor(1024, device=flag_gems.device),
        torch.scalar_tensor(1024, device=flag_gems.device),
        1024,
    ],
)
def test_accuracy_masked_fill_(shape, dtype, threshold, value):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randn(shape, dtype=dtype, device=flag_gems.device) < threshold

    ref_inp = to_reference(inp)
    ref_mask = to_reference(mask)
    if torch.is_tensor(value):
        ref_inp.masked_fill_(ref_mask, to_reference(value))
    else:
        ref_inp.masked_fill_(ref_mask, value)
    with flag_gems.use_gems():
        inp.masked_fill_(mask, value)

    gems_assert_equal(inp, ref_inp)


MNK_SHAPES = (
    (4, 4096, 4096), (4, 4096, 24576),
    (4, 12288, 4096), (4, 4096, 6144),
    (4,4096,151936), (3,4096,6144),
    (3,4096,4096), (3,4096,24576),
    (3,12288,4096), (5,4096,6144),
    (3,4096,151936), (2048,4096,6144),
    (2048,4096,4096), (2048,4096,24576),
    (2048,12288,4096), (256,4096,151936),
    (5,4096,4096), (5,4096,24576),
    (5,12288,4096), (1,4096,151936),
    (22,4096,6144), (22,4096,4096),
    (22,4096,24576), (22,12288,4096),
)
@pytest.mark.mm
@pytest.mark.parametrize("M, K, N", MNK_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_accuracy_mm(M, N, K, dtype):
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)

    ref_out = torch.mm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.mm(mat1, mat2)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


MUL_SHAPES = (
    ((64), ()),
    ((40960,1), (1, 64)),
    ((256,151936), ()),
)
@pytest.mark.mul
@pytest.mark.parametrize("lhs_shape, rhs_shape", MUL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_mul(lhs_shape, rhs_shape, dtype):
    inp1 = torch.randn(lhs_shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(rhs_shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.mul(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.mul(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pow
@pytest.mark.parametrize("shape", [(64,)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_pow(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    if flag_gems.vendor_name == "kunlunxin":
        inp1 = inp1.uniform_(-1, 1)
        inp2 = inp2.uniform_(-1, 1)

    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.pow(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.pow(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.inplace
@pytest.mark.pow_
@pytest.mark.parametrize("shape", [(64,)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_pow_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    if flag_gems.vendor_name == "kunlunxin":
        inp1 = inp1.uniform_(-1, 1)
        inp2 = inp2.uniform_(-1, 1)

    ref_inp1 = to_reference(inp1.clone(), True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = ref_inp1.pow_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.pow_(inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.reciprocal
@pytest.mark.parametrize("shape", [(64,)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_reciprocal(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.reciprocal(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.reciprocal(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.inplace
@pytest.mark.reciprocal_
@pytest.mark.parametrize("shape", [(64,)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_reciprocal_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.clone(), True)

    ref_out = torch.reciprocal_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.reciprocal_(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.scatter
@pytest.mark.parametrize("src_shape", [(256,151936)])
@pytest.mark.parametrize("inp_shape", [(256,151936)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_scatter_src(src_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    import random

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            ii = [i, j]
            ii[dim] = slice(0, index.size(dim) + 1)
            index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_src = to_reference(src)
    ref_out = torch.scatter(ref_inp, dim, ref_index, ref_src)
    with flag_gems.use_gems():
        res_out = torch.scatter(inp, dim, index, src)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.scatter
@pytest.mark.parametrize("src_shape", [(256,151936)])
@pytest.mark.parametrize("inp_shape", [(256,151936)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_scatter_add(src_shape, inp_shape, dim, dtype):
    init_seed(0)
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    import random

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
                ii = [i, j]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = to_reference(inp, upcast=True)
    ref_index = to_reference(index)
    ref_src = to_reference(src, upcast=True)
    ref_out = torch.scatter(ref_inp, dim, ref_index, ref_src, reduce="add")
    with flag_gems.use_gems():
        res_out = torch.scatter(inp, dim, index, src, reduce="add")

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter
@pytest.mark.parametrize("src_shape", [(256,151936)])
@pytest.mark.parametrize("inp_shape", [(256,151936)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_scatter_mul(src_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    import random

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
                ii = [i, j]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_src = to_reference(src)
    ref_out = torch.scatter(ref_inp, dim, ref_index, ref_src, reduce="multiply")
    with flag_gems.use_gems():
        res_out = torch.scatter(inp, dim, index, src, reduce="multiply")

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_
@pytest.mark.parametrize("src_shape", [(256,151936)])
@pytest.mark.parametrize("inp_shape", [(256,151936)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_inplace_scatter_src(src_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    import random

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
                ii = [i, j]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_src = to_reference(src)
    ref_out = ref_inp.clone().scatter_(dim, ref_index, ref_src)
    with flag_gems.use_gems():
        res_out = inp.clone().scatter_(dim, index, src)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.scatter_
@pytest.mark.parametrize("src_shape", [(256,151936)])
@pytest.mark.parametrize("inp_shape", [(256,151936)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_inplace_scatter_add(src_shape, inp_shape, dim, dtype):
    init_seed(0)
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    import random

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
                ii = [i, j]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = to_reference(inp, upcast=True)
    ref_index = to_reference(index)
    ref_src = to_reference(src, upcast=True)
    ref_out = ref_inp.clone().scatter_(dim, ref_index, ref_src, reduce="add")
    with flag_gems.use_gems():
        res_out = inp.clone().scatter_(dim, index, src, reduce="add")

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_
@pytest.mark.parametrize("src_shape", [(256,151936)])
@pytest.mark.parametrize("inp_shape", [(256,151936)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_inplace_scatter_mul(src_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    import random

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
                ii = [i, j]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_src = to_reference(src)
    ref_out = ref_inp.clone().scatter_(dim, ref_index, ref_src, reduce="multiply")
    with flag_gems.use_gems():
        res_out = inp.clone().scatter_(dim, index, src, reduce="multiply")

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sin
@pytest.mark.parametrize("shape", [(40960, 64)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_sin(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.sin(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sin(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.sin_
@pytest.mark.parametrize("shape", [(40960, 64)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_sin_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.clone(), True)

    ref_out = torch.sin_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sin_(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sort
@pytest.mark.parametrize("shape", [(256, 151936)])
@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("dim", [0, -1])
def test_sort(shape, descending, dtype, dim):
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_y = to_reference(y)
    # we only implement stable sort, non-stable sort is undefined
    ref_value, ref_index = torch.sort(
        ref_y, dim=dim, stable=True, descending=descending
    )

    with flag_gems.use_gems():
        res_value, res_index = torch.sort(
            y, dim=dim, stable=True, descending=descending
        )

    gems_assert_close(res_value, ref_value, dtype)
    gems_assert_equal(res_index, ref_index)


SUB_SHAPES = [
    ((256,), torch.int32),
    ((256,1), torch.bfloat16),
    ((256,151936), torch.float32),
]
@pytest.mark.sub
@pytest.mark.parametrize("shape, dtype", SUB_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
def test_accuracy_sub(shape, alpha, dtype):
    if dtype == torch.int32:
        inp1 = torch.randint(0, 255, shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randint(0, 255, shape, dtype=dtype, device=flag_gems.device)
    else:
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.sub(ref_inp1, ref_inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.sub_
@pytest.mark.parametrize("shape, dtype", SUB_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
def test_accuracy_sub_(shape, alpha, dtype):
    if dtype == torch.int32:
        inp1 = torch.randint(0, 255, shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randint(0, 255, shape, dtype=dtype, device=flag_gems.device)
    else:
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1.clone(), True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = ref_inp1.sub_(ref_inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = inp1.sub_(inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


FUSED_ADD_RMS_NORM_SHAPES = (
    (4, 4096), (3, 4096),
    (2048, 4096), (5, 4096),
    (22, 4096)
)
@pytest.mark.fused_add_rms_norm
@pytest.mark.parametrize("shape", FUSED_ADD_RMS_NORM_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_accuracy_fused_add_rms_norm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    residual = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(layer_shape, dtype=dtype, device=flag_gems.device)
    eps = 1e-5

    ref_inp = to_reference(inp, True)
    ref_residual = to_reference(residual, True)
    ref_weight = to_reference(weight, True)

    def _torch_fused_add_rms_norm(x, residual, weight, eps):
        x = x + residual
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + eps)
        return weight * hidden_states, x

    ref_out, ref_new_residual = _torch_fused_add_rms_norm(
        ref_inp,
        ref_residual,
        weight=ref_weight,
        eps=eps,
    )

    res_out, res_new_residual = flag_gems.fused_add_rms_norm(
        inp, residual, list(layer_shape), weight=weight, eps=eps
    )

    gems_assert_close(res_out, ref_out, dtype)
    gems_assert_close(res_new_residual, ref_new_residual, dtype)


RMS_NORM_SHAPES = (
    (128,128), (32,128),
    (4,4096), (3,4096),
    (96,128), (24,128),
    (2048,4096),(176,128),
    (16384,128), (5,4096),
    (160,128), (40,128),
    (22,4096), (704,128),
)
@pytest.mark.rms_norm
@pytest.mark.parametrize("shape", RMS_NORM_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_accuracy_rmsnorm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    np.random.seed(0)
    np_inp = np.random.uniform(-0.1, 0.1, shape).astype(np.float32)
    np_grad = np.random.uniform(-0.01, 0.01, shape).astype(np.float32)
    np_weight = np.random.uniform(-0.1, 0.1, layer_shape).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=flag_gems.device, requires_grad=True)
    weight = torch.tensor(
        np_weight, dtype=dtype, device=flag_gems.device, requires_grad=True
    )

    eps = 1e-5

    ref_inp = to_reference(inp)
    ref_weight = to_reference(weight)

    def _torch_rms_norm(x, weight, eps):
        upcast_x = x.to(torch.float32)
        variance = upcast_x.pow(2).mean(-1, keepdim=True)
        hidden_states = upcast_x * torch.rsqrt(variance + eps).to(torch.float32)
        hidden_states = hidden_states.to(x.dtype)
        return weight * hidden_states

    ref_out = _torch_rms_norm(ref_inp, weight=ref_weight, eps=eps)
    res_out = flag_gems.rms_norm(inp, list(layer_shape), weight=weight, eps=eps)

    res_grad = torch.tensor(
        np_grad, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_grad = to_reference(res_grad)

    res_grad, res_weight_grad = torch.autograd.grad(res_out, (inp, weight), res_grad)
    ref_grad, ref_weight_grad = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight), ref_grad
    )

    gems_assert_close(res_out, ref_out, dtype)
    if flag_gems.vendor_name == "kunlunxin" and shape == (200, 40999, 3):
        pytest.skip("wait for backward support")
    gems_assert_close(res_grad, ref_grad, dtype)
    gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=N)


# Copied from transformers.models.llama.modeling_llama.rotate_half
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.cohere.modeling_cohere.rotate_half
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere/modeling_cohere.py
def rotate_interleave(x):
    """Rotates interleave the hidden dims of the input."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def get_rope_cos_sin(max_seq_len, dim, dtype, base=10000, device=flag_gems.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin

def torch_apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids: Optional[torch.Tensor] = None,
    rotary_interleaved: bool = False,
):
    q = q.float()
    k = k.float()
    if position_ids is None:
        cos = cos[None, : q.size(-3), None, :]
        sin = sin[None, : q.size(-3), None, :]
    else:
        cos = cos[position_ids].unsqueeze(-2)  # [bs, seq_len, 1, dim/2]
        sin = sin[position_ids].unsqueeze(-2)  # [bs, seq_len, 1, dim/2]
    if rotary_interleaved:
        cos = torch.repeat_interleave(cos, 2, dim=-1)  # [bs, seq_len, 1, dim]
        sin = torch.repeat_interleave(sin, 2, dim=-1)  # [bs, seq_len, 1, dim]
        rotate_fn = rotate_interleave
    else:
        cos = torch.cat([cos, cos], dim=-1)  # [bs, seq_len, 1, dim]
        sin = torch.cat([sin, sin], dim=-1)  # [bs, seq_len, 1, dim]
        rotate_fn = rotate_half

    q_embed = (q * cos) + (rotate_fn(q) * sin)
    k_embed = (k * cos) + (rotate_fn(k) * sin)

    return q_embed, k_embed


@pytest.mark.apply_rotary_pos_emb
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("max_seq_len", [40960])
@pytest.mark.parametrize("seq_len", [4, 3, 2048, 5, 22])
@pytest.mark.parametrize("q_heads,k_heads", [(4096, 1024)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("rotary_interleaved", [True, False])
@pytest.mark.parametrize("has_pos_id", [True])
def test_apply_rotary_pos_emb(
    batch_size,
    max_seq_len,
    seq_len,
    q_heads,
    k_heads,
    head_dim,
    dtype,
    has_pos_id,
    rotary_interleaved,
):
    q = torch.randn(
        (batch_size, seq_len, q_heads, head_dim), dtype=dtype, device=flag_gems.device
    )
    k = torch.randn(
        (batch_size, seq_len, k_heads, head_dim), dtype=dtype, device=flag_gems.device
    )

    position_ids = torch.randint(
        0, max_seq_len, (batch_size, seq_len), device=flag_gems.device
    )
    cos, sin = get_rope_cos_sin(max_seq_len, head_dim, dtype, device=flag_gems.device)

    ref_q = to_reference(q, True)
    ref_k = to_reference(k, True)
    ref_cos = to_reference(cos, True)
    ref_sin = to_reference(sin, True)
    ref_position_ids = to_reference(position_ids)

    q_embed_ref, k_embed_ref = torch_apply_rotary_pos_emb(
        q=ref_q,
        k=ref_k,
        cos=ref_cos,
        sin=ref_sin,
        position_ids=ref_position_ids if has_pos_id else None,
        rotary_interleaved=rotary_interleaved,
    )
    q_embed_out, k_embed_out = flag_gems.apply_rotary_pos_emb(
        q=q,
        k=k,
        cos=cos,
        sin=sin,
        position_ids=position_ids if has_pos_id else None,
        rotary_interleaved=rotary_interleaved,
    )

    gems_assert_close(q_embed_out, q_embed_ref, dtype)
    gems_assert_close(k_embed_out, k_embed_ref, dtype)


SILU_AND_MUL_SHAPES = (
    ((4,24576), (3,24576), (2048,24576),
    (5,24576), (22,24576))
)
@pytest.mark.silu_and_mul
@pytest.mark.parametrize("shape", SILU_AND_MUL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_accuracy_silu_and_mul(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    # inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    d = shape[-1] // 2
    inp1, inp2 = inp[..., :d], inp[..., d:]
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    out = torch.nn.functional.silu(ref_inp1)
    ref_out = torch.mul(torch.nn.functional.silu(ref_inp1), ref_inp2)
    with flag_gems.use_gems():
        res_out = flag_gems.silu_and_mul(inp1, inp2)

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)

    (ref_inp1_grad, ref_inp2_grad) = torch.autograd.grad(
        ref_out, (ref_inp1, ref_inp2), ref_grad
    )

    (res_inp1_grad, res_inp2_grad) = torch.autograd.grad(
        res_out, (inp1, inp2), out_grad
    )

    gems_assert_close(res_out, ref_out, dtype)
    gems_assert_close(res_inp1_grad, ref_inp1_grad, dtype)
    gems_assert_close(res_inp2_grad, ref_inp2_grad, dtype)


TO_DTYPE_SHAPES = (
    ((4,151936), torch.bfloat16, torch.float32),
    ((3,151936), torch.bfloat16, torch.float32),
    ((40960, 128), torch.float32, torch.bfloat16),
    ((256, 151936), torch.bfloat16, torch.float32),
    ((1, 151936), torch.bfloat16, torch.float32),
)
@pytest.mark.to_dtype
@pytest.mark.parametrize("shape, in_dtype, out_dtype", TO_DTYPE_SHAPES)
def test_accuracy_to_dtype(shape, in_dtype, out_dtype):
    x = torch.randn(shape, dtype=in_dtype, device=flag_gems.device)
    ref_x = to_reference(x)
    ref_out = ref_x.to(out_dtype)
    with flag_gems.use_gems():
        out = x.to(out_dtype)
    gems_assert_equal(out, ref_out)