import flag_gems
import numpy as np
import paddle
import pytest
import os
import random
import time

import numpy as np
import pytest
import torch

from .accuracy_utils import (
    CONTIGUOUS_SHAPE_STRIDES_2D,
    FLOAT_DTYPES,
    SCALARS,
    INT_DTYPES,
    IRREGULAR_SHAPE_STRIDES,
    REDUCTION_SHAPES,
    REDUCTION_SMALL_SHAPES,
    SHAPE_STRIDES,
    ARANGE_START,
    ALL_INT_DTYPES,
    ALL_FLOAT_DTYPES,
    UT_SHAPES_1D,
    UT_SHAPES_2D,
    BOOL_TYPES,
    SkipVersion,
    POINTWISE_SHAPES,
    gems_assert_close,
    gems_assert_equal,
    init_seed,
    to_reference,
    unsqueeze_tensor,
)
from .conftest import QUICK_MODE, TO_CPU
# Make sure every thread has same seed.
random.seed(time.time() // 100)

FLOAT_DTYPES = [torch.float32] if QUICK_MODE else FLOAT_DTYPES
DIM_LIST = [1] if QUICK_MODE else [0, 1]
DIMS_LIST = [1] if QUICK_MODE else [0, 1, [0, 1], [1, 0]]
KEEPDIM_DIMS_SHAPE = (
    [(True, DIMS_LIST[0], REDUCTION_SHAPES[0])]
    if QUICK_MODE
    else list(zip([True, False] * 2, DIMS_LIST, REDUCTION_SHAPES + [(7, 4, 11, 1)]))
)
KEEPDIM_DIM = (
    [(True, DIM_LIST[0])] if QUICK_MODE else list(zip([True, False], DIM_LIST))
)
SMOOTH_IGNORE_SHAPE = (
    [(0.1, 1, REDUCTION_SHAPES[0])]
    if QUICK_MODE
    else list(zip([0, 0.1, 1], [1, 200, -100], REDUCTION_SHAPES))
)
SMOOTH_SHAPE = (
    [(0.1, REDUCTION_SHAPES[0])]
    if QUICK_MODE
    else list(zip([1, 0.1, 0], REDUCTION_SHAPES))
)
DIM_SHAPE_STRIDES = (
    [(1, *CONTIGUOUS_SHAPE_STRIDES_2D[1])]
    if QUICK_MODE
    else list(
        (random.randint(0, len(shape) - 1), shape, stride)
        for shape, stride in SHAPE_STRIDES
    )
)
REGULAR_DIM_SHAPE_STRIDES = (
    [(1, *CONTIGUOUS_SHAPE_STRIDES_2D[1])]
    if QUICK_MODE
    else list(
        (random.randint(0, len(shape) - 1), shape, stride)
        for shape, stride in CONTIGUOUS_SHAPE_STRIDES_2D
    )
)
IRREGULAR_DIM_SHAPE_STRIDES = [(3, *IRREGULAR_SHAPE_STRIDES)]

THRESHOLD_SHAPE = (
    [(0.3, REDUCTION_SHAPES[0])]
    if QUICK_MODE
    else list(zip([0.3, 0.5, 0.7], REDUCTION_SHAPES))
)
CROSS_ENTROPY_LOSS_REDUCTION = ["mean"] if QUICK_MODE else ["mean", "none", "sum"]


MN_SHAPES = [(1, 32)] if QUICK_MODE else [(1, 32), (160, 1024), (5333, 497)]
MNK_SHAPES = (
    [(1, 1, 32)] if QUICK_MODE else [(2,3,4), (15, 160, 1024), (495, 5333, 71)]
)



@pytest.mark.softmax
@pytest.mark.parametrize(
    "shape", [(1, 256)] if QUICK_MODE else [(1, 256), (4096, 256), (200, 2560, 3)]
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("neg_inf", [True, False])
def test_accuracy_softmax(shape, dtype, dim, neg_inf):
    if dtype in [torch.bfloat16]:
        pytest.skip(f"{dtype} not supported")
        return
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if neg_inf:
        inp = torch.where(inp < 0.0, float("-inf"), inp).to(dtype)
    inp.requires_grad_()
    ref_inp = to_reference(inp, True).clone().detach()
    ref_inp.requires_grad_()

    ref_out = torch.nn.functional.softmax(ref_inp, dim=dim)
    ref_out.backward(torch.ones_like(ref_out))
    with flag_gems.use_gems():
        res_out = torch.nn.functional.softmax(inp, dim=dim)
        res_out.backward(torch.ones_like(res_out))
    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    gems_assert_close(inp.grad, ref_inp.grad, dtype, equal_nan=True)

@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="temp disable for updating",
)
@pytest.mark.bmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_bmm(M, N, K, dtype):
    torch.cuda.manual_seed_all(42)
    if flag_gems.vendor_name == "mthreads":
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)

    ref_out = torch.bmm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.bmm(mat1, mat2)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)

    if flag_gems.vendor_name == "mthreads":
        del os.environ["MUSA_ENABLE_SQMMA"]

@pytest.mark.sum
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sum_without_dim(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.sum(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sum(inp)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.size)

@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="temp disable for updating",
)
@pytest.mark.addmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("b_column_major", [True, False])
def test_accuracy_addmm(M, N, K, scalar, dtype, b_column_major):
    torch.cuda.manual_seed_all(42)
    if flag_gems.vendor_name == "mthreads":
        os.environ["MUSA_ENABLE_SQMMA"] = "1"
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    if b_column_major:
        mat2 = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    else:
        mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias1 = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)
    ref_bias1 = to_reference(bias1, True)

    alpha = beta = scalar

    ref_out1 = torch.addmm(ref_bias1, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    with flag_gems.use_gems():
        res_out1 = torch.addmm(bias1, mat1, mat2, alpha=alpha, beta=beta)

    gems_assert_close(res_out1, ref_out1, dtype, reduce_dim=K)

    bias2 = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_bias2 = to_reference(bias2, True)

    ref_out2 = torch.addmm(ref_bias2, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    with flag_gems.use_gems():
        res_out2 = torch.addmm(bias2, mat1, mat2, alpha=alpha, beta=beta)

    gems_assert_close(res_out2, ref_out2, dtype, reduce_dim=K)

    if flag_gems.vendor_name == "mthreads":
        del os.environ["MUSA_ENABLE_SQMMA"]

@pytest.mark.mean
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mean_without_dim(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.mean(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.mean(inp)

    gems_assert_close(res_out, ref_out, dtype)

SHAPE_DIAGONAL = list(zip(POINTWISE_SHAPES, [-2, -2, -1, 0, 1, 3]))
@pytest.mark.triu
@pytest.mark.parametrize("shape, diagonal", SHAPE_DIAGONAL)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_triu(shape, diagonal, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = unsqueeze_tensor(inp, 2)
    ref_inp = to_reference(inp)

    ref_out = torch.triu(ref_inp, diagonal)
    with flag_gems.use_gems():
        res_out = torch.triu(inp, diagonal)

    gems_assert_equal(res_out, ref_out)

@pytest.mark.all
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allTrue"])
def test_accuracy_all_without_dim(shape, dtype, kind):
    if dtype in [torch.float16, torch.bfloat16]: # paddle.all暂时不支持
        pytest.skip("Skip fp16 type.")
    if kind == "allTrue":
        inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 2, shape).to(flag_gems.device).to(dtype)
    ref_inp = to_reference(inp)

    ref_out = torch.all(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.all(inp)

    gems_assert_equal(res_out, ref_out)

@pytest.mark.amax
@pytest.mark.parametrize("keepdim, dim, shape", KEEPDIM_DIMS_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_amax(shape, dim, keepdim, dtype):
    if dtype in [torch.float16, torch.bfloat16]: # paddle.amax暂时不支持
        pytest.skip("Skip fp16 type.")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.amax(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.amax(inp, dim=dim, keepdim=keepdim)

    gems_assert_equal(res_out, ref_out)

@pytest.mark.argmax
@pytest.mark.parametrize("shape", REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_argmax(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.argmax(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.argmax(inp, dim=dim, keepdim=keepdim)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.min
@pytest.mark.parametrize("shape", REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("keepdim, dim", KEEPDIM_DIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + ALL_INT_DTYPES)
def test_accuracy_min_dim(shape, dim, keepdim, dtype):
    if dtype in [torch.int16]: # paddle.min暂时不支持
        pytest.skip(f"Skip {dtype} type.")
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype).to(
            flag_gems.device
        )
    ref_inp = to_reference(inp)

    ref_out_value = torch.min(ref_inp, dim, keepdim)
    with flag_gems.use_gems():
        res_out_value = torch.min(inp, dim, keepdim)

    gems_assert_equal(res_out_value, ref_out_value)


@pytest.mark.index_select
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_index_select(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    index_size = inp.size(dim)
    from math import floor

    index = torch.randint(
        0, index_size, [floor(index_size * 0.8)]
    ).to(flag_gems.device)

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_out = torch.index_select(ref_inp, dim, ref_index)
    with flag_gems.use_gems():
        res_out = torch.index_select(inp, dim, index)

    gems_assert_equal(res_out, ref_out)

@pytest.mark.diag
@pytest.mark.parametrize("shape", UT_SHAPES_1D + UT_SHAPES_2D)
@pytest.mark.parametrize("diagonal", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES + BOOL_TYPES)
def test_accuracy_diag(shape, diagonal, dtype):
    if dtype in [torch.int16, torch.bool]: # paddle.diag暂时不支持
        pytest.skip("Skip  type.")
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    elif dtype in BOOL_TYPES:
        inp = paddle.randint(0, 2, shape=shape, dtype='int64').astype('bool').to(
            flag_gems.device
        )
    else:
        inp = torch.randint(0, 0x7FFF, shape=shape, dtype=dtype).to(
            flag_gems.device
        )

    ref_inp = to_reference(inp)

    ref_out = torch.diag(ref_inp, diagonal)
    with flag_gems.use_gems():
        res_out = torch.diag(inp, diagonal)
    gems_assert_equal(res_out, ref_out)

@pytest.mark.dot
@pytest.mark.parametrize("shape", UT_SHAPES_1D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_dot_tensor_tensor(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.dot(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.dot(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)

@pytest.mark.embedding
@pytest.mark.parametrize("EmbeddingSize", [1024] if TO_CPU else [4096])
@pytest.mark.parametrize("Batch", [2] if TO_CPU else [2, 4])
@pytest.mark.parametrize("M", [4] if TO_CPU else [4, 8])
@pytest.mark.parametrize("N", [8] if TO_CPU else [128, 256, 4096])
@pytest.mark.parametrize("padding_idx", [None, -1, 1, 2])
@pytest.mark.parametrize("scale_grad_by_freq", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_embedding(EmbeddingSize, Batch, M, N, padding_idx, scale_grad_by_freq, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    res_indices = torch.randint(
        0, EmbeddingSize, (Batch, M)
    ).to(flag_gems.device)
    res_indices.requires_grad_()

    res_embedding = torch.randn(
        (EmbeddingSize, N), device=flag_gems.device, dtype=dtype, 
    )
    res_embedding.requires_grad_()


    ref_embedding = to_reference(res_embedding).clone().detach()
    ref_embedding.requires_grad_()
    ref_indices = to_reference(res_indices)

    ref_out = torch.nn.functional.embedding(
        ref_indices, ref_embedding, padding_idx, scale_grad_by_freq=scale_grad_by_freq
    )
    ref_out.backward(torch.ones_like(ref_out))
    with flag_gems.use_gems():
        res_out = torch.nn.functional.embedding(
            res_indices,
            res_embedding,
            padding_idx,
            scale_grad_by_freq=scale_grad_by_freq,
        )
        res_out.backward(torch.ones_like(res_out))
    gems_assert_close(res_out, ref_out, dtype)
    gems_assert_close(res_embedding.grad, ref_embedding.grad, dtype)

@pytest.mark.max
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + ALL_INT_DTYPES)
def test_accuracy_max_without_dim(shape, dtype):
    if dtype in [torch.int16]: 
        pytest.skip(f"Skip {dtype} type.")
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype).to(
            flag_gems.device
        )
    ref_inp = to_reference(inp)

    ref_out = torch.max(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.max(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.ones
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", BOOL_TYPES + ALL_INT_DTYPES + ALL_FLOAT_DTYPES)
def test_accuracy_ones(shape, dtype):
    # without dtype
    with flag_gems.use_gems():
        res_out = torch.ones(shape, device=flag_gems.device)
    gems_assert_equal(res_out, torch.ones(shape, device="cpu" if TO_CPU else flag_gems.device))

    # with dtype
    with flag_gems.use_gems():
        res_out = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    gems_assert_equal(
        res_out, torch.ones(shape, dtype=dtype, device="cpu" if TO_CPU else flag_gems.device)
    )

@pytest.mark.zeros
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", BOOL_TYPES + ALL_INT_DTYPES + ALL_FLOAT_DTYPES)
def test_accuracy_zeros(shape, dtype):
    # without dtype
    with flag_gems.use_gems():
        res_out = torch.zeros(shape, device=flag_gems.device)
    gems_assert_equal(res_out, torch.zeros(shape, device="cpu" if TO_CPU else flag_gems.device))

    # with dtype
    with flag_gems.use_gems():
        res_out = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    gems_assert_equal(
        res_out, torch.zeros(shape, dtype=dtype, device="cpu" if TO_CPU else flag_gems.device)
    )

@pytest.mark.zeros_like
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_zeros_like(shape, dtype):
    inp = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    ref_out = torch.zeros_like(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.zeros_like(inp)
    gems_assert_equal(res_out, ref_out)

@pytest.mark.topk
@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("hiddensize", [128, 256])
@pytest.mark.parametrize("topk", [5])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_topk(
    batch_size,
    hiddensize,
    topk,
    largest,
    dtype,
):
    x = torch.arange(hiddensize, dtype=dtype, device=flag_gems.device)
    x = x.repeat(batch_size).reshape(batch_size, hiddensize)

    # Each row use different shuffled index.
    for bsz in range(batch_size):
        col_indices = torch.randperm(x.size(1))
        x[bsz, :] = x[bsz, col_indices]
    ref_x = to_reference(x)

    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float16:
        ref_x = ref_x.cuda()

    ref_value, ref_index = torch.topk(ref_x, topk, largest=largest)

    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float16:
        if TO_CPU:
            ref_value = ref_value.cpu()
            ref_index = ref_index.cpu()

    with flag_gems.use_gems():
        res_value, res_index = torch.topk(x, topk, largest=largest)

    gems_assert_close(res_value, ref_value, dtype)
    gems_assert_equal(res_index, ref_index)


@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="temp disable for updating",
)
@pytest.mark.mv
@pytest.mark.parametrize("M, N", MN_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mv(M, N, dtype):
    if dtype in [torch.float16, torch.bfloat16]:
        pytest.skip(f"not support {dtype}")
    matrix = torch.randn((N, M), dtype=dtype, device=flag_gems.device)
    vector = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    ref_matrix = to_reference(matrix, True)
    ref_vector = to_reference(vector, True)

    ref_out = torch.mv(ref_matrix, ref_vector)
    with flag_gems.use_gems():
        res_out = torch.mv(matrix, vector)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=M)

if __name__ == "__main__":
    # pytest.main(["-svx", __file__, "::test_accuracy_argmin"])
    file_path = os.path.abspath(__file__)
    pytest.main(["-svx", f"{file_path}"])

