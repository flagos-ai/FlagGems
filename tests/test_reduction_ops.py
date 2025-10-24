import os
import random
import time

import numpy as np
import pytest
import torch

import flag_gems
from flag_gems import topk_softmax

from .accuracy_utils import (
    CONTIGUOUS_SHAPE_STRIDES_2D,
    FLOAT_DTYPES,
    INT_DTYPES,
    IRREGULAR_SHAPE_STRIDES,
    REDUCTION_SHAPES,
    REDUCTION_SMALL_SHAPES,
    SHAPE_STRIDES,
    SkipVersion,
    gems_assert_close,
    gems_assert_equal,
    init_seed,
    to_reference,
)
from .conftest import QUICK_MODE

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


@pytest.mark.amax
@pytest.mark.parametrize("keepdim, dim, shape", KEEPDIM_DIMS_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_amax(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.amax(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.amax(inp, dim=dim, keepdim=keepdim)

    gems_assert_equal(res_out, ref_out)


# TODO: There are some bugs in argmax with large size.
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


@pytest.mark.argmin
@pytest.mark.parametrize("shape", REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST + [None])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_argmin(shape, dim, keepdim, dtype):
    if dtype in INT_DTYPES:
        inp = torch.randint(-1024, 1024, size=shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.argmin(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.argmin(inp, dim=dim, keepdim=keepdim)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.cross_entropy_loss
@pytest.mark.parametrize("label_smoothing, ignore_index, shape", SMOOTH_IGNORE_SHAPE)
@pytest.mark.parametrize("reduction", CROSS_ENTROPY_LOSS_REDUCTION)
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cross_entropy_loss_indices(
    shape, dtype, weight, ignore_index, reduction, label_smoothing
):
    dim = 1
    up_limit = shape[dim] - 1
    target_shape = list(shape)
    del target_shape[dim]

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    target = torch.randint(0, up_limit, target_shape, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target)

    if weight:
        wgt = torch.randn(shape[dim], dtype=dtype, device=flag_gems.device)
        ref_wgt = to_reference(wgt, True)
    else:
        wgt = None
        ref_wgt = None
    ref_out = torch.nn.functional.cross_entropy(
        ref_inp,
        ref_target,
        weight=ref_wgt,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
    res_out = flag_gems.cross_entropy_loss(
        inp,
        target,
        weight=wgt,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])


@pytest.mark.cross_entropy_loss
@pytest.mark.parametrize("label_smoothing, shape", SMOOTH_SHAPE)
@pytest.mark.parametrize("reduction", CROSS_ENTROPY_LOSS_REDUCTION)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cross_entropy_loss_probabilities(
    shape, dtype, reduction, label_smoothing
):
    dim = 1
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(shape[dim], dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target, True)
    ref_weight = to_reference(weight, True)
    ref_out = torch.nn.functional.cross_entropy(
        ref_inp,
        ref_target,
        weight=ref_weight,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
    res_out = flag_gems.cross_entropy_loss(
        inp, target, weight=weight, reduction=reduction, label_smoothing=label_smoothing
    )
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])


@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.nll_loss
@pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("ignore_index", [1, 200, -100])
def test_accuracy_nll_loss(shape, dtype, ignore_index, reduction, weight):
    dim = 1
    target_shape = list(shape)
    del target_shape[dim]

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    target = torch.randint(0, shape[dim], target_shape, device=flag_gems.device)
    if weight:
        weight = torch.randn(shape[dim], dtype=dtype, device=flag_gems.device)
    else:
        weight = None
    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target)
    ref_weight = to_reference(weight, True)

    ref_out = torch.nn.functional.nll_loss(
        ref_inp, ref_target, ref_weight, reduction=reduction, ignore_index=ignore_index
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.nll_loss(
            inp, target, weight, reduction=reduction, ignore_index=ignore_index
        )
    reduce_dim = 1 if reduction == "none" else target.numel()
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim, equal_nan=True)

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    with flag_gems.use_gems():
        (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])


CUMSUM_SHAPES = (
    [(2, 32)] if QUICK_MODE else REDUCTION_SHAPES + [(2637,), (16, 1025, 255)]
)


@pytest.mark.cumsum
@pytest.mark.parametrize("shape", CUMSUM_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_cumsum(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    dim = 1 if shape == REDUCTION_SHAPES[-1] else -1
    if dtype in INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
        ref_inp = to_reference(inp)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = to_reference(inp, True)

    ref_out = torch.cumsum(ref_inp, dim=dim)
    if flag_gems.vendor_name == "kunlunxin":
        from flag_gems.runtime.backend._kunlunxin import ops as kl_ops

        res_out = kl_ops.cumsum(inp, dim=dim)
    else:
        with flag_gems.use_gems():
            res_out = torch.cumsum(inp, dim=dim)

    # we should use ref's output type, since cumsum of int dtype results in int64
    check_dtype = ref_out.dtype if dtype in INT_DTYPES else dtype
    gems_assert_close(res_out, ref_out, check_dtype, reduce_dim=shape[dim])


CUMMIN_SHAPES = (
    [(2, 32)] if QUICK_MODE else REDUCTION_SHAPES + [(2637,), (16, 1025, 255)]
)


@pytest.mark.skipif(
    SkipVersion("triton", "<3.0"),
    reason="Skipping when associative_scan only support single tensor input.",
)
@pytest.mark.cummin
@pytest.mark.parametrize("shape", CUMMIN_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_cummin(shape, dtype):
    if flag_gems.vendor_name == "mthreads":
        # Compatible with older versions of LLVM
        os.environ["DISABLE_LLVM_OPT"] = "1"

    dim = 1 if shape == REDUCTION_SHAPES[-1] else -1
    if dtype in INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.cummin(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.cummin(inp, dim=dim)
    gems_assert_close(res_out.values, ref_out.values, dtype, reduce_dim=shape[dim])
    gems_assert_equal(res_out.indices, ref_out.indices)

    if flag_gems.vendor_name == "mthreads":
        del os.environ["DISABLE_LLVM_OPT"]


@pytest.mark.cummin
@pytest.mark.parametrize("shape", CUMMIN_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("nan_ratio", [0.1, 0.3, 0.5])
def test_accuracy_cummin_with_nan(shape, dtype, nan_ratio):
    """Test cummin with NaN values at different ratios"""
    if flag_gems.vendor_name == "mthreads":
        # Compatible with older versions of LLVM
        os.environ["DISABLE_LLVM_OPT"] = "1"

    dim = 1 if shape == REDUCTION_SHAPES[-1] else -1

    # Create tensor with some NaN values
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # Randomly set some values to NaN
    total_elements = inp.numel()
    nan_count = int(total_elements * nan_ratio)
    nan_indices = torch.randperm(total_elements)[:nan_count]
    flat_inp = inp.flatten()
    flat_inp[nan_indices] = float("nan")
    inp = flat_inp.view(shape)

    ref_inp = to_reference(inp, True)

    ref_out = torch.cummin(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.cummin(inp, dim=dim)

    gems_assert_close(
        res_out.values, ref_out.values, dtype, reduce_dim=shape[dim], equal_nan=True
    )
    gems_assert_equal(res_out.indices, ref_out.indices)

    if flag_gems.vendor_name == "mthreads":
        del os.environ["DISABLE_LLVM_OPT"]


CUMMAX_SHAPES = (
    [(2, 32)] if QUICK_MODE else REDUCTION_SHAPES + [(2637,), (16, 1025, 255)]
)


@pytest.mark.skipif(
    SkipVersion("triton", "<3.0"),
    reason="Skipping when associative_scan only support single tensor input.",
)
@pytest.mark.cummax
@pytest.mark.parametrize("shape", CUMMAX_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_cummax(shape, dtype):
    if flag_gems.vendor_name == "mthreads":
        # Compatible with older versions of LLVM
        os.environ["DISABLE_LLVM_OPT"] = "1"

    dim = 1 if shape == REDUCTION_SHAPES[-1] else -1
    if dtype in INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.cummax(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.cummax(inp, dim=dim)
    gems_assert_close(res_out.values, ref_out.values, dtype, reduce_dim=shape[dim])
    gems_assert_equal(res_out.indices, ref_out.indices)

    if flag_gems.vendor_name == "mthreads":
        del os.environ["DISABLE_LLVM_OPT"]


@pytest.mark.cummax
@pytest.mark.parametrize("shape", CUMMAX_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("nan_ratio", [0.1, 0.3, 0.5])
def test_accuracy_cummax_with_nan(shape, dtype, nan_ratio):
    """Test cummax with NaN values at different ratios"""
    if flag_gems.vendor_name == "mthreads":
        # Compatible with older versions of LLVM
        os.environ["DISABLE_LLVM_OPT"] = "1"

    dim = 1 if shape == REDUCTION_SHAPES[-1] else -1

    # Create tensor with some NaN values
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # Randomly set some values to NaN
    total_elements = inp.numel()
    nan_count = int(total_elements * nan_ratio)
    nan_indices = torch.randperm(total_elements)[:nan_count]
    flat_inp = inp.flatten()
    flat_inp[nan_indices] = float("nan")
    inp = flat_inp.view(shape)

    ref_inp = to_reference(inp, True)

    ref_out = torch.cummax(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.cummax(inp, dim=dim)

    gems_assert_close(
        res_out.values, ref_out.values, dtype, reduce_dim=shape[dim], equal_nan=True
    )
    gems_assert_equal(res_out.indices, ref_out.indices)

    if flag_gems.vendor_name == "mthreads":
        del os.environ["DISABLE_LLVM_OPT"]


NONZERO_SHAPES = [(2, 32)] if QUICK_MODE else REDUCTION_SHAPES + [(2637,)]


@pytest.mark.nonzero
@pytest.mark.parametrize("shape", NONZERO_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES + [torch.bool])
def test_accuracy_nonzero(shape, dtype):
    if dtype == torch.bool:
        inp = torch.randint(0, 2, shape, dtype=torch.int, device=flag_gems.device).to(
            dtype
        )
    elif dtype in INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, False)

    ref_out = torch.nonzero(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nonzero(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.count_nonzero
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES + [torch.bool])
def test_accuracy_count_nonzero(shape, dtype):
    if dtype == torch.bool:
        inp = torch.randint(0, 2, shape, dtype=torch.int, device=flag_gems.device).to(
            dtype
        )
    elif dtype in INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, False)
    dim = random.choice([None] + list(range(inp.ndim)))
    ref_out = torch.count_nonzero(ref_inp, dim)
    with flag_gems.use_gems():
        res_out = torch.count_nonzero(inp, dim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.log_softmax
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1] if flag_gems.vendor_name == "cambricon" else [1])
def test_accuracy_log_softmax(shape, dtype, dim):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.log_softmax(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.log_softmax(inp, dim=dim)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log_softmax
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1] if flag_gems.vendor_name == "cambricon" else [1])
def test_accuracy_log_softmax_backward(shape, dtype, dim):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)
    res_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_out = torch.randn_like(res_grad)
    ref_grad = to_reference(res_grad, True)
    ref_out = to_reference(res_out, True)

    ref_in_grad = torch.ops.aten._log_softmax_backward_data(
        ref_grad, ref_out, dim, ref_grad.dtype
    )
    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten._log_softmax_backward_data(
            res_grad, res_out, dim, dtype
        )
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])


# TODO: failed at (1, 2) (200, 40999, 3)
@pytest.mark.softmax
@pytest.mark.parametrize(
    "shape", [(1, 256)] if QUICK_MODE else [(1, 256), (4096, 256), (200, 2560, 3)]
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
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


@pytest.mark.softmax
@pytest.mark.parametrize(
    "shape", [(1, 256)] if QUICK_MODE else [(1, 256), (4096, 256), (200, 2560, 3)]
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("neg_inf", [True, False])
def test_accuracy_softmax_backward(shape, dtype, dim, neg_inf):
    res_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if neg_inf:
        res_grad = torch.where(res_grad < 0.0, float("-inf"), res_grad)
    res_out = torch.randn_like(res_grad)

    ref_grad = to_reference(res_grad, True)
    ref_out = to_reference(res_out, True)

    ref_in_grad = torch.ops.aten._softmax_backward_data(
        ref_grad, ref_out, dim, ref_grad.dtype
    )
    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten._softmax_backward_data(
            res_grad, res_out, dim, dtype
        )
    gems_assert_close(
        res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim], equal_nan=True
    )


@pytest.mark.var_mean
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIMS_LIST)
@pytest.mark.parametrize("correction", [1] if QUICK_MODE else [0, 1])
@pytest.mark.parametrize("keepdim", [True] if QUICK_MODE else [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_varmean(shape, dim, correction, keepdim, dtype):
    if shape[0] == 1:  # TODO: res is inf, while ref is nan
        shape = (2, 2)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_var, ref_mean = torch.var_mean(
        ref_inp, dim, correction=correction, keepdim=keepdim
    )
    with flag_gems.use_gems():
        res_var, res_mean = torch.var_mean(
            inp, dim, correction=correction, keepdim=keepdim
        )

    gems_assert_close(res_mean, ref_mean, dtype)
    gems_assert_close(res_var, ref_var, dtype)


@pytest.mark.scatter
@pytest.mark.parametrize(
    "src_shape", [(32, 8, 4)] if QUICK_MODE else [(128, 16, 4), (256, 32, 8)]
)
@pytest.mark.parametrize(
    "inp_shape", [(64, 16, 8)] if QUICK_MODE else [(512, 128, 32), (1024, 64, 16)]
)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_scatter_src(src_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    import random

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

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_src = to_reference(src)
    ref_out = torch.scatter(ref_inp, dim, ref_index, ref_src)
    with flag_gems.use_gems():
        res_out = torch.scatter(inp, dim, index, src)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.scatter
@pytest.mark.parametrize(
    "src_shape", [(32, 8, 4)] if QUICK_MODE else [(128, 16, 4), (256, 32, 8)]
)
@pytest.mark.parametrize(
    "inp_shape", [(64, 16, 8)] if QUICK_MODE else [(512, 128, 32), (1024, 64, 16)]
)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_scatter_add(src_shape, inp_shape, dim, dtype):
    init_seed(0)
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    import random

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

    ref_inp = to_reference(inp, upcast=True)
    ref_index = to_reference(index)
    ref_src = to_reference(src, upcast=True)
    ref_out = torch.scatter(ref_inp, dim, ref_index, ref_src, reduce="add")
    with flag_gems.use_gems():
        res_out = torch.scatter(inp, dim, index, src, reduce="add")

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.scatter
@pytest.mark.parametrize(
    "src_shape", [(32, 8, 4)] if QUICK_MODE else [(128, 16, 4), (256, 32, 8)]
)
@pytest.mark.parametrize(
    "inp_shape", [(64, 16, 8)] if QUICK_MODE else [(512, 128, 32), (1024, 64, 16)]
)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_scatter_mul(src_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    import random

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

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_src = to_reference(src)
    ref_out = torch.scatter(ref_inp, dim, ref_index, ref_src, reduce="multiply")
    with flag_gems.use_gems():
        res_out = torch.scatter(inp, dim, index, src, reduce="multiply")

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_
@pytest.mark.parametrize(
    "src_shape", [(32, 8, 4)] if QUICK_MODE else [(128, 16, 4), (256, 32, 8)]
)
@pytest.mark.parametrize(
    "inp_shape", [(64, 16, 8)] if QUICK_MODE else [(512, 128, 32), (1024, 64, 16)]
)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_inplace_scatter_src(src_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    import random

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

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_src = to_reference(src)
    ref_out = ref_inp.clone().scatter_(dim, ref_index, ref_src)
    with flag_gems.use_gems():
        res_out = inp.clone().scatter_(dim, index, src)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.scatter_
@pytest.mark.parametrize(
    "src_shape", [(32, 8, 4)] if QUICK_MODE else [(128, 16, 4), (256, 32, 8)]
)
@pytest.mark.parametrize(
    "inp_shape", [(64, 16, 8)] if QUICK_MODE else [(512, 128, 32), (1024, 64, 16)]
)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_inplace_scatter_add(src_shape, inp_shape, dim, dtype):
    init_seed(0)
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    import random

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

    ref_inp = to_reference(inp, upcast=True)
    ref_index = to_reference(index)
    ref_src = to_reference(src, upcast=True)
    ref_out = ref_inp.clone().scatter_(dim, ref_index, ref_src, reduce="add")
    with flag_gems.use_gems():
        res_out = inp.clone().scatter_(dim, index, src, reduce="add")

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.scatter_
@pytest.mark.parametrize(
    "src_shape", [(32, 8, 4)] if QUICK_MODE else [(128, 16, 4), (256, 32, 8)]
)
@pytest.mark.parametrize(
    "inp_shape", [(64, 16, 8)] if QUICK_MODE else [(512, 128, 32), (1024, 64, 16)]
)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_inplace_scatter_mul(src_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    import random

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

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_src = to_reference(src)
    ref_out = ref_inp.clone().scatter_(dim, ref_index, ref_src, reduce="multiply")
    with flag_gems.use_gems():
        res_out = inp.clone().scatter_(dim, index, src, reduce="multiply")

    gems_assert_close(res_out, ref_out, dtype)


TRACE_SHAPES = [
    (1, 1),
    (5, 5),
    (10, 20),
    (30, 15),
    (1, 100),
    (100, 1),
    (128, 256),
    (256, 128),
    (0, 10),  # empty diagonal
    (10, 0),  # empty diagonal
    (1500, 1200),  # Larger shape
]


@pytest.mark.trace
@pytest.mark.parametrize("shape", TRACE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES + [torch.bool])
def test_accuracy_trace(shape, dtype):
    if dtype == torch.bool:
        inp = torch.randint(0, 2, size=shape, device=flag_gems.device).to(dtype)
    elif dtype in INT_DTYPES:
        inp = torch.randint(-100, 100, size=shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(inp)
    if dtype == torch.bool and ref_inp.device.type == "cpu":
        pytest.skip("skipping bool on CPU reference.")

    ref_out = torch.trace(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.trace(inp)

    if dtype in FLOAT_DTYPES:
        gems_assert_close(res_out, ref_out, dtype)
    else:
        gems_assert_equal(res_out, ref_out)


@pytest.mark.gather
@pytest.mark.parametrize(
    "inp_shape",
    [(32, 8, 4)] if QUICK_MODE else [(512, 128, 32), (1024, 64, 16), (128, 32, 256)],
)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_gather(inp_shape, dim, dtype):
    inp = torch.randn(
        inp_shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    size_dim = inp_shape[dim]

    import random

    index_shape = [
        random.randint(1, inp_shape[0]),
        random.randint(1, inp_shape[1]),
        random.randint(1, inp_shape[2]),
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


@pytest.mark.select_scatter
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_select_scatter(shape, dim, dtype):
    import random

    index = random.randint(0, shape[dim] - 1)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    src_shape = list(inp.shape)
    del src_shape[dim]
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(inp)
    ref_src = to_reference(src)
    ref_out = torch.select_scatter(ref_inp, dim=dim, index=index, src=ref_src)
    with flag_gems.use_gems():
        res_out = torch.select_scatter(inp, dim=dim, index=index, src=src)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.select_scatter
def test_accuracy_select_scatter_with_self_overlapping_input():
    dim = 0
    index = 1
    inp = torch.randn((1, 4), device=flag_gems.device).broadcast_to((3, 4))
    src = torch.randn((4,), device=flag_gems.device)

    ref_inp = to_reference(inp)
    ref_src = to_reference(src)
    ref_out = torch.select_scatter(ref_inp, dim=dim, index=index, src=ref_src)
    with flag_gems.use_gems():
        res_out = torch.select_scatter(inp, dim=dim, index=index, src=src)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_scatter
@pytest.mark.parametrize(("dim", "shape", "stride"), REGULAR_DIM_SHAPE_STRIDES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("start", [16, 64])
@pytest.mark.parametrize("end", [1024, 256])
@pytest.mark.parametrize("step", [1, 2])
def test_accuracy_slice_scatter(shape, stride, dim, dtype, start, end, step):
    inp = torch.empty_strided(shape, stride, dtype=dtype, device=flag_gems.device)
    inp.copy_(1)

    valid_shape = list(inp.shape)
    size = valid_shape[dim]

    start = start % size
    end = end % (size + 1)

    if end < start:
        end, start = start, end
    elif end == start:
        end = size

    valid_shape[dim] = (end - start + step - 1) // step

    src = torch.rand(valid_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(inp)
    ref_src = to_reference(src)
    ref_out = torch.slice_scatter(
        ref_inp, dim=dim, src=ref_src, start=start, end=end, step=step
    )

    if flag_gems.vendor_name == "kunlunxin":
        from flag_gems.runtime.backend._kunlunxin import ops as kl_ops

        res_out = kl_ops.slice_scatter(
            inp, dim=dim, src=src, start=start, end=end, step=step
        )
    elif flag_gems.vendor_name == "cambricon":
        from flag_gems.runtime.backend._cambricon import ops as cam_ops

        res_out = cam_ops.slice_scatter(
            inp, dim=dim, src=src, start=start, end=end, step=step
        )
    else:
        res_out = flag_gems.ops.slice_scatter(
            inp, dim=dim, src=src, start=start, end=end, step=step
        )

    gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_scatter
def test_accuracy_slice_scatter_with_self_overlapping_input():
    inp = torch.randn((3, 1), device=flag_gems.device).broadcast_to((3, 8))
    src = torch.rand((3, 4), device=flag_gems.device)

    start = 0
    end = 8
    step = 2
    dim = 1
    ref_inp = to_reference(inp)
    ref_src = to_reference(src)
    ref_out = torch.slice_scatter(
        ref_inp, dim=dim, src=ref_src, start=start, end=end, step=step
    )
    if flag_gems.vendor_name == "kunlunxin":
        from flag_gems.runtime.backend._kunlunxin import ops as kl_ops

        res_out = kl_ops.slice_scatter(
            inp, dim=dim, src=src, start=start, end=end, step=step
        )
    else:
        res_out = flag_gems.ops.slice_scatter(
            inp, dim=dim, src=src, start=start, end=end, step=step
        )

    gems_assert_equal(res_out, ref_out)


@pytest.mark.index_add
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_index_add(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max
    index = torch.randperm(index_len, device=flag_gems.device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    alpha = 2

    ref_inp = to_reference(inp)
    ref_src = to_reference(src)
    ref_index = to_reference(index)
    ref_out = torch.index_add(ref_inp, dim, ref_index, ref_src, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.index_add(inp, dim, index, src, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype=dtype, reduce_dim=dim)


@pytest.mark.index_add_
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_index_add_(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max
    index = torch.randperm(index_len, device=flag_gems.device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    alpha = 2

    ref_inp = to_reference(inp)
    ref_src = to_reference(src)
    ref_index = to_reference(index)
    ref_inp.index_add_(dim, ref_index, ref_src, alpha=alpha)
    with flag_gems.use_gems():
        inp.index_add_(dim, index, src, alpha=alpha)

    gems_assert_close(inp, ref_inp, dtype=dtype, reduce_dim=dim)


@pytest.mark.index_select
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_index_select(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    index_size = inp.size(dim)
    from math import floor

    index = torch.randint(
        0, index_size, [floor(index_size * 0.8)], device=flag_gems.device
    )

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_out = torch.index_select(ref_inp, dim, ref_index)
    with flag_gems.use_gems():
        res_out = torch.index_select(inp, dim, index)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.skipif(flag_gems.vendor_name == "mthreads", reason="AssertionError")
@pytest.mark.masked_select
@pytest.mark.parametrize("threshold, shape", THRESHOLD_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_masked_select(shape, dtype, threshold):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randn(shape, dtype=dtype, device=flag_gems.device) < threshold

    ref_inp = to_reference(inp)
    ref_mask = to_reference(mask)
    ref_out = torch.masked_select(ref_inp, ref_mask)
    with flag_gems.use_gems():
        res_out = torch.masked_select(inp, mask)

    gems_assert_equal(res_out, ref_out)


SHAPE_CONV1D = [
    ((32, 2, 4), (17, 2, 2)),
    ((32, 15, 6), (17, 15, 2)),
    ((32, 16, 1024), (1024, 16, 8)),
    ((64, 64, 64), (128, 64, 7)),
    ((32, 12, 9), (17, 12, 3)),
    ((32, 6, 6), (64, 6, 2)),
]


@pytest.mark.skipif(flag_gems.vendor_name == "mthreads", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.conv1d
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV1D)
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("padding", [1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_accuracy_conv1d(shape, kernel, stride, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, True)
    ref_out = torch.nn.functional.conv1d(
        ref_inp, ref_weight, bias=None, stride=stride, padding=padding, dilation=1
    )

    res_out = flag_gems.conv1d(
        inp, weight, bias=None, stride=stride, padding=padding, dilation=1
    )
    gems_assert_close(res_out, ref_out, dtype)


SHAPE_CONV2D = [
    ((1, 2, 5, 5), (1, 2, 3, 3), 1),
    ((2, 3, 9, 9), (1, 3, 3, 3), 1),
    ((2, 2, 3, 3), (1, 2, 2, 2), 1),
    ((32, 8, 8, 8), (32, 8, 2, 2), 1),
    ((18, 16, 4, 4), (16, 16, 2, 2), 1),
    ((9, 16, 4, 4), (128, 4, 2, 2), 4),
    ((32, 16, 8, 8), (32, 4, 4, 4), 4),
    ((18, 16, 4, 4), (16, 8, 2, 2), 2),
    ((9, 16, 4, 4), (128, 8, 2, 2), 2),
    ((32, 8, 8, 8), (32, 8, 3, 3), 1),
    ((18, 16, 5, 5), (16, 16, 3, 3), 1),
    ((9, 16, 7, 7), (128, 4, 3, 3), 4),
    ((32, 16, 9, 9), (32, 4, 5, 5), 4),
    ((18, 16, 11, 11), (16, 8, 3, 3), 2),
    ((9, 16, 6, 6), (128, 8, 3, 3), 2),
]


@pytest.mark.skipif(flag_gems.vendor_name == "mthreads", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RESULT TODOFIX")
@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.conv2d
@pytest.mark.parametrize("shape, kernel,groups", SHAPE_CONV2D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_accuracy_conv2d(shape, kernel, stride, padding, groups, dtype, dilation, bias):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(
        kernel, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    if bias is True:
        bias = torch.randn(
            [weight.shape[0]], dtype=dtype, device=flag_gems.device, requires_grad=True
        )
        bias_ref = to_reference(bias, True)
    else:
        bias = None
        bias_ref = None

    ref_weight = to_reference(weight, True)
    ref_out = torch.nn.functional.conv2d(
        ref_inp,
        ref_weight,
        bias=bias_ref,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    ).to(dtype)

    res_out = flag_gems.conv2d(
        inp,
        weight,
        bias=bias,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(ref_out).to(flag_gems.device)

    ref_grad = to_reference(out_grad, True)
    if bias is not None:
        (ref_in_grad, ref_weight_grad, ref_bias_grad) = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight, bias_ref), ref_grad
        )
        (res_in_grad, res_weight_grad, res_bias_grad) = torch.autograd.grad(
            res_out, (inp, weight, bias), out_grad
        )
    else:
        (ref_in_grad, ref_weight_grad) = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight), ref_grad
        )
        (res_in_grad, res_weight_grad) = torch.autograd.grad(
            res_out, (inp, weight), out_grad
        )

    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=weight.shape[2])

    gems_assert_close(
        res_weight_grad, ref_weight_grad, dtype, reduce_dim=weight.shape[0]
    )
    if bias is not None:
        gems_assert_close(res_bias_grad, ref_bias_grad, dtype)


SHAPE_CONV3D = [
    ((1, 2, 5, 5, 5), (1, 2, 3, 3, 3), 1),
    ((2, 3, 9, 9, 9), (1, 3, 3, 3, 3), 1),
    ((2, 2, 3, 3, 3), (1, 2, 2, 2, 2), 1),
    ((32, 8, 8, 8, 8), (32, 8, 2, 2, 2), 1),
    ((18, 16, 4, 4, 4), (16, 16, 2, 2, 2), 1),
    ((9, 16, 4, 4, 4), (128, 4, 2, 2, 2), 4),
    ((32, 16, 8, 8, 8), (32, 4, 4, 4, 4), 4),
    ((18, 16, 4, 4, 4), (16, 8, 2, 2, 2), 2),
    ((9, 16, 4, 4, 4), (128, 8, 2, 2, 2), 2),
    ((32, 8, 8, 8, 8), (32, 8, 3, 3, 3), 1),
    ((18, 16, 5, 5, 5), (16, 16, 3, 3, 3), 1),
    ((9, 16, 7, 7, 7), (128, 4, 3, 3, 3), 4),
    ((32, 16, 9, 9, 9), (32, 4, 5, 5, 5), 4),
    ((18, 16, 11, 11, 11), (16, 8, 3, 3, 3), 2),
    ((9, 16, 6, 6, 6), (128, 8, 3, 3, 3), 2),
]


@pytest.mark.skipif(flag_gems.vendor_name == "mthreads", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.conv3d
@pytest.mark.parametrize("shape, kernel,groups", SHAPE_CONV3D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_accuracy_conv3d(shape, kernel, stride, padding, groups, dtype, dilation, bias):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(
        kernel, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    if bias is True:
        bias = torch.randn(
            [weight.shape[0]], dtype=dtype, device=flag_gems.device, requires_grad=False
        )
        bias_ref = to_reference(bias, True)
    else:
        bias = None
        bias_ref = None

    ref_weight = to_reference(weight, True)
    ref_out = torch.nn.functional.conv3d(
        ref_inp,
        ref_weight,
        bias=bias_ref,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    ).to(dtype)

    res_out = flag_gems.conv3d(
        inp,
        weight,
        bias=bias,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    gems_assert_close(res_out, ref_out, dtype)


SHAPE_DEPTHWISE = [
    ((32, 4, 8, 8), (32, 1, 2, 2), (2, 2)),
    ((18, 16, 4, 4), (16, 1, 2, 2), (2, 2)),
    ((9, 32, 4, 4), (128, 1, 2, 2), (2, 2)),
    ((32, 16, 8, 8), (32, 1, 4, 4), (4, 4)),
    ((18, 8, 4, 4), (16, 1, 2, 2), (2, 2)),
    ((9, 4, 4, 4), (128, 1, 2, 2), (2, 2)),
    ((32, 4, 8, 8), (32, 1, 3, 3), (3, 3)),
    ((18, 16, 13, 13), (16, 1, 5, 5), (5, 5)),
    ((9, 32, 8, 8), (128, 1, 3, 3), (3, 3)),
    ((32, 16, 9, 9), (32, 1, 5, 5), (5, 5)),
    ((18, 8, 7, 7), (16, 1, 3, 3), (3, 3)),
    ((9, 4, 6, 6), (128, 1, 3, 3), (3, 3)),
]


# test for depthwise depends on specific device
@pytest.mark.skip("conv_depthwise2d introduces failures, disable it temporarily")
@pytest.mark.conv_depthwise2d
@pytest.mark.parametrize("shape_input, shape_weight,kernel ", SHAPE_DEPTHWISE)
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("padding", [2])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_depthwise2d(
    shape_input, shape_weight, kernel, stride, padding, dtype
):
    inp = torch.randn(
        shape_input, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_inp = to_reference(inp, False)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(shape_weight, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight, False)
    ref_out = torch._C._nn._conv_depthwise2d(
        ref_inp,
        ref_weight,
        kernel,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=1,
    )

    res_out = flag_gems._conv_depthwise2d(
        inp, weight, kernel, bias=None, stride=stride, padding=padding, dilation=1
    )
    gems_assert_close(res_out, ref_out, dtype)


INDEX_PUT_SHAPE_ACC_FALSE = (
    ((2**28,), ((2**16,),), (2**16,)),
    ((32, 32), ((8,), (8,)), (8,)),
    ((32, 32), ((8,), (2, 8)), (8,)),
    ((32, 32), ((2, 8),), (32,)),
    ((512, 512, 512), ((128,), (128,), (128,)), (128,)),
    ((512, 512, 512), ((2, 128), (128,), (128,)), (128,)),
    ((512, 512, 512), ((2, 128),), (512,)),
    (
        (64, 64, 64),
        (
            (2, 8),
            (2, 8),
        ),
        (2, 8, 64),
    ),
)

INDEX_ACC_SHAPE = (
    ((2**28,), ((2**16,),)),
    ((32, 32), ((8,), (8,))),
    ((32, 32), ((8,), (2, 8))),
    ((32, 32), ((2, 8),)),
    ((512, 512, 512), ((128,), (128,), (128,))),
    ((512, 512, 512), ((2, 128), (128,), (128,))),
    ((512, 512, 512), ((2, 128),)),
    (
        (64, 64, 64),
        (
            (2, 8),
            (2, 8),
        ),
    ),
)


def gen_indices(input_shape, indices_shape, accumulate):
    indices = []
    for i, shape in enumerate(indices_shape):
        index = np.random.choice(
            np.arange(input_shape[i]), size=shape, replace=accumulate
        )
        indices.append(torch.tensor(index, device=flag_gems.device))
    return indices


@pytest.mark.index_put
@pytest.mark.parametrize(
    "input_shape, indices_shape, values_shape", INDEX_PUT_SHAPE_ACC_FALSE
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_index_put_acc_false(input_shape, indices_shape, values_shape, dtype):
    accumulate = False
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    indices = gen_indices(input_shape, indices_shape, accumulate)
    values = torch.randn(
        values_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )

    ref_inp = to_reference(inp)
    ref_indices = [to_reference(index) for index in indices]
    ref_values = to_reference(values)
    ref_out = torch.index_put(ref_inp, ref_indices, ref_values, accumulate)
    out = flag_gems.index_put(inp, indices, values, accumulate)
    gems_assert_close(out, ref_out, dtype)


INDEX_PUT_SHAPE_ACC_TRUE = (
    ((2**28,), ((2**16,),), (2**16,)),
    ((32, 32), ((8,), (8,)), (8,)),
    ((512, 512, 512), ((128,), (128,), (128,)), (128,)),
    ((64, 64, 64), ((2, 8), (2, 8), (2, 8)), (2, 8)),
)


@pytest.mark.index_put
@pytest.mark.parametrize(
    "input_shape, indices_shape, values_shape", INDEX_PUT_SHAPE_ACC_TRUE
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_put_acc_true(input_shape, indices_shape, values_shape, dtype):
    init_seed(0)
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(24)
        torch.mlu.manual_seed_all(24)
    accumulate = True
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    indices = gen_indices(input_shape, indices_shape, accumulate)
    values = torch.randn(
        values_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )

    ref_inp = to_reference(inp, upcast=True)
    ref_indices = [to_reference(index) for index in indices]
    ref_values = to_reference(values, upcast=True)
    ref_out = torch.index_put(ref_inp, ref_indices, ref_values, accumulate)
    out = flag_gems.index_put(inp, indices, values, accumulate)
    gems_assert_close(out, ref_out, dtype)


@pytest.mark.index_put_
@pytest.mark.parametrize(
    "input_shape, indices_shape, values_shape", INDEX_PUT_SHAPE_ACC_FALSE
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_index_put__acc_false(input_shape, indices_shape, values_shape, dtype):
    accumulate = False
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    indices = gen_indices(input_shape, indices_shape, accumulate)
    values = torch.randn(
        values_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )

    ref_inp = to_reference(inp)
    ref_indices = [to_reference(index) for index in indices]
    ref_values = to_reference(values)
    torch.index_put_(ref_inp, ref_indices, ref_values, accumulate)
    flag_gems.index_put_(inp, indices, values, accumulate)
    gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.index_put_
@pytest.mark.parametrize(
    "input_shape, indices_shape, values_shape", INDEX_PUT_SHAPE_ACC_TRUE
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_put__acc_true(input_shape, indices_shape, values_shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
    if flag_gems.vendor_name == "mthreads":
        torch.manual_seed(0)
        torch.musa.manual_seed_all(0)
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)
    accumulate = True
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    indices = gen_indices(input_shape, indices_shape, accumulate)
    values = torch.randn(
        values_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )

    ref_inp = to_reference(inp, upcast=True)
    ref_indices = [to_reference(index) for index in indices]
    ref_values = to_reference(values, upcast=True)
    torch.index_put_(ref_inp, ref_indices, ref_values, accumulate)
    flag_gems.index_put_(inp, indices, values, accumulate)
    if flag_gems.vendor_name == "cambricon" and dtype == torch.float16:
        from .accuracy_utils import to_cpu

        inp = to_cpu(inp, ref_inp)
        ref_inp = ref_inp.to(dtype)
        torch.testing.assert_close(inp, ref_inp, atol=3e-3, rtol=3e-2)
    else:
        gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.index
@pytest.mark.parametrize("input_shape, indices_shape", INDEX_ACC_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
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


@pytest.mark.mse_loss
@pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mse_loss(shape, dtype, reduction):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    dim = 1
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target, True)

    ref_out = torch.nn.functional.mse_loss(ref_inp, ref_target, reduction=reduction)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.mse_loss(inp, target, reduction=reduction)
    gems_assert_close(res_out, ref_out, dtype, equal_nan=True, reduce_dim=shape[dim])


def topk_softmax_torch_reference(gating_output: torch.Tensor, topk: int):
    probs = torch.softmax(gating_output, dim=-1)
    topk_values, topk_indices = torch.topk(
        probs, k=topk, dim=-1, largest=True, sorted=True
    )
    num_tokens = gating_output.shape[0]
    source_rows = torch.arange(topk, device=gating_output.device).view(
        1, -1
    ) * num_tokens + torch.arange(num_tokens, device=gating_output.device).view(-1, 1)
    return topk_values, topk_indices, source_rows


def generate_test_params():
    params = [torch.int32, torch.int64]
    if SkipVersion("torch", ">2.2"):
        params.append(torch.uint32)
    return params


@pytest.mark.skipif(flag_gems.vendor_name == "metax", reason="RunetimeError")
@pytest.mark.topk_softmax
@pytest.mark.parametrize("index_dtype", generate_test_params())
@pytest.mark.parametrize(
    "num_tokens, num_experts, topk",
    [
        (1, 4, 2),
        (4, 8, 2),
        (8, 16, 4),
        (32, 64, 8),
        (128, 128, 16),
        (500, 255, 30),
        (512, 256, 32),
        (1024, 512, 32),
    ],
)
def test_topk_softmax(num_tokens, num_experts, topk, index_dtype):
    if flag_gems.vendor_name == "mthreads" and index_dtype == torch.uint32:
        # torch musa unsupport uint32
        index_dtype = torch.int64

    torch.manual_seed(42)
    device = flag_gems.device

    gating_output = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )

    topk_weights = torch.empty((num_tokens, topk), device=device, dtype=torch.float32)
    topk_indices = torch.empty((num_tokens, topk), device=device, dtype=index_dtype)
    token_expert_indices = torch.empty(
        (num_tokens, topk), device=device, dtype=torch.int32
    )

    topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output)

    ref_weights, ref_indices, ref_source_rows = topk_softmax_torch_reference(
        gating_output, topk
    )

    assert topk_weights.shape == (num_tokens, topk)
    assert topk_indices.shape == (num_tokens, topk)
    assert token_expert_indices.shape == (num_tokens, topk)

    assert torch.allclose(topk_weights, ref_weights, atol=1e-5)
    assert torch.equal(topk_indices.cpu(), ref_indices.to(index_dtype).cpu())
    assert torch.equal(token_expert_indices.cpu(), ref_source_rows.cpu())
