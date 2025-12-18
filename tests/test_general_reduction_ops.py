import pytest
import torch

import flag_gems

from .accuracy_utils import (
    ALL_FLOAT_DTYPES,
    ALL_INT_DTYPES,
    FLOAT_DTYPES,
    POINTWISE_SHAPES,
    REDUCTION_SHAPES,
    REDUCTION_SMALL_SHAPES,
    SkipVersion,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
)
from .conftest import QUICK_MODE

FLOAT_DTYPES = [torch.float32] if QUICK_MODE else FLOAT_DTYPES
DIM_LIST = [1] if QUICK_MODE else [0, 1]
DIMS_LIST = [1] if QUICK_MODE else [0, 1, [0, 1], [1, 0]]
KIND_KEEPDIM_DIMS_SHAPE = (
    [("normal", True, DIMS_LIST[0], REDUCTION_SHAPES[0])]
    if QUICK_MODE
    else list(
        zip(
            ["normal", "allTrue"] * 2,
            [True, False] * 2,
            DIMS_LIST,
            REDUCTION_SHAPES + [(7, 4, 11, 1)],
        )
    )
)
KEEPDIM_DIMS = (
    [(True, DIMS_LIST[0])] if QUICK_MODE else list(zip([True, False] * 2, DIMS_LIST))
)
KEEPDIM_DIM = (
    [(True, DIM_LIST[0])] if QUICK_MODE else list(zip([True, False], DIM_LIST))
)


@pytest.mark.all
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allTrue"])
def test_accuracy_all_without_dim(shape, dtype, kind):
    if kind == "allTrue":
        inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.all(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.all(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.all
@pytest.mark.skipif(SkipVersion("torch", "<2.2") and flag_gems.vendor_name != "kunlunxin", reason="Skipping Pytorch version.")
@pytest.mark.parametrize("kind, keepdim, dim, shape", KIND_KEEPDIM_DIMS_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
def test_accuracy_all_dims(shape, dim, keepdim, dtype, kind):
    if kind == "allTrue":
        inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.all(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.all(inp, dim=dim, keepdim=keepdim)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.allclose
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_FLOAT_DTYPES + ALL_INT_DTYPES)
@pytest.mark.parametrize("equal_nan", [False, True])
@pytest.mark.parametrize("gen_nan", [0, 1, 2, 3, 4])
def test_accuracy_allclose(shape, dtype, equal_nan, gen_nan):
    # [gen_nan] 1: nan, 2: inf, 3: -inf, 4: inf vs -inf
    rtol = torch.rand(1, dtype=torch.float32, device=flag_gems.device).item() * (
        0.0001 if dtype in [torch.bfloat16, torch.float16] else 0.01
    )
    if dtype in ALL_FLOAT_DTYPES:
        atol = (
            torch.finfo(dtype).tiny
            * torch.randint(0, 4, (1,), device=flag_gems.device).item()
        )
        inp1 = torch.full(shape, 1.234, dtype=dtype, device=flag_gems.device)
        inp2 = torch.full(shape, 1.234, dtype=dtype, device=flag_gems.device)
        if gen_nan:
            nan_num = torch.full(
                (1,),
                float("nan" if gen_nan == 1 else "inf"),
                dtype=dtype,
                device=flag_gems.device,
            )
            # FIXME: Neg doesn't support double on torch_musa, so workaround temporarily.
            inp1.view(-1)[0] = (
                (-nan_num.cpu()).to(flag_gems.device) if gen_nan == 3 else nan_num
            )
            inp2.view(-1)[0] = (
                (-nan_num.cpu()).to(flag_gems.device) if gen_nan >= 3 else nan_num
            )
    else:
        atol = (
            torch.finfo(torch.float16).eps
            * torch.randint(0, 10, (1,), device=flag_gems.device).item()
        )
        inp1 = torch.randint(-1000, 1000, shape, device=flag_gems.device).to(dtype)
        inp2 = torch.randint(-1000, 1000, shape, device=flag_gems.device).to(dtype)

    ref_inp1 = to_reference(inp1, False)
    ref_inp2 = to_reference(inp2, False)

    with flag_gems.use_gems():
        res_out = torch.allclose(inp1, inp2, rtol, atol, equal_nan=equal_nan)
    ref_out = torch.allclose(ref_inp1, ref_inp2, rtol, atol, equal_nan=equal_nan)

    assert res_out == ref_out


@pytest.mark.any
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allFalse"])
def test_accuracy_any_without_dim(shape, dtype, kind):
    if kind == "allFalse":
        inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.any(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.any(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.any
@pytest.mark.skipif(SkipVersion("torch", "<2.2") and flag_gems.vendor_name != "kunlunxin", reason="Skipping Pytorch version.")
@pytest.mark.parametrize("kind, keepdim, dim, shape", KIND_KEEPDIM_DIMS_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
def test_accuracy_any_dims(shape, dim, keepdim, dtype, kind):
    if kind == "allFalse":
        inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.any(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.any(inp, dim=dim, keepdim=keepdim)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.max
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + ALL_INT_DTYPES)
def test_accuracy_max_without_dim(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = to_reference(inp)

    ref_out = torch.max(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.max(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.max
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_max_without_dim_all_neg_inf(shape, dtype):
    inp = torch.full(
        shape, fill_value=float("-inf"), dtype=dtype, device=flag_gems.device
    )
    ref_inp = to_reference(inp)

    ref_out = torch.max(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.max(inp)

    gems_assert_equal(res_out, ref_out)


# cambricon add
@pytest.mark.max
@pytest.mark.skipif(
    flag_gems.vendor_name != "cambricon" and flag_gems.vendor_name != "metax",
    reason="cambricon and metax test only",
)
@pytest.mark.parametrize("shape", REDUCTION_SHAPES + [[1]])
@pytest.mark.parametrize("dtype", ALL_INT_DTYPES)
def test_accuracy_max_int(shape, dtype):
    inp = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    ref_inp = to_reference(inp)

    ref_out = torch.max(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.max(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.max
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + ALL_INT_DTYPES)
def test_accuracy_max_without_dim_uncontiguous(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device="cpu")[::2, ::2].to(
            flag_gems.device
        )
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu")[
            ::2, ::2
        ].to(flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.max(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.max(inp)

    gems_assert_equal(res_out, ref_out)


# TODO: failed at (200, 40999, 3), while successed at this shape in mean_dim
@pytest.mark.max
@pytest.mark.parametrize("shape", REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("keepdim, dim", KEEPDIM_DIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + ALL_INT_DTYPES)
def test_accuracy_max_dim(shape, dim, keepdim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = to_reference(inp)

    ref_out_value, ref_out_index = torch.max(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out_value, res_out_index = torch.max(inp, dim=dim, keepdim=keepdim)

    gems_assert_equal(res_out_index, ref_out_index)
    gems_assert_equal(res_out_value, ref_out_value)


@pytest.mark.max
@pytest.mark.skipif(
    flag_gems.vendor_name == "aipu",
    reason="Big shape run slowly.",
)
@pytest.mark.parametrize("shape", [(4, 1048577, 4)])
@pytest.mark.parametrize("keepdim, dim", [(True, 1), (False, 1)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + ALL_INT_DTYPES)
def test_accuracy_max_dim_big_shape(shape, dim, keepdim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = to_reference(inp)

    ref_out_value, ref_out_index = torch.max(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out_value, res_out_index = torch.max(inp, dim=dim, keepdim=keepdim)

    gems_assert_equal(res_out_index, ref_out_index)
    gems_assert_equal(res_out_value, ref_out_value)


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


@pytest.mark.mean
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("keepdim, dim", KEEPDIM_DIMS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mean_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.mean(ref_inp, dim, keepdim)
    with flag_gems.use_gems():
        res_out = torch.mean(inp, dim, keepdim)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.min
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + ALL_INT_DTYPES)
def test_accuracy_min_without_dim(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = to_reference(inp)

    ref_out = torch.min(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.min(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.min
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_min_without_dim_all_inf(shape, dtype):
    # ensure that padding value used in min is inf, not max value
    inp = torch.full(
        shape, fill_value=float("inf"), dtype=dtype, device=flag_gems.device
    )
    ref_inp = to_reference(inp)

    ref_out = torch.min(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.min(inp)

    gems_assert_equal(res_out, ref_out)


# TODO: failed at (200, 40999, 3), while successed at this shape in mean_dim
@pytest.mark.min
@pytest.mark.parametrize("shape", REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("keepdim, dim", KEEPDIM_DIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + ALL_INT_DTYPES)
def test_accuracy_min_dim(shape, dim, keepdim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = to_reference(inp)

    ref_out_value, ref_out_index = torch.min(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out_value, res_out_index = torch.min(inp, dim=dim, keepdim=keepdim)

    gems_assert_equal(res_out_index, ref_out_index)
    gems_assert_equal(res_out_value, ref_out_value)


@pytest.mark.prod
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_prod_without_dim(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.prod(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.prod(inp)

    gems_assert_close(res_out, ref_out, dtype)


# TODO: failed at (200, 40999, 3), while successed at this shape in mean_dim
@pytest.mark.prod
@pytest.mark.parametrize("shape", REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("keepdim, dim", KEEPDIM_DIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_prod_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.prod(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.prod(inp, dim=dim, keepdim=keepdim)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sum
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sum_without_dim(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.sum(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sum(inp)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.numel())


@pytest.mark.sum
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("keepdim, dim", KEEPDIM_DIM + [(False, []), (True, [])])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sum_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.sum(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.sum(inp, dim=dim, keepdim=keepdim)

    if isinstance(dim, int):
        dim = [dim]
    dim = [d % inp.ndim for d in dim]
    _dim = 1
    for d in dim:
        _dim *= shape[d]
    if dim == []:
        _dim = inp.numel()
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=_dim)


QUANTILE_SHAPES = REDUCTION_SMALL_SHAPES + [(10, 64, 196), (65535, 1)]
QUANTILE_FLOAT_DTYPES = [torch.float32]
QUANTILE_Q = (
    [(0.2, 0.5, 0.8)]
    if QUICK_MODE
    else [(0.4), (0.0, 0.2, 0.5, 0.8, 1.0), (0.662, 0.8, 0.104, 0.99, 0.347, 0.255)]
)
QUANTILE_INTERPOLATION = (
    ["linear"] if QUICK_MODE else ["linear", "lower", "higher", "nearest", "midpoint"]
)


@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RESULT TODOFIX")
@pytest.mark.skipif(SkipVersion("triton", "<3.0"), reason="Skipping Triton version.")
@pytest.mark.quantile
@pytest.mark.parametrize("shape", QUANTILE_SHAPES)
@pytest.mark.parametrize("dtype", QUANTILE_FLOAT_DTYPES)
@pytest.mark.parametrize("q", QUANTILE_Q)
@pytest.mark.parametrize("interpolation", QUANTILE_INTERPOLATION)
def test_accuracy_quantile_without_dim(shape, dtype, q, interpolation):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    q = torch.tensor(q, dtype=dtype, device=inp.device)
    ref_q = to_reference(q)

    ref_out = torch.quantile(ref_inp, ref_q, interpolation=interpolation)
    with flag_gems.use_gems():
        res_out = torch.quantile(inp, q, interpolation=interpolation)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.numel())


@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RESULT TODOFIX")
@pytest.mark.skipif(SkipVersion("triton", "<3.0"), reason="Skipping Triton version.")
@pytest.mark.quantile
@pytest.mark.parametrize("shape", QUANTILE_SHAPES)
@pytest.mark.parametrize("keepdim, dim", KEEPDIM_DIM)
@pytest.mark.parametrize("dtype", QUANTILE_FLOAT_DTYPES)
@pytest.mark.parametrize("q", QUANTILE_Q)
@pytest.mark.parametrize("interpolation", QUANTILE_INTERPOLATION)
def test_accuracy_quantile_dim(shape, dim, keepdim, dtype, q, interpolation):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    q = torch.tensor(q, dtype=dtype, device=inp.device)
    ref_q = to_reference(q)

    ref_out = torch.quantile(
        ref_inp, ref_q, dim=dim, keepdim=keepdim, interpolation=interpolation
    )
    with flag_gems.use_gems():
        res_out = torch.quantile(
            inp, q, dim=dim, keepdim=keepdim, interpolation=interpolation
        )

    if isinstance(dim, int):
        dim = [dim]
    dim = [d % inp.ndim for d in dim]
    _dim = 1
    for d in dim:
        _dim *= shape[d]
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=_dim)
