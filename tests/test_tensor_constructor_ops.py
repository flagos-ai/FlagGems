import math

import pytest
import torch
from packaging import version

import flag_gems

from .accuracy_utils import (
    ALL_FLOAT_DTYPES,
    ALL_INT_DTYPES,
    BOOL_TYPES,
    DISTRIBUTION_SHAPES,
    FLOAT_DTYPES,
    POINTWISE_SHAPES,
    gems_assert_equal,
    to_reference,
)
from .conftest import TO_CPU

device = flag_gems.device


@pytest.mark.rand
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rand(shape, dtype):
    with flag_gems.use_gems():
        res_out = torch.rand(shape, dtype=dtype, device=device)
    ref_out = to_reference(res_out)
    assert (ref_out <= 1.0).all()
    assert (ref_out >= 0.0).all()


@pytest.mark.randn
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_randn(shape, dtype):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
    with flag_gems.use_gems():
        res_out = torch.randn(shape, dtype=dtype, device=device)
    ref_out = to_reference(res_out)
    mean = torch.mean(ref_out)
    std = torch.std(ref_out)
    assert torch.abs(mean) < 0.01
    assert torch.abs(std - 1) < 0.01


@pytest.mark.rand_like
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rand_like(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=device)
    with flag_gems.use_gems():
        res_out = torch.rand_like(x)
    ref_out = to_reference(res_out)
    assert (ref_out <= 1.0).all()
    assert (ref_out >= 0.0).all()


@pytest.mark.randn_like
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_randn_like(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=device)
    with flag_gems.use_gems():
        res_out = torch.randn_like(x)
    ref_out = to_reference(res_out)
    mean = torch.mean(ref_out)
    std = torch.std(ref_out)
    assert torch.abs(mean) < 0.01
    assert torch.abs(std - 1) < 0.01


@pytest.mark.zeros
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", BOOL_TYPES + ALL_INT_DTYPES + ALL_FLOAT_DTYPES)
def test_accuracy_zeros(shape, dtype):
    # without dtype
    with flag_gems.use_gems():
        res_out = torch.zeros(shape, device=flag_gems.device)
    gems_assert_equal(res_out, torch.zeros(shape, device="cpu" if TO_CPU else device))

    # with dtype
    with flag_gems.use_gems():
        res_out = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    gems_assert_equal(
        res_out, torch.zeros(shape, dtype=dtype, device="cpu" if TO_CPU else device)
    )


@pytest.mark.ones
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", BOOL_TYPES + ALL_INT_DTYPES + ALL_FLOAT_DTYPES)
def test_accuracy_ones(shape, dtype):
    # without dtype
    with flag_gems.use_gems():
        res_out = torch.ones(shape, device=flag_gems.device)
    gems_assert_equal(res_out, torch.ones(shape, device="cpu" if TO_CPU else device))

    # with dtype
    with flag_gems.use_gems():
        res_out = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    gems_assert_equal(
        res_out, torch.ones(shape, dtype=dtype, device="cpu" if TO_CPU else device)
    )


@pytest.mark.full
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", BOOL_TYPES + ALL_INT_DTYPES + ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("fill_value", [3.1415926, 2, False])
def test_accuracy_full(shape, dtype, fill_value):
    # without dtype
    ref_out = torch.full(shape, fill_value, device="cpu" if TO_CPU else device)
    with flag_gems.use_gems():
        res_out = torch.full(shape, fill_value, device=flag_gems.device)
    gems_assert_equal(res_out, ref_out)

    # with dtype
    ref_out = torch.full(
        shape, fill_value, dtype=dtype, device="cpu" if TO_CPU else device
    )
    with flag_gems.use_gems():
        res_out = torch.full(shape, fill_value, dtype=dtype, device=flag_gems.device)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.zeros_like
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_zeros_like(shape, dtype):
    inp = torch.empty(size=shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp)
    ref_out = torch.zeros_like(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.zeros_like(inp)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.ones_like
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_ones_like(shape, dtype):
    inp = torch.empty(size=shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp)
    ref_out = torch.ones_like(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ones_like(inp)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.full_like
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", BOOL_TYPES + ALL_INT_DTYPES + ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("xdtype", BOOL_TYPES + ALL_INT_DTYPES + ALL_FLOAT_DTYPES)
@pytest.mark.parametrize(
    "fill_value", [3.1415926, 2, False, float("inf"), float("nan")]
)
def test_accuracy_full_like(shape, dtype, xdtype, fill_value):
    if isinstance(fill_value, float) and (
        math.isinf(fill_value) or math.isnan(fill_value)
    ):
        if dtype not in ALL_FLOAT_DTYPES:
            pytest.skip("Skipping inf/nan test for non-float dtypes")
    inp = torch.empty(size=shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp)

    # without dtype
    ref_out = torch.full_like(ref_inp, fill_value)
    with flag_gems.use_gems():
        res_out = torch.full_like(inp, fill_value)
    gems_assert_equal(res_out, ref_out, equal_nan=True)

    # with dtype
    ref_out = torch.full_like(ref_inp, fill_value, dtype=dtype)
    with flag_gems.use_gems():
        res_out = torch.full_like(inp, fill_value, dtype=dtype)
    gems_assert_equal(res_out, ref_out, equal_nan=True)


# @pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RESULT TODOFIX")
@pytest.mark.randperm
@pytest.mark.parametrize("n", [123, 12345, 123456])
@pytest.mark.parametrize("dtype", ALL_INT_DTYPES)
def test_accuracy_randperm(n, dtype):
    if n > torch.iinfo(torch.int16).max and dtype == torch.int16:
        return

    ref_out = torch.randperm(n, dtype=dtype, device="cpu" if TO_CPU else device)
    with flag_gems.use_gems():
        res_out = torch.randperm(n, dtype=dtype, device=flag_gems.device)
    sorted_ref, _ = torch.sort(ref_out)
    sorted_res, _ = torch.sort(res_out)
    gems_assert_equal(sorted_res, sorted_ref)


@pytest.mark.eye
@pytest.mark.parametrize(
    "shape",
    [
        (256, 1024),
        (1024, 256),
        (8192, 4096),
        (4096, 8192),
    ]
    + [(2**d, 2**d) for d in range(7, 13)],
)
@pytest.mark.parametrize("dtype", ALL_INT_DTYPES + ALL_FLOAT_DTYPES + BOOL_TYPES)
def test_accuracy_eye(shape, dtype):
    if (
        TO_CPU
        and dtype == torch.bfloat16
        and version.parse(torch.__version__) < version.parse("2.5.0")
    ):
        pytest.skip("BFloat16 not supported on CPU in torch<2.5.0")
    n, m = shape

    # test eye(n, m) without dtype
    with flag_gems.use_gems():
        res_out = torch.eye(n, m, device=flag_gems.device)
    gems_assert_equal(res_out, torch.eye(n, m, device="cpu" if TO_CPU else device))

    # with dtype
    with flag_gems.use_gems():
        res_out = torch.eye(n, m, dtype=dtype, device=flag_gems.device)
    gems_assert_equal(
        res_out,
        torch.eye(n, m, dtype=dtype, device="cpu" if TO_CPU else device),
    )

    # test eye(n)
    with flag_gems.use_gems():
        res_out = torch.eye(n, device=flag_gems.device)
    gems_assert_equal(res_out, torch.eye(n, device="cpu" if TO_CPU else device))

    # with dtype
    with flag_gems.use_gems():
        res_out = torch.eye(n, dtype=dtype, device=flag_gems.device)
    gems_assert_equal(
        res_out,
        torch.eye(n, dtype=dtype, device="cpu" if TO_CPU else device),
    )
