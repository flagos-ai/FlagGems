import pytest
import torch

import flag_gems

from .accuracy_utils import (
    DISTRIBUTION_SHAPES,
    FLOAT_DTYPES,
    POINTWISE_SHAPES,
    gems_assert_equal,
)


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_rand(shape, dtype):
    with flag_gems.use_gems():
        res_out = torch.rand(shape, dtype=dtype, device="cuda")
    assert (res_out <= 1.0).all()
    assert (res_out >= 0.0).all()


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_randn(shape, dtype):
    with flag_gems.use_gems():
        res_out = torch.randn(shape, dtype=dtype, device="cuda")
    mean = torch.mean(res_out)
    std = torch.std(res_out)
    assert torch.abs(mean) < 0.01
    assert torch.abs(std - 1) < 0.01


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rand_like(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device="cuda")
    with flag_gems.use_gems():
        res_out = torch.rand_like(x)
    assert (res_out <= 1.0).all()
    assert (res_out >= 0.0).all()


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_zeros(shape, dtype):
    with flag_gems.use_gems():
        res_out = torch.zeros(shape, dtype=dtype, device="cuda")
    gems_assert_equal(res_out, torch.zeros(shape, dtype=dtype, device="cuda"))


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_ones(shape, dtype):
    with flag_gems.use_gems():
        res_out = torch.ones(shape, dtype=dtype, device="cuda")
    gems_assert_equal(res_out, torch.ones(shape, dtype=dtype, device="cuda"))


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_full(shape, dtype):
    with flag_gems.use_gems():
        res_out = torch.full(shape, 3.1415926, dtype=dtype, device="cuda")
    gems_assert_equal(res_out, torch.full(shape, 3.1415926, dtype=dtype, device="cuda"))


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_zeros_like(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device="cuda")
    with flag_gems.use_gems():
        res_out = torch.zeros_like(x)
    gems_assert_equal(res_out, torch.zeros_like(x))


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_ones_like(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device="cuda")
    with flag_gems.use_gems():
        res_out = torch.ones_like(x)
    gems_assert_equal(res_out, torch.ones_like(x))


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_full_like(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device="cuda")
    with flag_gems.use_gems():
        res_out = torch.full_like(x, 3.1415926)
    gems_assert_equal(res_out, torch.full_like(x, 3.1415926))
