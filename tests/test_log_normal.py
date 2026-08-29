import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SHAPES = [
    (64, 64),
    (100, 1, 100),
    (10000, 1),
    (100, 256, 100),
    (10000, 256),
    (20, 320, 15),
    (1024, 1024),
]
MOMENT_SHAPES = [(256, 1024), (1024, 1024), (4096, 4096), (64, 512, 512)]
LARGE_SHAPES = [
    (268435456,),
    (1073741824,),
    (10000, 65536),
    (100, 65536, 100),
    (1024, 1024, 1024),
]
MOMENT_PARAMS = [(0.0, 0.5), (1.0, 1.0)]
MOMENT_PARAMS_NO_FP16 = [(1.0, 2.0)]


@pytest.mark.log_normal
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log_normal(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    x_ref = utils.to_reference(x.clone())
    with flag_gems.use_gems():
        y = torch.ops.aten.log_normal(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.data_ptr() != x.data_ptr()
    y_res = utils.to_reference(y)

    assert (y_res > 0).all()

    torch.testing.assert_close(utils.to_reference(x), x_ref)


@pytest.mark.log_normal
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("mean,std", [(0.0, 1.0), (2.0, 0.5), (-1.0, 1.0)])
def test_log_normal_params(dtype, mean, std):
    shape = (100, 256, 100)
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        y = torch.ops.aten.log_normal(x, mean=mean, std=std)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    y_res = utils.to_reference(y)
    assert (y_res > 0).all()

    mean_res = torch.mean(y_res.to(torch.float32))
    expected_mean = torch.tensor(
        np.exp(mean + std**2 / 2), device=mean_res.device, dtype=torch.float32
    )

    mean_tol = 0.15 * expected_mean.item()
    utils.gems_assert_close(mean_res, expected_mean, torch.float32, atol=mean_tol)


@pytest.mark.log_normal
@pytest.mark.parametrize("shape", MOMENT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("mean,std", MOMENT_PARAMS)
def test_log_normal_moments(shape, dtype, mean, std):
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        y = torch.ops.aten.log_normal(x, mean=mean, std=std)

    y_res = utils.to_reference(y).to(torch.float32)
    mean_res = torch.mean(y_res)
    var_res = torch.var(y_res)
    expected_mean = torch.tensor(
        np.exp(mean + std**2 / 2), device=mean_res.device, dtype=torch.float32
    )
    expected_var = torch.tensor(
        (np.exp(std**2) - 1) * np.exp(2 * mean + std**2),
        device=var_res.device,
        dtype=torch.float32,
    )

    mean_tol = 0.15 * expected_mean.item()
    var_tol = 0.25 * expected_var.item()
    utils.gems_assert_close(mean_res, expected_mean, torch.float32, atol=mean_tol)
    utils.gems_assert_close(var_res, expected_var, torch.float32, atol=var_tol)


@pytest.mark.log_normal
@pytest.mark.parametrize("shape", MOMENT_SHAPES)
@pytest.mark.parametrize("dtype", [d for d in utils.FLOAT_DTYPES if d != torch.float16])
@pytest.mark.parametrize("mean,std", MOMENT_PARAMS_NO_FP16)
def test_log_normal_moments_large_std(shape, dtype, mean, std):
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        y = torch.ops.aten.log_normal(x, mean=mean, std=std)

    y_res = utils.to_reference(y).to(torch.float32)
    mean_res = torch.mean(y_res)
    expected_mean = torch.tensor(
        np.exp(mean + std**2 / 2), device=mean_res.device, dtype=torch.float32
    )
    mean_tol = 0.15 * expected_mean.item()
    utils.gems_assert_close(mean_res, expected_mean, torch.float32, atol=mean_tol)


@pytest.mark.log_normal
@pytest.mark.parametrize("shape", LARGE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log_normal_large_shapes(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        y = torch.ops.aten.log_normal(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    y_res = utils.to_reference(y)
    assert (y_res > 0).all()
