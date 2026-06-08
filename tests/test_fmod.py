import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    POINTWISE_SHAPES,
    SCALARS,
    gems_assert_close,
    to_reference,
)


@pytest.mark.fmod_tensor
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fmod_tensor(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.where(inp2 == 0, torch.ones_like(inp2), inp2)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)
    ref_out = torch.fmod(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.fmod(inp1, inp2)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.fmod_scalar
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fmod_scalar(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar if scalar != 0 else 1.0
    ref_inp1 = to_reference(inp1, True)
    ref_out = torch.fmod(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.fmod(inp1, inp2)
    atol = 1e-3 if abs(scalar) < 0.01 else 1e-4
    gems_assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.fmod_tensor_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fmod_tensor_inplace(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.where(inp2 == 0, torch.ones_like(inp2), inp2)
    ref_inp1 = to_reference(inp1.clone(), True)
    ref_inp2 = to_reference(inp2, True)
    ref_out = ref_inp1.fmod_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.fmod_(inp2)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.fmod_scalar_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fmod_scalar_inplace(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar if scalar != 0 else 1.0
    ref_inp1 = to_reference(inp1.clone(), True)
    ref_out = ref_inp1.fmod_(inp2)
    with flag_gems.use_gems():
        res_out = inp1.fmod_(inp2)
    atol = 1e-3 if abs(scalar) < 0.01 else 1e-4
    gems_assert_close(res_out, ref_out, dtype, atol=atol)
