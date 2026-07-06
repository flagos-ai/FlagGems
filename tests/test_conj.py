import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.conj_physical
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_conj_physical(shape, dtype):
    float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    real = torch.randn(shape, dtype=float_dtype, device=flag_gems.device)
    imag = torch.randn(shape, dtype=float_dtype, device=flag_gems.device)
    inp = torch.complex(real, imag).to(dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.conj_physical(ref_inp)
    with flag_gems.use_gems():
        res_out = flag_gems.conj_physical(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.conj_physical
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_conj_physical_real_input(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.conj_physical(ref_inp)
    with flag_gems.use_gems():
        res_out = flag_gems.conj_physical(inp)

    utils.gems_assert_equal(res_out, ref_out)
