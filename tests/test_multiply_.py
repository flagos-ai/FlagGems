import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.multiply_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_multiply_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = ref_inp1.multiply_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.multiply_(inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.multiply_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_multiply_tensor_scalar_(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1.clone(), True)

    ref_out = ref_inp1.multiply_(inp2)
    with flag_gems.use_gems():
        res_out = inp1.multiply_(inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)
