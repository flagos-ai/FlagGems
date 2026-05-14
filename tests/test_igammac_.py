import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.igammac
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_igammac(shape, dtype):
    # igammac requires positive inputs
    # Generate positive random values for a and x
    inp1 = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    inp2 = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.igammac(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.igammac(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.igammac_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_igammac_(shape, dtype):
    # igammac_ is the in-place version
    # Generate positive random values for a and x
    inp1 = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    inp2 = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    ref_inp1 = to_reference(inp1.clone(), True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = ref_inp1.igammac_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.igammac_(inp2)

    gems_assert_close(res_out, ref_out, dtype)
