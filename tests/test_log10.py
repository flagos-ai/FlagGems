import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    POINTWISE_SHAPES,
    gems_assert_close,
    to_reference,
)


@pytest.mark.log10
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_log10(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    
    ref_inp = to_reference(inp, True)
    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)
    
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.log10_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_log10_(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.clone(), True)
    
    ref_out = torch.log10_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10_(inp)
    
    gems_assert_close(res_out, ref_out, dtype)
