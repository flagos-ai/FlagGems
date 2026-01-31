import logging
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
@pytest.mark.parametrize(shape, POINTWISE_SHAPES)
@pytest.mark.parametrize(dtype, FLOAT_DTYPES)
def test_accuracy_log10(shape, dtype):
    """Test log10 accuracy across different shapes and dtypes"""
    # Generate positive random inputs (log10 requires positive inputs)
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 100 + 1e-6
    ref_inp = to_reference(inp, True)

    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    gems_assert_close(res_out, ref_out, dtype, rtol=1e-4, atol=1.3e-6)


@pytest.mark.log10
@pytest.mark.parametrize(dtype, FLOAT_DTYPES)
def test_accuracy_log10_known_values(dtype):
    """Test log10 with known mathematical values"""
    # Test known values: log10(0.001)=-3, log10(0.1)=-1, log10(1)=0, log10(10)=1, log10(100)=2, log10(1000)=3
    known_values = torch.tensor([0.001, 0.1, 1.0, 10.0, 100.0, 1000.0], dtype=dtype, device=flag_gems.device)
    expected = torch.tensor([-3.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=dtype, device=flag_gems.device)
    
    ref_inp = to_reference(known_values, True)
    ref_out = torch.log10(ref_inp)
    
    with flag_gems.use_gems():
        res_out = torch.log10(known_values)

    gems_assert_close(res_out, ref_out, dtype, rtol=1e-4, atol=1.3e-6)


@pytest.mark.log10
def test_accuracy_log10_edge_cases():
    """Test log10 with edge cases"""
    # Test very small and very large values
    edge_cases = torch.tensor([1e-10, 1e-5, 1.0, 1e5, 1e10], dtype=torch.float32, device=flag_gems.device)
    ref_inp = to_reference(edge_cases, True)

    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(edge_cases)

    gems_assert_close(res_out, ref_out, torch.float32, rtol=1e-4, atol=1.3e-6)


@pytest.mark.log10
@pytest.mark.parametrize(shape, [(64, 64), (256, 256), (1024, 1024)])
@pytest.mark.parametrize(dtype, [torch.float16, torch.float32])
def test_performance_log10(shape, dtype):
    """Performance test for log10 operator across different sizes"""
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 100 + 1e-6
    ref_inp = to_reference(inp, True)

    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    gems_assert_close(res_out, ref_out, dtype, rtol=1e-4, atol=1.3e-6)
