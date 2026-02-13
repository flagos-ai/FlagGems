import pytest
import torch

import flag_gems

# 1. Improved import grouping (Standard -> Third-party -> Internal)
# 2. All comments converted to English as requested.

def get_tolerances(dtype):
    """
    Returns the absolute and relative tolerances based on the 
    official FlagGems precision standards.
    """
    if dtype == torch.float16:
        return {"atol": 1e-3, "rtol": 1e-4}
    elif dtype == torch.float32:
        return {"atol": 1.3e-6, "rtol": 1e-4}
    elif dtype == torch.bfloat16:
        return {"atol": 0.016, "rtol": 1e-4}
    return {"atol": 1e-5, "rtol": 1e-4}

@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [
    (1, 1), 
    (1024, 1024), 
    (16, 3, 224, 224)
])
def test_accuracy_asinh(dtype, shape):
    """
    Verify correctness of asinh across various scales and data types.
    """
    x = torch.randn(shape, device="cuda", dtype=dtype)
    
    # Official PyTorch reference implementation
    ref_out = torch.asinh(x)
    
    # FlagGems Triton-optimized implementation
    with flag_gems.use_gems():
        res_out = torch.asinh(x)
        
    tol = get_tolerances(dtype)
    torch.testing.assert_close(res_out, ref_out, **tol, equal_nan=True)

@pytest.mark.parametrize("dtype", [torch.float32])
def test_asinh_edge_cases(dtype):
    """
    Ensure the operator handles special IEEE 754 values correctly.
    """
    x = torch.tensor([float('nan'), float('inf'), float('-inf'), 0.0], device="cuda", dtype=dtype)
    
    ref_out = torch.asinh(x)
    with flag_gems.use_gems():
        res_out = torch.asinh(x)
        
    torch.testing.assert_close(res_out, ref_out, equal_nan=True)