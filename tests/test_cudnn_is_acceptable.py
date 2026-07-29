import pytest
import torch

import flag_gems


@pytest.mark.cudnn_is_acceptable
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_cudnn_is_acceptable_cuda(dtype):
    """Test cudnn_is_acceptable returns True for valid CUDA float tensors."""
    inp = torch.randn(64, 64, dtype=dtype, device=flag_gems.device)

    ref_result = torch.backends.cudnn.is_available() and torch.backends.cudnn.enabled
    with flag_gems.use_gems():
        result = torch.ops.aten.cudnn_is_acceptable.default(inp)

    assert result == ref_result, f"Expected {ref_result}, got {result}"


@pytest.mark.cudnn_is_acceptable
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool])
def test_cudnn_is_acceptable_non_float(dtype):
    """Test cudnn_is_acceptable returns False for non-float dtypes."""
    inp = torch.zeros(64, 64, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        result = torch.ops.aten.cudnn_is_acceptable.default(inp)

    assert result is False, f"Expected False for {dtype}, got {result}"
