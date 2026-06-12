import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

if QUICK_MODE:
    # Minimal shapes and dtype for quick validation
    SPARSE_LINEAR_SHAPES = [
        (16, 32),
    ]
    # Use float32 only in quick mode for faster validation
    FLOAT_DTYPES = [torch.float32]
else:
    # Representative shapes covering small to medium matrix dimensions
    SPARSE_LINEAR_SHAPES = [
        (16, 32),
        (32, 64),
        (64, 128),
        (128, 256),
    ]
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.sparse_semi_structured_linear
@pytest.mark.parametrize("M, K", SPARSE_LINEAR_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__sparse_semi_structured_linear(M, K, dtype):
    """Test for sparse semi-structured linear layer.

    Uses meta tensor filled with ones so no elements are masked,
    allowing dense matmul as the reference comparison.
    """
    N = K  # output features equal to input features

    # Use fixed seed for reproducibility
    torch.manual_seed(12345)
    input = torch.randn(M, K, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(N, K, dtype=dtype, device=flag_gems.device)
    meta = torch.ones(N // 4, K, dtype=torch.int8, device=flag_gems.device)

    ref_input = utils.to_reference(input, True)
    ref_weight = utils.to_reference(weight, True)

    # Reference: dense matmul (treating weight as dense since meta=1 everywhere)
    ref_out = torch.matmul(ref_input, ref_weight.t())

    with flag_gems.use_gems():
        res_out = flag_gems._sparse_semi_structured_linear(input, weight, meta)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sparse_semi_structured_linear
@pytest.mark.parametrize("M, K", [(32, 64)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__sparse_semi_structured_linear_with_bias(M, K, dtype):
    """Test sparse semi-structured linear with bias."""
    N = K

    # Use fixed seed for reproducibility
    torch.manual_seed(12345)
    input = torch.randn(M, K, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(N, K, dtype=dtype, device=flag_gems.device)
    bias = torch.randn(N, dtype=dtype, device=flag_gems.device)
    meta = torch.ones(N // 4, K, dtype=torch.int8, device=flag_gems.device)

    ref_input = utils.to_reference(input, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    # Reference: dense matmul + bias
    ref_out = torch.matmul(ref_input, ref_weight.t()) + ref_bias

    with flag_gems.use_gems():
        res_out = flag_gems._sparse_semi_structured_linear(
            input, weight, meta, bias=bias
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
