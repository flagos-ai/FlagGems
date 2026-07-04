import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.matmul_layernorm
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_matmul_layernorm(shape, dtype):
    # shape is (*, K) where last dim is input features
    # Weight shape is (N, K) where N is output features
    K = shape[-1]
    N = K  # Use same dimension for simplicity

    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(N, K, dtype=dtype, device=flag_gems.device)
    bias = torch.randn(N, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input)
    ref_weight = utils.to_reference(weight)
    ref_bias = utils.to_reference(bias)

    # Reference: matmul then layer_norm
    matmul_result = torch.matmul(ref_input, ref_weight.t())
    if ref_bias is not None:
        matmul_result = matmul_result + ref_bias
    ref_out = torch.nn.functional.layer_norm(matmul_result, [N], eps=1e-5)

    with flag_gems.use_gems():
        res_out = flag_gems.matmul_layernorm(input, weight, bias, eps=1e-5)

    # Use appropriate tolerance based on dtype (larger for complex fused ops)
    atol = (
        0.01 if dtype == torch.float16 else (0.05 if dtype == torch.bfloat16 else 1e-4)
    )
    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.matmul_layernorm
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_matmul_layernorm_no_bias(shape, dtype):
    # Test without bias
    K = shape[-1]
    N = K

    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(N, K, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input)
    ref_weight = utils.to_reference(weight)

    # Reference: matmul then layer_norm (no bias)
    matmul_result = torch.matmul(ref_input, ref_weight.t())
    ref_out = torch.nn.functional.layer_norm(matmul_result, [N], eps=1e-5)

    with flag_gems.use_gems():
        res_out = flag_gems.matmul_layernorm(input, weight, bias=None, eps=1e-5)

    # Use appropriate tolerance based on dtype (larger for complex fused ops)
    atol = (
        0.01 if dtype == torch.float16 else (0.05 if dtype == torch.bfloat16 else 1e-4)
    )
    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)
