import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shapes for matmul-bias-activation tests
# Shape format: (M, N, K) for matmul: (M, K) x (K, N)
MNK_SHAPES = [
    (1, 1, 32),
    (15, 160, 1024),
    (495, 5333, 71),
]


@pytest.mark.matmul_bias_activation
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_matmul_bias_activation(M, N, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skipping fp32 matmul_bias_activation test on tsingmicro platform")

    # Create input tensors
    input_tensor = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)

    # Reference: matmul + bias + relu
    ref_input = utils.to_reference(input_tensor, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.relu(torch.mm(ref_input, ref_weight) + ref_bias)
    with flag_gems.use_gems():
        from flag_gems.ops.matmul_bias_activation import matmul_bias_activation

        res_out = matmul_bias_activation(input_tensor, weight, bias)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)
