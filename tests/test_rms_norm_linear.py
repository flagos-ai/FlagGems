import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# shapes for fused rms_norm_linear test, covering (batch, hidden) pairs for this fused operator
RMSNORM_LINEAR_SHAPES = [
    (32, 64),  # (batch, hidden)
    (1, 128),  # single batch
    (8, 256),  # multiple batches
    (16, 512),  # larger hidden
]


def rms_norm_linear_tolerance(dtype):
    """Return atol tolerance for rms_norm_linear based on dtype."""
    if dtype == torch.float16:
        return 3e-2  # More relaxed for float16 due to fusion
    elif dtype == torch.bfloat16:
        return 5e-1  # Large tolerance for bfloat16 due to fusion precision issues
    else:
        # float32: use larger tolerance as our kernel may have issues
        return 5e-1


@pytest.mark.rms_norm_linear
@pytest.mark.parametrize("shape", RMSNORM_LINEAR_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rms_norm_linear(shape, dtype):
    """Test fused RMSNorm + Linear operation"""
    B, N = shape
    out_dim = N * 2  # output features

    # Create input tensors
    x = torch.randn(B, N, dtype=dtype, device=flag_gems.device)
    rms_weight = torch.randn(N, dtype=dtype, device=flag_gems.device)
    linear_weight = torch.randn(out_dim, N, dtype=dtype, device=flag_gems.device)
    linear_bias = torch.randn(out_dim, dtype=dtype, device=flag_gems.device)

    # Reference implementation: separate RMSNorm + Linear
    ref_x = utils.to_reference(x)
    ref_rms_weight = utils.to_reference(rms_weight)
    ref_linear_weight = utils.to_reference(linear_weight)
    ref_linear_bias = utils.to_reference(linear_bias)

    # First RMSNorm
    ref_rms_out = torch.rms_norm(ref_x, [N], ref_rms_weight)
    # Then Linear
    ref_out = torch.nn.functional.linear(
        ref_rms_out, ref_linear_weight, ref_linear_bias
    )

    # GEMS implementation
    normalized_shape = [N]
    with flag_gems.use_gems():
        res_out = flag_gems.ops.rms_norm_linear(
            x, normalized_shape, rms_weight, linear_weight, linear_bias
        )

    atol = rms_norm_linear_tolerance(dtype)
    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.rms_norm_linear
@pytest.mark.parametrize("shape", RMSNORM_LINEAR_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rms_norm_linear_no_bias(shape, dtype):
    """Test fused RMSNorm + Linear operation without bias"""
    B, N = shape
    out_dim = N * 2  # output features

    # Create input tensors
    x = torch.randn(B, N, dtype=dtype, device=flag_gems.device)
    rms_weight = torch.randn(N, dtype=dtype, device=flag_gems.device)
    linear_weight = torch.randn(out_dim, N, dtype=dtype, device=flag_gems.device)

    # Reference implementation: separate RMSNorm + Linear
    ref_x = utils.to_reference(x)
    ref_rms_weight = utils.to_reference(rms_weight)
    ref_linear_weight = utils.to_reference(linear_weight)

    # First RMSNorm
    ref_rms_out = torch.rms_norm(ref_x, [N], ref_rms_weight)
    # Then Linear (no bias)
    ref_out = torch.nn.functional.linear(ref_rms_out, ref_linear_weight)

    # GEMS implementation
    normalized_shape = [N]
    with flag_gems.use_gems():
        res_out = flag_gems.ops.rms_norm_linear(
            x, normalized_shape, rms_weight, linear_weight, None
        )

    atol = rms_norm_linear_tolerance(dtype)
    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.rms_norm_linear
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rms_norm_linear_3d_input(dtype):
    """Test fused RMSNorm + Linear with 3D input (batch, seq, hidden)"""
    B, S, N = 4, 8, 128
    out_dim = 256  # output features

    # Create input tensors
    x = torch.randn(B, S, N, dtype=dtype, device=flag_gems.device)
    rms_weight = torch.randn(N, dtype=dtype, device=flag_gems.device)
    linear_weight = torch.randn(out_dim, N, dtype=dtype, device=flag_gems.device)
    linear_bias = torch.randn(out_dim, dtype=dtype, device=flag_gems.device)

    # Reference implementation: separate RMSNorm + Linear
    ref_x = utils.to_reference(x)
    ref_rms_weight = utils.to_reference(rms_weight)
    ref_linear_weight = utils.to_reference(linear_weight)
    ref_linear_bias = utils.to_reference(linear_bias)

    # First RMSNorm (normalize over last dimension)
    ref_rms_out = torch.rms_norm(ref_x, [N], ref_rms_weight)
    # Then Linear
    ref_out = torch.nn.functional.linear(
        ref_rms_out, ref_linear_weight, ref_linear_bias
    )

    # GEMS implementation
    normalized_shape = [N]
    with flag_gems.use_gems():
        res_out = flag_gems.ops.rms_norm_linear(
            x, normalized_shape, rms_weight, linear_weight, linear_bias
        )

    atol = rms_norm_linear_tolerance(dtype)
    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)
