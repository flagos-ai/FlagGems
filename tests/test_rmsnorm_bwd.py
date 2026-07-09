import pytest
import torch

import flag_gems
from flag_gems.ops.rmsnorm_bwd import rmsnorm_bwd
from flag_gems.ops.rmsnorm_fwd import rmsnorm_fwd

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    SHAPES = [(32, 128)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    SHAPES = [(32, 128), (64, 256), (128, 512), (256, 1024), (1024, 2048)]


# Check if TransformerEngine is available
try:
    import transformer_engine.pytorch.cpp_extensions as tex
    from transformer_engine.pytorch import DType as TEDType

    HAS_TE = True

    TORCH_TO_TE_DTYPE = {
        torch.float32: TEDType.kFloat32,
        torch.float16: TEDType.kFloat16,
        torch.bfloat16: TEDType.kBFloat16,
    }
except ImportError:
    HAS_TE = False


def _torch_rmsnorm_bwd(dz, x, weight, eps, zero_centered_gamma=False):
    """Reference implementation for RMSNorm backward."""
    x_fp32 = x.to(torch.float32)
    dz_fp32 = dz.to(torch.float32)
    w_fp32 = weight.to(torch.float32)

    if zero_centered_gamma:
        w_fp32 = w_fp32 + 1.0

    # Recompute forward values
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    rsigma = torch.rsqrt(variance + eps)
    x_hat = x_fp32 * rsigma

    # Compute gradients
    # dx = rsigma * (dz * w - x_hat * mean(x_hat * dz * w))
    dz_w = dz_fp32 * w_fp32
    c1 = (x_hat * dz_w).mean(dim=-1, keepdim=True)
    dx = rsigma * (dz_w - x_hat * c1)

    # dgamma = sum(dz * x_hat) over batch dimension
    dgamma = (dz_fp32 * x_hat).sum(dim=0)

    return dx.to(x.dtype), dgamma.to(weight.dtype)


# =============================================================================
# Tests with torch reference implementation
# =============================================================================


@pytest.mark.rmsnorm_bwd
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_rmsnorm_bwd(shape, dtype, zero_centered_gamma):
    M, N = shape
    eps = 1e-5

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(N, dtype=dtype, device=flag_gems.device)
    if zero_centered_gamma:
        weight = weight * 0.1
    dz = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # Forward pass to get rsigma
    out, _, rsigma = rmsnorm_fwd(
        x,
        weight,
        eps,
        ln_out=None,
        quantizer=None,
        otype=dtype,
        sm_margin=0,
        zero_centered_gamma=zero_centered_gamma,
    )

    # Reference backward
    ref_x = utils.to_reference(x)
    ref_weight = utils.to_reference(weight)
    ref_dz = utils.to_reference(dz)
    ref_dx, ref_dgamma = _torch_rmsnorm_bwd(
        ref_dz, ref_x, ref_weight, eps, zero_centered_gamma=zero_centered_gamma
    )

    # FlagGems backward
    res_dx, res_dgamma = rmsnorm_bwd(
        dz, x, rsigma, weight, sm_margin=0, zero_centered_gamma=zero_centered_gamma
    )

    utils.gems_assert_close(res_dx, ref_dx, dtype)
    utils.gems_assert_close(res_dgamma, ref_dgamma, dtype)


# =============================================================================
# Tests with TransformerEngine reference
# =============================================================================


@pytest.mark.rmsnorm_bwd
@pytest.mark.skipif(not HAS_TE, reason="TransformerEngine not available")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_rmsnorm_bwd_vs_te(shape, dtype, zero_centered_gamma):
    M, N = shape
    eps = 1e-5

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(N, dtype=dtype, device=flag_gems.device)
    if zero_centered_gamma:
        weight = weight * 0.1
    dz = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # TransformerEngine forward to get rsigma
    te_otype = TORCH_TO_TE_DTYPE[dtype]
    te_fwd_result = tex.rmsnorm_fwd(
        x,
        weight,
        eps,
        None,  # ln_out
        None,  # quantizer
        te_otype,
        0,  # sm_margin
        zero_centered_gamma,
    )
    te_out, _, te_rsigma = te_fwd_result

    # TransformerEngine backward
    te_bwd_result = tex.rmsnorm_bwd(
        dz, x, te_rsigma, weight, 0, zero_centered_gamma  # sm_margin
    )
    te_dx, te_dgamma = te_bwd_result

    # FlagGems forward to get rsigma
    gems_out, _, gems_rsigma = rmsnorm_fwd(
        x,
        weight,
        eps,
        ln_out=None,
        quantizer=None,
        otype=dtype,
        sm_margin=0,
        zero_centered_gamma=zero_centered_gamma,
    )

    # FlagGems backward
    res_dx, res_dgamma = rmsnorm_bwd(
        dz, x, gems_rsigma, weight, sm_margin=0, zero_centered_gamma=zero_centered_gamma
    )

    utils.gems_assert_close(res_dx, te_dx, dtype)
    utils.gems_assert_close(res_dgamma, te_dgamma, dtype)
