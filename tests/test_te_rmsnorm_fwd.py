import pytest
import torch

import flag_gems
from flag_gems.ops.te_rmsnorm_fwd import te_rmsnorm_fwd

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

    # Mapping from torch dtype to TE DType
    TORCH_TO_TE_DTYPE = {
        torch.float32: TEDType.kFloat32,
        torch.float16: TEDType.kFloat16,
        torch.bfloat16: TEDType.kBFloat16,
    }
except ImportError:
    HAS_TE = False


def _torch_rmsnorm_fwd(x, weight, eps, zero_centered_gamma=False, otype=None):
    """Reference implementation matching TransformerEngine's rmsnorm_fwd."""
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    rsigma = torch.rsqrt(variance + eps)
    x_norm = x_fp32 * rsigma

    if zero_centered_gamma:
        gamma = weight.to(torch.float32) + 1.0
    else:
        gamma = weight.to(torch.float32)

    if otype is None:
        otype = x.dtype

    output = (x_norm * gamma).to(otype)
    rsigma = rsigma.squeeze(-1)
    return output, rsigma


# =============================================================================
# Tests with torch reference implementation
# =============================================================================


@pytest.mark.rmsnorm_fwd
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
@pytest.mark.parametrize("use_ln_out", [False, True])
def test_rmsnorm_fwd(shape, dtype, zero_centered_gamma, use_ln_out):
    M, N = shape
    eps = 1e-5

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(N, dtype=dtype, device=flag_gems.device)
    if zero_centered_gamma:
        weight = weight * 0.1

    ln_out = (
        torch.empty(shape, dtype=dtype, device=flag_gems.device) if use_ln_out else None
    )

    ref_inp = utils.to_reference(inp)
    ref_weight = utils.to_reference(weight)

    ref_out, ref_rsigma = _torch_rmsnorm_fwd(
        ref_inp, ref_weight, eps, zero_centered_gamma=zero_centered_gamma
    )

    res_out, _, res_rsigma = te_rmsnorm_fwd(
        inp,
        weight,
        eps,
        ln_out=ln_out,
        quantizer=None,
        otype=dtype,
        sm_margin=0,
        zero_centered_gamma=zero_centered_gamma,
    )

    if use_ln_out:
        assert res_out.data_ptr() == ln_out.data_ptr()
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_rsigma, ref_rsigma, torch.float32)


# =============================================================================
# Tests with TransformerEngine reference
# =============================================================================


@pytest.mark.rmsnorm_fwd
@pytest.mark.skipif(not HAS_TE, reason="TransformerEngine not available")
@pytest.mark.skipif(cfg.TO_CPU, reason="TransformerEngine not available on CPU")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
@pytest.mark.parametrize("use_ln_out", [False, True])
def test_rmsnorm_fwd_vs_te(shape, dtype, zero_centered_gamma, use_ln_out):
    M, N = shape
    eps = 1e-5

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(N, dtype=dtype, device=flag_gems.device)
    if zero_centered_gamma:
        weight = weight * 0.1

    ln_out_te = (
        torch.empty(shape, dtype=dtype, device=flag_gems.device) if use_ln_out else None
    )
    ln_out_gems = (
        torch.empty(shape, dtype=dtype, device=flag_gems.device) if use_ln_out else None
    )

    # TransformerEngine reference
    te_otype = TORCH_TO_TE_DTYPE[dtype]
    te_result = tex.rmsnorm_fwd(
        inp,
        weight,
        eps,
        ln_out_te,
        None,  # quantizer
        te_otype,
        0,  # sm_margin
        zero_centered_gamma,
    )
    te_out, _, te_rsigma = te_result

    # FlagGems implementation
    res_out, _, res_rsigma = te_rmsnorm_fwd(
        inp,
        weight,
        eps,
        ln_out=ln_out_gems,
        quantizer=None,
        otype=dtype,
        sm_margin=0,
        zero_centered_gamma=zero_centered_gamma,
    )

    if use_ln_out:
        assert res_out.data_ptr() == ln_out_gems.data_ptr()
    utils.gems_assert_close(res_out, te_out, dtype)
    utils.gems_assert_close(res_rsigma, te_rsigma, torch.float32)
