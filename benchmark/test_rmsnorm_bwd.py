import pytest
import torch

from flag_gems.ops.rmsnorm_bwd import rmsnorm_bwd

from . import base, consts

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


def rmsnorm_bwd_input_fn(shape, dtype, device):
    M, N = shape
    x = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(N, dtype=dtype, device=device)
    dz = torch.randn(shape, dtype=dtype, device=device)

    # Compute rsigma using forward pass
    eps = 1e-5
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    rsigma = torch.rsqrt(variance + eps).squeeze(-1).to(torch.float32)

    yield dz, x, rsigma, weight


def te_rmsnorm_bwd(dz, x, rsigma, weight, sm_margin=0, zero_centered_gamma=False):
    result = tex.rmsnorm_bwd(dz, x, rsigma, weight, sm_margin, zero_centered_gamma)
    return result[0]


def gems_rmsnorm_bwd(dz, x, rsigma, weight, sm_margin=0, zero_centered_gamma=False):
    result = rmsnorm_bwd(dz, x, rsigma, weight, sm_margin, zero_centered_gamma)
    return result[0]


@pytest.mark.rmsnorm_bwd
@pytest.mark.skipif(not HAS_TE, reason="TransformerEngine not available")
def test_rmsnorm_bwd():
    bench = base.GenericBenchmark2DOnly(
        input_fn=rmsnorm_bwd_input_fn,
        op_name="rmsnorm_bwd",
        torch_op=te_rmsnorm_bwd,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(gems_rmsnorm_bwd)
    bench.run()
