import pytest
import torch

from flag_gems.ops.te_rmsnorm import te_rmsnorm_fwd as gems_te_rmsnorm_fwd

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


def rmsnorm_fwd_input_fn(shape, dtype, device):
    M, N = shape
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(N, dtype=dtype, device=device)
    yield inp, weight


def te_rmsnorm_fwd(inp, weight, eps=1e-5):
    te_otype = TORCH_TO_TE_DTYPE[inp.dtype]
    result = tex.rmsnorm_fwd(
        inp,
        weight,
        eps,
        None,  # ln_out
        None,  # quantizer
        te_otype,
        0,  # sm_margin
        False,  # zero_centered_gamma
    )
    return result[0]


def gems_rmsnorm_fwd(inp, weight, eps=1e-5):
    result = gems_te_rmsnorm_fwd(
        inp,
        weight,
        eps,
        ln_out=None,
        quantizer=None,
        otype=inp.dtype,
        sm_margin=0,
        zero_centered_gamma=False,
    )
    return result[0]


@pytest.mark.rmsnorm_fwd
@pytest.mark.skipif(not HAS_TE, reason="TransformerEngine not available")
def test_rmsnorm_fwd():
    bench = base.GenericBenchmark2DOnly(
        input_fn=rmsnorm_fwd_input_fn,
        op_name="rmsnorm_fwd",
        torch_op=te_rmsnorm_fwd,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(gems_rmsnorm_fwd)
    bench.run()
