# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch

from flag_gems.ops.te_rmsnorm import te_rmsnorm_bwd as gems_te_rmsnorm_bwd
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


# =============================================================================
# Forward Benchmark
# =============================================================================


def te_rmsnorm_fwd_input_fn(shape, dtype, device):
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


@pytest.mark.te_rmsnorm_fwd
@pytest.mark.skipif(not HAS_TE, reason="TransformerEngine not available")
def test_te_rmsnorm_fwd():
    bench = base.GenericBenchmark2DOnly(
        input_fn=te_rmsnorm_fwd_input_fn,
        op_name="te_rmsnorm_fwd",
        torch_op=te_rmsnorm_fwd,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(gems_rmsnorm_fwd)
    bench.run()


# =============================================================================
# Backward Benchmark
# =============================================================================


def te_rmsnorm_bwd_input_fn(shape, dtype, device):
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
    result = gems_te_rmsnorm_bwd(dz, x, rsigma, weight, sm_margin, zero_centered_gamma)
    return result[0]


@pytest.mark.te_rmsnorm_bwd
@pytest.mark.skipif(not HAS_TE, reason="TransformerEngine not available")
def test_te_rmsnorm_bwd():
    bench = base.GenericBenchmark2DOnly(
        input_fn=te_rmsnorm_bwd_input_fn,
        op_name="te_rmsnorm_bwd",
        torch_op=te_rmsnorm_bwd,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(gems_rmsnorm_bwd)
    bench.run()
