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

try:
    from vllm.platforms import current_platform
    from vllm.third_party.deep_gemm.utils import per_custom_dims_cast_to_fp8
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits as vllm_fp8_fp4_mqa_logits

    VLLM_AVAILABLE = True
    SM90_AVAILABLE = current_platform.has_device_capability(90)
except ImportError:
    VLLM_AVAILABLE = False
    SM90_AVAILABLE = False

import flag_gems
from flag_gems.fused.fp8_fp4_mqa_logits import fp8_fp4_mqa_logits

from .accuracy_utils import gems_assert_close, to_reference

device = flag_gems.device

# DeepSeek V4 production config
H = 64
D = 128

# Test shapes: (M, N) covering decode and prefill workloads
DECODE_SHAPES = [(1, 1024), (1, 2048), (1, 4096), (4, 2048), (4, 4096)]
PREFILL_SHAPES = [
    (64, 4096),
    (256, 4096),
    (1024, 4096),
    (2048, 4096),
    (1024, 8192),
]


def _build_inputs(M, N, device):
    """Build FP8 quantized inputs matching vLLM DeepGEMM conventions."""
    torch.manual_seed(42)

    q_bf16 = torch.randn(M, H, D, device=device, dtype=torch.bfloat16)
    k_bf16 = torch.randn(N, D, device=device, dtype=torch.bfloat16)
    weights = torch.randn(M, H, device=device, dtype=torch.float32).abs()

    q_fp8 = q_bf16.to(torch.float8_e4m3fn)
    k_fp8, k_scale = per_custom_dims_cast_to_fp8(k_bf16, (0,), False)

    ks = torch.zeros(M, dtype=torch.int32, device=device)
    ke = torch.full((M,), N, dtype=torch.int32, device=device)

    return q_fp8, k_fp8, k_scale, weights, ks, ke


@pytest.mark.fp8_fp4_mqa_logits
@pytest.mark.skipif(
    not (torch.cuda.is_available() and SM90_AVAILABLE),
    reason="requires CUDA with Hopper architecture (SM90+)",
)
@pytest.mark.skipif(
    not VLLM_AVAILABLE,
    reason="requires vLLM with DeepGEMM and FP8 quantization support",
)
@pytest.mark.parametrize(
    "M, N",
    DECODE_SHAPES + PREFILL_SHAPES,
    ids=[f"{m}x{n}" for m, n in DECODE_SHAPES + PREFILL_SHAPES],
)
@pytest.mark.parametrize("clean_logits", [True, False])
def test_fp8_fp4_mqa_logits(M, N, clean_logits):
    q_fp8, k_fp8, k_scale, weights, ks, ke = _build_inputs(M, N, device)

    ref_out = vllm_fp8_fp4_mqa_logits(
        q=(q_fp8, None),
        kv=(k_fp8, k_scale),
        weights=weights,
        cu_seqlen_ks=ks,
        cu_seqlen_ke=ke,
        clean_logits=clean_logits,
    )
    ref_out = to_reference(ref_out)

    with flag_gems.use_gems():
        res_out = fp8_fp4_mqa_logits(
            q=(q_fp8, None),
            kv=(k_fp8, k_scale),
            weights=weights,
            cu_seqlen_ks=ks,
            cu_seqlen_ke=ke,
            clean_logits=clean_logits,
        )

    gems_assert_close(
        res_out, ref_out, res_out.dtype, equal_nan=True, atol=5e-2, reduce_dim=1
    )
