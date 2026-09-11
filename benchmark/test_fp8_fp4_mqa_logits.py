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

from . import base

# DeepSeek V4 production config
H = 64
D = 128


def _build_case(M, N, dtype, device):
    """Build FP8 quantized inputs for benchmarking."""
    q_bf16 = torch.randn(M, H, D, device=device, dtype=dtype)
    k_bf16 = torch.randn(N, D, device=device, dtype=dtype)
    weights = torch.randn(M, H, device=device, dtype=torch.float32).abs()

    q_fp8 = q_bf16.clamp(
        min=torch.finfo(torch.float8_e4m3fn).min,
        max=torch.finfo(torch.float8_e4m3fn).max,
    ).to(torch.float8_e4m3fn)
    k_fp8, k_scale = per_custom_dims_cast_to_fp8(k_bf16, (0,), False)

    ks = torch.zeros(M, dtype=torch.int32, device=device)
    ke = torch.full((M,), N, dtype=torch.int32, device=device)

    return q_fp8, k_fp8, k_scale, weights, ks, ke


class FP8FP4MQALogitsBenchmark(base.Benchmark):
    """Benchmark for fp8_fp4_mqa_logits: FlagGems Triton vs vLLM DeepGEMM."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 1024),
            (1, 2048),
            (1, 4096),
            (4, 2048),
            (4, 4096),
            (64, 4096),
            (256, 4096),
            (1024, 4096),
            (2048, 4096),
            (4096, 8192),
            (1024, 8192),
        ]

    def get_input_iter(self, dtype):
        for M, N in self.shapes:
            case = _build_case(M, N, dtype, self.device)
            q_fp8, k_fp8, k_scale, weights, ks, ke = case
            yield (q_fp8, k_fp8, k_scale, weights, ks, ke, dtype)


def _vllm_wrapper(q_fp8, k_fp8, k_scale, weights, ks, ke, dtype):
    return vllm_fp8_fp4_mqa_logits(
        q=(q_fp8, None),
        kv=(k_fp8, k_scale),
        weights=weights,
        cu_seqlen_ks=ks,
        cu_seqlen_ke=ke,
        clean_logits=True,
    )


def _gems_wrapper(q_fp8, k_fp8, k_scale, weights, ks, ke, dtype):
    from flag_gems.fused import fp8_fp4_mqa_logits

    return fp8_fp4_mqa_logits(
        q=(q_fp8, None),
        kv=(k_fp8, k_scale),
        weights=weights,
        cu_seqlen_ks=ks,
        cu_seqlen_ke=ke,
        clean_logits=True,
    )


@pytest.mark.skipif(
    not (torch.cuda.is_available() and SM90_AVAILABLE),
    reason="requires CUDA with Hopper architecture (SM90+)",
)
@pytest.mark.skipif(
    not VLLM_AVAILABLE,
    reason="requires vLLM with DeepGEMM and FP8 quantization support",
)
@pytest.mark.fp8_fp4_mqa_logits
def test_fp8_fp4_mqa_logits():
    bench = FP8FP4MQALogitsBenchmark(
        op_name="fp8_fp4_mqa_logits",
        torch_op=_vllm_wrapper,
        gems_op=_gems_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.run()
