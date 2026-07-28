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

import random

import pytest
import torch

import flag_gems

from . import conftest as cfg
from .accuracy_utils import gems_assert_close

try:
    from vllm.utils.deep_gemm import fp8_mqa_logits
except ImportError:
    fp8_mqa_logits = None


try:
    import vllm  # noqa: F401

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


def is_hopper_available() -> bool:
    """Check if the current device is NVIDIA Hopper architecture (SM90+)."""
    if flag_gems.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major * 10 + minor) >= 90


HOPPER_AVAILABLE = is_hopper_available()


def has_deep_gemm() -> bool:
    """Check if vLLM's DeepGEMM is available."""
    try:
        from vllm.utils.import_utils import has_deep_gemm

        return has_deep_gemm()
    except ImportError:
        return False


DEEPGEMM_AVAILABLE = has_deep_gemm()

device = flag_gems.device

# Shape configs for QUICK_MODE
if cfg.QUICK_MODE:
    MN_SHAPES = [(32, 2048)]
    HD_SHAPES = [(32, 128)]
else:
    MN_SHAPES = [(32, 2048), (32, 4096), (32, 1024)]
    HD_SHAPES = [(32, 128)]


@pytest.mark.fp8_mqa_logits
@pytest.mark.skipif(
    not (torch.cuda.is_available() and HOPPER_AVAILABLE),
    reason="requires CUDA with Hopper architecture (SM90+)",
)
@pytest.mark.skipif(
    not (VLLM_AVAILABLE and DEEPGEMM_AVAILABLE),
    reason="requires vLLM with DeepGEMM support",
)
@pytest.mark.parametrize("clean_logits", [True, False])
def test_fp8_mqa_logits(clean_logits: bool):
    torch.manual_seed(0)
    random.seed(0)

    M = 4
    H = 32
    D = 128
    N = 4096

    q = torch.randn((M, H, D), device=device, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    k_fp8 = torch.randn((N, D), device=device, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    k_scales = torch.rand((N,), device=device, dtype=torch.float32) * 0.01 + 0.001
    weights = torch.randn((M, H), device=device, dtype=torch.float32)

    cu_seqlen_ks = torch.tensor([0, 1024, 2048, 3072], device=device, dtype=torch.int32)
    cu_seqlen_ke = torch.tensor(
        [1024, 2048, 3072, 4096], device=device, dtype=torch.int32
    )

    ref_out = fp8_mqa_logits(
        q, (k_fp8, k_scales), weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits
    )

    with flag_gems.use_gems():
        res_out = fp8_mqa_logits(
            q=q,
            kv=(k_fp8, k_scales),
            weights=weights,
            cu_seqlen_ks=cu_seqlen_ks,
            cu_seqlen_ke=cu_seqlen_ke,
            clean_logits=clean_logits,
        )

    gems_assert_close(
        res_out, ref_out, res_out.dtype, equal_nan=True, atol=5e-2, reduce_dim=1
    )


@pytest.mark.fp8_mqa_logits
@pytest.mark.skipif(
    not (torch.cuda.is_available() and HOPPER_AVAILABLE),
    reason="requires CUDA with Hopper architecture (SM90+)",
)
@pytest.mark.skipif(
    not (VLLM_AVAILABLE and DEEPGEMM_AVAILABLE),
    reason="requires vLLM with DeepGEMM support",
)
@pytest.mark.parametrize("M, N", MN_SHAPES)
@pytest.mark.parametrize("H, D", HD_SHAPES)
def test_fp8_mqa_logits_param(M: int, N: int, H: int, D: int):
    torch.manual_seed(0)
    random.seed(0)
    clean_logits = True

    q = torch.randn((M, H, D), device=device, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    k_fp8 = torch.randn((N, D), device=device, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    k_scales = torch.rand((N,), device=device, dtype=torch.float32) * 0.01 + 0.001
    weights = torch.randn((M, H), device=device, dtype=torch.float32)

    cu_seqlen_ks = torch.zeros(M, device=device, dtype=torch.int32)
    cu_seqlen_ke = torch.ones(M, device=device, dtype=torch.int32) * N
    for m in range(M):
        cu_seqlen_ks[m] = m * (N // M // 2)
        cu_seqlen_ke[m] = N - (M - m - 1) * (N // M // 2)

    ref_out = fp8_mqa_logits(
        q, (k_fp8, k_scales), weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits
    )

    with flag_gems.use_gems():
        res_out = fp8_mqa_logits(
            q=q,
            kv=(k_fp8, k_scales),
            weights=weights,
            cu_seqlen_ks=cu_seqlen_ks,
            cu_seqlen_ke=cu_seqlen_ke,
            clean_logits=clean_logits,
        )

    gems_assert_close(
        res_out, ref_out, res_out.dtype, equal_nan=True, atol=5e-2, reduce_dim=1
    )
