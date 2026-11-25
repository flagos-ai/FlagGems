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

from . import accuracy_utils as utils
from . import conftest as cfg
from .accuracy_utils import gems_assert_close

try:
    from vllm.utils.deep_gemm import fp8_mqa_logits
except ImportError:
    fp8_mqa_logits = None


FP8_DTYPES = [torch.float8_e4m3fn] if utils.fp8_is_supported else []

pytestmark = pytest.mark.skipif(
    not FP8_DTYPES,
    reason="kunlunxin does not support FP8 (no FP8 hardware)",
)


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


def _torch_fp8_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits):
    """Pure-torch reference for fp8 MQA logits.

    ``logits[m, n] = sum_h ReLU(sum_d q[m, h, d] * k[n, d] * scale[n]) * w[m, h]``
    for ``ks[m] <= n < ke[m]``; positions outside that range are ``-inf`` when
    ``clean_logits`` is set. Same order of operations as the kernel: the key
    scale is applied to K before the dot product and the ReLU is per head.
    """
    k_fp8, k_scales = kv
    k = k_fp8.to(torch.float32) * k_scales.to(torch.float32)[:, None]
    scores = torch.einsum("mhd,nd->mhn", q.to(torch.float32), k)
    logits = (torch.clamp(scores, min=0.0) * weights.to(torch.float32)[:, :, None]).sum(
        dim=1
    )
    if clean_logits:
        cols = torch.arange(k_fp8.shape[0], device=q.device)
        inside = (cols[None, :] >= cu_seqlen_ks[:, None]) & (
            cols[None, :] < cu_seqlen_ke[:, None]
        )
        logits = torch.where(inside, logits, float("-inf"))
    return logits


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

    # Build the reference inputs on the reference device: under --ref cpu the
    # pure-torch formula runs on CPU, while the DeepGEMM reference is CUDA-only.
    ref_q = utils.to_reference(q)
    ref_kv = (utils.to_reference(k_fp8), utils.to_reference(k_scales))
    ref_weights = utils.to_reference(weights)
    ref_ks = utils.to_reference(cu_seqlen_ks)
    ref_ke = utils.to_reference(cu_seqlen_ke)

    if DEEPGEMM_AVAILABLE and ref_q.device.type == "cuda":
        ref_out = fp8_mqa_logits(
            ref_q, ref_kv, ref_weights, ref_ks, ref_ke, clean_logits
        )
    else:
        # vLLM's DeepGEMM reference is not available (or the CPU lane is
        # requested); use the pure-torch reference instead of skipping the lane.
        ref_out = _torch_fp8_mqa_logits(
            ref_q, ref_kv, ref_weights, ref_ks, ref_ke, clean_logits
        )

    res_out = flag_gems.fp8_mqa_logits(
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

    # Build the reference inputs on the reference device: under --ref cpu the
    # pure-torch formula runs on CPU, while the DeepGEMM reference is CUDA-only.
    ref_q = utils.to_reference(q)
    ref_kv = (utils.to_reference(k_fp8), utils.to_reference(k_scales))
    ref_weights = utils.to_reference(weights)
    ref_ks = utils.to_reference(cu_seqlen_ks)
    ref_ke = utils.to_reference(cu_seqlen_ke)

    if DEEPGEMM_AVAILABLE and ref_q.device.type == "cuda":
        ref_out = fp8_mqa_logits(
            ref_q, ref_kv, ref_weights, ref_ks, ref_ke, clean_logits
        )
    else:
        # vLLM's DeepGEMM reference is not available (or the CPU lane is
        # requested); use the pure-torch reference instead of skipping the lane.
        ref_out = _torch_fp8_mqa_logits(
            ref_q, ref_kv, ref_weights, ref_ks, ref_ke, clean_logits
        )

    res_out = flag_gems.fp8_mqa_logits(
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
