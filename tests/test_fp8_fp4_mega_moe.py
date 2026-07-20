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
    from vllm.third_party import deep_gemm

    VLLM_AVAILABLE = True
    SM100_AVAILABLE = current_platform.has_device_capability(100)
except ImportError:
    deep_gemm = None
    VLLM_AVAILABLE = False
    SM100_AVAILABLE = False

import flag_gems
from flag_gems.fused.fp8_fp4_mega_moe import (
    fp8_fp4_mega_moe,
    fp8_fp4_mega_moe_torch_ref,
)

from .accuracy_utils import gems_assert_close, to_reference

device = flag_gems.device

# Functional local-expert shapes: (num_tokens, hidden, intermediate, num_experts, top_k)
TEST_SHAPES = [
    (1, 32, 32, 2, 1),
    (3, 64, 32, 4, 2),
]


def _has_native_fp8():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def _pack_fp4_e2m1(x):
    ax = x.abs()
    code = torch.zeros_like(x, dtype=torch.uint8)
    for boundary in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        code += (ax > boundary).to(torch.uint8)
    code |= ((x < 0) & (code != 0)).to(torch.uint8) << 3
    packed = (code[..., 0::2] & 0x0F) | ((code[..., 1::2] & 0x0F) << 4)
    return packed.contiguous().view(torch.int8)


def _quantize_fp4(x):
    groups = x.float().view(*x.shape[:-1], x.shape[-1] // 32, 32)
    scale = groups.abs().amax(dim=-1).clamp_min(1e-4) / 6.0
    x_scaled = (groups / scale.unsqueeze(-1)).reshape_as(x.float())
    return _pack_fp4_e2m1(x_scaled), scale.contiguous()


def _quantize_fp8(x):
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    groups = x.float().view(x.shape[0], x.shape[1] // 32, 32)
    scale = groups.abs().amax(dim=-1).clamp_min(1e-4) / fp8_max
    x_scaled = (groups / scale.unsqueeze(-1)).reshape_as(x.float())
    x_fp8 = x_scaled.clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return x_fp8, scale.contiguous()


def _build_inputs(num_tokens, hidden, intermediate, num_experts, top_k, device):
    """Build FP8/FP4 inputs for the functional local MegaMoE interface."""
    torch.manual_seed(42)

    x = torch.randn((num_tokens, hidden), device=device, dtype=torch.bfloat16)
    l1 = torch.randn(
        (num_experts, 2 * intermediate, hidden),
        device=device,
        dtype=torch.bfloat16,
    )
    l2 = torch.randn(
        (num_experts, hidden, intermediate),
        device=device,
        dtype=torch.bfloat16,
    )

    x_fp8, x_scale = _quantize_fp8(x)
    l1_fp4, l1_scale = _quantize_fp4(l1)
    l2_fp4, l2_scale = _quantize_fp4(l2)

    scores = torch.randn((num_tokens, num_experts), device=device, dtype=torch.float32)
    topk_weights, topk_idx = torch.topk(scores, top_k, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1)

    return x_fp8, x_scale, topk_idx, topk_weights, l1_fp4, l1_scale, l2_fp4, l2_scale


@pytest.mark.fp8_fp4_mega_moe
@pytest.mark.skipif(
    not (torch.cuda.is_available() and SM100_AVAILABLE),
    reason="requires CUDA with Blackwell architecture (SM100+)",
)
@pytest.mark.skipif(
    not VLLM_AVAILABLE,
    reason="requires vLLM with DeepGEMM MegaMoE support",
)
def test_fp8_fp4_mega_moe():
    pytest.skip(
        "vLLM/DeepGEMM fp8_fp4_mega_moe requires a torch.distributed "
        "symmetric-memory group and cannot be run as a single-process tensor "
        "unit test. See DeepGEMM tests/test_mega_moe.py for the production "
        "interface."
    )


@pytest.mark.fp8_fp4_mega_moe
@pytest.mark.skipif(
    not _has_native_fp8(),
    reason="requires CUDA with native FP8 support (SM90+)",
)
@pytest.mark.parametrize(
    "num_tokens, hidden, intermediate, num_experts, top_k",
    TEST_SHAPES,
    ids=[f"{m}x{h}x{i}-e{e}-top{k}" for m, h, i, e, k in TEST_SHAPES],
)
def test_fp8_fp4_mega_moe_torch_ref(
    num_tokens, hidden, intermediate, num_experts, top_k
):
    args = _build_inputs(num_tokens, hidden, intermediate, num_experts, top_k, device)
    ref_out = to_reference(fp8_fp4_mega_moe_torch_ref(*args))

    with flag_gems.use_gems():
        res_out = fp8_fp4_mega_moe(*args)

    gems_assert_close(
        res_out,
        ref_out,
        res_out.dtype,
        equal_nan=True,
        atol=2e-1,
        reduce_dim=1,
    )
