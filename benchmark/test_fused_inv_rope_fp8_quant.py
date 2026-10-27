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

import flag_gems

from . import base

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
QUANT_GROUP_SIZE = 128

HAS_NATIVE_FP8 = hasattr(torch, "float8_e4m3fn") and (
    flag_gems.SUPPORTED_FP8_DTYPE == torch.float8_e4m3fn
)

try:
    from vllm.v1.attention.ops.deepseek_v4_ops import (
        fused_inv_rope_fp8_quant as vllm_fused_inv_rope_fp8_quant,
    )

    HAS_VLLM_FUSED_INV_ROPE_FP8_QUANT = True
except ImportError:
    HAS_VLLM_FUSED_INV_ROPE_FP8_QUANT = False


def _make_cos_sin_cache(max_pos, rope_dim, device):
    half = rope_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _input_fn(shape, dtype, device):
    num_tokens, num_heads, n_groups, tma_aligned_scales = shape
    heads_per_group = num_heads // n_groups
    max_pos = max(4096, num_tokens * 2)

    o = torch.randn(num_tokens, num_heads, HEAD_DIM, dtype=dtype, device=device)
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.long, device=device
    )
    cos_sin_cache = _make_cos_sin_cache(max_pos, ROPE_DIM, torch.device(device))

    yield (
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        NOPE_DIM,
        ROPE_DIM,
        QUANT_GROUP_SIZE,
        tma_aligned_scales,
    )


def _gems_fused_inv_rope_fp8_quant(
    o,
    positions,
    cos_sin_cache,
    n_groups,
    heads_per_group,
    nope_dim,
    rope_dim,
    quant_group_size,
    tma_aligned_scales,
):
    return flag_gems.fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=quant_group_size,
        tma_aligned_scales=tma_aligned_scales,
    )


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(not HAS_NATIVE_FP8, reason="requires native float8_e4m3fn support")
@pytest.mark.skipif(
    not HAS_VLLM_FUSED_INV_ROPE_FP8_QUANT,
    reason="vLLM fused_inv_rope_fp8_quant not installed",
)
def test_fused_inv_rope_fp8_quant():
    bench = base.GenericBenchmark(
        op_name="fused_inv_rope_fp8_quant",
        input_fn=_input_fn,
        torch_op=vllm_fused_inv_rope_fp8_quant,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_fused_inv_rope_fp8_quant)
    bench.run()
