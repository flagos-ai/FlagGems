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

import logging

import pytest
import torch

from flag_gems.fused import fused_deepseek_v4_qnorm_rope_kv_rope_insert

from . import base

logger = logging.getLogger(__name__)

# DeepSeek-V4 model constants (hardcoded in kernel)
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM  # 448

# Benchmark shapes: (num_tokens, num_heads)
# Representing production decode/prefill workloads.
BENCH_SHAPES = [
    (1, 128),
    (4, 128),
    (17, 128),
    (64, 128),
    (1024, 128),
    (2048, 128),
    (4096, 128),
]


# ── Load baseline: prefer vLLM CUDA, fallback to PyTorch ref ──
_BASELINE_NAME = "unknown"

try:
    import vllm._C  # noqa: F401

    _cuda_fn = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_insert
    assert callable(_cuda_fn)
    _BASELINE_NAME = "vLLM CUDA"
    logger.info("Benchmark baseline: vLLM CUDA kernel")
except Exception:
    _cuda_fn = None
    logger.info("vLLM CUDA not available, falling back to PyTorch reference")


def _rms_norm_no_weight(x, eps):
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    return (x_f32 * torch.rsqrt(variance + eps)).to(torch.bfloat16)


def _gptj_rope(x, cos_sin_cache, position_ids, nope_dim=448):
    half_rope = (x.shape[-1] - nope_dim) // 2
    cos = cos_sin_cache[position_ids, :half_rope]
    sin = cos_sin_cache[position_ids, half_rope:]
    if x.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    x_even = x[..., nope_dim::2].float()
    x_odd = x[..., nope_dim + 1 :: 2].float()
    result = x.clone()
    result[..., nope_dim::2] = (x_even * cos - x_odd * sin).to(torch.bfloat16)
    result[..., nope_dim + 1 :: 2] = (x_even * sin + x_odd * cos).to(torch.bfloat16)
    return result


def _pytorch_ref(
    q,
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    eps=1e-6,
    cache_block_size=16,
):
    """PyTorch reference (fallback baseline). Modifies q and k_cache in-place."""
    N_insert = slot_mapping.shape[0]
    q_normed = _rms_norm_no_weight(q, eps)
    q.copy_(_gptj_rope(q_normed, cos_sin_cache, position_ids))
    kv_roped = _gptj_rope(kv[:N_insert], cos_sin_cache, position_ids[:N_insert])
    for i in range(N_insert):
        slot_id = slot_mapping[i].item()
        if slot_id < 0:
            continue
        block_idx = slot_id // cache_block_size
        pos_in_block = slot_id % cache_block_size
        k_cache[block_idx, pos_in_block, :] = kv_roped[i]


def _cuda_baseline(
    q,
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    eps=1e-6,
    cache_block_size=16,
):
    """vLLM CUDA baseline wrapper with matching signature."""
    _cuda_fn(
        q,
        kv,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        eps,
        cache_block_size,
    )


if _cuda_fn is not None:
    _baseline_fn = _cuda_baseline
    _BASELINE_NAME = "vLLM CUDA"
else:
    _baseline_fn = _pytorch_ref
    _BASELINE_NAME = "PyTorch ref"


def _make_cos_sin_cache(max_pos, rope_dim, device):
    base = 10000.0
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _make_inputs(N, H, cache_block_size, device):
    torch.manual_seed(0)
    q = torch.randn(N, H, HEAD_DIM, device=device, dtype=torch.bfloat16)
    kv = torch.randn(N, HEAD_DIM, device=device, dtype=torch.bfloat16)
    position_ids = torch.arange(N, device=device, dtype=torch.int64)
    slot_mapping = torch.arange(N, device=device, dtype=torch.int64)
    max_pos = max(4096, N)
    num_blocks = (N + cache_block_size - 1) // cache_block_size + 1
    k_cache = torch.zeros(
        num_blocks, cache_block_size, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    cos_sin_cache = _make_cos_sin_cache(max_pos, ROPE_DIM, device)
    return q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache


class FusedQnormRopeKvRopeInsertBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "num_tokens, num_heads"

    def set_shapes(self, shape_file_path=None):
        self.shapes = BENCH_SHAPES

    def get_input_iter(self, dtype):
        cbs = 16  # cache_block_size, hardcoded for DeepSeek-V4
        for N, H in self.shapes:
            q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache = _make_inputs(
                N, H, cbs, self.device
            )
            yield (
                q,
                kv,
                k_cache,
                slot_mapping,
                position_ids,
                cos_sin_cache,
                {"eps": 1e-6, "cache_block_size": cbs},
            )


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_insert
def test_fused_deepseek_v4_qnorm_rope_kv_rope_insert():
    bench = FusedQnormRopeKvRopeInsertBenchmark(
        op_name="fused_deepseek_v4_qnorm_rope_kv_rope_insert",
        torch_op=_baseline_fn,
        gems_op=fused_deepseek_v4_qnorm_rope_kv_rope_insert,
        dtypes=[torch.bfloat16],
    )
    bench.run()
