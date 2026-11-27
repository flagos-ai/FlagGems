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
from itertools import product

import pytest
import torch

try:
    from vllm.utils.deep_gemm import fp8_paged_mqa_logits as vllm_fp8_paged_mqa_logits
    from vllm.utils.deep_gemm import get_num_sms, get_paged_mqa_logits_metadata
    from vllm.utils.import_utils import has_deep_gemm

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

import flag_gems
from flag_gems import fp8_paged_mqa_logits as gems_fp8_paged_mqa_logits

from . import base

random.seed(42)


def is_hopper_available():
    if flag_gems.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major * 10 + minor) >= 90


DEEPGEMM_AVAILABLE = VLLM_AVAILABLE and has_deep_gemm()
HOPPER_AVAILABLE = is_hopper_available()

# The vLLM/DeepGEMM baseline is CUDA Hopper only. Vendors listed here are
# timed against the dequantized torch reference instead of skipping the case.
TORCH_REFERENCE_VENDORS = ("enflame",)
USE_DEEPGEMM_BASELINE = (
    DEEPGEMM_AVAILABLE
    and HOPPER_AVAILABLE
    and flag_gems.vendor_name not in TORCH_REFERENCE_VENDORS
)
BASELINE_UNAVAILABLE = (
    flag_gems.vendor_name not in TORCH_REFERENCE_VENDORS and not USE_DEEPGEMM_BASELINE
)


def kv_cache_cast_to_fp8_deepgemm(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)

    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(torch.uint8)

    sf_scaled = sf.squeeze(-1).squeeze(-1)
    sf_bytes = sf_scaled.view(torch.int32).view(torch.uint8)
    x_fp8[:, block_size * head_dim :] = sf_bytes
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


def kv_cache_cast_to_fp8_triton(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)

    out = torch.empty(
        (num_blocks, block_size, num_heads, head_dim + 4),
        device=x.device,
        dtype=torch.uint8,
    )
    out[..., :head_dim] = x_scaled.view(torch.uint8)

    sf_scaled = sf.squeeze(-1).squeeze(-1)
    sf_bytes = sf_scaled.view(torch.int32).view(torch.uint8)
    out[..., head_dim:] = sf_bytes.view(num_blocks, block_size, num_heads, 4)
    return out


def _build_case(
    batch_size, next_n, heads, head_dim, avg_kv, blocksize, q_dtype, max_model_len=4096
):
    num_blocks = max_model_len * 2

    q = torch.randn(
        (batch_size, next_n, heads, head_dim),
        device=flag_gems.device,
        dtype=q_dtype,
    )
    q_fp8 = q.to(torch.float8_e4m3fn)

    kv_cache = torch.randn(
        (num_blocks, blocksize, 1, head_dim),
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )

    weights = torch.randn(
        (batch_size * next_n, heads),
        device=flag_gems.device,
        dtype=torch.float32,
    )

    context_lens = torch.randint(
        int(0.8 * avg_kv),
        int(1.2 * avg_kv),
        (batch_size,),
        device=flag_gems.device,
        dtype=torch.int32,
    )

    max_num_blocks_per_seq = (
        int(context_lens.max().item()) + blocksize - 1
    ) // blocksize
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq),
        device=flag_gems.device,
        dtype=torch.int32,
    )

    counter = 0
    block_idx_pool = list(range(num_blocks))
    random.shuffle(block_idx_pool)
    for i in range(batch_size):
        ctx_len = int(context_lens[i].item())
        for j in range((ctx_len + blocksize - 1) // blocksize):
            block_tables[i, j] = block_idx_pool[counter]
            counter += 1

    kv_cache_fp8_deepgemm = kv_cache_cast_to_fp8_deepgemm(kv_cache)
    kv_cache_fp8_triton = kv_cache_cast_to_fp8_triton(kv_cache)

    return (
        q_fp8,
        kv_cache_fp8_deepgemm,
        kv_cache_fp8_triton,
        weights,
        context_lens,
        block_tables,
        max_model_len,
    )


def torch_reference_fp8_paged_mqa_logits(
    q_fp8, kv_cache, weights, context_lens, block_tables, max_model_len
):
    """Dequantized torch reference for the paged MQA logits baseline."""
    batch_size, next_n, heads, dim = q_fp8.shape
    block_size = kv_cache.shape[1]
    device = q_fp8.device
    q_float = q_fp8.to(torch.float32)
    logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )
    for batch_idx in range(batch_size):
        context_len = int(context_lens[batch_idx])
        for next_n_idx in range(next_n):
            query_pos = context_len - next_n + next_n_idx
            keep = min(context_len, query_pos + 1)
            if keep <= 0:
                continue
            positions = torch.arange(keep, device=device)
            phys_blocks = block_tables[batch_idx, positions // block_size].long()
            intra_positions = positions % block_size
            kv_rows = kv_cache[phys_blocks, intra_positions, 0]
            kv_values = (
                kv_rows[:, :dim]
                .contiguous()
                .view(torch.float8_e4m3fn)
                .to(torch.float32)
            )
            kv_scales = (
                kv_rows[:, dim : dim + 4].contiguous().view(torch.float32).squeeze(-1)
            )
            dots = (kv_values @ q_float[batch_idx, next_n_idx].T) * kv_scales.unsqueeze(
                1
            )
            dots = torch.clamp(dots, min=0.0)
            row_idx = batch_idx * next_n + next_n_idx
            logits[row_idx, :keep] = (dots * weights[row_idx].unsqueeze(0)).sum(-1)
    return logits


class FP8PagedMQACompareBenchmark(base.Benchmark):
    def __init__(self):
        baseline = self._vllm_wrapper if USE_DEEPGEMM_BASELINE else self._torch_wrapper
        super().__init__(
            "fp8_paged_mqa_logits_gems_vs_deepgemm",
            baseline,
            [torch.bfloat16],
        )
        self.set_gems(self._gems_wrapper)

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, _dtype):
        compare_shapes = [
            (1, 1, 16, 64, 1024),
            (2, 1, 32, 128, 2048),
            (4, 1, 32, 128, 2048),
            (2, 2, 32, 128, 2048),
            (8, 1, 32, 128, 3072),
        ]
        q_dtypes = [torch.bfloat16, torch.float16]
        blocksize = 64

        for (bs, nn, h, d, avg_kv), q_dtype in product(compare_shapes, q_dtypes):
            case = _build_case(bs, nn, h, d, avg_kv, blocksize, q_dtype)
            (
                q_fp8,
                kv_dg,
                kv_tr,
                weights,
                context_lens,
                block_tables,
                max_model_len,
            ) = case
            if USE_DEEPGEMM_BASELINE:
                schedule_metadata = get_paged_mqa_logits_metadata(
                    context_lens, blocksize, get_num_sms()
                )
            else:
                schedule_metadata = None
            yield (
                q_fp8,
                kv_dg,
                kv_tr,
                weights,
                context_lens,
                block_tables,
                schedule_metadata,
                max_model_len,
                q_dtype,
                blocksize,
            )

    @staticmethod
    def _vllm_wrapper(
        q_fp8,
        kv_cache_fp8_deepgemm,
        kv_cache_fp8_triton,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        q_dtype,
        blocksize,
    ):
        return vllm_fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8_deepgemm,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_model_len,
            clean_logits=True,
        )

    @staticmethod
    def _torch_wrapper(
        q_fp8,
        kv_cache_fp8_deepgemm,
        kv_cache_fp8_triton,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        q_dtype,
        blocksize,
    ):
        return torch_reference_fp8_paged_mqa_logits(
            q_fp8.cpu(),
            kv_cache_fp8_triton.cpu(),
            weights.cpu(),
            context_lens.cpu(),
            block_tables.cpu(),
            max_model_len,
        )

    @staticmethod
    def _gems_wrapper(
        q_fp8,
        kv_cache_fp8_deepgemm,
        kv_cache_fp8_triton,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        q_dtype,
        blocksize,
    ):
        return gems_fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8_triton,
            weights,
            context_lens,
            block_tables,
            max_model_len,
        )


@pytest.mark.skipif(
    BASELINE_UNAVAILABLE,
    reason="requires CUDA + vLLM + DeepGEMM + Hopper baseline",
)
@pytest.mark.performance
@pytest.mark.fp8_paged_mqa_logits
def test_perf_fp8_paged_mqa_logits_gems_vs_deepgemm():
    bench = FP8PagedMQACompareBenchmark()
    bench.run()
