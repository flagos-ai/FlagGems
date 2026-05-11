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
from flag_gems.ops.fp8_paged_mqa_logits import (
    fp8_paged_mqa_logits as gems_fp8_paged_mqa_logits,
)

from . import base

random.seed(42)


def is_hopper_available():
    if flag_gems.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major * 10 + minor) >= 90


DEEPGEMM_AVAILABLE = VLLM_AVAILABLE and has_deep_gemm()
HOPPER_AVAILABLE = is_hopper_available()


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


class FP8PagedMQACompareBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "fp8_paged_mqa_logits_gems_vs_deepgemm",
            self._vllm_wrapper,
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
            schedule_metadata = get_paged_mqa_logits_metadata(
                context_lens, blocksize, get_num_sms()
            )
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
    not (
        torch.cuda.is_available()
        and VLLM_AVAILABLE
        and DEEPGEMM_AVAILABLE
        and HOPPER_AVAILABLE
    ),
    reason="requires CUDA + vLLM + DeepGEMM + Hopper",
)
@pytest.mark.performance
@pytest.mark.fp8_paged_mqa_logits
def test_perf_fp8_paged_mqa_logits_gems_vs_deepgemm():
    bench = FP8PagedMQACompareBenchmark()
    bench.run()
