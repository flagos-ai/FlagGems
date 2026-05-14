import random

import numpy as np
import pytest
import torch

import flag_gems
from flag_gems.fused.DSA.indexer_k_tiled import (
    triton_lighting_indexer_k_tiled_interface,
)

from . import base

random.seed(42)


def generate_random_cu_seqlens_for_bench(
    per_cp_seqlen, cp_size=3, cp_rank=4, kv_stride=1, average_q_len=512
):
    """Generate random cumulative sequence lengths for benchmarking."""
    from tests.test_DSA.utils import generate_random_cu_seqlens

    return generate_random_cu_seqlens(
        per_cp_seqlen=per_cp_seqlen,
        cp_size=cp_size,
        cp_rank=cp_rank,
        kv_stride=kv_stride,
        average_q_len=average_q_len,
    )


def ref_lighting_indexer(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke):
    """Reference implementation using PyTorch."""
    k = kv
    q_f = q.float()
    k_f = k.float()

    seq_len_kv = kv.shape[0]
    mask_lo = (
        torch.arange(0, seq_len_kv, device=q.device)[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device=q.device)[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q_f, k_f)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))
    return logits


CONFIGS = [
    (256, 1024, 8, 64),
    (512, 2048, 16, 64),
    (1024, 4096, 16, 64),
    (1024, 4096, 32, 128),
]


def input_fn(shape, dtype, device):
    seq_len_q, seq_len_kv, num_heads, qk_dim = shape
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    q = torch.randn((seq_len_q, num_heads, qk_dim), dtype=dtype, device=device)
    kv = torch.randn((seq_len_kv, qk_dim), dtype=dtype, device=device)
    weights = torch.randn((seq_len_q, num_heads), dtype=torch.float32, device=device)
    ks, ke = generate_random_cu_seqlens_for_bench(
        per_cp_seqlen=seq_len_q,
        cp_size=3,
        cp_rank=4,
        kv_stride=1,
        average_q_len=max(32, seq_len_q // 4),
    )
    yield q, kv, weights, ks, ke


class LightingIndexerBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = CONFIGS
    DEFAULT_SHAPE_DESC = "S, SKV, H, D"

    def set_more_shapes(self):
        return []

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES


@pytest.mark.triton_lighting_indexer_k_tiled_interface
def test_perf_triton_lighting_indexer_k_tiled_interface():
    bench = LightingIndexerBenchmark(
        op_name="triton_lighting_indexer_k_tiled_interface",
        torch_op=ref_lighting_indexer,
        input_fn=input_fn,
        gems_op=triton_lighting_indexer_k_tiled_interface,
        dtypes=[torch.bfloat16],
    )
    bench.run()
