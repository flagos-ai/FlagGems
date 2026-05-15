from pathlib import Path

import pytest
import torch

from flag_gems.runtime.backend._nvidia.fused.dsv4_attention_triton import (
    dsv4_flash_mla_sparse_decode,
    dsv4_flash_mla_sparse_prefill,
)

from . import base

ORACLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "flag_gems"
    / "runtime"
    / "backend"
    / "_nvidia"
    / "fused"
    / "verify"
    / "dsv4_attention_oracle.pt"
)


def _prefill_op(q, kv, indices, sm_scale, attn_sink, topk_length):
    return dsv4_flash_mla_sparse_prefill(
        q,
        kv,
        indices,
        sm_scale,
        d_v=512,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )


def _decode_op(
    q,
    cache,
    indices,
    sm_scale,
    attn_sink,
    extra_cache,
    extra_indices,
    topk_length,
    extra_topk_length,
):
    return dsv4_flash_mla_sparse_decode(
        q,
        cache,
        indices,
        sm_scale,
        head_dim_v=512,
        attn_sink=attn_sink,
        extra_k_cache=extra_cache,
        extra_indices_in_kvcache=extra_indices,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
        block_size=64,
        rope_dim=64,
    )


class DSV4PrefillBenchmark(base.Benchmark):
    def __init__(self, case):
        super().__init__(
            "dsv4_flash_mla_sparse_prefill",
            _prefill_op,
            [torch.bfloat16],
            gems_op=_prefill_op,
        )
        self.case = case

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, dtype):
        _ = dtype
        yield (
            self.case["q"],
            self.case["kv"],
            self.case["indices"],
            self.case["sm_scale"],
            self.case["attn_sink"],
            self.case["topk_length"],
        )


class DSV4DecodeBenchmark(base.Benchmark):
    def __init__(self, case):
        super().__init__(
            "dsv4_flash_mla_sparse_decode",
            _decode_op,
            [torch.bfloat16],
            gems_op=_decode_op,
        )
        self.case = case

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, dtype):
        _ = dtype
        yield (
            self.case["q"],
            self.case["cache"],
            self.case["indices"],
            self.case["sm_scale"],
            self.case["attn_sink"],
            self.case["extra_cache"],
            self.case["extra_indices"],
            self.case["topk_length"],
            self.case["extra_topk_length"],
        )


@pytest.mark.dsv4_attention_prefill
@pytest.mark.skipif(not torch.cuda.is_available(), reason="DSV4 benchmark requires CUDA")
def test_dsv4_attention_prefill_benchmark():
    if not ORACLE_PATH.exists():
        pytest.skip(f"oracle not found: {ORACLE_PATH}")
    oracle = torch.load(ORACLE_PATH, map_location="cuda")
    bench = DSV4PrefillBenchmark(oracle["prefill_576"])
    bench.run()


@pytest.mark.dsv4_attention_decode
@pytest.mark.skipif(not torch.cuda.is_available(), reason="DSV4 benchmark requires CUDA")
def test_dsv4_attention_decode_benchmark():
    if not ORACLE_PATH.exists():
        pytest.skip(f"oracle not found: {ORACLE_PATH}")
    oracle = torch.load(ORACLE_PATH, map_location="cuda")
    bench = DSV4DecodeBenchmark(oracle["decode"])
    bench.run()
