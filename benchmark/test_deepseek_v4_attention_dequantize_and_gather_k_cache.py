import pytest
import torch

from flag_gems.fused.deepseek_v4_attention_dequantize_and_gather_k_cache import (
    dequantize_and_gather_k_cache,
)
from flag_gems.utils.device_info import get_device_capability

try:
    from vllm.v1.attention.ops.deepseek_v4_ops import (
        dequantize_and_gather_k_cache as vllm_dequantize_and_gather_k_cache,
    )

    _HAS_VLLM_DEQUANTIZE_AND_GATHER_K_CACHE = True
except Exception:
    vllm_dequantize_and_gather_k_cache = None
    _HAS_VLLM_DEQUANTIZE_AND_GATHER_K_CACHE = False

from . import base


def is_support_fp8e4nv():
    major, minor = get_device_capability()
    return major * 10 + minor >= 89


class DequantizeAndGatherKCacheBenchmark(base.Benchmark):
    def __init__(self):
        def vllm_dequantize_and_gather_k_cache_adapter(
            out,
            k_cache,
            seq_lens,
            gather_lens,
            block_table,
            block_size,
            offset=0,
            rope_dim=64,
            nope_dim=None,
            scale_slots=None,
        ):
            _ = (rope_dim, nope_dim, scale_slots)
            return vllm_dequantize_and_gather_k_cache(
                out,
                k_cache,
                seq_lens,
                gather_lens,
                block_table,
                block_size,
                offset,
            )

        super().__init__(
            "dequantize_and_gather_k_cache",
            vllm_dequantize_and_gather_k_cache_adapter,
            [torch.bfloat16],
            gems_op=dequantize_and_gather_k_cache,
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [
            (1, 512, 128, 512, 448, 64),
            (2, 1024, 256, 512, 448, 64),
            (4, 2048, 512, 512, 448, 64),
            (4, 2048, 2048, 512, 448, 64),
            (8, 4096, 1024, 512, 448, 64),
        ]

    def get_input_iter(self, dtype):
        _ = dtype
        for batch, seq_len, gather_len, dim, nope_dim, rope_dim in self.shapes:
            scale_slots = (nope_dim + 63) // 64 + (1 if nope_dim % 64 == 0 else 0)
            block_size = 64
            token_data_size = nope_dim + rope_dim * 2
            block_stride = block_size * token_data_size + block_size * scale_slots
            num_blocks = batch * ((seq_len + block_size - 1) // block_size)
            out = torch.empty(
                (batch, gather_len, dim), device="cuda", dtype=torch.bfloat16
            )
            k_cache = torch.zeros(
                (num_blocks, block_stride), device="cuda", dtype=torch.uint8
            )
            seq_lens = torch.full((batch,), seq_len, device="cuda", dtype=torch.int32)
            gather_lens = torch.full(
                (batch,), gather_len, device="cuda", dtype=torch.int32
            )
            block_table = torch.arange(
                num_blocks, device="cuda", dtype=torch.int32
            ).view(batch, -1)
            yield (
                out,
                k_cache,
                seq_lens,
                gather_lens,
                block_table,
                block_size,
                0,
                rope_dim,
                nope_dim,
                scale_slots,
            )


@pytest.mark.skipif(
    (not torch.cuda.is_available())
    or (not is_support_fp8e4nv())
    or (not _HAS_VLLM_DEQUANTIZE_AND_GATHER_K_CACHE),
    reason="requires cuda with fp8e4nv support and vllm deepseek_v4_ops.dequantize_and_gather_k_cache",
)
def test_dequantize_and_gather_k_cache_benchmark():
    DequantizeAndGatherKCacheBenchmark().run()
