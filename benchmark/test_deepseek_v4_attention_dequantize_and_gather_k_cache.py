import pytest
import torch

from flag_gems.fused.deepseek_v4_attention_dequantize_and_gather_k_cache import (
    dequantize_and_gather_k_cache,
)

from . import base


class DequantizeAndGatherKCacheBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "dequantize_and_gather_k_cache",
            dequantize_and_gather_k_cache,
            [torch.bfloat16],
            gems_op=dequantize_and_gather_k_cache,
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [(4, 2048, 576)]

    def get_input_iter(self, dtype):
        _ = dtype
        for batch, gather_len, dim in self.shapes:
            rope_dim = 64
            nope_dim = dim - rope_dim
            scale_slots = (nope_dim + 63) // 64 + (1 if nope_dim % 64 == 0 else 0)
            block_size = 64
            token_data_size = nope_dim + rope_dim * 2
            block_stride = block_size * token_data_size + block_size * scale_slots
            num_blocks = batch * ((gather_len + block_size - 1) // block_size)
            out = torch.empty(
                (batch, gather_len, dim), device="cuda", dtype=torch.bfloat16
            )
            k_cache = torch.zeros(
                (num_blocks, block_stride), device="cuda", dtype=torch.uint8
            )
            seq_lens = torch.full(
                (batch,), gather_len, device="cuda", dtype=torch.int32
            )
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_dequantize_and_gather_k_cache_benchmark():
    DequantizeAndGatherKCacheBenchmark().run()
