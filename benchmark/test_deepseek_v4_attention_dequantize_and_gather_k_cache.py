import pytest
import torch

from flag_gems.fused.deepseek_v4_attention_dequantize_and_gather_k_cache import (
    dequantize_and_gather_k_cache,
)

from . import base


def torch_dequantize_and_gather_k_cache(
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
    if nope_dim is None:
        nope_dim = out.shape[-1] - rope_dim
    if scale_slots is None:
        scale_slots = (nope_dim + 63) // 64 + (1 if nope_dim % 64 == 0 else 0)
    k_cache_2d = k_cache.view(k_cache.shape[0], -1) if k_cache.ndim == 3 else k_cache
    token_data_size = nope_dim + rope_dim * 2
    for req_idx in range(seq_lens.numel()):
        seq_len = int(seq_lens[req_idx].item())
        gather_len = (
            seq_len if gather_lens is None else int(gather_lens[req_idx].item())
        )
        start_pos = seq_len - gather_len
        for local_i in range(gather_len):
            pos = start_pos + local_i
            block_in_seq = pos // block_size
            pos_in_block = pos % block_size
            physical_block = int(block_table[req_idx, block_in_seq].item())
            block = k_cache_2d[physical_block]
            token_base = pos_in_block * token_data_size
            scale_base = block_size * token_data_size + pos_in_block * scale_slots
            for qblock in range(scale_slots):
                start = qblock * 64
                end = min(start + 64, nope_dim)
                if start >= end:
                    continue
                values = block[token_base + start : token_base + end].view(
                    torch.float8_e4m3fn
                )
                scale = torch.exp2(block[scale_base + qblock].float() - 127.0)
                out[req_idx, offset + local_i, start:end] = (values.float() * scale).to(
                    torch.bfloat16
                )
            rope_bytes = block[
                token_base + nope_dim : token_base + nope_dim + rope_dim * 2
            ]
            out[
                req_idx, offset + local_i, nope_dim : nope_dim + rope_dim
            ] = rope_bytes.view(torch.bfloat16)[:rope_dim]


class DequantizeAndGatherKCacheBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "dequantize_and_gather_k_cache",
            torch_dequantize_and_gather_k_cache,
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
