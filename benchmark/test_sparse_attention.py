import math

import pytest
import torch

import flag_gems

from . import base

# sparse_attention shape layout:
# (batch, seq_len, kv_len, topk, heads, dim)
SPARSE_ATTENTION_SHAPES = [
    (16, 1, 136, 136, 8, 512),
    (16, 1, 392, 385, 8, 512),
    (16, 1, 392, 386, 8, 512),
    (16, 1, 392, 387, 8, 512),
    (32, 1, 392, 388, 8, 512),
    (32, 1, 392, 389, 8, 512),
    (32, 1, 392, 390, 8, 512),
    (32, 1, 392, 391, 8, 512),
    (64, 1, 136, 136, 8, 512),
    (64, 1, 392, 385, 8, 512),
    (64, 1, 392, 388, 8, 512),
    (64, 1, 392, 389, 8, 512),
]


def torch_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale):
    batch, seq_len, heads, dim = q.shape
    topk = topk_idxs.shape[-1]

    kv_expanded = kv[:, None, :, :].expand(batch, seq_len, -1, dim)
    idx_expanded = topk_idxs[:, :, :, None].expand(batch, seq_len, topk, dim).long()
    gathered_kv = torch.gather(kv_expanded, 2, idx_expanded)

    scores = (
        torch.einsum("bmhd,bmtd->bmht", q.float(), gathered_kv.float()) * softmax_scale
    )
    sink = attn_sink[None, None, :, None].expand(batch, seq_len, heads, 1)
    attn = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)

    out = torch.einsum("bmht,bmtd->bmhd", attn[:, :, :, :-1], gathered_kv.float())
    return out.to(q.dtype)


class SparseAttentionBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SPARSE_ATTENTION_SHAPES[:]
        self.shape_desc = "B, M, KV_LEN, TOPK, H, D"

    def set_more_shapes(self):
        return None

    def get_input_iter(self, dtype):
        for seed, (batch, seq_len, kv_len, topk, heads, dim) in enumerate(self.shapes):
            torch.manual_seed(2026 + seed)
            q = torch.randn(
                (batch, seq_len, heads, dim), dtype=dtype, device=self.device
            )
            kv = torch.randn((batch, kv_len, dim), dtype=dtype, device=self.device)
            attn_sink = torch.zeros((heads,), dtype=torch.float32, device=self.device)
            topk_idxs = torch.randint(
                0,
                kv_len,
                (batch, seq_len, topk),
                dtype=torch.int32,
                device=self.device,
            )
            yield q, kv, attn_sink, topk_idxs, 1.0 / math.sqrt(dim)


@pytest.mark.skip(reason="The test case fails with many exceptions: #2669")
@pytest.mark.skipif(flag_gems.device == "cpu", reason="Unsupported in CPU mode")
@pytest.mark.sparse_attention
def test_sparse_attention():
    bench = SparseAttentionBenchmark(
        op_name="sparse_attention",
        torch_op=torch_sparse_attention,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flag_gems.sparse_attn_triton)
    bench.run()
