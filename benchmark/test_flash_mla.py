import math

import pytest
import torch
import triton

import flag_gems

from . import base

vendor_name = flag_gems.vendor_name


class FlashMLABenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (batch, num_heads, seq_len, head_size).
        return []


@pytest.mark.skipif(vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.flash_mla
def test_flash_mla(monkeypatch):
    def flash_mla_kwargs(shape, dtype, device):
        seqlen = shape[0]
        b = 128
        s_q = 1
        h_q = 128
        h_kv = 1
        d = 576
        dv = 512
        causal = True
        block_size = 64
        cache_seqlens = torch.tensor(
            [seqlen + 2 * i for i in range(b)], dtype=torch.int32, device=device
        )
        max_seqlen = cache_seqlens.max().item()
        max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256

        q = torch.randn([b, s_q, h_q, d], dtype=dtype, device=device)
        block_table = torch.arange(
            b * max_seqlen_pad // block_size, dtype=torch.int32, device=device
        ).view(b, max_seqlen_pad // block_size)
        blocked_k = torch.randn(
            [block_table.numel(), block_size, h_kv, d], dtype=dtype, device=device
        )
        yield q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal

    def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
        query = query.float()
        key = key.float()
        value = value.float()
        key = key.repeat_interleave(h_q // h_kv, dim=0)
        value = value.repeat_interleave(h_q // h_kv, dim=0)
        attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        if is_causal:
            s_q = query.shape[-2]
            s_k = key.shape[-2]
            attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype, device=query.device)
            temp_mask = torch.ones(
                s_q, s_k, dtype=torch.bool, device=query.device
            ).tril(diagonal=s_k - s_q)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)
            attn_weight += attn_bias
        lse = attn_weight.logsumexp(dim=-1)
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
        return attn_weight @ value, lse

    def ref_mla(
        q,
        block_table,
        blocked_k,
        max_seqlen_pad,
        block_size,
        b,
        s_q,
        cache_seqlens,
        h_q,
        h_kv,
        d,
        dv,
        causal,
    ):
        device = q.device
        blocked_v = blocked_k[..., :dv]
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32, device=device)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32, device=device)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            O, LSE = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
                h_q=h_q,
                h_kv=h_kv,
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out, lse

    if flag_gems.vendor_name == "mthreads":
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    bench = FlashMLABenchmark(
        op_name="flash_mla",
        input_fn=flash_mla_kwargs,
        torch_op=ref_mla,
        gems_op=flag_gems.flash_mla,
        dtypes=[
            torch.bfloat16,
        ],
    )
    bench.run()
