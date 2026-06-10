import math

import pytest
import torch
import triton

import flag_gems

from . import accuracy_utils as utils

device = flag_gems.device
vendor_name = flag_gems.vendor_name


def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str) -> None:
    x, y = x.double(), y.double()
    x = x.to(y.device)
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()

    assert cos_diff < 1e-5, f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}"


def _scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
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
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool, device=query.device).tril(
            diagonal=s_k - s_q
        )
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
        O, LSE = _scaled_dot_product_attention(
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


@pytest.mark.skipif(vendor_name == "hygon", reason="#2817: RuntimeError")
@pytest.mark.flash_mla
@pytest.mark.parametrize("seqlen", [1024, 2048, 4096, 8192])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_flash_mla(monkeypatch, seqlen, dtype):
    if vendor_name == "mthreads":
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

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

    ref_q = utils.to_reference(q)
    ref_block_table = utils.to_reference(block_table)
    ref_blocked_k = utils.to_reference(blocked_k)
    ref_cache_seqlens = utils.to_reference(cache_seqlens)

    ref_out, _ = ref_mla(
        ref_q,
        ref_block_table,
        ref_blocked_k,
        max_seqlen_pad,
        block_size,
        b,
        s_q,
        ref_cache_seqlens,
        h_q,
        h_kv,
        d,
        dv,
        causal,
    )
    res_out = flag_gems.flash_mla(
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
    )

    cal_diff(utils.to_reference(res_out), ref_out, "out")
