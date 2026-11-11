# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import math

import pytest
try:
    import torch_fl  # noqa: F401  # loads the Hygon runtime before torch.cuda use
except ImportError:
    torch_fl = None
import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import random_utils


pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon-specific FP32 FA path"
)


@triton.jit
def _dropout_mask_kernel(out, seed, base_offset, n_elements, dropout_p, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rng_offsets = (base_offset + offsets).to(tl.uint32)
    keep = tl.rand(seed, rng_offsets) > dropout_p
    tl.store(out + offsets, keep.to(tl.int8), mask=offsets < n_elements)


def _reference(q, k, v, dout, scale, causal, dropout_p, mask):
    qf, kf, vf = q.double(), k.double(), v.double()
    scores = torch.einsum("bqhd,bkhd->bhqk", qf, kf) * scale
    if causal:
        q_idx = torch.arange(q.shape[1])[:, None]
        k_idx = torch.arange(k.shape[1])[None, :]
        scores = scores.masked_fill(k_idx > q_idx + k.shape[1] - q.shape[1], -float("inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    if dropout_p > 0:
        probs_drop = probs * mask.double() / (1.0 - dropout_p)
    else:
        probs_drop = probs
    out = torch.einsum("bhqk,bkhd->bqhd", probs_drop, vf)

    do = dout.double()
    d_out_bh = do.permute(0, 2, 1, 3)
    out_bh = out.permute(0, 2, 1, 3)
    d_v = torch.einsum("bhqk,bqhd->bkhd", probs_drop, d_out_bh)
    dp = torch.einsum("bhqd,bkhd->bhqk", d_out_bh, vf)
    d = (out_bh * d_out_bh).sum(dim=-1, keepdim=True)
    ds = probs_drop * (dp - d) * scale
    d_q = torch.einsum("bhqk,bkhd->bqhd", ds, kf)
    d_k = torch.einsum("bhqk,bqhd->bkhd", ds, qf)
    return out.float(), lse.float(), d_q.float(), d_k.float(), d_v.float()


def _make_mask(device, seed, offset, batch, heads, seqlen_q, seqlen_k, dropout_p):
    n = batch * heads * seqlen_q * seqlen_k
    mask = torch.empty(n, dtype=torch.int8, device=device)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    _dropout_mask_kernel[grid](
        mask, seed, offset, n, dropout_p, BLOCK=256, num_warps=4
    )
    return mask.reshape(batch, heads, seqlen_q, seqlen_k).to(torch.bool)


def _run_forward(q, k, v, scale, causal, dropout_p):
    return flag_gems._flash_attention_forward(
        q,
        k,
        v,
        None,
        None,
        q.shape[1],
        k.shape[1],
        dropout_p,
        causal,
        False,
        scale=scale,
    )


@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_hygon_fp32_flash_attention_forward_and_backward_dropout0(head_size, causal):
    device = flag_gems.device
    batch, seqlen, heads = 1, 64, 2
    scale = 1.0 / math.sqrt(head_size)
    torch.manual_seed(20261111)
    q = torch.randn((batch, seqlen, heads, head_size), dtype=torch.float32).to(device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dout = torch.randn_like(q)

    out, lse, seed, offset, _ = _run_forward(q, k, v, scale, causal, 0.0)
    ref_out, ref_lse, ref_dq, ref_dk, ref_dv = _reference(
        q.cpu(), k.cpu(), v.cpu(), dout.cpu(), scale, causal, 0.0, None
    )
    torch.testing.assert_close(out.cpu(), ref_out, rtol=2e-4, atol=3e-4)
    torch.testing.assert_close(lse.cpu(), ref_lse, rtol=2e-4, atol=3e-4)

    dq, dk, dv = flag_gems.flash_attention_backward(
        dout,
        q,
        k,
        v,
        out,
        lse,
        None,
        None,
        seqlen,
        seqlen,
        0.0,
        causal,
        seed,
        offset,
        scale=scale,
    )
    torch.testing.assert_close(dq.cpu(), ref_dq, rtol=2e-3, atol=3e-3)
    torch.testing.assert_close(dk.cpu(), ref_dk, rtol=2e-3, atol=3e-3)
    torch.testing.assert_close(dv.cpu(), ref_dv, rtol=2e-3, atol=3e-3)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_size", [64, 128])
def test_hygon_fp32_flash_attention_forward_and_backward_dropout(causal, head_size):
    device = flag_gems.device
    batch, seqlen, heads = 1, 64, 2
    dropout_p = 0.2
    scale = 1.0 / math.sqrt(head_size)
    torch.manual_seed(20261112)
    q = torch.randn((batch, seqlen, heads, head_size), dtype=torch.float32).to(device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dout = torch.randn_like(q)

    random_utils.set_philox_state(20261113, 0, torch_device_fn.current_device())
    out, lse, seed, offset, _ = _run_forward(q, k, v, scale, causal, dropout_p)
    seed_i, offset_i = int(seed.item()), int(offset.item())
    mask = _make_mask(device, seed_i, offset_i, batch, heads, seqlen, seqlen, dropout_p)
    ref = _reference(
        q.cpu(), k.cpu(), v.cpu(), dout.cpu(), scale, causal, dropout_p, mask.cpu()
    )
    torch.testing.assert_close(out.cpu(), ref[0], rtol=3e-4, atol=5e-4)
    torch.testing.assert_close(lse.cpu(), ref[1], rtol=3e-4, atol=5e-4)

    dq, dk, dv = flag_gems.flash_attention_backward(
        dout,
        q,
        k,
        v,
        out,
        lse,
        None,
        None,
        seqlen,
        seqlen,
        dropout_p,
        causal,
        seed,
        offset,
        scale=scale,
    )
    torch.testing.assert_close(dq.cpu(), ref[2], rtol=4e-3, atol=5e-3)
    torch.testing.assert_close(dk.cpu(), ref[3], rtol=4e-3, atol=5e-3)
    torch.testing.assert_close(dv.cpu(), ref[4], rtol=4e-3, atol=5e-3)

    # The forward reservation must advance the generator by the dense matrix
    # footprint, rounded to a Philox group of four values.
    expected_increment = (batch * heads * seqlen * seqlen + 3) // 4 * 4
    _, next_offset, _, _, _ = _run_forward(q, k, v, scale, causal, dropout_p)
    assert int(next_offset.item()) - offset_i == expected_increment
