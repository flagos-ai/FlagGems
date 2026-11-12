# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import math

import pytest
import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import random_utils

pytestmark = [
    pytest.mark._flash_attention_forward,
    pytest.mark.flash_attention_backward,
    pytest.mark.skipif(
        flag_gems.vendor_name != "hygon", reason="Hygon-specific FP16 FA path"
    ),
]


@triton.jit
def _dropout_mask_kernel(
    out, seed, base_offset, n_elements, dropout_p, BLOCK: tl.constexpr
):
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
        scores = scores.masked_fill(
            k_idx > q_idx + k.shape[1] - q.shape[1], -float("inf")
        )
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)

    do = dout.double()
    d_out_bh = do.permute(0, 2, 1, 3)
    dp = torch.einsum("bhqd,bkhd->bhqk", d_out_bh, vf)
    if dropout_p > 0:
        mask_f = mask.double()
        probs_drop = probs * mask_f / (1.0 - dropout_p)
        dp_drop = dp * mask_f / (1.0 - dropout_p)
    else:
        probs_drop = probs
        dp_drop = dp

    out = torch.einsum("bhqk,bkhd->bqhd", probs_drop, vf)
    d_v = torch.einsum("bhqk,bhqd->bkhd", probs_drop, d_out_bh)
    d = (probs_drop * dp).sum(dim=-1, keepdim=True)
    ds = probs * (dp_drop - d) * scale
    d_q = torch.einsum("bhqk,bkhd->bqhd", ds, kf)
    d_k = torch.einsum("bhqk,bqhd->bkhd", ds, qf)
    return out.float(), lse.float(), d_q.float(), d_k.float(), d_v.float()


def _make_mask(device, seed, offset, batch, heads, seqlen_q, seqlen_k, dropout_p):
    n = batch * heads * seqlen_q * seqlen_k
    mask = torch.empty(n, dtype=torch.int8, device=device)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    _dropout_mask_kernel[grid](mask, seed, offset, n, dropout_p, BLOCK=256, num_warps=4)
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


def _assert_finite(*tensors):
    for tensor in tensors:
        assert bool(
            torch.isfinite(tensor).all()
        ), "FA result contains non-finite values"


@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_hygon_fp16_flash_attention_forward_and_backward_dropout0(head_size, causal):
    device = flag_gems.device
    batch, seqlen, heads = 1, 64, 2
    scale = 1.0 / math.sqrt(head_size)
    torch.manual_seed(20261114)
    q = torch.randn((batch, seqlen, heads, head_size), dtype=torch.float16).to(device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dout = torch.randn_like(q)

    out, lse, seed, offset, _ = _run_forward(q, k, v, scale, causal, 0.0)
    ref_out, ref_lse, ref_dq, ref_dk, ref_dv = _reference(
        q.cpu(), k.cpu(), v.cpu(), dout.cpu(), scale, causal, 0.0, None
    )
    _assert_finite(out, lse)
    assert out.shape == q.shape
    assert out.stride() == q.stride()
    torch.testing.assert_close(out.cpu().float(), ref_out, rtol=8e-3, atol=8e-3)
    torch.testing.assert_close(lse.cpu(), ref_lse, rtol=5e-3, atol=5e-3)

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
    _assert_finite(dq, dk, dv)
    torch.testing.assert_close(dq.cpu().float(), ref_dq, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(dk.cpu().float(), ref_dk, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(dv.cpu().float(), ref_dv, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_hygon_fp16_flash_attention_forward_and_backward_dropout(head_size, causal):
    device = flag_gems.device
    batch, seqlen, heads = 1, 64, 2
    dropout_p = 0.2
    scale = 1.0 / math.sqrt(head_size)
    torch.manual_seed(20261115)
    q = torch.randn((batch, seqlen, heads, head_size), dtype=torch.float16).to(device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dout = torch.randn_like(q)

    random_utils.set_philox_state(20261116, 0, torch_device_fn.current_device())
    out, lse, seed, offset, _ = _run_forward(q, k, v, scale, causal, dropout_p)
    seed_i, offset_i = int(seed.item()), int(offset.item())
    mask = _make_mask(device, seed_i, offset_i, batch, heads, seqlen, seqlen, dropout_p)
    ref = _reference(
        q.cpu(), k.cpu(), v.cpu(), dout.cpu(), scale, causal, dropout_p, mask.cpu()
    )
    _assert_finite(out, lse)
    assert out.shape == q.shape
    assert out.stride() == q.stride()
    torch.testing.assert_close(out.cpu().float(), ref[0], rtol=1.2e-2, atol=1.2e-2)
    torch.testing.assert_close(lse.cpu(), ref[1], rtol=5e-3, atol=5e-3)

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
    _assert_finite(dq, dk, dv)
    torch.testing.assert_close(dq.cpu().float(), ref[2], rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(dk.cpu().float(), ref[3], rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(dv.cpu().float(), ref[4], rtol=3e-2, atol=3e-2)

    expected_increment = (batch * heads * seqlen * seqlen + 3) // 4 * 4
    _, _, _, next_offset, _ = _run_forward(q, k, v, scale, causal, dropout_p)
    assert int(next_offset.item()) - offset_i == expected_increment


FP16_EXTENDED_CASES = [
    pytest.param(1, 32, 32, 2, 64, False, id="b1-s32-h2-d64-noncausal"),
    pytest.param(1, 32, 32, 2, 64, True, id="b1-s32-h2-d64-causal"),
    pytest.param(2, 64, 64, 2, 64, False, id="b2-s64-h2-d64-noncausal"),
    pytest.param(2, 64, 64, 4, 128, True, id="b2-s64-h4-d128-causal"),
    pytest.param(1, 128, 128, 2, 64, False, id="b1-s128-h2-d64-noncausal"),
    pytest.param(1, 128, 128, 2, 128, True, id="b1-s128-h2-d128-causal"),
]


@pytest.mark.parametrize(
    "batch, seqlen_q, seqlen_k, heads, head_size, causal", FP16_EXTENDED_CASES
)
@pytest.mark.parametrize("dropout_p", [0.1, 0.2, 0.5])
def test_hygon_fp16_flash_attention_dropout_extended(
    batch, seqlen_q, seqlen_k, heads, head_size, causal, dropout_p
):
    """Exercise multiple batch/head sizes, causal modes and dropout ratios."""
    device = flag_gems.device
    scale = 1.0 / math.sqrt(head_size)
    seed_base = 20261200 + batch + seqlen_q + seqlen_k + heads + head_size
    torch.manual_seed(seed_base)
    q = torch.randn(
        (batch, seqlen_q, heads, head_size), dtype=torch.float16, device=device
    )
    k = torch.randn(
        (batch, seqlen_k, heads, head_size), dtype=torch.float16, device=device
    )
    v = torch.randn_like(k)
    dout = torch.randn_like(q)

    random_utils.set_philox_state(seed_base + 1, 0, torch_device_fn.current_device())
    out, lse, seed, offset, _ = _run_forward(q, k, v, scale, causal, dropout_p)
    seed_i, offset_i = int(seed.item()), int(offset.item())
    mask = _make_mask(
        device, seed_i, offset_i, batch, heads, seqlen_q, seqlen_k, dropout_p
    )
    ref = _reference(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        dout.cpu(),
        scale,
        causal,
        dropout_p,
        mask.cpu(),
    )

    _assert_finite(out, lse)
    assert out.shape == q.shape
    assert out.stride() == q.stride()
    torch.testing.assert_close(out.cpu().float(), ref[0], rtol=1.5e-2, atol=1.5e-2)
    torch.testing.assert_close(lse.cpu(), ref[1], rtol=8e-3, atol=8e-3)

    dq, dk, dv = flag_gems.flash_attention_backward(
        dout,
        q,
        k,
        v,
        out,
        lse,
        None,
        None,
        seqlen_q,
        seqlen_k,
        dropout_p,
        causal,
        seed,
        offset,
        scale=scale,
    )
    _assert_finite(dq, dk, dv)
    torch.testing.assert_close(dq.cpu().float(), ref[2], rtol=4e-2, atol=4e-2)
    torch.testing.assert_close(dk.cpu().float(), ref[3], rtol=4e-2, atol=4e-2)
    torch.testing.assert_close(dv.cpu().float(), ref[4], rtol=4e-2, atol=4e-2)

    _, _, _, next_offset, _ = _run_forward(q, k, v, scale, causal, dropout_p)
    expected_increment = (batch * heads * seqlen_q * seqlen_k + 3) // 4 * 4
    assert int(next_offset.item()) - offset_i == expected_increment


@pytest.mark.parametrize("causal", [False, True])
def test_hygon_fp16_flash_attention_dropout_replay(causal):
    """Verify deterministic Philox replay for the direct low-level FA entry."""
    device = flag_gems.device
    batch, seqlen, heads, head_size = 1, 32, 2, 64
    dropout_p = 0.2
    scale = 1.0 / math.sqrt(head_size)
    torch.manual_seed(20261240 + int(causal))
    q = torch.randn(
        (batch, seqlen, heads, head_size), dtype=torch.float16, device=device
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    random_utils.set_philox_state(
        20261250 + int(causal), 0, torch_device_fn.current_device()
    )
    first = _run_forward(q, k, v, scale, causal, dropout_p)
    random_utils.set_philox_state(
        20261250 + int(causal), 0, torch_device_fn.current_device()
    )
    second = _run_forward(q, k, v, scale, causal, dropout_p)

    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
    assert int(first[2].item()) == int(second[2].item())
    assert int(first[3].item()) == int(second[3].item())
