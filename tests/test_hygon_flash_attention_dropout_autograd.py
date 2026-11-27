# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import math

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import random_utils

from .test_hygon_fp16_flash_attention_dropout import (
    _make_mask,
    _reference,
    _run_forward,
)

pytestmark = [
    pytest.mark.underscore_flash_attention_forward,
    pytest.mark.flash_attention_backward,
]


def _autograd_reference(q, k, v, dout, scale, causal, dropout_p, mask):
    # Differentiate only the forward expression. Do not encode the backward
    # formula under test, use a native attention backward, or redraw the mask.
    q, k, v = [x.detach().cpu().double().requires_grad_(True) for x in (q, k, v)]
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    if causal:
        q_idx = torch.arange(q.shape[1])[:, None]
        k_idx = torch.arange(k.shape[1])[None, :]
        scores = scores.masked_fill(
            k_idx > q_idx + k.shape[1] - q.shape[1], -float("inf")
        )
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    if dropout_p:
        probs = probs * mask.detach().cpu().double() / (1.0 - dropout_p)
    out = torch.einsum("bhqk,bkhd->bqhd", probs, v)
    grads = torch.autograd.grad(out, (q, k, v), dout.detach().cpu().double())
    return tuple(x.detach() for x in (out, lse, *grads))


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dropout_p", [0.0, 0.5])
def test_flash_attention_dropout_formula_autograd_control(causal, dropout_p):
    generator = torch.Generator().manual_seed(20261117)
    q, k, v, dout = [
        torch.randn((1, 8, 2, 8), generator=generator, dtype=torch.float64)
        for _ in range(4)
    ]
    mask = torch.rand((1, 2, 8, 8), generator=generator) > dropout_p
    scale = 1.0 / math.sqrt(q.shape[-1])
    ref = _autograd_reference(q, k, v, dout, scale, causal, dropout_p, mask)
    manual = _reference(q, k, v, dout, scale, causal, dropout_p, mask)
    for actual, expected in zip(manual, ref):
        torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-6)

    if dropout_p:
        # Negative control: the former bug replaces the outer softmax P with
        # P_drop. It changes dQ/dK even though the dV expression is correct.
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
        if causal:
            scores = scores.masked_fill(
                torch.ones(8, 8, dtype=torch.bool).triu(1), -float("inf")
            )
        probs = torch.softmax(scores, dim=-1)
        probs_drop = probs * mask / (1.0 - dropout_p)
        dp = torch.einsum("bqhd,bkhd->bhqk", dout, v)
        delta = (probs_drop * dp).sum(dim=-1, keepdim=True)
        wrong_ds = probs_drop * (dp * mask / (1.0 - dropout_p) - delta) * scale
        wrong_dq = torch.einsum("bhqk,bkhd->bqhd", wrong_ds, k)
        wrong_dk = torch.einsum("bhqk,bqhd->bkhd", wrong_ds, q)
        assert not torch.allclose(wrong_dq, ref[2], rtol=3e-2, atol=3e-2)
        assert not torch.allclose(wrong_dk, ref[3], rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon dense FA path")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dropout_p", [0.2, 0.5])
@pytest.mark.parametrize("seqlen,head_size", [(64, 64), (128, 128)])
def test_hygon_dense_flash_attention_dropout_autograd(
    dtype, causal, dropout_p, seqlen, head_size
):
    generator = torch.Generator().manual_seed(20261117)
    q, k, v, dout = [
        torch.randn((1, seqlen, 2, head_size), generator=generator, dtype=dtype).to(
            flag_gems.device
        )
        for _ in range(4)
    ]
    scale = 1.0 / math.sqrt(head_size)
    # A nonzero reservation also checks mask replay beyond counter zero.
    random_utils.set_philox_state(20261117, 1048576, torch_device_fn.current_device())
    out, lse, seed, offset, _ = _run_forward(q, k, v, scale, causal, dropout_p)
    mask = _make_mask(
        q.device, int(seed.item()), int(offset.item()), 1, 2, seqlen, seqlen, dropout_p
    )
    assert bool(mask.any()) and bool((~mask).any())
    ref = _autograd_reference(q, k, v, dout, scale, causal, dropout_p, mask)
    grads = flag_gems.flash_attention_backward(
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
    for i, (actual, expected) in enumerate(zip((out, lse, *grads), ref)):
        assert bool(torch.isfinite(actual).all())
        assert actual.shape == expected.shape
        if dtype == torch.float16:
            rtol, atol = (1.2e-2, 1.2e-2) if i < 2 else (3e-2, 3e-2)
        else:
            rtol, atol = (3e-4, 5e-4) if i < 2 else (4e-3, 5e-3)
        torch.testing.assert_close(
            actual.cpu().double(), expected, rtol=rtol, atol=atol
        )
