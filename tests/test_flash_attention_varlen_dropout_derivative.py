# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import pytest
import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.ops.flash_attention_backward import flash_attn_backward


@triton.jit
def _backward_mask(
    output,
    seed,
    M: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    head, row, column = i // (M * N), i // N % M, i % N
    offset = head * N + row * N + column
    tl.store(output + i, tl.rand(seed, offset) > 0.5, i < H * M * N)


@pytest.mark.flash_attention_backward
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia", reason="direct generic NVIDIA core regression"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dropout_p", [0.0, 0.5])
def test_varlen_dropout_derivative_identical_keys(dtype, dropout_p):
    m, n, heads, width, seed = 128, 256, 2, 32, 7331
    q = torch.zeros((1, m, heads, width), dtype=dtype, device=flag_gems.device)
    k = torch.full((1, n, heads, width), 0.125, dtype=dtype, device=flag_gems.device)
    v = torch.ones_like(k)
    grad = torch.full_like(q, 1 / q.numel())
    keep = torch.ones((1, heads, m, n), dtype=torch.bool, device=flag_gems.device)
    if dropout_p:
        _backward_mask[(triton.cdiv(keep.numel(), 256),)](keep, seed, m, n, heads, 256)
    weights = keep.double() / (n * (1 - dropout_p))
    out = (
        weights.sum(-1)
        .permute(0, 2, 1)
        .unsqueeze(-1)
        .expand_as(q)
        .to(dtype)
        .contiguous()
    )
    lse = torch.full(
        (1, heads, m), math.log(n), dtype=torch.float32, device=flag_gems.device
    )
    expected_dv = (
        weights.sum(-2).permute(0, 2, 1).unsqueeze(-1).expand_as(v) / q.numel()
    )
    options = dict(
        max_seqlen_q=m,
        max_seqlen_k=n,
        is_dropout=bool(dropout_p),
        dropout_p=dropout_p,
        rng_state=(seed, 0),
        softmax_scale=width**-0.5,
    )
    args = (grad, q, k, v, out, lse)
    args = tuple(t.squeeze(0) for t in args[:5]) + (lse.squeeze(0).T.contiguous(),)
    options.update(
        cu_seq_q=torch.tensor([0, m], dtype=torch.int32, device=flag_gems.device),
        cu_seq_k=torch.tensor([0, n], dtype=torch.int32, device=flag_gems.device),
    )
    dq, dk, dv, _ = flash_attn_backward(*args, **options)
    dv = dv.unsqueeze(0)
    # Identical keys make Q irrelevant for a fixed mask; Q=0 also makes dK zero.
    torch.testing.assert_close(dq, torch.zeros_like(dq), rtol=0, atol=1e-6)
    torch.testing.assert_close(dk, torch.zeros_like(dk), rtol=0, atol=1e-6)
    torch.testing.assert_close(
        dv.double(),
        expected_dv,
        rtol=0.016 if dtype == torch.bfloat16 else 0.001,
        atol=1e-7,
    )


@pytest.mark.flash_attention_backward
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia", reason="direct generic NVIDIA core regression"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dropout_p", [0.0, 0.5])
def test_varlen_dropout_derivative_nonzero_qkv(dtype, dropout_p):
    torch.manual_seed(197)
    m, n, heads, width, seed = 128, 256, 2, 32, 7331
    q = 0.5 * torch.randn((1, m, heads, width), dtype=dtype, device=flag_gems.device)
    k = 0.5 * torch.randn((1, n, heads, width), dtype=dtype, device=flag_gems.device)
    v = 0.5 * torch.randn_like(k)
    # An unscaled cotangent directly constrains the nonzero dK computation.
    grad = 0.5 * torch.randn_like(q)
    keep = torch.ones((1, heads, m, n), dtype=torch.bool, device=flag_gems.device)
    if dropout_p:
        _backward_mask[(triton.cdiv(keep.numel(), 256),)](keep, seed, m, n, heads, 256)
    q_ref, k_ref, v_ref = [x.double().requires_grad_() for x in (q, k, v)]
    scores = (
        q_ref.transpose(1, 2) @ k_ref.transpose(1, 2).transpose(-1, -2)
    ) * width**-0.5
    probability = scores.softmax(-1)
    dropped = probability * keep.double() / (1 - dropout_p)
    reference = (dropped @ v_ref.transpose(1, 2)).transpose(1, 2)
    expected = torch.autograd.grad(reference, (q_ref, k_ref, v_ref), grad.double())
    out = reference.detach().to(dtype).contiguous()
    lse = scores.detach().logsumexp(-1).float()
    options = dict(
        max_seqlen_q=m,
        max_seqlen_k=n,
        is_dropout=bool(dropout_p),
        dropout_p=dropout_p,
        rng_state=(seed, 0),
        softmax_scale=width**-0.5,
    )
    args = (grad, q, k, v, out, lse)
    args = tuple(t.squeeze(0) for t in args[:5]) + (lse.squeeze(0).T.contiguous(),)
    options.update(
        cu_seq_q=torch.tensor([0, m], dtype=torch.int32, device=flag_gems.device),
        cu_seq_k=torch.tensor([0, n], dtype=torch.int32, device=flag_gems.device),
    )
    actual = flash_attn_backward(*args, **options)[:3]
    for name, result, target in zip(("dQ", "dK"), actual[:2], expected[:2]):
        result = result.unsqueeze(0)
        torch.testing.assert_close(
            result.double(),
            target,
            rtol=0.016 if dtype == torch.bfloat16 else 0.001,
            atol=1e-4,
            msg=lambda message: name + ": " + message,
        )
    result = actual[2].unsqueeze(0)
    # dV rounds P_drop to the input dtype before its tensor-core dot. For
    # normal probabilities the cast contributes at most u * sum(abs(P_drop*dO)).
    # Keep the ordinary matmul/output tolerance in addition to that component.
    nonzero = dropped.detach()[dropped.detach() != 0]
    assert nonzero.min() >= torch.finfo(dtype).tiny
    unit_roundoff = torch.finfo(dtype).eps / 2
    cast_budget = unit_roundoff * (
        dropped.detach().abs().transpose(-1, -2) @ grad.double().transpose(1, 2).abs()
    ).transpose(1, 2)
    rtol = 0.016 if dtype == torch.bfloat16 else 0.001
    allowance = cast_budget + 1e-4 + rtol * expected[2].abs()
    error = (result.double() - expected[2]).abs()
    assert torch.all(
        error <= allowance
    ), "dV exceeds the probability-cast and output-rounding budget"
