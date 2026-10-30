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

"""CPU-reference regressions for Ascend attention UB and tail handling."""

import math

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend attention regression"
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", [64, 72, 128, 256])
def test_varlen_forward_backward_padded_heads(dtype, causal, head_dim):
    # Irregular lengths exercise both masked and interior KV tiles, plus the
    # early return for shorter sequences. Four query heads also exercise GQA.
    lengths = [7, 65, 257]
    heads, kv_heads = 4, 2
    torch.manual_seed(17)
    total = sum(lengths)
    cpu_inputs = [
        torch.randn(total, h, head_dim).to(dtype).float().requires_grad_()
        for h in (heads, kv_heads, kv_heads)
    ]
    q, k, v = [
        x.detach().to(device=flag_gems.device, dtype=dtype) for x in cpu_inputs
    ]
    cu_cpu = torch.tensor([0, 7, 72, total], dtype=torch.int32)
    cu = cu_cpu.to(flag_gems.device)
    out, lse = flag_gems.flash_attn_varlen_func(
        q,
        k,
        v,
        max(lengths),
        cu,
        max(lengths),
        cu,
        causal=causal,
        return_softmax_lse=True,
    )

    ref_out, ref_lse = [], []
    for start, end in zip(cu_cpu.tolist()[:-1], cu_cpu.tolist()[1:]):
        qr, kr, vr = [x[start:end].transpose(0, 1) for x in cpu_inputs]
        kr = kr.repeat_interleave(heads // kv_heads, dim=0)
        vr = vr.repeat_interleave(heads // kv_heads, dim=0)
        scores = qr @ kr.transpose(-1, -2) * head_dim**-0.5
        if causal:
            mask = torch.ones(end - start, end - start, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        ref_out.append((scores.softmax(-1) @ vr).transpose(0, 1))
        ref_lse.append(scores.logsumexp(-1))
    ref_out = torch.cat(ref_out)
    ref_lse = torch.cat(ref_lse, dim=1)
    atol, rtol = (0.02, 0.02) if dtype == torch.bfloat16 else (0.003, 0.003)
    torch.testing.assert_close(out.cpu().float(), ref_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(lse.cpu(), ref_lse, atol=0.002, rtol=0.002)

    grad_cpu = torch.randn_like(ref_out).to(dtype).float()
    ref_out.backward(grad_cpu)
    grads = flag_gems.flash_attention_backward(
        grad_cpu.to(device=flag_gems.device, dtype=dtype),
        q,
        k,
        v,
        out,
        lse.transpose(0, 1).contiguous(),
        cu,
        cu,
        max(lengths),
        max(lengths),
        0.0,
        causal,
        None,
        None,
        scale=head_dim**-0.5,
    )
    for name, actual, source in zip(("dq", "dk", "dv"), grads, cpu_inputs):
        torch.testing.assert_close(
            actual.cpu().float(),
            source.grad,
            atol=atol,
            rtol=rtol,
            msg=lambda msg: f"{name}: {msg}",
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "head_dim,seq_len,heads,kv_heads",
    [
        (16, 65, 2, 2),
        (64, 65, 4, 2),
        (128, 65, 4, 2),
        (256, 65, 6, 1),
        (256, 256, 6, 1),
    ],
)
def test_dense_forward_backward(dtype, causal, head_dim, seq_len, heads, kv_heads):
    # Qwen's local text heads are Q:KV=6:1 with D=256. Tail lengths also
    # exercise masked dO loads in the backward diagonal and interior tiles.
    torch.manual_seed(29)
    cpu_inputs = [
        torch.randn(1, h, seq_len, head_dim).to(dtype).float().requires_grad_()
        for h in (heads, kv_heads, kv_heads)
    ]
    q, k, v = [
        x.detach().to(device=flag_gems.device, dtype=dtype) for x in cpu_inputs
    ]
    out, lse_log2 = flag_gems.scaled_dot_product_attention_forward(
        q, k, v, is_causal=causal, enable_gqa=True
    )
    qr, kr, vr = cpu_inputs
    kr = kr.repeat_interleave(heads // kv_heads, dim=1)
    vr = vr.repeat_interleave(heads // kv_heads, dim=1)
    scores = qr @ kr.transpose(-1, -2) * head_dim**-0.5
    if causal:
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    ref_out = scores.softmax(-1) @ vr
    atol, rtol = (0.03, 0.03) if dtype == torch.bfloat16 else (0.004, 0.004)
    torch.testing.assert_close(out.cpu().float(), ref_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        lse_log2.cpu() * math.log(2), scores.logsumexp(-1), atol=0.002, rtol=0.002
    )
    grad_cpu = torch.randn_like(ref_out).to(dtype).float()
    ref_out.backward(grad_cpu)
    grads = flag_gems.scaled_dot_product_attention_backward(
        grad_cpu.to(device=flag_gems.device, dtype=dtype),
        q,
        k,
        v,
        out,
        lse_log2,
        is_causal=causal,
        enable_gqa=True,
    )
    for name, actual, source in zip(("dq", "dk", "dv"), grads, cpu_inputs):
        torch.testing.assert_close(
            actual.cpu().float(),
            source.grad,
            atol=atol,
            rtol=rtol,
            msg=lambda msg: f"{name}: {msg}",
        )
