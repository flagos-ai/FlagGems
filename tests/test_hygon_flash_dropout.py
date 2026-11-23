import math

import pytest
import torch

import flag_gems

pytestmark = [
    pytest.mark.flash_attention_backward,
    pytest.mark.skipif(
        flag_gems.vendor_name != "hygon", reason="Hygon native dropout protocol"
    ),
]


def _backward(*args, **kwargs):
    from flag_gems.runtime.backend._hygon.gfx936.ops.flash_attention_backward import (
        flash_attention_backward,
    )

    return flash_attention_backward(*args, **kwargs)


CASES = [
    (1, 32, 2, 2, 0.2, False, False),
    (1, 64, 2, 2, 0.0, False, False),
    (1, 64, 2, 2, 0.1, False, False),
    (1, 64, 2, 2, 0.2, False, False),
    (1, 64, 2, 2, 0.5, False, False),
    (2, 64, 3, 3, 0.2, False, False),
    (1, 64, 2, 2, 0.2, True, False),
    (1, 64, 2, 2, 0.2, False, True),
    (1, 64, 4, 2, 0.2, False, False),
    (1, 128, 2, 2, 0.2, False, False),
    (1, 256, 2, 2, 0.2, False, False),
    (1, 128, 2, 2, 0.5, True, False),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch,length,heads,kv_heads,dropout,causal,sliced", CASES)
def test_hygon_native_dropout_backward(
    dtype, batch, length, heads, kv_heads, dropout, causal, sliced
):
    generator = torch.Generator().manual_seed(859400 + length)
    dim = 64
    backing_length = length * (2 if sliced else 1)
    q_base = torch.randn(batch, backing_length, heads, dim, generator=generator) * 0.25
    k_base = (
        torch.randn(batch, backing_length, kv_heads, dim, generator=generator) * 0.25
    )
    if length > dim:
        # Orthogonal feature ranges give uniform logits while all three gradients
        # remain sensitive to the dropout mask.
        q_base[..., dim // 2 :] = 0
        k_base[..., : dim // 2] = 0
    v_base = torch.zeros(batch, backing_length, kv_heads, dim)
    for key in range(length):
        v_base[:, key * (2 if sliced else 1), :, key % dim] = 2 ** (key // dim)

    def move(base):
        lowp = base.to(dtype)
        device = lowp.to(flag_gems.device)
        if sliced:
            return lowp[:, ::2], device[:, ::2]
        return lowp, device

    cpu_q, q = move(q_base)
    cpu_k, k = move(k_base)
    cpu_v, v = move(v_base)
    forward = torch.ops.aten._flash_attention_forward.default(
        q, k, v, None, None, length, length, dropout, causal, False
    )
    output, lse, seed, offset, _ = forward
    actual = output.detach().cpu().float().transpose(1, 2)
    if length <= dim:
        mask = (actual[..., :length] > 0).float()
    else:
        # Each value column encodes at most four mask bits. With uniform logits,
        # native output directly reveals the actual forward draw; no RNG replica
        # or independently sampled dropout mask enters the reference.
        denominator = (
            torch.arange(1, length + 1).view(1, 1, length, 1) if causal else length
        )
        encoded = actual * denominator * (1 - dropout)
        rounded = encoded.round()
        assert (encoded - rounded).abs().max().item() < 0.2
        keys = torch.arange(length)
        mask = ((rounded.long()[..., keys % dim] >> (keys // dim)) & 1).float()

    qr = cpu_q.float().transpose(1, 2).detach().requires_grad_()
    kr = cpu_k.float().transpose(1, 2).detach().requires_grad_()
    vr = cpu_v.float().transpose(1, 2).detach().requires_grad_()
    repeated_k = kr.repeat_interleave(heads // kv_heads, dim=1)
    repeated_v = vr.repeat_interleave(heads // kv_heads, dim=1)
    logits = qr @ repeated_k.transpose(-1, -2) / math.sqrt(dim)
    if causal:
        logits = logits.masked_fill(
            torch.ones(length, length, dtype=torch.bool).triu(1), -float("inf")
        )
    reference = (logits.softmax(-1) * mask / (1 - dropout)) @ repeated_v
    forward_atol = 0.002 if dtype == torch.float16 else 0.012
    torch.testing.assert_close(actual, reference.detach(), rtol=0.04, atol=forward_atol)

    grad_cpu = torch.randn(batch, length, heads, dim, generator=generator).to(dtype)
    reference.backward(grad_cpu.float().transpose(1, 2))
    gradients = _backward(
        grad_cpu.to(flag_gems.device),
        q,
        k,
        v,
        output,
        lse,
        None,
        None,
        length,
        length,
        dropout,
        causal,
        seed,
        offset,
    )
    backward_atol = 0.02 if dtype == torch.float16 else 0.12
    for actual_grad, ref_input in zip(gradients, (qr, kr, vr)):
        result = actual_grad.cpu().float()
        assert torch.isfinite(result).all()
        torch.testing.assert_close(
            result, ref_input.grad.transpose(1, 2), rtol=0.08, atol=backward_atol
        )


@pytest.mark.parametrize("guard", ["fp32", "packed", "nonpositive_scale"])
def test_hygon_dropout_capability_guards(guard):
    dtype = torch.float32 if guard == "fp32" else torch.float16
    shape = (4, 1, 64) if guard == "packed" else (1, 4, 1, 64)
    q = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    lse = torch.zeros((1, 1, 4), dtype=torch.float32, device=flag_gems.device)
    seed = torch.zeros((), dtype=torch.int64, device=flag_gems.device)
    scale = 0.0 if guard == "nonpositive_scale" else None
    error = ValueError if guard == "fp32" else NotImplementedError
    message = (
        "matching float16 or bfloat16" if guard == "fp32" else "dense positive-scale"
    )
    with pytest.raises(error, match=message):
        _backward(
            q, q, q, q, q, lse, None, None, 4, 4, 0.2, False, seed, seed, scale=scale
        )
