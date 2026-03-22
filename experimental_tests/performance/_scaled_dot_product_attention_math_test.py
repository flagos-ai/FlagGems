# _scaled_dot_product_attention_math performance benchmark

import math
import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops._scaled_dot_product_attention_math import (
    _scaled_dot_product_attention_math as gems_sdpa_math,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import TO_CPU  # noqa: E402
except ImportError:
    TO_CPU = False


def ref_sdpa_math(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """PyTorch reference (math backend) for benchmark baseline."""
    q_fp = q.to(torch.float32)
    k_fp = k.to(torch.float32)
    v_fp = v.to(torch.float32)

    logits = torch.matmul(q_fp, k_fp.transpose(-2, -1))
    if scale is None:
        logits = logits * (1.0 / math.sqrt(q_fp.size(-1)))
    else:
        logits = logits * scale

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            logits = logits.masked_fill(~attn_mask, float("-inf"))
        else:
            logits = logits + attn_mask.to(dtype=logits.dtype)

    if is_causal:
        L_q = logits.size(-2)
        L_k = logits.size(-1)
        causal_mask = torch.tril(
            torch.ones((L_q, L_k), dtype=torch.bool, device=logits.device)
        )
        logits = logits.masked_fill(~causal_mask, float("-inf"))

    attn = torch.softmax(logits, dim=-1)
    out = torch.matmul(attn, v_fp)
    return out.to(dtype=q.dtype)


@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.parametrize(
    "B, H, S, D",
    [
        (1, 8, 64, 64),
        (2, 8, 128, 64),
        (2, 8, 256, 64),
        (2, 8, 512, 64),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_perf__scaled_dot_product_attention_math(B, H, S, D, dtype):
    q = torch.randn((B, H, S, D), dtype=dtype, device=flag_gems.device)
    k = torch.randn((B, H, S, D), dtype=dtype, device=flag_gems.device)
    v = torch.randn((B, H, S, D), dtype=dtype, device=flag_gems.device)

    # Warmup
    for _ in range(3):
        _ = gems_sdpa_math(q, k, v, dropout_p=0.0, is_causal=False)
        _ = ref_sdpa_math(q, k, v, dropout_p=0.0, is_causal=False)
    torch.cuda.synchronize()

    quantiles = [0.5, 0.2, 0.8]
    ms_triton, _, _ = triton.testing.do_bench(
        lambda: gems_sdpa_math(q, k, v, dropout_p=0.0, is_causal=False),
        rep=100,
        quantiles=quantiles,
    )
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: ref_sdpa_math(q, k, v, dropout_p=0.0, is_causal=False),
        rep=100,
        quantiles=quantiles,
    )

    speedup = ms_torch / ms_triton if ms_triton > 0 else float("inf")
    print(
        f"\n[B={B}, H={H}, S={S}, D={D}, dtype={dtype}] "
        f"PyTorch: {ms_torch:.4f}ms | Triton: {ms_triton:.4f}ms | Speedup: {speedup:.2f}x"
    )
