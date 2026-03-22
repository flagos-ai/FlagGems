# _scaled_dot_product_attention_math operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops._scaled_dot_product_attention_math import (
    _scaled_dot_product_attention_math as gems_sdpa_math,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close  # noqa: E402
except ImportError:
    TO_CPU = False

    def gems_assert_close(res, ref, dtype, **kwargs):
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    if TO_CPU:
        ref_inp = inp.to("cpu")
    else:
        ref_inp = inp.clone()
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


def ref_sdpa_math(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """PyTorch reference implementation for accuracy comparison."""
    import math

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
    "B, H, S_q, S_kv, D",
    [
        (1, 1, 8, 8, 16),
        (2, 4, 16, 32, 64),
        (2, 2, 32, 16, 32),
        (1, 8, 64, 64, 64),
        (2, 4, 128, 128, 128),
    ],
)
@pytest.mark.parametrize(
    "has_mask, is_causal",
    [(False, False), (True, False), (False, True)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test__scaled_dot_product_attention_math(
    B, H, S_q, S_kv, D, has_mask, is_causal, dtype
):
    q = torch.randn((B, H, S_q, D), dtype=dtype, device=flag_gems.device)
    k = torch.randn((B, H, S_kv, D), dtype=dtype, device=flag_gems.device)
    v = torch.randn((B, H, S_kv, D), dtype=dtype, device=flag_gems.device)

    attn_mask = None
    if has_mask:
        attn_mask = torch.zeros((B, H, S_q, S_kv), dtype=dtype, device=flag_gems.device)
        if S_kv > 1:
            attn_mask[:, :, :, S_kv // 2:] = float("-inf")

    ref_out = ref_sdpa_math(
        to_reference(q),
        to_reference(k),
        to_reference(v),
        to_reference(attn_mask),
        dropout_p=0.0,
        is_causal=is_causal,
        scale=None,
    )
    res_out = gems_sdpa_math(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=None,
    )

    if dtype == torch.float32:
        rtol, atol = 1e-2, 1e-2  # tl.dot tiling introduces ~1e-3 fp32 error
    elif dtype == torch.float16:
        rtol, atol = 1e-2, 1e-2
    else:  # bfloat16
        rtol, atol = 2e-2, 2e-2

    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)
