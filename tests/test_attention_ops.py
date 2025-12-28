import math
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import pytest
import torch
import triton

import flag_gems

try:
    import vllm.vllm_flash_attn.flash_attn_interface as _vllm  # noqa: F401

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import set_philox_state

from .accuracy_utils import gems_assert_close, init_seed, to_reference
from .conftest import TO_CPU

device = flag_gems.device


def make_input(
    batch,
    num_head,
    num_head_k,
    q_seq_len,
    kv_seq_len,
    head_size,
    dtype,
    device,
    requires_grad=False,
):
    set_philox_state(1234567890, 0, device)
    q_shape = (batch, num_head, q_seq_len, head_size)
    kv_shape = (batch, num_head_k, kv_seq_len, head_size)
    q = torch.empty(q_shape, dtype=dtype, device=device).uniform_(-0.05, 0.05)
    k = torch.empty(kv_shape, dtype=dtype, device=device).uniform_(-0.05, 0.05)
    v = torch.empty(kv_shape, dtype=dtype, device=device).uniform_(-0.05, 0.05)
    if requires_grad:
        q.requires_grad_()
        k.requires_grad_()
        v.requires_grad_()
    return q, k, v


# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py
def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    # row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    row_idx = torch.arange(seqlen_q, device=device, dtype=torch.long)[:, None]
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        # key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        key_leftpad = key_leftpad[:, None, None, None]
        # col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = col_idx.repeat(key_leftpad.shape[0], 1, 1, 1)
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        # else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
        else key_padding_mask.sum(-1)[:, None, None, None]
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        # else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
        else query_padding_mask.sum(-1)[:, None, None, None]
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_ref(
    q,
    k,
    v,
    scale,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        scale: float
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    import math

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    q *= scale

    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    g = q.shape[2] // k.shape[2]
    # k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    # v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    k = k.repeat_interleave(g, dim=2)
    v = v.repeat_interleave(g, dim=2)
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    if key_padding_mask is not None:
        scores.masked_fill_((~key_padding_mask)[:, None, None, :], float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(
            torch.all(local_mask, dim=-1, keepdim=True), 0.0
        )
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        mask = (~query_padding_mask)[:, None, :, None]
        attention = attention.masked_fill(mask, 0.0)

    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_((~query_padding_mask)[:, :, None, None], 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def torch_sdpa(q, k, v, scale, is_causal, enable_gqa=False):
    if torch.__version__ < "2.5":
        torch_result = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            scale=scale,
            is_causal=is_causal,
        )
    else:
        if flag_gems.vendor_name == "iluvatar" and TO_CPU:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            ctx = sdpa_kernel(backends=[SDPBackend.MATH])
        else:
            from contextlib import nullcontext

            ctx = nullcontext()
        with ctx:
            torch_result = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                scale=scale,
                is_causal=is_causal,
                enable_gqa=enable_gqa,
            )
    return torch_result


def torch_flash_fwd(
    q, k, v, scale, is_causal, dropout_p=0, return_debug_mask=False, **extra_kwargs
):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    (
        out,
        lse,
        seed,
        offset,
        debug_softmax,
    ) = torch.ops.aten._flash_attention_forward(
        q,
        k,
        v,
        None,
        None,
        q.shape[-3],
        k.shape[-3],
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=scale,
        **extra_kwargs,
    )

    return out, lse, seed, offset, debug_softmax


def gems_flash_fwd(
    q, k, v, scale, is_causal, dropout_p=0, return_debug_mask=False, **extra_kwargs
):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    (
        out,
        lse,
        seed,
        offset,
        debug_softmax,
    ) = flag_gems.ops.flash_attention_forward(
        q,
        k,
        v,
        None,
        None,
        q.shape[-3],
        k.shape[-3],
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=scale,
        **extra_kwargs,
    )

    return out, lse, seed, offset, debug_softmax


# @pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(
    torch.__version__ < "2.5", reason="Low Pytorch Version: enable_gqa not supported"
)
@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize(
    "batch, num_q_head, num_kv_head, q_seq_len, kv_seq_len, head_size, enable_gqa",
    [
        (4, 8, 8, 1024, 1024, 64, False),
        (4, 8, 8, 1024, 1024, 128, False),
        (4, 8, 8, 2048, 256, 64, False),
        (4, 8, 8, 2048, 256, 128, False),
        (4, 8, 8, 17, 1030, 64, False),
        (4, 8, 8, 17, 1030, 128, False),
        # adopted from FlagAttention `test_attention_fwd`:
        (2, 4, 4, 512, 612, 128, False),
        (2, 4, 4, 1024, 1034, 64, False),
        (2, 4, 4, 2048, 2048, 32, False),
        (2, 4, 4, 4096, 4096, 16, False),
        (2, 4, 4, 4001, 4001, 32, False),
        (2, 4, 4, 4001, 4096, 64, False),
        (2, 4, 4, 4096, 4000, 128, False),
        (1, 2, 2, 8192, 8202, 16, False),
        (1, 2, 2, 8192, 8192, 32, False),
        # test for mqa/gqa
        (2, 4, 2, 512, 612, 128, True),
        (2, 4, 1, 1024, 1034, 64, True),
        (2, 4, 2, 2048, 2048, 32, True),
        (2, 4, 1, 4096, 4096, 16, True),
        (2, 4, 2, 4001, 4001, 32, True),
        (2, 4, 1, 4001, 4096, 64, True),
        (2, 4, 2, 4096, 4000, 128, True),
        (1, 2, 1, 8192, 8202, 16, True),
        (1, 2, 1, 8192, 8192, 32, True),
    ],
)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sdpa_legacy(
    batch,
    num_q_head,
    num_kv_head,
    q_seq_len,
    kv_seq_len,
    head_size,
    is_causal,
    dtype,
    enable_gqa,
):
    if flag_gems.vendor_name == "hygon":
        os.environ["TRITON_HIP_USE_NEW_STREAM_PIPELINE"] = "0"
    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch,
        num_q_head,
        num_kv_head,
        q_seq_len,
        kv_seq_len,
        head_size,
        dtype,
        device,
        requires_grad=True,
    )
    ref_q = to_reference(q, False)
    ref_k = to_reference(k, False)
    ref_v = to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))

    # forward
    torch_result = torch_sdpa(
        ref_q, ref_k, ref_v, scale, is_causal, enable_gqa=enable_gqa
    )

    gems_result = flag_gems.ops.scaled_dot_product_attention(
        q, k, v, attn_mask=None, scale=scale, is_causal=is_causal, enable_gqa=enable_gqa
    )

    gems_assert_close(gems_result, torch_result, dtype)
    if flag_gems.vendor_name == "hygon":
        del os.environ["TRITON_HIP_USE_NEW_STREAM_PIPELINE"]


@pytest.mark.skipif(True, reason="something wrong here, disable it for temp")
@pytest.mark.skipif(flag_gems.vendor_name == "metax", reason="TODOFIX")
@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.skipif(
    torch.__version__ < "2.5", reason="Low Pytorch Version: enable_gqa not supported"
)
@pytest.mark.scaled_dot_product_attention_backward
@pytest.mark.parametrize(
    "batch, num_q_head, num_kv_head, q_seq_len, kv_seq_len, head_size, enable_gqa",
    [
        (4, 8, 8, 1024, 1024, 64, False),
        (4, 8, 8, 1024, 1024, 128, False),
        (4, 8, 8, 2048, 256, 64, False),
        (4, 8, 8, 2048, 256, 128, False),
        (4, 8, 8, 17, 1030, 64, False),
        (4, 8, 8, 17, 1030, 128, False),
        # adopted from FlagAttention `test_attention_fwd`:
        (2, 4, 4, 512, 612, 128, False),
        (2, 4, 4, 1024, 1034, 64, False),
        (2, 4, 4, 2048, 2048, 32, False),
        (2, 4, 4, 4096, 4096, 16, False),
        (2, 4, 4, 4001, 4001, 32, False),
        (2, 4, 4, 4001, 4096, 64, False),
        (2, 4, 4, 4096, 4000, 128, False),
        (1, 2, 2, 8192, 8202, 16, False),
        (1, 2, 2, 8192, 8192, 32, False),
        # test for mqa/gqa
        (2, 4, 2, 512, 612, 128, True),
        (2, 4, 1, 1024, 1034, 64, True),
        (2, 4, 2, 2048, 2048, 32, True),
        (2, 4, 1, 4096, 4096, 16, True),
        (2, 4, 2, 4001, 4001, 32, True),
        (2, 4, 1, 4001, 4096, 64, True),
        (2, 4, 2, 4096, 4000, 128, True),
        (1, 2, 1, 8192, 8202, 16, True),
        (1, 2, 1, 8192, 8192, 32, True),
    ],
)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sdpa_legacy_backward(
    batch,
    num_q_head,
    num_kv_head,
    q_seq_len,
    kv_seq_len,
    head_size,
    is_causal,
    dtype,
    enable_gqa,
):
    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch,
        num_q_head,
        num_kv_head,
        q_seq_len,
        kv_seq_len,
        head_size,
        dtype,
        device,
        requires_grad=True,
    )
    ref_q = to_reference(q, False)
    ref_k = to_reference(k, False)
    ref_v = to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))

    # forward
    torch_result = torch_sdpa(
        ref_q, ref_k, ref_v, scale, is_causal, enable_gqa=enable_gqa
    )

    gems_result = flag_gems.ops.scaled_dot_product_attention(
        q, k, v, attn_mask=None, scale=scale, is_causal=is_causal, enable_gqa=enable_gqa
    )

    gems_assert_close(gems_result, torch_result, dtype)

    # backward
    dout = torch.randn_like(ref_q)
    torch_result.backward(dout)
    gems_result.backward(dout)
    torch_q_grad = ref_q.grad.clone() if ref_q.grad is not None else None
    torch_k_grad = ref_k.grad.clone() if ref_k.grad is not None else None
    torch_v_grad = ref_v.grad.clone() if ref_v.grad is not None else None
    gems_q_grad = q.grad.clone() if q.grad is not None else None
    gems_k_grad = k.grad.clone() if k.grad is not None else None
    gems_v_grad = v.grad.clone() if v.grad is not None else None

    # NOTE: NaN may arise in the gradients, this behavior aligns with PyTorch's SDPA
    gems_assert_close(gems_q_grad, torch_q_grad, dtype, equal_nan=True)
    gems_assert_close(gems_k_grad, torch_k_grad, dtype, equal_nan=True)
    gems_assert_close(gems_v_grad, torch_v_grad, dtype, equal_nan=True)


@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize(
    ["batch", "num_head", "q_seq_len", "kv_seq_len"],
    [
        (4, 8, 1024, 1024),
    ],
)
@pytest.mark.parametrize("head_size", [64, 128, 192, 256])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sdpa_square_qk_even_mn(
    batch, num_head, q_seq_len, kv_seq_len, head_size, is_causal, dtype
):
    if flag_gems.vendor_name == "mthreads":
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch, num_head, num_head, q_seq_len, kv_seq_len, head_size, dtype, device
    )
    ref_q = to_reference(q, False)
    ref_k = to_reference(k, False)
    ref_v = to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))
    torch_result = torch_sdpa(ref_q, ref_k, ref_v, scale, is_causal)
    with flag_gems.use_gems():
        gems_result = torch_sdpa(q, k, v, scale, is_causal)
    gems_assert_close(gems_result, torch_result, dtype)

    if flag_gems.vendor_name == "mthreads":
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize(
    ["batch", "num_head", "q_seq_len", "kv_seq_len"],
    [(1, 1, 128, 2048), (4, 8, 1024, 128), (4, 8, 17, 1030)],
)
@pytest.mark.parametrize("head_size", [64, 128, 192, 256])
@pytest.mark.parametrize("is_causal", [False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sdpa_nonsquare_qk(
    batch, num_head, q_seq_len, kv_seq_len, head_size, is_causal, dtype
):
    if flag_gems.vendor_name == "mthreads":
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch, num_head, num_head, q_seq_len, kv_seq_len, head_size, dtype, device
    )
    ref_q = to_reference(q, False)
    ref_k = to_reference(k, False)
    ref_v = to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))
    torch_result = torch_sdpa(ref_q, ref_k, ref_v, scale, is_causal)
    with flag_gems.use_gems():
        gems_result = torch_sdpa(q, k, v, scale, is_causal)
    gems_assert_close(gems_result, torch_result, dtype)

    if flag_gems.vendor_name == "mthreads":
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.skipif(flag_gems.vendor_name == "metax", reason="TODOFIX")
@pytest.mark.skipif(TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(
    flag_gems.vendor_name == "mthreads", reason="Unsupported in CPU mode"
)
@pytest.mark.flash_attention_forward
@pytest.mark.parametrize(
    ["batch", "num_head", "q_seq_len", "kv_seq_len"],
    [(1, 1, 128, 2048), (4, 8, 1024, 128), (4, 8, 17, 1030)],
)
@pytest.mark.parametrize("head_size", [64, 128, 192, 256])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_fwd_nonsquare_qk(
    batch, num_head, q_seq_len, kv_seq_len, head_size, is_causal, dtype
):
    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch, num_head, num_head, q_seq_len, kv_seq_len, head_size, dtype, device
    )
    ref_q = to_reference(q, False)
    ref_k = to_reference(k, False)
    ref_v = to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))

    torch_out, torch_lse, _, _, _ = torch_flash_fwd(
        ref_q, ref_k, ref_v, scale, is_causal
    )
    gems_out, gems_lse, _, _, _ = gems_flash_fwd(q, k, v, scale, is_causal)

    gems_assert_close(gems_out, torch_out, dtype)
    if flag_gems.vendor_name == "iluvatar":
        return
    gems_assert_close(gems_lse, torch_lse, torch.float)


@pytest.mark.skipif(TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(
    flag_gems.vendor_name == "mthreads", reason="Unsupported in CPU mode"
)
@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.flash_attention_forward
@pytest.mark.parametrize(
    ["batch", "num_head", "num_head_k", "q_seq_len", "kv_seq_len"],
    [(4, 8, 2, 1024, 1024), (4, 4, 4, 1, 519)],
)
@pytest.mark.parametrize("head_size", [128, 192])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("alibi", [True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_fwd_gqa_alibi_softcap(
    batch,
    num_head,
    num_head_k,
    q_seq_len,
    kv_seq_len,
    head_size,
    is_causal,
    soft_cap,
    alibi,
    dtype,
):
    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch, num_head, num_head_k, q_seq_len, kv_seq_len, head_size, dtype, device
    )
    ref_q = to_reference(q, False)
    ref_k = to_reference(k, False)
    ref_v = to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))

    if alibi:
        # alibi_slopes = torch.rand(batch, num_head, device=device, dtype=torch.float32) * 0.3
        alibi_slopes = (
            torch.ones(batch, num_head, device=device, dtype=torch.float32) * 0.3
        )
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, q_seq_len, kv_seq_len, causal=is_causal
        )
    else:
        alibi_slopes, attn_bias = None, None

    torch_out, _ = attention_ref(
        ref_q,
        ref_k,
        ref_v,
        scale,
        None,
        None,
        attn_bias,
        0.0,
        None,
        causal=is_causal,
        window_size=(-1, -1),
        softcap=soft_cap if soft_cap is not None else 0,
    )

    gems_out, gems_lse, _, _, _ = gems_flash_fwd(
        q,
        k,
        v,
        scale,
        is_causal,
        alibi_slopes=alibi_slopes,
        softcap=soft_cap if soft_cap is not None else 0,
        disable_splitkv=True,
    )

    gems_assert_close(gems_out, torch_out, dtype)


@pytest.mark.skipif(flag_gems.vendor_name == "metax", reason="TODOFIX")
@pytest.mark.skipif(TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(
    flag_gems.vendor_name == "mthreads", reason="Unsupported in CPU mode"
)
@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.flash_attention_forward
@pytest.mark.parametrize(
    ["batch", "num_head", "num_head_k", "q_seq_len", "kv_seq_len"],
    [(1, 4, 1, 1, 1024), (4, 4, 4, 1, 519)],
)
@pytest.mark.parametrize("head_size", [128, 192])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_splitkv(
    batch,
    num_head,
    num_head_k,
    q_seq_len,
    kv_seq_len,
    head_size,
    is_causal,
    soft_cap,
    alibi,
    dtype,
):
    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch, num_head, num_head_k, q_seq_len, kv_seq_len, head_size, dtype, device
    )
    ref_q = to_reference(q, False)
    ref_k = to_reference(k, False)
    ref_v = to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))

    if alibi:
        # alibi_slopes = torch.rand(batch, num_head, device=device, dtype=torch.float32) * 0.3
        alibi_slopes = (
            torch.ones(batch, num_head, device=device, dtype=torch.float32) * 0.3
        )
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, q_seq_len, kv_seq_len, causal=is_causal
        )
    else:
        alibi_slopes, attn_bias = None, None

    torch_out, _ = attention_ref(
        ref_q,
        ref_k,
        ref_v,
        scale,
        None,
        None,
        attn_bias,
        0.0,
        None,
        causal=is_causal,
        window_size=(-1, -1),
        softcap=soft_cap if soft_cap is not None else 0,
    )

    gems_out, gems_lse, _, _, _ = gems_flash_fwd(
        q,
        k,
        v,
        scale,
        is_causal,
        alibi_slopes=alibi_slopes,
        softcap=soft_cap if soft_cap is not None else 0,
    )

    gems_assert_close(gems_out, torch_out, dtype)


@pytest.mark.skipif(flag_gems.vendor_name == "metax", reason="TODOFIX")
@pytest.mark.skipif(TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(torch.__version__ < "2.4", reason="Low Pytorch Version")
@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(
    flag_gems.vendor_name == "mthreads", reason="Unsupported in CPU mode"
)
@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.flash_attention_forward
@pytest.mark.parametrize(
    ["batch", "num_head", "q_seq_len", "kv_seq_len"],
    [(1, 1, 128, 2048), (8, 32, 1024, 1024), (8, 32, 1024, 128), (8, 32, 17, 1030)],
)
@pytest.mark.parametrize("head_size", [128, 192])
@pytest.mark.parametrize(
    ["window_size_left", "window_size_right"], [(256, 0), (128, 128)]
)
@pytest.mark.parametrize("is_causal", [False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_fwd_swa(
    batch,
    num_head,
    q_seq_len,
    kv_seq_len,
    head_size,
    is_causal,
    window_size_left,
    window_size_right,
    dtype,
):
    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch, num_head, num_head, q_seq_len, kv_seq_len, head_size, dtype, device
    )
    ref_q = to_reference(q, False)
    ref_k = to_reference(k, False)
    ref_v = to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))

    torch_out, torch_lse, _, _, _ = torch_flash_fwd(
        ref_q,
        ref_k,
        ref_v,
        scale,
        is_causal,
        dropout_p=0,
        return_debug_mask=False,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    gems_out, gems_lse, _, _, _ = gems_flash_fwd(
        q,
        k,
        v,
        scale,
        is_causal,
        dropout_p=0,
        return_debug_mask=False,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )

    gems_assert_close(gems_out, torch_out, dtype)
    if flag_gems.vendor_name == "iluvatar":
        return
    gems_assert_close(gems_lse, torch_lse, torch.float)


# Following varlen and paged attn tests are copied from
# https://github.com/vllm-project/flash-attention/blob/main/tests/test_vllm_flash_attn.py


def attn_bias_from_alibi_slopes(slopes, seqlen_q, seqlen_k, causal=False):
    batch, nheads = slopes.shape
    device = slopes.device
    slopes = slopes.unsqueeze(-1).unsqueeze(-1)
    if causal:
        return (
            torch.arange(-seqlen_k + 1, 1, device=device, dtype=torch.float32) * slopes
        )

    row_idx = torch.arange(seqlen_q, device=device, dtype=torch.long).unsqueeze(-1)
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    relative_pos = torch.abs(row_idx + seqlen_k - seqlen_q - col_idx)
    return -slopes * relative_pos.to(dtype=slopes.dtype)


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens: List[int],
    block_tables: torch.Tensor,
    scale: float,
    attn_bias: torch.Tensor = None,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        # clone to avoid clobbering the query tensor
        q = query[start_idx : start_idx + query_len].clone()
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k)
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))

        if attn_bias is not None:
            attn = attn + attn_bias[i, :, :query_len, :kv_len]

        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RESULT TODOFIX")
@pytest.mark.flash_attn_varlen_func
@pytest.mark.parametrize("seq_lens", [[(1, 1328), (5, 18), (129, 463)]])
@pytest.mark.parametrize("num_heads", [(4, 4), (8, 2), (16, 2)])
@pytest.mark.parametrize("head_size", [128, 192, 256])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("sliding_window", [None])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("num_blocks", [32768, 2048])
@torch.inference_mode()
def test_flash_attn_varlen_func(
    seq_lens: List[Tuple[int, int]],
    num_heads: Tuple[int, int],
    head_size: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    block_size: int,
    alibi: bool,
    soft_cap: Optional[float],
    num_blocks: int,
) -> None:
    if flag_gems.vendor_name == "mthreads":
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    # (Issue) numerical stability concern
    if alibi is True and soft_cap is not None:
        return

    with torch.device(flag_gems.device):
        init_seed(1234567890)
        if flag_gems.vendor_name == "cambricon":
            torch.manual_seed(123456)
            torch.mlu.manual_seed_all(123456)
        num_seqs = len(seq_lens)
        query_lens = [x[0] for x in seq_lens]
        kv_lens = [x[1] for x in seq_lens]
        num_query_heads = num_heads[0]
        num_kv_heads = num_heads[1]
        assert num_query_heads % num_kv_heads == 0
        max_query_len = max(query_lens)
        max_kv_len = max(kv_lens)
        window_size = (
            (sliding_window, sliding_window) if sliding_window is not None else (-1, -1)
        )
        scale = head_size**-0.5
        query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
        key_cache = torch.randn(
            num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
        )
        value_cache = torch.randn_like(key_cache)
        cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
            dim=0, dtype=torch.int32
        )
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(
            0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
        )

        causal = True

        if alibi:
            # alibi_slopes = torch.rand(num_seqs, num_query_heads, device=device, dtype=torch.float32) * 0.3
            alibi_slopes = (
                torch.ones(
                    num_seqs, num_query_heads, device=device, dtype=torch.float32
                )
                * 0.3
            )
            attn_bias = attn_bias_from_alibi_slopes(
                alibi_slopes, max_query_len, max_kv_len, causal=causal
            )
        else:
            alibi_slopes, attn_bias = None, None

        output = flag_gems.ops.flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_query_lens,
            seqused_k=seqused_k,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_kv_len,
            softmax_scale=scale,
            causal=causal,
            window_size=window_size,
            block_table=block_tables,
            softcap=soft_cap if soft_cap is not None else 0,
            alibi_slopes=alibi_slopes,
            fa_version=2,
        )

        ref_output = ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=block_tables,
            scale=scale,
            attn_bias=attn_bias,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
        )

        torch.testing.assert_close(
            output, ref_output, atol=2e-2, rtol=1e-2
        ), f"{torch.max(torch.abs(output - ref_output))}"

    if flag_gems.vendor_name == "mthreads":
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RESULT TODOFIX")
@pytest.mark.flash_attn_varlen_func
@pytest.mark.parametrize("seq_lens", [[(1, 1328), (1, 18), (1, 463)]])
@pytest.mark.parametrize("num_heads", [(8, 2)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("sliding_window", [None])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None, 10.0])
@pytest.mark.parametrize("num_blocks", [2048])
@torch.inference_mode()
def test_flash_attn_varlen_func_swap_qg(
    seq_lens: List[Tuple[int, int]],
    num_heads: Tuple[int, int],
    head_size: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
) -> None:
    if flag_gems.vendor_name == "mthreads":
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    with torch.device(flag_gems.device):
        init_seed(1234567890)
        num_seqs = len(seq_lens)
        query_lens = [x[0] for x in seq_lens]
        kv_lens = [x[1] for x in seq_lens]
        num_query_heads = num_heads[0]
        num_kv_heads = num_heads[1]
        assert num_query_heads % num_kv_heads == 0
        max_query_len = max(query_lens)
        max_kv_len = max(kv_lens)
        window_size = (
            (sliding_window, sliding_window) if sliding_window is not None else (-1, -1)
        )
        scale = head_size**-0.5
        query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
        key_cache = torch.randn(
            num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
        )
        value_cache = torch.randn_like(key_cache)
        cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
            dim=0, dtype=torch.int32
        )
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(
            0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
        )

        output = flag_gems.ops.flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_query_lens,
            seqused_k=seqused_k,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=window_size,
            block_table=block_tables,
            softcap=soft_cap if soft_cap is not None else 0,
            fa_version=2,
        )

        ref_output = ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=block_tables,
            scale=scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
        )

        torch.testing.assert_close(
            output, ref_output, atol=2e-2, rtol=1e-2
        ), f"{torch.max(torch.abs(output - ref_output))}"

    if flag_gems.vendor_name == "mthreads":
        del os.environ["MUSA_ENABLE_SQMMA"]


DTYPES = [torch.half, torch.bfloat16, torch.float]
# Parameters for MLA tests.
KV_LORA_RANKS = [512]
QK_ROPE_HEAD_DIMS = [64]
NUM_TOKENS_MLA = [42]
BLOCK_SIZES_MLA = [16]
NUM_BLOCKS_MLA = [8]
SEEDS = [0]
CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]
# We assume fp8 is always enabled for testing.
KV_CACHE_DTYPE = [
    "auto",
]


def _create_mla_cache(
    num_blocks: int,
    block_size: int,
    entry_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
) -> torch.Tensor:
    cache_dtype = torch.uint8 if kv_cache_dtype == "fp8" else dtype
    return torch.zeros(
        num_blocks, block_size, entry_size, dtype=cache_dtype, device=device
    )


# Custom implementation for FP8 conversion (only for testing)
def convert_fp8(
    dst: torch.Tensor, src: torch.Tensor, scale: float, kv_dtype: str
) -> None:
    if kv_dtype == "fp8":
        dst_ = (src / scale).to(torch.float8_e4m3fn).view(dst.dtype)
        dst.copy_(dst_)
    else:
        dst.copy_(src)


@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.concat_and_cache_mla
@pytest.mark.parametrize("kv_lora_rank", KV_LORA_RANKS)
@pytest.mark.parametrize("qk_rope_head_dim", QK_ROPE_HEAD_DIMS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS_MLA)
@pytest.mark.parametrize("block_size", BLOCK_SIZES_MLA)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS_MLA)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "device",
    [flag_gems.device] if flag_gems.vendor_name == "mthreads" else CUDA_DEVICES,
)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@torch.inference_mode()
def test_concat_and_cache_mla(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_tokens: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    kv_cache_dtype: str,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    with torch.device(device):
        total_slots = num_blocks * block_size
        slot_mapping_lst = random.sample(range(total_slots), num_tokens)
        slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

        kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype, device=device)
        k_pe = torch.randn(num_tokens, qk_rope_head_dim, dtype=dtype, device=device)
        entry_size = kv_lora_rank + qk_rope_head_dim

        scale = torch.tensor(0.1, dtype=torch.float32, device=device)
        kv_cache = _create_mla_cache(
            num_blocks, block_size, entry_size, dtype, kv_cache_dtype, device
        )
        ref_temp = to_reference(
            torch.zeros(*kv_cache.shape, dtype=dtype, device=device)
        )

        for i in range(num_tokens):
            slot = slot_mapping[i].item()
            block_idx = slot // block_size
            block_offset = slot % block_size
            ref_temp[block_idx, block_offset, :kv_lora_rank] = kv_c[i]
            ref_temp[block_idx, block_offset, kv_lora_rank:] = k_pe[i]

        if kv_cache_dtype == "fp8":
            ref_kv_cache = to_reference(
                torch.empty_like(ref_temp, dtype=kv_cache.dtype)
            )
            convert_fp8(ref_kv_cache, ref_temp, scale.item(), kv_dtype=kv_cache_dtype)
        else:
            ref_kv_cache = to_reference(ref_temp)
        with flag_gems.use_gems():
            flag_gems.concat_and_cache_mla(
                kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
            )

        if kv_cache_dtype == "fp8":
            result_temp = torch.empty_like(kv_cache, dtype=torch.uint8)
            convert_fp8(
                result_temp,
                kv_cache.contiguous(),
                scale.item(),
                kv_dtype=kv_cache_dtype,
            )
            expected_temp = to_reference(
                torch.empty_like(ref_kv_cache, dtype=torch.uint8)
            )
            convert_fp8(
                expected_temp, ref_kv_cache, scale.item(), kv_dtype=kv_cache_dtype
            )
            dtype = torch.float8_e4m3fn
            # TODO: RuntimeError: Comparing
            # maybe a bug in torch.testing.assert_close
            # gems_assert_close(kv_cache.view(dtype), ref_kv_cache.view(dtype), dtype)
            torch.testing.assert_close(result_temp, expected_temp, atol=0.001, rtol=0.1)
        else:
            gems_assert_close(kv_cache, ref_kv_cache, kv_cache.dtype)


def create_kv_caches_with_random(
    num_blocks,
    block_size,
    num_layers,
    num_heads,
    head_size,
    cache_dtype,
    model_dtype=None,
    seed=None,
):
    init_seed(seed)
    torch_dtype = model_dtype
    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=torch_dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches: list[torch.Tensor] = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_cache.uniform_(-scale, scale)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches: list[torch.Tensor] = []
    for _ in range(num_layers):
        value_cache = torch.empty(
            size=value_cache_shape, dtype=torch_dtype, device=device
        )
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            value_cache.uniform_(-scale, scale)
        else:
            raise ValueError(f"Does not support value cache of type {cache_dtype}")
        value_caches.append(value_cache)
    return key_caches, value_caches


@pytest.mark.reshape_and_cache
@pytest.mark.parametrize("num_tokens", [42])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_size", [64, 80, 120, 256])
@pytest.mark.parametrize("block_size", [8, 16, 32])
@pytest.mark.parametrize("num_blocks", [1024, 10000])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16, torch.float])
@pytest.mark.parametrize("kv_cache_dtype", ["auto"])
@pytest.mark.parametrize("seed", [2025])
def test_reshape_and_cache(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    seed: int,
) -> None:
    init_seed(seed)
    with torch.device(device):
        # Create a random slot mapping.
        num_slots = block_size * num_blocks
        slot_mapping_lst = random.sample(range(num_slots), num_tokens)
        slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long)

        qkv = torch.randn(num_tokens, 3, num_heads, head_size, dtype=dtype)
        _, key, value = qkv.unbind(dim=1)

        # Create the KV caches.
        key_caches, value_caches = create_kv_caches_with_random(
            num_blocks, block_size, 1, num_heads, head_size, kv_cache_dtype, dtype, seed
        )
        key_cache, value_cache = key_caches[0], value_caches[0]

        # Using default kv_scale
        k_scale = (key.amax() / 64.0).to(torch.float32)
        v_scale = (value.amax() / 64.0).to(torch.float32)

        # Clone the KV caches.
        cloned_key_cache = key_cache.clone()
        cloned_value_cache = value_cache.clone()

        # Call the reshape_and_cache kernel.
        flag_gems.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

        # Run the reference implementation.
        reshaped_key = key.reshape(num_tokens, *key_cache[0, :, :, 0, :].shape)
        block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
        block_indicies_lst = block_indicies.cpu().tolist()
        block_offsets = slot_mapping % block_size
        block_offsets_lst = block_offsets.cpu().tolist()
        for i in range(num_tokens):
            block_idx = block_indicies_lst[i]
            block_offset = block_offsets_lst[i]
            cloned_key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
            cloned_value_cache[block_idx, :, :, block_offset] = value[i]

        torch.testing.assert_close(key_cache.cpu(), cloned_key_cache.cpu())
        torch.testing.assert_close(value_cache.cpu(), cloned_value_cache.cpu())


@pytest.mark.skipif(TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(triton.__version__ < "3.1", reason="Low Triton Version")
@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "mthreads", reason="Low Triton Version")
@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.flash_attention_forward
@pytest.mark.parametrize(
    ["batch", "num_head", "q_seq_len", "kv_seq_len"],
    [
        (1, 1, 1024, 1024),
    ],
)
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_fwd_dropout(
    batch, num_head, q_seq_len, kv_seq_len, head_size, is_causal, dtype
):
    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch, num_head, num_head, q_seq_len, kv_seq_len, head_size, dtype, device
    )
    scale = float(1.0 / np.sqrt(head_size))
    dropout_p = 0.2
    gems_out, gems_lse, _, _, debug_softmax = gems_flash_fwd(
        q, k, v, scale, is_causal, dropout_p=dropout_p, return_debug_mask=True
    )

    dropout_ratio = torch.sum(debug_softmax < 0) / torch.sum(debug_softmax != 0)
    np.testing.assert_allclose(dropout_ratio.to("cpu"), dropout_p, rtol=5e-2)


def create_kv_caches_with_random_flash(
    num_blocks,
    block_size,
    num_layers,
    num_heads,
    head_size,
    cache_dtype,
    model_dtype,
    seed,
    device,
):
    init_seed(seed)
    torch_dtype = model_dtype
    key_value_cache_shape = (num_blocks, 2, block_size, num_heads, head_size)
    scale = head_size**-0.5

    key_caches: list[torch.Tensor] = []
    value_caches: list[torch.Tensor] = []

    for _ in range(num_layers):
        key_value_cache = torch.empty(
            size=key_value_cache_shape, dtype=torch_dtype, device=device
        )
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_value_cache.uniform_(-scale, scale)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_value_cache[:, 0])
        value_caches.append(key_value_cache[:, 1])
    return key_caches, value_caches


@pytest.mark.reshape_and_cache_flash
@pytest.mark.parametrize("num_tokens", [42])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_size", [64, 80, 120, 256])
@pytest.mark.parametrize("block_size", [8, 16, 32])
@pytest.mark.parametrize("num_blocks", [1024, 10000])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16, torch.float])
@pytest.mark.parametrize("kv_cache_dtype", ["auto"])
@pytest.mark.parametrize("seed", [2025])
@torch.inference_mode()
def test_reshape_and_cache_flash(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    seed: int,
) -> None:
    init_seed(seed)
    with torch.device(device):
        # Create a random slot mapping.
        num_slots = block_size * num_blocks
        slot_mapping_lst = random.sample(range(num_slots), num_tokens)
        slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)
        qkv = torch.randn(
            num_tokens, 3, num_heads, head_size, dtype=dtype, device=device
        )
        _, key, value = qkv.unbind(dim=1)

        # Create the KV caches.
        key_caches, value_caches = create_kv_caches_with_random_flash(
            num_blocks,
            block_size,
            1,
            num_heads,
            head_size,
            kv_cache_dtype,
            dtype,
            seed=seed,
            device=device,
        )
        key_cache, value_cache = (
            key_caches[0].contiguous(),
            value_caches[0].contiguous(),
        )
        del key_caches
        del value_caches

        k_scale = (key.amax() / 64.0).to(torch.float32)
        v_scale = (value.amax() / 64.0).to(torch.float32)

        # Clone the KV caches.
        cloned_key_cache = key_cache.clone()
        cloned_value_cache = value_cache.clone()

        # Call the reshape_and_cache kernel.
        flag_gems.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

        # Run the reference implementation.
        block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
        block_indicies_lst = block_indicies.cpu().tolist()
        block_offsets = slot_mapping % block_size
        block_offsets_lst = block_offsets.cpu().tolist()
        for i in range(num_tokens):
            block_idx = block_indicies_lst[i]
            block_offset = block_offsets_lst[i]
            cloned_key_cache[block_idx, block_offset, :, :] = key[i]
            cloned_value_cache[block_idx, block_offset, :, :] = value[i]

        torch.testing.assert_close(key_cache.cpu(), cloned_key_cache.cpu())
        torch.testing.assert_close(value_cache.cpu(), cloned_value_cache.cpu())


@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.flash_mla
@pytest.mark.parametrize("seqlen", [1024, 2048, 4096, 8192])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
    ],
)
def test_flash_mla(seqlen, dtype):
    if flag_gems.vendor_name == "mthreads":
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

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

    ref_q = to_reference(q)
    ref_block_table = to_reference(block_table)
    ref_blocked_k = to_reference(blocked_k)
    ref_cache_seqlens = to_reference(cache_seqlens)

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

    def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str) -> None:
        x, y = x.double(), y.double()
        x = x.to(y.device)
        RMSE = ((x - y) * (x - y)).mean().sqrt().item()
        cos_diff = 1 - 2 * (x * y).sum().item() / max(
            (x * x + y * y).sum().item(), 1e-12
        )
        amax_diff = (x - y).abs().max().item()
        assert cos_diff < 1e-5, f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}"

    cal_diff(to_reference(res_out), ref_out, "out")

    if flag_gems.vendor_name == "mthreads":
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.get_scheduler_metadata
@pytest.mark.parametrize("batch_size", [1, 8, 256, 512])
@pytest.mark.parametrize("max_seqlen_k", [512, 2048])
@pytest.mark.parametrize("headdim", [64, 128])
@pytest.mark.parametrize("num_splits_static", [0, 4])
@pytest.mark.parametrize("seed", [42])
@pytest.mark.skipif(not HAS_VLLM, reason="vllm not installed")
def test_scheduler_metadata_correctness(
    batch_size, max_seqlen_k, headdim, num_splits_static, seed
):
    if TO_CPU:
        pytest.skip("Skipping correctness test in CPU mode.")
    device = torch_device_fn.current_device()
    init_seed(seed)

    seqused_k = torch.randint(
        1, max_seqlen_k, (batch_size,), dtype=torch.int32, device=device
    )
    num_heads, num_heads_k = 32, 8
    headdim_v = headdim
    qkv_dtype = torch.float16
    # num_sm = torch.cuda.get_device_properties(device).multi_processor_count
    import vllm.vllm_flash_attn.flash_attn_interface as vllm_ops  # noqa: F401

    ref_metadata = torch.ops._vllm_fa3_C.get_scheduler_metadata(
        batch_size=batch_size,
        max_seqlen_q=1,
        max_seqlen_k=max_seqlen_k,
        num_heads=num_heads,
        num_heads_k=num_heads_k,
        headdim=headdim,
        headdim_v=headdim_v,
        qkv_dtype=qkv_dtype,
        seqused_k=seqused_k,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        cu_seqlens_k_new=None,
        seqused_q=None,
        leftpad_k=None,
        page_size=None,
        max_seqlen_k_new=0,
        is_causal=False,
        window_size_left=-1,
        window_size_right=-1,
        has_softcap=False,
        num_splits=num_splits_static,
        pack_gqa=True,
        sm_margin=0,
    )

    with flag_gems.use_gems():
        gems_metadata = flag_gems.get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k,
            num_heads=num_heads,
            num_heads_k=num_heads_k,
            headdim=headdim,
            headdim_v=headdim_v,
            qkv_dtype=qkv_dtype,
            seqused_k=seqused_k,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            cu_seqlens_k_new=None,
            seqused_q=None,
            leftpad_k=None,
            page_size=None,
            max_seqlen_k_new=0,
            is_causal=False,
            window_size_left=-1,
            window_size_right=-1,
            has_softcap=False,
            num_splits=num_splits_static,
            pack_gqa=True,
            sm_margin=0,
        )

    gems_assert_close(gems_metadata, ref_metadata, dtype=torch.int32)
