import logging

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

device = flag_gems.device

# Use the aten op for the PyTorch reference path to avoid depending on the
# private torch._... alias across PyTorch versions.
sdpa_math = torch.ops.aten._scaled_dot_product_attention_math
generic_math = (
    flag_gems._scaled_dot_product_attention_math.__module__
    == "flag_gems.ops._scaled_dot_product_attention_math"
)


@pytest.fixture
def math_caplog(caplog, monkeypatch):
    op_logger = logging.getLogger(
        utils.gems_log_logger(flag_gems._scaled_dot_product_attention_math)
    )
    # Exercise capture even when another test disabled log propagation.
    monkeypatch.setattr(op_logger, "propagate", False)
    op_logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        op_logger.removeHandler(caplog.handler)


# Cases cover: (1) equal q/kv seq lens with and without causal masking, (2)
# differing q/kv seq lens, (3) an explicit float attn_mask, and (4) custom
# scale values. attn_mask is only used when is_causal=False because the aten
# op rejects attn_mask+is_causal=True.
@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.parametrize(
    "batch, num_head, q_seq_len, kv_seq_len, head_size, use_mask, is_causal, scale",
    [
        # equal q/kv seq lens, both causal settings
        (2, 4, 8, 8, 16, False, False, None),
        (2, 4, 8, 8, 16, False, True, None),
        (2, 4, 32, 32, 32, False, False, None),
        (2, 4, 32, 32, 32, False, True, None),
        (4, 8, 64, 64, 64, False, False, None),
        (4, 8, 64, 64, 64, False, True, None),
        (2, 4, 128, 128, 128, False, False, None),
        (2, 4, 128, 128, 128, False, True, None),
        (1, 2, 256, 256, 64, False, False, None),
        (1, 2, 256, 256, 64, False, True, None),
        # differing q/kv seq lens (causal=False only)
        (2, 4, 32, 64, 32, False, False, None),
        (4, 8, 64, 128, 64, False, False, None),
        (2, 4, 128, 64, 64, False, False, None),
        # explicit float attn_mask
        (2, 4, 32, 32, 32, True, False, None),
        (4, 8, 64, 64, 64, True, False, None),
        # custom scale values
        (2, 4, 32, 32, 32, False, False, 0.5),
        (2, 4, 32, 32, 32, False, False, 1.0),
        (2, 4, 32, 32, 32, False, False, 2.0),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_scaled_dot_product_attention_math(
    batch,
    num_head,
    q_seq_len,
    kv_seq_len,
    head_size,
    use_mask,
    is_causal,
    scale,
    dtype,
):
    utils.init_seed(1234567890)

    q = torch.randn(batch, num_head, q_seq_len, head_size, dtype=dtype, device=device)
    k = torch.randn(batch, num_head, kv_seq_len, head_size, dtype=dtype, device=device)
    v = torch.randn(batch, num_head, kv_seq_len, head_size, dtype=dtype, device=device)
    attn_mask = (
        torch.randn(q_seq_len, kv_seq_len, dtype=dtype, device=device)
        if use_mask
        else None
    )

    ref_q = utils.to_reference(q)
    ref_k = utils.to_reference(k)
    ref_v = utils.to_reference(v)
    ref_mask = utils.to_reference(attn_mask)

    ref_out, ref_weights = sdpa_math(
        ref_q, ref_k, ref_v, attn_mask=ref_mask, is_causal=is_causal, scale=scale
    )

    res_out, res_weights = flag_gems._scaled_dot_product_attention_math(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale
    )

    utils.gems_assert_close(res_out, ref_out, dtype, atol=3e-2)
    utils.gems_assert_close(res_weights, ref_weights, dtype, atol=3e-2)


@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.parametrize(
    "heads,key_heads,value_heads,q_len,kv_len,head_dim,use_mask,is_causal",
    [
        (4, 2, 2, 8, 8, 16, False, False),
        (8, 1, 1, 1, 65, 16, False, False),
        pytest.param(
            8,
            2,
            4,
            33,
            65,
            24,
            False,
            False,
            marks=pytest.mark.skipif(
                not generic_math,
                reason="Independent K/V groups test the generic kernel",
            ),
        ),
        (8, 2, 2, 7, 11, 16, True, False),
        (8, 2, 2, 7, 11, 16, False, True),
        (4, 4, 4, 8, 8, 16, False, False),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_scaled_dot_product_attention_math_gqa(
    heads,
    key_heads,
    value_heads,
    q_len,
    kv_len,
    head_dim,
    use_mask,
    is_causal,
    dtype,
    math_caplog,
):
    utils.init_seed(1234567890)
    q = torch.randn(2, heads, q_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(2, key_heads, kv_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(2, value_heads, kv_len, head_dim, dtype=dtype, device=device)
    attn_mask = None
    if use_mask:
        attn_mask = torch.randn(2, heads, q_len, kv_len, dtype=dtype, device=device)
        attn_mask[0, 0, 0, :] = -float("inf")

    ref_out, ref_weights = sdpa_math(
        utils.to_reference(q),
        utils.to_reference(k),
        utils.to_reference(v),
        attn_mask=utils.to_reference(attn_mask),
        is_causal=is_causal,
        scale=0.25,
        enable_gqa=True,
    )
    gems_op = flag_gems._scaled_dot_product_attention_math
    with math_caplog.at_level("DEBUG", logger=utils.gems_log_logger(gems_op)):
        res_out, res_weights = gems_op(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            scale=0.25,
            enable_gqa=True,
        )

    assert (
        f"{utils.gems_log_prefix(gems_op)} _SCALED_DOT_PRODUCT_ATTENTION_MATH"
        in math_caplog.text
    )
    utils.gems_assert_close(res_out, ref_out, dtype, atol=3e-2)
    utils.gems_assert_close(res_weights, ref_weights, dtype, atol=3e-2)


@pytest.mark.scaled_dot_product_attention_math
def test_scaled_dot_product_attention_math_positional_dropout_mask_gqa(math_caplog):
    utils.init_seed(1234567890)
    q = torch.randn(2, 4, 8, 16, dtype=torch.float16, device=device)
    k = torch.randn(2, 2, 8, 16, dtype=q.dtype, device=device)
    v = torch.randn(2, 2, 8, 16, dtype=q.dtype, device=device)
    dropout_mask = torch.zeros(2, 4, 8, 8, dtype=torch.bool, device=device)

    ref_out, ref_weights = sdpa_math(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        None,
        0.0,
        False,
        dropout_mask.cpu(),
        scale=0.25,
        enable_gqa=True,
    )
    gems_op = flag_gems._scaled_dot_product_attention_math
    with math_caplog.at_level("DEBUG", logger=utils.gems_log_logger(gems_op)):
        res_out, res_weights = gems_op(
            q, k, v, None, 0.0, False, dropout_mask, scale=0.25, enable_gqa=True
        )

    assert (
        f"{utils.gems_log_prefix(gems_op)} _SCALED_DOT_PRODUCT_ATTENTION_MATH"
        in math_caplog.text
    )
    utils.gems_assert_close(res_out, ref_out, q.dtype, atol=3e-2)
    utils.gems_assert_close(res_weights, ref_weights, q.dtype, atol=3e-2)


@pytest.mark.scaled_dot_product_attention_math
def test_scaled_dot_product_attention_math_broadcast_heads(math_caplog):
    utils.init_seed(1234567890)
    q = torch.randn(2, 4, 8, 16, dtype=torch.float16, device=device)
    k = torch.randn(2, 1, 8, 16, dtype=q.dtype, device=device)
    v = torch.randn(2, 1, 8, 16, dtype=q.dtype, device=device)
    ref_out, ref_weights = sdpa_math(q.cpu(), k.cpu(), v.cpu(), enable_gqa=False)
    gems_op = flag_gems._scaled_dot_product_attention_math
    with math_caplog.at_level("DEBUG", logger=utils.gems_log_logger(gems_op)):
        res_out, res_weights = gems_op(q, k, v, enable_gqa=False)
    assert (
        f"{utils.gems_log_prefix(gems_op)} _SCALED_DOT_PRODUCT_ATTENTION_MATH"
        in math_caplog.text
    )

    utils.gems_assert_close(res_out, ref_out, q.dtype, atol=3e-2)
    utils.gems_assert_close(res_weights, ref_weights, q.dtype, atol=3e-2)


@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.skipif(not generic_math, reason="Legacy positional scale is a generic API")
def test_scaled_dot_product_attention_math_legacy_positional_scale(math_caplog):
    utils.init_seed(1234567890)
    q = torch.randn(2, 4, 8, 16, dtype=torch.float16, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    ref_out, ref_weights = sdpa_math(q.cpu(), k.cpu(), v.cpu(), scale=0.5)
    gems_op = flag_gems._scaled_dot_product_attention_math
    with math_caplog.at_level("DEBUG", logger=utils.gems_log_logger(gems_op)):
        res_out, res_weights = gems_op(q, k, v, None, 0.0, False, 0.5)
    assert (
        f"{utils.gems_log_prefix(gems_op)} _SCALED_DOT_PRODUCT_ATTENTION_MATH"
        in math_caplog.text
    )

    utils.gems_assert_close(res_out, ref_out, q.dtype, atol=3e-2)
    utils.gems_assert_close(res_weights, ref_weights, q.dtype, atol=3e-2)


@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.skipif(not generic_math, reason="Checks the generic kernel's empty heads")
@pytest.mark.parametrize("key_heads,value_heads", [(0, 0), (0, 1), (1, 0)])
def test_scaled_dot_product_attention_math_empty_heads(
    key_heads, value_heads, math_caplog
):
    q = torch.empty(2, 0, 8, 16, dtype=torch.float16, device=device)
    k = torch.empty(2, key_heads, 8, 16, dtype=q.dtype, device=device)
    v = torch.empty(2, value_heads, 8, 16, dtype=q.dtype, device=device)
    ref_out, ref_weights = sdpa_math(q.cpu(), k.cpu(), v.cpu())
    gems_op = flag_gems._scaled_dot_product_attention_math

    with math_caplog.at_level("DEBUG", logger=utils.gems_log_logger(gems_op)):
        res_out, res_weights = gems_op(q, k, v)

    assert (
        f"{utils.gems_log_prefix(gems_op)} _SCALED_DOT_PRODUCT_ATTENTION_MATH"
        in math_caplog.text
    )
    assert res_out.shape == ref_out.shape and res_out.dtype == ref_out.dtype
    assert (
        res_weights.shape == ref_weights.shape
        and res_weights.dtype == ref_weights.dtype
    )
    assert res_out.numel() == 0 and res_weights.numel() == 0

    # PyTorch 2.11 can crash for zero K/V heads with enable_gqa=True.
    with pytest.raises(RuntimeError, match="query heads must be divisible"):
        gems_op(q, k, v, enable_gqa=True)


@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.skipif(not generic_math, reason="Checks the generic kernel's empty heads")
@pytest.mark.parametrize("key_heads,value_heads", [(2, 1), (1, 2)])
def test_scaled_dot_product_attention_math_empty_heads_invalid_broadcast(
    key_heads, value_heads
):
    q = torch.empty(2, 0, 8, 16, dtype=torch.float16, device=device)
    k = torch.empty(2, key_heads, 8, 16, dtype=q.dtype, device=device)
    v = torch.empty(2, value_heads, 8, 16, dtype=q.dtype, device=device)
    with pytest.raises(RuntimeError):
        sdpa_math(q.cpu(), k.cpu(), v.cpu())
    with pytest.raises(RuntimeError, match="key and value heads must match"):
        flag_gems._scaled_dot_product_attention_math(q, k, v)


@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.skipif(not generic_math, reason="Checks generic GQA error handling")
@pytest.mark.parametrize("key_heads,value_heads", [(4, 2), (2, 4)])
def test_scaled_dot_product_attention_math_invalid_gqa(key_heads, value_heads):
    q = torch.randn(2, 6, 8, 16, dtype=torch.float16, device=device)
    k = torch.randn(2, key_heads, 8, 16, dtype=q.dtype, device=device)
    v = torch.randn(2, value_heads, 8, 16, dtype=q.dtype, device=device)
    with pytest.raises(RuntimeError, match="query heads must be divisible"):
        flag_gems._scaled_dot_product_attention_math(q, k, v, enable_gqa=True)


@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.skipif(device != "cuda", reason="Forces the CUDA SDPA math backend")
def test_flaggems_sdpa_math_dispatch_gqa(math_caplog):
    utils.init_seed(1234567890)
    q = torch.randn(2, 4, 8, 16, dtype=torch.float16, device=device)
    k = torch.randn(2, 2, 8, 16, dtype=q.dtype, device=device)
    v = torch.randn(2, 2, 8, 16, dtype=q.dtype, device=device)
    ref_out, _ = sdpa_math(q.cpu(), k.cpu(), v.cpu(), enable_gqa=True)
    gems_op = flag_gems._scaled_dot_product_attention_math
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    lib = torch.library.Library("aten", "IMPL")
    try:
        flag_gems.only_enable(lib=lib, include=["_scaled_dot_product_attention_math"])
        with math_caplog.at_level("DEBUG", logger=utils.gems_log_logger(gems_op)):
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                res_out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, enable_gqa=True
                )
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    assert (
        f"{utils.gems_log_prefix(gems_op)} _SCALED_DOT_PRODUCT_ATTENTION_MATH"
        in math_caplog.text
    )
    utils.gems_assert_close(res_out, ref_out, q.dtype, atol=3e-2)


@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.parametrize(
    "mask_kind", ["2d_float", "4d_broadcast_float", "all_neg_inf_row", "2d_bool"]
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_scaled_dot_product_attention_math_broadcast_and_all_masked_rows(
    mask_kind, dtype
):
    utils.init_seed(1234567890)
    batch, heads, q_len, kv_len, head_dim = 2, 4, 17, 23, 32
    q = torch.randn(batch, heads, q_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, heads, kv_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, heads, kv_len, head_dim, dtype=dtype, device=device)

    if mask_kind == "2d_float":
        keep = torch.randint(0, 2, (q_len, kv_len), device=device).to(torch.bool)
        keep[:, 0] = True
        attn_mask = torch.where(
            keep, torch.zeros((), dtype=dtype, device=device), -float("inf")
        )
    elif mask_kind == "4d_broadcast_float":
        keep = torch.randint(0, 2, (batch, 1, q_len, kv_len), device=device).to(
            torch.bool
        )
        keep[:, :, :, 0] = True
        attn_mask = torch.where(
            keep, torch.zeros((), dtype=dtype, device=device), -float("inf")
        )
    elif mask_kind == "all_neg_inf_row":
        attn_mask = torch.zeros((batch, 1, q_len, kv_len), dtype=dtype, device=device)
        attn_mask[0, 0, 0, :] = -float("inf")
    else:
        attn_mask = torch.randint(0, 2, (q_len, kv_len), device=device).to(torch.bool)

    ref_out, ref_weights = sdpa_math(
        utils.to_reference(q),
        utils.to_reference(k),
        utils.to_reference(v),
        attn_mask=utils.to_reference(attn_mask),
        is_causal=False,
    )
    res_out, res_weights = flag_gems._scaled_dot_product_attention_math(
        q, k, v, attn_mask=attn_mask, is_causal=False
    )

    utils.gems_assert_close(res_out, ref_out, dtype, atol=3e-2)
    utils.gems_assert_close(res_weights, ref_weights, dtype, atol=3e-2)


@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.parametrize("q_len,kv_len", [(0, 8), (8, 0)])
def test_scaled_dot_product_attention_math_empty_seq(q_len, kv_len):
    q = torch.empty(2, 4, q_len, 16, dtype=torch.float16, device=device)
    k = torch.empty(2, 4, kv_len, 16, dtype=torch.float16, device=device)
    v = torch.empty(2, 4, kv_len, 16, dtype=torch.float16, device=device)
    ref_out, ref_weights = sdpa_math(
        utils.to_reference(q), utils.to_reference(k), utils.to_reference(v)
    )
    res_out, res_weights = flag_gems._scaled_dot_product_attention_math(q, k, v)
    assert res_out.shape == ref_out.shape
    assert res_weights.shape == ref_weights.shape
    if res_out.numel() > 0:
        utils.gems_assert_close(res_out, ref_out, torch.float16, atol=1e-3)


@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_scaled_dot_product_attention_math_non_contiguous_float_broadcast_mask(dtype):
    utils.init_seed(1234567890)
    batch, heads, q_len, kv_len, head_dim = 2, 4, 16, 20, 32
    q = torch.randn(batch, heads, q_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, heads, kv_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, heads, kv_len, head_dim, dtype=dtype, device=device)
    attn_mask = torch.randn(
        batch, 1, kv_len, q_len, dtype=dtype, device=device
    ).transpose(2, 3)
    assert not attn_mask.is_contiguous()

    ref_out, ref_weights = sdpa_math(
        utils.to_reference(q),
        utils.to_reference(k),
        utils.to_reference(v),
        attn_mask=utils.to_reference(attn_mask),
    )
    res_out, res_weights = flag_gems._scaled_dot_product_attention_math(
        q, k, v, attn_mask=attn_mask
    )

    utils.gems_assert_close(res_out, ref_out, dtype, atol=3e-2)
    utils.gems_assert_close(res_weights, ref_weights, dtype, atol=3e-2)
