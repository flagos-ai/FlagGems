import numpy as np
import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import random_utils

from . import accuracy_utils as utils
from . import conftest as cfg

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
    random_utils.set_philox_state(1234567890, 0, device)
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


def torch_sdpa(q, k, v, scale, is_causal, enable_gqa=False):
    if torch.__version__ < "2.5":
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            scale=scale,
            is_causal=is_causal,
        )

    if flag_gems.vendor_name == "iluvatar" and cfg.TO_CPU:
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
def test_scaled_dot_product_attention_legacy(
    monkeypatch,
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
        monkeypatch.env("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

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
    ref_q = utils.to_reference(q, False)
    ref_k = utils.to_reference(k, False)
    ref_v = utils.to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))

    # forward
    torch_result = torch_sdpa(
        ref_q, ref_k, ref_v, scale, is_causal, enable_gqa=enable_gqa
    )

    if flag_gems.vendor_name == "cambricon":
        gems_result = flag_gems.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            scale=scale,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )
    else:
        gems_result = flag_gems.ops.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            scale=scale,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )

    utils.gems_assert_close(gems_result, torch_result, dtype)


@pytest.mark.skipif(True, reason="Something wrong here, disable it for temp")
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
def test_scaled_dot_product_attention_legacy_legacy_backward(
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
    ref_q = utils.to_reference(q, False)
    ref_k = utils.to_reference(k, False)
    ref_v = utils.to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))

    # forward
    torch_result = torch_sdpa(
        ref_q, ref_k, ref_v, scale, is_causal, enable_gqa=enable_gqa
    )

    if flag_gems.vendor_name == "cambricon":
        gems_result = flag_gems.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            scale=scale,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )
    else:
        gems_result = flag_gems.ops.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            scale=scale,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )

    utils.gems_assert_close(gems_result, torch_result, dtype)

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
    utils.gems_assert_close(gems_q_grad, torch_q_grad, dtype, equal_nan=True)
    utils.gems_assert_close(gems_k_grad, torch_k_grad, dtype, equal_nan=True)
    utils.gems_assert_close(gems_v_grad, torch_v_grad, dtype, equal_nan=True)


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
def test_scaled_dot_product_attention_square_qk_even_mn(
    monkeypatch, batch, num_head, q_seq_len, kv_seq_len, head_size, is_causal, dtype
):
    if flag_gems.vendor_name == "mthreads":
        monkeypatch.env("MUSA_ENABLE_SQMMA", "1")

    device = torch_device_fn.current_device()

    q, k, v = make_input(
        batch, num_head, num_head, q_seq_len, kv_seq_len, head_size, dtype, device
    )
    ref_q = utils.to_reference(q, False)
    ref_k = utils.to_reference(k, False)
    ref_v = utils.to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))
    torch_result = torch_sdpa(ref_q, ref_k, ref_v, scale, is_causal)

    with flag_gems.use_gems():
        gems_result = torch_sdpa(q, k, v, scale, is_causal)

    utils.gems_assert_close(gems_result, torch_result, dtype)


@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize(
    ["batch", "num_head", "q_seq_len", "kv_seq_len"],
    [(1, 1, 128, 2048), (4, 8, 1024, 128), (4, 8, 17, 1030)],
)
@pytest.mark.parametrize("head_size", [64, 128, 192, 256])
@pytest.mark.parametrize("is_causal", [False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_dot_product_attention_nonsquare_qk(
    monkeypatch, batch, num_head, q_seq_len, kv_seq_len, head_size, is_causal, dtype
):
    if flag_gems.vendor_name == "mthreads":
        monkeypatch.env("MUSA_ENABLE_SQMMA", "1")

    if flag_gems.vendor_name == "hygon":
        monkeypatch.env("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    device = torch_device_fn.current_device()

    q, k, v = make_input(
        batch, num_head, num_head, q_seq_len, kv_seq_len, head_size, dtype, device
    )

    ref_q = utils.to_reference(q, False)
    ref_k = utils.to_reference(k, False)
    ref_v = utils.to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))
    torch_result = torch_sdpa(ref_q, ref_k, ref_v, scale, is_causal)

    with flag_gems.use_gems():
        gems_result = torch_sdpa(q, k, v, scale, is_causal)

    utils.gems_assert_close(gems_result, torch_result, dtype)


@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.parametrize(
    "batch, num_head, seq_len, head_size",
    [
        (2, 4, 8, 16),
        (2, 4, 32, 32),
        (4, 8, 64, 64),
        (2, 4, 128, 128),
        (1, 2, 256, 64),
    ],
)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_accuracy_scaled_dot_product_attention_math(
    batch,
    num_head,
    seq_len,
    head_size,
    is_causal,
    dtype,
):
    """Test _scaled_dot_product_attention_math accuracy against PyTorch reference."""
    init_seed(1234567890)

    q = torch.randn(batch, num_head, seq_len, head_size, dtype=dtype, device=device)
    k = torch.randn(batch, num_head, seq_len, head_size, dtype=dtype, device=device)
    v = torch.randn(batch, num_head, seq_len, head_size, dtype=dtype, device=device)

    ref_q = to_reference(q)
    ref_k = to_reference(k)
    ref_v = to_reference(v)

    # Reference result
    ref_out, ref_weights = torch._scaled_dot_product_attention_math(
        ref_q, ref_k, ref_v, is_causal=is_causal
    )

    # FlagGems result
    with flag_gems.use_gems():
        res_out, res_weights = torch._scaled_dot_product_attention_math(
            q, k, v, is_causal=is_causal
        )

    # Use higher tolerance for attention ops due to cumulative numerical errors
    gems_assert_close(res_out, ref_out, dtype, atol=3e-2)
    gems_assert_close(res_weights, ref_weights, dtype, atol=3e-2)


@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.parametrize(
    "batch, num_head, q_seq_len, kv_seq_len, head_size",
    [
        (2, 4, 32, 64, 32),
        (2, 4, 64, 32, 64),
        (4, 8, 128, 256, 64),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_scaled_dot_product_attention_math_different_seqlen(
    batch,
    num_head,
    q_seq_len,
    kv_seq_len,
    head_size,
    dtype,
):
    """Test _scaled_dot_product_attention_math with different Q and KV sequence lengths."""
    init_seed(1234567890)

    q = torch.randn(batch, num_head, q_seq_len, head_size, dtype=dtype, device=device)
    k = torch.randn(batch, num_head, kv_seq_len, head_size, dtype=dtype, device=device)
    v = torch.randn(batch, num_head, kv_seq_len, head_size, dtype=dtype, device=device)

    ref_q = to_reference(q)
    ref_k = to_reference(k)
    ref_v = to_reference(v)

    # Reference result (non-causal only for different sequence lengths)
    ref_out, ref_weights = torch._scaled_dot_product_attention_math(
        ref_q, ref_k, ref_v, is_causal=False
    )

    # FlagGems result
    with flag_gems.use_gems():
        res_out, res_weights = torch._scaled_dot_product_attention_math(
            q, k, v, is_causal=False
        )

    # Use higher tolerance for attention ops due to cumulative numerical errors
    gems_assert_close(res_out, ref_out, dtype, atol=3e-2)
    gems_assert_close(res_weights, ref_weights, dtype, atol=3e-2)


@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.parametrize(
    "batch, num_head, seq_len, head_size",
    [
        (2, 4, 32, 32),
        (4, 8, 64, 64),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_scaled_dot_product_attention_math_with_mask(
    batch,
    num_head,
    seq_len,
    head_size,
    dtype,
):
    """Test _scaled_dot_product_attention_math with attention mask."""
    init_seed(1234567890)

    q = torch.randn(batch, num_head, seq_len, head_size, dtype=dtype, device=device)
    k = torch.randn(batch, num_head, seq_len, head_size, dtype=dtype, device=device)
    v = torch.randn(batch, num_head, seq_len, head_size, dtype=dtype, device=device)
    attn_mask = torch.randn(seq_len, seq_len, dtype=dtype, device=device)

    ref_q = to_reference(q)
    ref_k = to_reference(k)
    ref_v = to_reference(v)
    ref_mask = to_reference(attn_mask)

    # Reference result
    ref_out, ref_weights = torch._scaled_dot_product_attention_math(
        ref_q, ref_k, ref_v, attn_mask=ref_mask
    )

    # FlagGems result
    with flag_gems.use_gems():
        res_out, res_weights = torch._scaled_dot_product_attention_math(
            q, k, v, attn_mask=attn_mask
        )

    # Use higher tolerance for attention ops due to cumulative numerical errors
    gems_assert_close(res_out, ref_out, dtype, atol=3e-2)
    gems_assert_close(res_weights, ref_weights, dtype, atol=3e-2)


@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.parametrize(
    "batch, num_head, seq_len, head_size",
    [
        (2, 4, 32, 32),
    ],
)
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_scaled_dot_product_attention_math_with_scale(
    batch,
    num_head,
    seq_len,
    head_size,
    scale,
    dtype,
):
    """Test _scaled_dot_product_attention_math with custom scale."""
    init_seed(1234567890)

    q = torch.randn(batch, num_head, seq_len, head_size, dtype=dtype, device=device)
    k = torch.randn(batch, num_head, seq_len, head_size, dtype=dtype, device=device)
    v = torch.randn(batch, num_head, seq_len, head_size, dtype=dtype, device=device)

    ref_q = to_reference(q)
    ref_k = to_reference(k)
    ref_v = to_reference(v)

    # Reference result
    ref_out, ref_weights = torch._scaled_dot_product_attention_math(
        ref_q, ref_k, ref_v, scale=scale
    )

    # FlagGems result
    with flag_gems.use_gems():
        res_out, res_weights = torch._scaled_dot_product_attention_math(
            q, k, v, scale=scale
        )

    # Use higher tolerance for attention ops due to cumulative numerical errors
    gems_assert_close(res_out, ref_out, dtype, atol=3e-2)
    gems_assert_close(res_weights, ref_weights, dtype, atol=3e-2)
