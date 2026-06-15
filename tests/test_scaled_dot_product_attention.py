import numpy as np
import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import random_utils

from . import accuracy_utils as utils
from . import conftest as cfg
from .conftest import QUICK_MODE

device = flag_gems.device

if QUICK_MODE:
    LEGACY_SHAPES = [
        (4, 8, 8, 1024, 1024, 64, False),
    ]
    CAUSAL_CHOICES = [False]
    FLOAT_DTYPES = [torch.float16]
    HEAD_SIZES = [64]
    NONSQUARE_SHAPES = [(4, 8, 1024, 128)]
else:
    LEGACY_SHAPES = [
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
    ]
    CAUSAL_CHOICES = [False, True]
    FLOAT_DTYPES = [torch.float16, torch.bfloat16]
    HEAD_SIZES = [64, 128, 192, 256]
    NONSQUARE_SHAPES = [(1, 1, 128, 2048), (4, 8, 1024, 128), (4, 8, 17, 1030)]

SQUARE_SHAPES = [(4, 8, 1024, 1024)]


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
@pytest.mark.scaled_dot_product_attention_forward
@pytest.mark.parametrize(
    "batch, num_q_head, num_kv_head, q_seq_len, kv_seq_len, head_size, enable_gqa",
    LEGACY_SHAPES,
)
@pytest.mark.parametrize("is_causal", CAUSAL_CHOICES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
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
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

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

    if flag_gems.vendor_name in ["cambricon", "sunrise"]:
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


@pytest.mark.skipif(flag_gems.vendor_name == "metax", reason="Issue #2849: Not working")
@pytest.mark.skipif(
    flag_gems.vendor_name == "hygon", reason="Issue #2849: RuntimeError"
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin", reason="Issue #2849: Not working"
)
@pytest.mark.skipif(flag_gems.vendor_name == "sunrise", reason="Compiler Error")
@pytest.mark.skipif(
    torch.__version__ < "2.5", reason="Low Pytorch Version: enable_gqa not supported"
)
@pytest.mark.scaled_dot_product_attention_backward
@pytest.mark.parametrize(
    "batch, num_q_head, num_kv_head, q_seq_len, kv_seq_len, head_size, enable_gqa",
    LEGACY_SHAPES,
)
@pytest.mark.parametrize("is_causal", CAUSAL_CHOICES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scaled_dot_product_attention_legacy_backward(
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
    ref_q = utils.to_reference(q, False).detach().requires_grad_(True)
    ref_k = utils.to_reference(k, False).detach().requires_grad_(True)
    ref_v = utils.to_reference(v, False).detach().requires_grad_(True)
    scale = float(1.0 / np.sqrt(head_size))

    # forward
    torch_result = torch_sdpa(
        ref_q, ref_k, ref_v, scale, is_causal, enable_gqa=enable_gqa
    )

    if flag_gems.vendor_name in ["cambricon", "sunrise"]:
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
    ref_dout = torch.randn_like(ref_q)
    torch_result.backward(ref_dout)
    gems_result.backward(ref_dout.to(gems_result.device))
    torch_q_grad = ref_q.grad.clone() if ref_q.grad is not None else None
    torch_k_grad = ref_k.grad.clone() if ref_k.grad is not None else None
    torch_v_grad = ref_v.grad.clone() if ref_v.grad is not None else None
    gems_q_grad = q.grad.clone() if q.grad is not None else None
    gems_k_grad = k.grad.clone() if k.grad is not None else None
    gems_v_grad = v.grad.clone() if v.grad is not None else None

    # NOTE: NaN may arise in the gradients, this behavior aligns with PyTorch's SDPA
    utils.gems_assert_close(gems_q_grad, torch_q_grad, dtype, equal_nan=True)
    utils.gems_assert_close(gems_k_grad, torch_k_grad, dtype, equal_nan=True)

    # dV is more sensitive to softmax recomputation errors in flash attention backward
    # because it lacks the centering term (dP - D) that suppresses errors in dK/dQ.
    # GQA: different float accumulation order across Q heads vs PyTorch kernel
    # bf16: only 8 mantissa bits → largest recomputation error
    # fp16: 11 mantissa bits → moderate error
    is_gqa = enable_gqa and num_q_head != num_kv_head
    if is_gqa:
        if dtype == torch.bfloat16:
            v_atol = 2e-2
        elif dtype == torch.float16:
            v_atol = 4e-3
        else:
            v_atol = 5e-4
    else:
        if dtype == torch.bfloat16:
            v_atol = 5e-3
        elif dtype == torch.float16:
            v_atol = 2e-3
        else:
            v_atol = 3e-4
    utils.gems_assert_close(
        gems_v_grad, torch_v_grad, dtype, equal_nan=True, atol=v_atol
    )


@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize(
    ["batch", "num_head", "q_seq_len", "kv_seq_len"],
    SQUARE_SHAPES,
)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("is_causal", CAUSAL_CHOICES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scaled_dot_product_attention_square_qk_even_mn(
    monkeypatch, batch, num_head, q_seq_len, kv_seq_len, head_size, is_causal, dtype
):
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
    NONSQUARE_SHAPES,
)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("is_causal", [False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scaled_dot_product_attention_nonsquare_qk(
    monkeypatch, batch, num_head, q_seq_len, kv_seq_len, head_size, is_causal, dtype
):
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

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
