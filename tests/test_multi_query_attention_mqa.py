import numpy as np
import pytest
import torch

from flag_gems.fused import multi_query_attention_mqa
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import set_philox_state

from . import accuracy_utils as utils


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


def torch_sdpa_mqa(q, k, v, scale, is_causal=False):
    """Reference MQA implementation using PyTorch's SDPA with GQA enabled."""
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=scale, is_causal=is_causal, enable_gqa=True
    )


@pytest.mark.multi_query_attention_mqa
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("num_q_head", [8, 16, 32])
@pytest.mark.parametrize("q_seq_len", [128, 256])
@pytest.mark.parametrize("kv_seq_len", [128, 256])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
# MQA attention follows the generated fp16/bf16 coverage; not all utils.FLOAT_DTYPES are valid.
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_multi_query_attention_mqa(
    batch, num_q_head, q_seq_len, kv_seq_len, head_size, is_causal, dtype
):
    """Test Multi-Query Attention (MQA) accuracy.

    In MQA, key and value have only 1 head while query has multiple heads.
    """
    device = torch_device_fn.current_device()

    # MQA: num_kv_head = 1
    num_kv_head = 1
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

    # Reference: PyTorch's SDPA with GQA enabled (equivalent to MQA when KV has 1 head)
    ref_result = torch_sdpa_mqa(ref_q, ref_k, ref_v, scale, is_causal)

    # GEMS implementation
    gems_result = multi_query_attention_mqa(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
    )

    # Use appropriate tolerance based on dtype
    if dtype == torch.float32:
        utils.gems_assert_close(gems_result, ref_result, dtype=dtype, atol=1e-5)
    elif dtype == torch.float16:
        utils.gems_assert_close(gems_result, ref_result, dtype=dtype, atol=1e-3)
    else:  # bfloat16
        utils.gems_assert_close(gems_result, ref_result, dtype=dtype, atol=2e-2)
