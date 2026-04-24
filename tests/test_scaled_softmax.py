import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    ATTN_HEADS = [2]
else:
    ATTN_HEADS = [2, 4, 8, 16, 32]

# Make sure every thread has same seed.
random.seed(time.time() // 100)


@pytest.mark.scaled_softmax_forward
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("attn_heads", ATTN_HEADS)
@pytest.mark.parametrize("query_seq_len", [64, 128])
@pytest.mark.parametrize("key_seq_len", [128, 256, 512, 1024])
@pytest.mark.parametrize("scale_factor", [0.1])
def test_scaled_softmax_forward(
    batch_size, attn_heads, query_seq_len, key_seq_len, scale_factor, dtype
):
    try:
        from transformer_engine.pytorch import cpp_extensions as tex
    except ImportError:
        pytest.skip("transformer_engine_torch is not available, skipping accuracy test")

    s = torch.randn(
        (batch_size, attn_heads, query_seq_len, key_seq_len),
        dtype=dtype,
        device=flag_gems.device,
    )

    p_ref = tex.scaled_softmax_forward(s, scale_factor)
    p_ref = utils.to_reference(p_ref)
    with flag_gems.use_gems():
        p = flag_gems.scaled_softmax_forward(s, scale_factor)

    utils.gems_assert_close(p, p_ref, dtype, equal_nan=True)


@pytest.mark.scaled_softmax_backward
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("attn_heads", ATTN_HEADS)
@pytest.mark.parametrize("query_seq_len", [64, 128])
@pytest.mark.parametrize("key_seq_len", [128, 256, 512, 1024])
@pytest.mark.parametrize("scale_factor", [0.1])
def test_scaled_softmax_backward(
    batch_size, attn_heads, query_seq_len, key_seq_len, scale_factor, dtype
):
    try:
        from transformer_engine.pytorch import cpp_extensions as tex
    except ImportError:
        pytest.skip("transformer_engine_torch is not available, skipping accuracy test")

    out_grad = torch.randn(
        (batch_size, attn_heads, query_seq_len, key_seq_len),
        dtype=dtype,
        device=flag_gems.device,
    )
    s = torch.randn(
        (batch_size, attn_heads, query_seq_len, key_seq_len),
        dtype=dtype,
        device=flag_gems.device,
    )

    p_ref = tex.scaled_softmax_forward(s, scale_factor)
    with flag_gems.use_gems():
        p = flag_gems.scaled_softmax_forward(s, scale_factor)
        in_grad = flag_gems.scaled_softmax_backward(out_grad, p, scale_factor)
    in_grad_ref = tex.scaled_softmax_backward(out_grad, p_ref, scale_factor)
    in_grad_ref = utils.to_reference(in_grad_ref)

    utils.gems_assert_close(
        in_grad, in_grad_ref, dtype, equal_nan=True, reduce_dim=s.shape[-1]
    )
