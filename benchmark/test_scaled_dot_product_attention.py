import math

import pytest
import torch

import flag_gems

from . import base


class AttentionBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


def sdpa_flash(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )


@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize("dropout_p", [0.0])
@pytest.mark.parametrize("is_causal", [True, False])
def test_scaled_dot_product_attention(monkeypatch, dropout_p, is_causal):
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    def scaled_dot_product_attention_kwargs(shape, dtype, device):
        query = torch.randn(shape, device=device, dtype=dtype)
        key = torch.randn(shape, device=device, dtype=dtype)
        value = torch.randn(shape, device=device, dtype=dtype)
        yield query, key, value, None, dropout_p, is_causal

    bench = AttentionBenchmark(
        op_name="scaled_dot_product_attention",
        input_fn=scaled_dot_product_attention_kwargs,
        torch_op=sdpa_flash,
        gems_op=flag_gems.scaled_dot_product_attention,
        dtypes=[
            torch.float16,
            torch.bfloat16,
        ],
    )
    bench.run()


@pytest.mark.scaled_dot_product_attention_backward
@pytest.mark.xfail(
    reason="Operator bug: backward kernel triggers CUDA illegal memory access"
)
@pytest.mark.parametrize("dropout_p", [0.0])
@pytest.mark.parametrize("is_causal", [True, False])
def test_scaled_dot_product_attention_backward(monkeypatch, dropout_p, is_causal):
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    def scaled_dot_product_attention_backward_kwargs(shape, dtype, device):
        query = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        key = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        value = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        yield query, key, value, None, dropout_p, is_causal

    bench = AttentionBenchmark(
        op_name="scaled_dot_product_attention_backward",
        input_fn=scaled_dot_product_attention_backward_kwargs,
        torch_op=sdpa_flash,
        gems_op=flag_gems.scaled_dot_product_attention,
        dtypes=[
            torch.float16,
            torch.bfloat16,
        ],
        is_backward=True,
    )
    bench.run()


class ScaledDotProductAttentionForwardBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []

    def set_shapes(self, shape_file_path=None):
        self.shapes = []
        for head_size in (64, 128, 256):
            for is_causal in (False, True):
                self.shapes.append(
                    (4, 8, 8, 1024, 1024, head_size, is_causal, False)
                )
        for batch, num_q_head, num_kv_head, q_seq_len, kv_seq_len in (
            (1, 1, 1, 128, 2048),
            (4, 8, 8, 17, 1030),
        ):
            for is_causal in (False, True):
                self.shapes.append(
                    (
                        batch,
                        num_q_head,
                        num_kv_head,
                        q_seq_len,
                        kv_seq_len,
                        128,
                        is_causal,
                        False,
                    )
                )
        for head_size in (64, 128):
            for is_causal in (False, True):
                self.shapes.append(
                    (4, 8, 2, 1024, 1024, head_size, is_causal, True)
                )


def sdpa_forward_input_fn(shape, dtype, device):
    (
        batch,
        num_q_head,
        num_kv_head,
        q_seq_len,
        kv_seq_len,
        head_size,
        is_causal,
        enable_gqa,
    ) = shape
    q = torch.empty(
        (batch, num_q_head, q_seq_len, head_size), device=device, dtype=dtype
    ).uniform_(-0.05, 0.05)
    k = torch.empty(
        (batch, num_kv_head, kv_seq_len, head_size), device=device, dtype=dtype
    ).uniform_(-0.05, 0.05)
    v = torch.empty(
        (batch, num_kv_head, kv_seq_len, head_size), device=device, dtype=dtype
    ).uniform_(-0.05, 0.05)
    scale = float(1.0 / math.sqrt(head_size))
    yield q, k, v, None, 0.0, is_causal, scale, enable_gqa


def torch_sdpa_forward(
    query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
):
    return torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


@pytest.mark.scaled_dot_product_attention_forward
def test_scaled_dot_product_attention_forward(monkeypatch):
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    bench = ScaledDotProductAttentionForwardBenchmark(
        op_name="scaled_dot_product_attention_forward",
        input_fn=sdpa_forward_input_fn,
        torch_op=torch_sdpa_forward,
        gems_op=flag_gems.scaled_dot_product_attention_forward,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()
