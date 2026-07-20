import math

import pytest
import torch

from flag_gems.fused import multi_query_attention_mqa

from . import base


@pytest.mark.multi_query_attention_mqa
# MQA attention follows the generated fp16/bf16 coverage; not all consts.FLOAT_DTYPES are valid.
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_multi_query_attention_mqa(dtype):
    """Benchmark for Multi-Query Attention (MQA).

    In MQA, key and value have a single head (num_kv_head=1),
    while query can have multiple heads (num_q_head > 1).
    """

    def mqa_kwargs(shape, dtype, device):
        # shape is (batch, num_q_head, seq_len, head_size)
        batch, num_q_head, seq_len, head_size = shape
        # For MQA: KV has 1 head
        kv_head = 1
        query = torch.randn(shape, device=device, dtype=dtype)
        key = torch.randn(
            (batch, kv_head, seq_len, head_size), device=device, dtype=dtype
        )
        value = torch.randn(
            (batch, kv_head, seq_len, head_size), device=device, dtype=dtype
        )
        scale = float(1.0 / math.sqrt(head_size))
        yield query, key, value, None, 0.0, False, scale

    def torch_mqa_ref(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
    ):
        # Reference using PyTorch's SDPA with enable_gqa=True (MQA when KV has 1 head)
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=True,
        )

    bench = base.GenericBenchmark4DOnly(
        op_name="multi_query_attention_mqa",
        input_fn=mqa_kwargs,
        torch_op=torch_mqa_ref,
        dtypes=[dtype],
    )
    bench.set_gems(multi_query_attention_mqa)
    bench.run()
