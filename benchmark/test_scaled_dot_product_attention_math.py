import pytest
import torch

import flag_gems

from . import base, consts

# Call the op through torch.ops.aten instead of the private torch._... alias so
# flag_gems.use_gems() intercepts it and it stays version-stable.
sdpa_math = torch.ops.aten._scaled_dot_product_attention_math


class ScaledDotProductAttentionMathBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Override set_shapes (not set_more_shapes) so CI's generic core_shapes
        # can't replace these 4D attention shapes. (batch, num_head, seq_len,
        # head_size); head_size fixed at 64 (a common attention head dim) while
        # batch/heads/seq_len vary to sweep sizes.
        self.shapes = [
            (2, 4, 64, 64),
            (2, 8, 128, 64),
            (4, 8, 256, 64),
            (2, 8, 512, 64),
            (1, 8, 1024, 64),
        ]


class ScaledDotProductAttentionMathGqaBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # (batch, query_heads, kv_heads, query_len, kv_len, head_size).
        # Compare GQA and MQA for equal-length prefill and single-token decode.
        self.shapes = [
            (2, 8, 2, 128, 128, 64),
            (2, 8, 1, 128, 128, 64),
            (2, 8, 2, 1, 1024, 64),
            (2, 8, 1, 1, 1024, 64),
        ]


def scaled_dot_product_attention_math_input_fn(shape, dtype, device):
    query = torch.randn(shape, dtype=dtype, device=device)
    key = torch.randn(shape, dtype=dtype, device=device)
    value = torch.randn(shape, dtype=dtype, device=device)
    yield query, key, value


def scaled_dot_product_attention_math_gqa_input_fn(shape, dtype, device):
    batch, query_heads, kv_heads, query_len, kv_len, head_size = shape
    query = torch.randn(
        (batch, query_heads, query_len, head_size), dtype=dtype, device=device
    )
    key = torch.randn((batch, kv_heads, kv_len, head_size), dtype=dtype, device=device)
    value = torch.randn(
        (batch, kv_heads, kv_len, head_size), dtype=dtype, device=device
    )
    yield query, key, value, {"enable_gqa": True}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(flag_gems.device == "cpu", reason="Unsupported in CPU mode")
@pytest.mark.scaled_dot_product_attention_math
def test_scaled_dot_product_attention_math():
    bench = ScaledDotProductAttentionMathBenchmark(
        op_name="scaled_dot_product_attention_math",
        input_fn=scaled_dot_product_attention_math_input_fn,
        torch_op=sdpa_math,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems._scaled_dot_product_attention_math)
    bench.run()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(flag_gems.device == "cpu", reason="Unsupported in CPU mode")
@pytest.mark.scaled_dot_product_attention_math
def test_scaled_dot_product_attention_math_gqa():
    bench = ScaledDotProductAttentionMathGqaBenchmark(
        op_name="scaled_dot_product_attention_math",
        input_fn=scaled_dot_product_attention_math_gqa_input_fn,
        torch_op=sdpa_math,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems._scaled_dot_product_attention_math)
    bench.run()
