import os

import pytest
import torch

import flag_gems

from . import base, consts


class ScaledDotProductAttentionMathBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        self.shapes = [
            (2, 4, 64, 64),
            (2, 8, 128, 64),
            (4, 8, 256, 64),
            (2, 8, 512, 64),
            (1, 8, 1024, 64),
        ]
        return self.shapes


@pytest.mark.scaled_dot_product_attention_math
@pytest.mark.parametrize("is_causal", [True, False])
def test_perf_scaled_dot_product_attention_math(monkeypatch, is_causal):
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    def scaled_dot_product_attention_math_kwargs(shape, dtype, device):
        query = torch.randn(shape, device=device, dtype=dtype)
        key = torch.randn(shape, device=device, dtype=dtype)
        value = torch.randn(shape, device=device, dtype=dtype)
        yield query, key, value, None, 0.0, is_causal

    bench = ScaledDotProductAttentionMathBenchmark(
        op_name="_scaled_dot_product_attention_math",
        input_fn=scaled_dot_product_attention_math_kwargs,
        torch_op=torch._scaled_dot_product_attention_math,
        dtypes=[
            torch.float16,
            torch.bfloat16,
        ],
    )
    bench.set_gems(flag_gems._scaled_dot_product_attention_math)
    bench.run()
