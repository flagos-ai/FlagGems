import pytest
import torch

import flag_gems

from . import base, consts


class ScaledDotProductFusedAttentionOverrideableBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (2, 8, 64, 64),
            (4, 8, 128, 64),
            (2, 8, 128, 128),
            (1, 16, 256, 64),
        ]


def _case_fn(shape, dtype):
    dims = tuple(shape)
    for is_causal in (False, True):
        yield base.BenchmarkCasePlan(
            shape={"query": list(dims), "key": list(dims), "value": list(dims)},
            params={"is_causal": is_causal},
            builder_args=(dims,),
        )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    query, key, value = (
        torch.randn(shape, device=device, dtype=dtype) for _ in range(3)
    )
    return (
        query,
        key,
        value,
        None,
        0.0,
        plan.params["is_causal"],
        False,
        {"scale": None},
    )


def torch_ref(
    query,
    key,
    value,
    attn_bias=None,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
    scale=None,
):
    return flag_gems.ops.scaled_dot_product_attention_forward(
        query, key, value, is_causal=is_causal
    )


@pytest.mark.underscore_scaled_dot_product_fused_attention_overrideable
def test_scaled_dot_product_fused_attention_overrideable():
    def stock_attention(*args, **kwargs):
        return flag_gems._scaled_dot_product_fused_attention_overrideable(*args, **kwargs)

    bench = ScaledDotProductFusedAttentionOverrideableBenchmark(
        op_name="_scaled_dot_product_fused_attention_overrideable",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(stock_attention)
    bench.run()
