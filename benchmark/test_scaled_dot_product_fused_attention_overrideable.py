import math

import pytest
import torch

import flag_gems

from . import base, consts


class ScaledDotProductFusedAttentionOverrideableBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [
        (2, 8, 64, 64),
        (4, 8, 128, 64),
        (2, 8, 128, 128),
        (1, 16, 256, 64),
    ]
    DEFAULT_SHAPE_DESC = "B, H, S, D"


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
    if flag_gems.vendor_name == "metax":
        # The generic fused comparator exceeds the shared-memory limit on
        # some master core shapes. Keep the same attention math on device.
        scores = (query.float() @ key.float().transpose(-2, -1)) * (
            query.shape[-1] ** -0.5 if scale is None else scale
        )
        if is_causal:
            rows, cols = scores.shape[-2:]
            mask = torch.ones(rows, cols, device=query.device, dtype=torch.bool)
            scores = scores.masked_fill(~mask.tril(), float("-inf"))
        weights = scores.softmax(dim=-1)
        output = (weights @ value.float()).to(query.dtype)
        logsumexp_base2 = scores.logsumexp(dim=-1) * math.log2(math.e)
        return output, logsumexp_base2

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
