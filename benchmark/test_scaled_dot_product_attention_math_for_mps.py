import pytest
import torch

import flag_gems

from . import base, consts


class AttentionBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(2, 8, 64, 64), (4, 8, 128, 64), (2, 8, 128, 128), (1, 16, 256, 64)]


def _case_fn_for(is_causal):
    def case_fn(shape, dtype):
        del dtype
        dims = tuple(shape)
        yield base.BenchmarkCasePlan(shape={"query": list(dims), "key": list(dims), "value": list(dims)}, params={"is_causal": is_causal}, builder_args=(dims,))
    return case_fn


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    query, key, value = (torch.randn(shape, device=device, dtype=dtype) for _ in range(3))
    return query, key, value, None, 0.0, plan.params["is_causal"], None


def torch_ref(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, dropout_mask=None, scale=None):
    scale_factor = 1.0 / (query.shape[-1] ** 0.5) if scale is None else scale
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    if is_causal:
        seq_q, seq_k = query.shape[-2], key.shape[-2]
        mask = torch.ones(seq_q, seq_k, dtype=torch.bool, device=query.device).tril(diagonal=seq_k - seq_q)
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, value), weights


@pytest.mark.scaled_dot_product_attention_math_for_mps
@pytest.mark.underscore_scaled_dot_product_attention_math_for_mps
@pytest.mark.parametrize("is_causal", [False, True])
def test_scaled_dot_product_attention_math_for_mps(is_causal):
    bench = AttentionBenchmark(op_name="_scaled_dot_product_attention_math_for_mps", case_fn=_case_fn_for(is_causal), build_inputs_fn=_build_inputs_fn, torch_op=torch_ref, dtypes=consts.FLOAT_DTYPES)
    bench.set_gems(flag_gems._scaled_dot_product_attention_math_for_mps)
    bench.run()
