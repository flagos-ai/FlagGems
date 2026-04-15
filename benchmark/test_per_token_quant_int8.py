import pytest
import torch

import flag_gems

from . import base


class PerTokenQuantInt8Benchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


def _input_fn(shape, dtype, device):
    num_tokens, d = shape
    x = torch.rand(num_tokens, d, dtype=dtype, device=device)
    yield (x,)


def torch_per_token_quant_int8_ref(x):
    x_2d = x.float().reshape(-1, x.shape[-1]).contiguous()
    int8_min = torch.iinfo(torch.int8).min
    int8_max = torch.iinfo(torch.int8).max
    scale = x_2d.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-10)
    scale = scale / int8_max
    x_q = torch.round(x_2d / scale).clamp(min=int8_min, max=int8_max).to(torch.int8)
    x_q = x_q.reshape(x.shape)
    scale = scale.reshape(*x.shape[:-1], 1)
    return x_q, scale


@pytest.mark.per_token_quant_int8
def test_per_token_quant_int8():
    bench = PerTokenQuantInt8Benchmark(
        op_name="per_token_quant_int8",
        input_fn=_input_fn,
        torch_op=torch_per_token_quant_int8_ref,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flag_gems.per_token_quant_int8)
    bench.run()

