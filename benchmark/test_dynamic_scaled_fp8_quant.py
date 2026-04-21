import pytest
import torch

import flag_gems

from . import base


class DynamicScaledFp8QuantBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


def _input_fn(shape, dtype, device):
    num_tokens, d = shape
    x = torch.rand(num_tokens, d, dtype=dtype, device=device)
    yield (x,)


def torch_dynamic_scaled_fp8_quant_ref(x):
    dtype = flag_gems.SUPPORTED_FP8_DTYPE
    assert x.ndim == 2 and x.stride(-1) == 1
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    fp8_min = float(torch.finfo(torch.float8_e4m3fn).min)
    min_scale = 1.0 / (fp8_max * 512.0)

    scale = (x.abs().max().clamp(min=1e-10).to(torch.float32) / fp8_max).clamp(
        min=min_scale
    )
    x_q = (x.float() / scale).clamp(min=fp8_min, max=fp8_max).to(dtype)
    return x_q, scale.reshape(1)


@pytest.mark.dynamic_scaled_fp8_quant
def test_dynamic_scaled_fp8_quant():
    bench = DynamicScaledFp8QuantBenchmark(
        op_name="dynamic_scaled_fp8_quant",
        input_fn=_input_fn,
        torch_op=torch_dynamic_scaled_fp8_quant_ref,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flag_gems.dynamic_scaled_fp8_quant)
    bench.run()

