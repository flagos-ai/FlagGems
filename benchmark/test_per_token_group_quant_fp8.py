import random

import pytest
import torch

import flag_gems

from . import performance_utils as utils


class PerTokenGroupQuantFp8Benchmark(utils.GenericBenchmark):
    def set_more_shapes(self):
        return None


def _input_fn(shape, dtype, device):
    (num_tokens, d, group_size) = shape
    scale_ue8m0 = random.choice([True, False])
    x = torch.rand(num_tokens, d, dtype=dtype, device=device)

    yield (x, group_size, scale_ue8m0)


def torch_per_token_group_quant_fp8_ref(x, group_size, scale_ue8m0):
    dtype = flag_gems.SUPPORTED_FP8_DTYPE
    eps = 1e-10
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / fp8_max
    if scale_ue8m0:
        min_val = torch.tensor(1e-10, dtype=x_s.dtype, device=x_s.device)
        x_s = torch.exp2(torch.ceil(torch.log2(torch.maximum(x_s.abs(), min_val))))
    x_q = (x_ / x_s).clamp(min=fp8_min, max=fp8_max).to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))
    return x_q, x_s


@pytest.mark.per_token_group_quant_fp8
def test_per_token_group_quant_fp8():
    bench = PerTokenGroupQuantFp8Benchmark(
        op_name="per_token_group_quant_fp8",
        input_fn=_input_fn,
        torch_op=torch_per_token_group_quant_fp8_ref,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flag_gems.per_token_group_quant_fp8)
    bench.run()
