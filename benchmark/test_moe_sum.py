import pytest
import torch

import flag_gems

from . import base, consts


def _input_fn(shape, dtype, device):
    shape = (shape[0], 1, shape[1]) if len(shape) == 2 else shape
    num_tokens, topk, hidden_size = shape
    input_tensor = torch.randn(
        num_tokens,
        topk,
        hidden_size,
        dtype=dtype,
        device=device,
        requires_grad=False,
    )

    output_tensor = torch.empty(
        num_tokens, hidden_size, dtype=dtype, device=device, requires_grad=False
    )
    yield input_tensor, output_tensor


@pytest.mark.moe_sum
def test_moe_sum():
    def torch_op(input_tensor, output_tensor):
        output_tensor.copy_(input_tensor.sum(dim=1))

    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_input_fn,
        op_name="moe_sum",
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.moe_sum)
    bench.run()
