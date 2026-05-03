import pytest
import torch

import flag_gems

from . import base, consts

vendor_name = flag_gems.vendor_name


class LerpBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (N, C, H, W).
        return []


def lerp_input_fn(shape, dtype, device):
    input = torch.randn(*shape, device=device, dtype=dtype)
    end = input + 10
    weight = torch.randn(*shape, device=device, dtype=dtype)
    yield {"input": input, "end": end, "weight": weight},


@pytest.mark.lerp_tensor
def test_lerp_tensor():
    bench = LerpBenchmark(
        input_fn=lerp_input_fn,
        op_name="lerp_tensor",
        torch_op=torch.lerp,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


@pytest.mark.lerp_tensor_
def test_lerp_tensor_inplace():
    bench = LerpBenchmark(
        input_fn=lerp_input_fn,
        op_name="lerp_tensor_",
        torch_op=lambda input, end, weight: input.lerp_(end, weight),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
