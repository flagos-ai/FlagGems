import pytest
import torch

from . import base, consts


class HardsigmoidBackwardBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return None


def hardsigmoid_backward_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    yield grad_output, inp


@pytest.mark.hardsigmoid_backward
def test_hardsigmoid_backward():
    bench = HardsigmoidBackwardBenchmark(
        input_fn=hardsigmoid_backward_input_fn,
        op_name="hardsigmoid_backward",
        torch_op=torch.ops.aten.hardsigmoid_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
