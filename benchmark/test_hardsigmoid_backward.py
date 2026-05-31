from typing import Generator

import pytest
import torch

from . import base, consts, utils


class HardsigmoidBackwardBenchmark(base.Benchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            input = utils.generate_tensor_input(shape, dtype, self.device, requires_grad=True)
            yield input

    def get_bw_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            input = utils.generate_tensor_input(shape, dtype, self.device, requires_grad=True)
            grad_output = utils.generate_tensor_input(shape, dtype, self.device)
            yield grad_output, input


@pytest.mark.hardsigmoid_backward
def test_hardsigmoid_backward():
    bench = HardsigmoidBackwardBenchmark(
        op_name="hardsigmoid_backward",
        torch_op=lambda x: torch.ops.aten.hardsigmoid(x).backward(torch.ones_like(torch.ops.aten.hardsigmoid(x))),
        dtypes=consts.FLOAT_DTYPES,
        shapes=[(256,), (1024,), (4096,)],
    )
    bench.run()
