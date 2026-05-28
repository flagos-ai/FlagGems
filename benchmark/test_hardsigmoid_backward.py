from typing import Generator

import pytest
import torch

from . import base, consts, utils


class HardsigmoidBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            grad_out = torch.randn_like(inp)
            yield grad_out, inp


@pytest.mark.hardsigmoid_backward
def test_hardsigmoid_backward():
    bench = HardsigmoidBackwardBenchmark(
        op_name="hardsigmoid_backward",
        torch_op=torch.ops.aten.hardsigmoid_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
