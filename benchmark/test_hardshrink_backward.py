from typing import Generator

import pytest
import torch

from . import base, consts, utils


class HardshrinkBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            grad_out = torch.randn_like(inp)
            lambd = 0.5
            yield grad_out, inp, lambd


@pytest.mark.hardshrink_backward
def test_hardshrink_backward():
    bench = HardshrinkBackwardBenchmark(
        op_name="hardshrink_backward",
        torch_op=torch.ops.aten.hardshrink_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
