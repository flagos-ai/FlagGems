from typing import Generator

import pytest
import torch

from . import base, consts, utils


class SoftshrinkBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            grad_out = torch.randn_like(inp)
            lambd = 0.5
            yield grad_out, inp, lambd


@pytest.mark.softshrink_backward
def test_softshrink_backward():
    bench = SoftshrinkBackwardBenchmark(
        op_name="softshrink_backward",
        torch_op=torch.ops.aten.softshrink_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
