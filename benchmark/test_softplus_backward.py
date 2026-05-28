from typing import Generator

import pytest
import torch

from . import base, consts, utils


class SoftplusBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            grad_out = torch.randn_like(inp)
            beta = 1.0
            threshold = 20.0
            yield grad_out, inp, beta, threshold


@pytest.mark.softplus_backward
def test_softplus_backward():
    bench = SoftplusBackwardBenchmark(
        op_name="softplus_backward",
        torch_op=torch.ops.aten.softplus_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
