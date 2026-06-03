from typing import Generator

import pytest
import torch

from . import base, consts, utils


class CeluBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            grad_out = torch.randn_like(inp)
            alpha = 1.0
            yield grad_out, inp, alpha


@pytest.mark.celu_backward
def test_celu_backward():
    bench = CeluBackwardBenchmark(
        op_name="celu_backward",
        torch_op=torch.ops.aten.celu_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
