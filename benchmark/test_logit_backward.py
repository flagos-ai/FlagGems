from typing import Generator

import pytest
import torch

from . import base, consts, utils


class LogitBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            # Use values in (0, 1) range for logit
            inp = torch.clamp(inp, 0.1, 0.9)
            grad_out = torch.randn_like(inp)
            eps = 1e-5
            yield grad_out, inp, eps


@pytest.mark.logit_backward
def test_logit_backward():
    bench = LogitBackwardBenchmark(
        op_name="logit_backward",
        torch_op=torch.ops.aten.logit_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
