from typing import Generator

import pytest
import torch

from . import base, consts, utils

# SELU constants from PyTorch
ALPHA = 1.6732632423543772848170429916717
SCALE = 1.0507009873554804934193349852946


class SeluBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            grad_out = torch.randn_like(inp)
            yield grad_out, inp, ALPHA, SCALE


@pytest.mark.selu_backward
def test_selu_backward():
    bench = SeluBackwardBenchmark(
        op_name="selu_backward",
        torch_op=torch.ops.aten.selu_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
