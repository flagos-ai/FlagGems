from typing import Generator

import pytest
import torch

from . import base, consts, utils


class MinimumBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            x = utils.generate_tensor_input(shape, dtype, self.device)
            y = utils.generate_tensor_input(shape, dtype, self.device)
            grad_out = torch.randn_like(x)
            yield grad_out, x, y


@pytest.mark.minimum_backward
def test_minimum_backward():
    bench = MinimumBackwardBenchmark(
        op_name="minimum_backward",
        torch_op=torch.ops.aten.minimum_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
