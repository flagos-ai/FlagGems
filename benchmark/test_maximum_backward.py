from typing import Generator

import pytest
import torch

from . import base, consts, utils


class MaximumBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            x = utils.generate_tensor_input(shape, dtype, self.device)
            y = utils.generate_tensor_input(shape, dtype, self.device)
            grad_out = torch.randn_like(x)
            yield grad_out, x, y


@pytest.mark.maximum_backward
def test_maximum_backward():
    bench = MaximumBackwardBenchmark(
        op_name="maximum_backward",
        torch_op=torch.ops.aten.maximum_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
