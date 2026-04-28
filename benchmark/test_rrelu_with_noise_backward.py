from typing import Generator

import pytest
import torch

from . import base, consts, utils


class RreluWithNoiseBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            grad_out = torch.randn_like(inp)
            noise = torch.rand_like(inp)
            lower = 0.125
            upper = 1.0 / 3.0
            training = True
            self_is_result = False
            yield grad_out, inp, noise, lower, upper, training, self_is_result


@pytest.mark.rrelu_with_noise_backward
def test_rrelu_with_noise_backward():
    bench = RreluWithNoiseBackwardBenchmark(
        op_name="rrelu_with_noise_backward",
        torch_op=torch.ops.aten.rrelu_with_noise_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
