from typing import Generator

import pytest
import torch

from . import base, consts, utils


class RreluWithNoiseFunctionalBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            noise = torch.rand_like(inp)
            lower = 0.125
            upper = 1.0 / 3.0
            training = True
            generator = None
            yield inp, noise, lower, upper, training, generator


@pytest.mark.rrelu_with_noise_functional
def test_rrelu_with_noise_functional():
    bench = RreluWithNoiseFunctionalBenchmark(
        op_name="rrelu_with_noise_functional",
        torch_op=torch.ops.aten.rrelu_with_noise_functional,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
