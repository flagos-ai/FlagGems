from typing import Generator

import pytest
import torch

from . import base, consts, utils


class PreluBackwardBenchmark(base.Benchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            # Test with both scalar and per-channel weight
            x = utils.generate_tensor_input(shape, dtype, self.device)
            grad_output = torch.randn_like(x)

            # Scalar weight
            scalar_weight = torch.tensor(0.25, dtype=dtype, device=self.device)
            yield grad_output, x, scalar_weight

            # Per-channel weight (only if shape has channel dim)
            if len(shape) >= 2:
                num_channels = shape[1]
                per_channel_weight = torch.randn(num_channels, dtype=dtype, device=self.device)
                yield grad_output, x, per_channel_weight


@pytest.mark.prelu_backward
def test_prelu_backward():
    bench = PreluBackwardBenchmark(
        op_name="prelu_backward",
        torch_op=torch.ops.aten.prelu_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
