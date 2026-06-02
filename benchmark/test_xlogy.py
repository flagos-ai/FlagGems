from typing import Generator

import pytest
import torch

from . import base, consts, utils


class XlogyBenchmark(base.Benchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            x = utils.generate_tensor_input(shape, dtype, self.device)
            y = torch.randn(shape, dtype=dtype, device=self.device).abs() + 0.1
            yield x, y


@pytest.mark.xlogy
def test_xlogy():
    bench = XlogyBenchmark(
        op_name="xlogy",
        torch_op=lambda x, y: torch.ops.aten.xlogy(x, y),
        dtypes=consts.FLOAT_DTYPES,
        shapes=[(256,), (1024,), (4096,), (16384,)],
    )
    bench.run()
