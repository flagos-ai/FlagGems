from typing import Generator

import pytest
import torch

from . import base, consts, utils


class HeavisideBenchmark(base.Benchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            x = utils.generate_tensor_input(shape, dtype, self.device)
            values = torch.zeros(shape, dtype=dtype, device=self.device)
            yield x, values


@pytest.mark.heaviside
def test_heaviside():
    bench = HeavisideBenchmark(
        op_name="heaviside",
        torch_op=lambda x, v: torch.ops.aten.heaviside(x, v),
        dtypes=consts.FLOAT_DTYPES,
        shapes=[(256,), (1024,), (4096,), (16384,)],
    )
    bench.run()
