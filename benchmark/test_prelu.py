from typing import Generator

import pytest
import torch

from . import attri_util as consts
from . import performance_utils as base
from . import utils


class PreluBenchmark(base.Benchmark):
    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            x = utils.generate_tensor_input(shape, dtype, self.device)
            if len(shape) == 1:
                w = torch.randn((), dtype=dtype, device=self.device)
            else:
                w = torch.randn((shape[1],), dtype=dtype, device=self.device)
            yield x, w


@pytest.mark.prelu
def test_prelu():
    bench = PreluBenchmark(
        op_name="prelu",
        torch_op=torch.ops.aten.prelu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
