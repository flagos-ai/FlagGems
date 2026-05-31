from typing import Generator

import pytest
import torch

from . import base, consts, utils


class HardshrinkBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            lambd = 0.5
            yield inp, lambd


@pytest.mark.hardshrink
def test_hardshrink():
    bench = HardshrinkBenchmark(
        op_name="hardshrink",
        torch_op=torch.ops.aten.hardshrink,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
