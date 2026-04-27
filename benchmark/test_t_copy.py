from typing import Generator

import pytest
import torch

from . import attri_util as consts
from . import performance_utils as base
from . import utils


class TCopyBenchmark(base.Benchmark):
    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            if len(shape) == 2:
                inp = utils.generate_tensor_input(shape, dtype, self.device)
                yield inp,


@pytest.mark.t_copy
def test_t_copy():
    bench = TCopyBenchmark(
        op_name="t_copy",
        torch_op=torch.ops.aten.t_copy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
