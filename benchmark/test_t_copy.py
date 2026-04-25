from typing import Generator

import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


class TCopyBenchmark(utils.Benchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            if len(shape) == 2:
                inp = utils.generate_tensor_input(shape, cur_dtype, self.device)
                yield inp,


@pytest.mark.t_copy
def test_t_copy():
    bench = TCopyBenchmark(
        op_name="t_copy",
        torch_op=torch.ops.aten.t_copy,
        dtypes=attr_utils.FLOAT_DTYPES,
    )
    bench.run()
