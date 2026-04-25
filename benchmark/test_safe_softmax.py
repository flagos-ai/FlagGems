from typing import Generator

import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


class SafeSoftmaxBenchmark(utils.Benchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, cur_dtype, self.device)
            yield inp, -1, None


@pytest.mark.safe_softmax
def test_safe_softmax():
    bench = SafeSoftmaxBenchmark(
        op_name="_safe_softmax",
        torch_op=torch.ops.aten._safe_softmax,
        dtypes=attr_utils.FLOAT_DTYPES,
    )

    bench.run()
