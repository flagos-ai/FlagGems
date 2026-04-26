import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark, binary_input_fn


@pytest.mark.gelu_and_mul
def test_gelu_and_mul():
    def torch_op(x, y):
        return torch.mul(torch.nn.functional.gelu(x), y)

    bench = GenericBenchmark(
        input_fn=binary_input_fn,
        op_name="gelu_and_mul",
        torch_op=torch_op,
        gems_op=flag_gems.gelu_and_mul,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
