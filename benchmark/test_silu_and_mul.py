import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark, binary_input_fn


@pytest.mark.silu_and_mul
def test_silu_and_mul():
    def torch_op(x, y):
        return torch.mul(torch.nn.functional.silu(x), y)

    bench = GenericBenchmark(
        input_fn=binary_input_fn,
        op_name="silu_and_mul",
        gems_op=flag_gems.silu_and_mul,
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
