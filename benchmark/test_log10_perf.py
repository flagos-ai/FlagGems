import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import (
    GenericBenchmarkExcluse1D,
    unary_input_fn,
)


@ pytest.mark.log10
def test_perf_log10():
    bench = GenericBenchmarkExcluse1D(
        input_fn=unary_input_fn, op_name="log10", torch_op=torch.log10
    )
    bench.run()
