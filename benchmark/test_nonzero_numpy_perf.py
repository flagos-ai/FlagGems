import pytest
import torch

from benchmark.attri_util import BOOL_DTYPES, FLOAT_DTYPES, INT_DTYPES
from benchmark.performance_utils import GenericBenchmark, unary_input_fn


@pytest.mark.nonzero_numpy
def test_perf_nonzero_numpy():
    bench = GenericBenchmark(
        input_fn=unary_input_fn,
        op_name="nonzero_numpy",
        torch_op=torch.ops.aten.nonzero_numpy,
        dtypes=FLOAT_DTYPES + INT_DTYPES + BOOL_DTYPES,
    )
    bench.run()
