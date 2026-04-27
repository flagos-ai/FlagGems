import pytest
import torch

from . import performance_utils as base
from . import utils


@pytest.mark.vector_norm
def test_vector_norm():
    bench = base.GenericBenchmarkExcluse1D(
        op_name="vector_norm",
        input_fn=utils.unary_input_fn,
        torch_op=torch.linalg.vector_norm,
    )
    bench.run()
