import pytest
import torch

from . import performance_utils as base


@pytest.mark.zeros_like
def test_zeros_like():
    bench = base.GenericBenchmark(
        op_name="zeros_like", input_fn=base.unary_input_fn, torch_op=torch.zeros_like
    )
    bench.run()
