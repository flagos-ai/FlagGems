import pytest
import torch

from . import performance_utils as base


@pytest.mark.rand_like
def test_rand_like():
    bench = base.GenericBenchmark(
        op_name="rand_like", input_fn=base.unary_input_fn, torch_op=torch.rand_like
    )
    bench.run()
