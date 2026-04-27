import pytest
import torch

from . import performance_utils as base


@pytest.mark.randn_like
def test_randn_like():
    bench = base.GenericBenchmark(
        op_name="randn_like", input_fn=base.unary_input_fn, torch_op=torch.randn_like
    )
    bench.run()
