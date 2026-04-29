import pytest
import torch

from . import base, utils


@pytest.mark.ones_like
def test_ones_like():
    bench = base.GenericBenchmark(
        op_name="ones_like", input_fn=utils.unary_input_fn, torch_op=torch.ones_like
    )
    bench.run()
