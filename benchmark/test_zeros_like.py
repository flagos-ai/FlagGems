import pytest
import torch

from . import base, utils


@pytest.mark.zeros_like
def test_zeros_like():
    bench = base.GenericBenchmark(
        op_name="zeros_like", input_fn=utils.unary_input_fn, torch_op=torch.zeros_like
    )
    bench.run()
