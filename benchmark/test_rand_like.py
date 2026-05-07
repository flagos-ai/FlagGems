import pytest
import torch

from . import base, utils


@pytest.mark.rand_like
def test_rand_like():
    bench = base.GenericBenchmark(
        op_name="rand_like", input_fn=utils.unary_input_fn, torch_op=torch.rand_like
    )
    bench.run()
