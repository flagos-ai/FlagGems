import pytest
import torch

from . import base, consts, utils


@pytest.mark.dist
def test_dist():
    bench = base.GenericBenchmark(
        op_name="dist",
        input_fn=utils.binary_input_fn,
        torch_op=torch.dist,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
