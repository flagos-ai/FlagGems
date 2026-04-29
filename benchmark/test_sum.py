import pytest
import torch

from . import base, consts


@pytest.mark.sum
def test_sum():
    bench = base.UnaryReductionBenchmark(
        op_name="sum", torch_op=torch.sum, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
