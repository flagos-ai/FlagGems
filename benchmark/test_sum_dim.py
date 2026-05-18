import pytest
import torch

from . import base, consts


@pytest.mark.sum_dim
def test_sum_dim():
    bench = base.UnaryReductionBenchmark(
        op_name="sum_dim", torch_op=torch.sum, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
