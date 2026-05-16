import pytest
import torch

from . import base, consts


@pytest.mark.max
def test_max():
    bench = base.UnaryReductionBenchmark(
        op_name="max", torch_op=torch.max, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.max_dim
def test_max_dim():
    bench = base.UnaryReductionBenchmark(
        op_name="max_dim", torch_op=torch.max, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
