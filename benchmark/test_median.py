import pytest
import torch

from . import base, consts


@pytest.mark.median
def test_median():
    bench = base.UnaryReductionBenchmark(
        op_name="median", torch_op=torch.median, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.median_dim
def test_median_dim():
    bench = base.UnaryReductionBenchmark(
        op_name="median_dim", torch_op=torch.median, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
